from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from core.ml.prob_calibration import CalibratedProbabilityModel, fit_calibrated_probability_model
from core.ml.ranking_metrics import daily_ranking_metrics
from core.ml.splitters.purged_kfold import PurgedKFoldEmbargo


@dataclass
class ListNetConfig:
    model_id: str = "alpha_listnet_v2"
    horizon_days: int = 21
    tau: float = 1.0
    lambda_l2: float = 1e-3
    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    epochs: int = 50
    batch_dates: int = 8
    seed: int = 42
    cv_splits: int = 5
    embargo_days: int = 5
    purge_horizon_days: int = 21
    early_stop_patience: int = 5


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = z - float(np.max(z))
    e = np.exp(z)
    denom = float(np.sum(e))
    if denom <= 0 or not np.isfinite(denom):
        return np.full_like(e, 1.0 / max(len(e), 1), dtype=float)
    return e / denom


def percentile_gain(y_excess: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(y_excess, dtype=float))
    n = len(s)
    if n <= 1:
        return np.zeros(n, dtype=float)
    r = s.rank(method="average", ascending=True)
    g = (r - 1.0) / float(n - 1)
    return g.to_numpy(dtype=float)


def listnet_loss_and_grad(
    x: np.ndarray,
    y_gain: np.ndarray,
    w: np.ndarray,
    tau: float = 1.0,
    lambda_l2: float = 1e-3,
) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    x_arr = np.asarray(x, dtype=float)
    y_gain_arr = np.asarray(y_gain, dtype=float)
    w_arr = np.asarray(w, dtype=float)

    logits_true = y_gain_arr / float(tau)
    p_true = _softmax(logits_true)

    score = x_arr @ w_arr
    logits_pred = score / float(tau)
    p_pred = _softmax(logits_pred)

    ce = -float(np.sum(p_true * np.log(np.clip(p_pred, 1e-12, None))))
    l2 = 0.5 * float(lambda_l2) * float(np.sum(w_arr * w_arr))
    loss = ce + l2

    g = (p_pred - p_true) / float(tau)
    grad = x_arr.T @ g + float(lambda_l2) * w_arr

    return float(loss), np.asarray(grad, dtype=float), {"p_true": p_true, "p_pred": p_pred, "score": score}


def _fill_and_add_missing_count(x_frame: pd.DataFrame) -> pd.DataFrame:
    x = x_frame.copy()
    missing_count = x.isna().sum(axis=1).astype(float)
    x = x.fillna(0.0)
    x["missing_count"] = missing_count
    return x


def _standardize_fit_transform(x_frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = x_frame.astype(float)
    means = x.mean(axis=0).to_numpy(dtype=float)
    stds = x.std(axis=0, ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    z = (x.to_numpy(dtype=float) - means) / stds
    return pd.DataFrame(z, columns=x.columns, index=x.index), means, stds


def _standardize_transform(x_frame: pd.DataFrame, means: np.ndarray, stds: np.ndarray) -> pd.DataFrame:
    x = x_frame.astype(float).to_numpy(dtype=float)
    st = np.where(np.asarray(stds, dtype=float) <= 0, 1.0, np.asarray(stds, dtype=float))
    z = (x - np.asarray(means, dtype=float)) / st
    return pd.DataFrame(z, columns=x_frame.columns, index=x_frame.index)


def prepare_listnet_training_frame(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if data.empty:
        return pd.DataFrame(), []

    keep = data.copy()
    keep["as_of_date"] = pd.to_datetime(keep["as_of_date"]).dt.date
    keep = keep.dropna(subset=["as_of_date", "y_excess"])

    if feature_columns is None:
        exclude = {"symbol", "date", "as_of_date", "y_excess", "y_rank_z", "label_version", "feature_version"}
        feature_columns = [c for c in keep.columns if c not in exclude and pd.api.types.is_numeric_dtype(keep[c])]

    x = _fill_and_add_missing_count(keep[feature_columns])
    used_cols = [*feature_columns, "missing_count"]
    keep = pd.concat([keep[["symbol", "as_of_date", "y_excess"]], x], axis=1)

    keep["y_gain"] = (
        keep.groupby("as_of_date", group_keys=False)["y_excess"]
        .transform(lambda s: pd.Series(percentile_gain(s.to_numpy()), index=s.index))
        .astype(float)
    )
    return keep, used_cols


@dataclass
class ListNetModel:
    config: ListNetConfig
    feature_columns: list[str]
    w: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    training_log: list[dict[str, float]]
    calib_a: float = 1.0
    calib_b: float = 0.0
    calib_iso_x: np.ndarray | None = None
    calib_iso_y: np.ndarray | None = None

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "as_of_date", "score_raw", "score_z", "p_raw", "p_cal"])

        x = frame.reindex(columns=[c for c in self.feature_columns if c != "missing_count"], fill_value=0.0)
        x = _fill_and_add_missing_count(x)
        x = x.reindex(columns=self.feature_columns, fill_value=0.0)
        z = _standardize_transform(x, self.feature_means, self.feature_stds)

        out = frame[["symbol", "as_of_date"]].copy()
        out["score_raw"] = z.to_numpy(dtype=float) @ self.w
        out["score_z"] = out.groupby("as_of_date", group_keys=False)["score_raw"].transform(
            lambda s: ((s - float(s.mean())) / max(float(s.std(ddof=0)), 1e-12)) if len(s) > 1 else 0.0
        )
        out["score_raw"] = pd.to_numeric(out["score_raw"], errors="coerce").fillna(0.0)
        out["score_z"] = pd.to_numeric(out["score_z"], errors="coerce").fillna(0.0)
        iso_x = np.asarray(self.calib_iso_x if self.calib_iso_x is not None else [], dtype=float)
        iso_y = np.asarray(self.calib_iso_y if self.calib_iso_y is not None else [], dtype=float)
        calib = CalibratedProbabilityModel(a=float(self.calib_a), b=float(self.calib_b), iso_x=iso_x, iso_y=iso_y)
        out["p_raw"] = calib.p_raw(out["score_z"].to_numpy(dtype=float))
        out["p_cal"] = calib.p_cal(out["score_z"].to_numpy(dtype=float))
        return out

    def save_npz(self, artifact_path: Path) -> None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "w": self.w,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "feature_columns": np.array(self.feature_columns, dtype=object),
            "config_json": np.array(json.dumps(asdict(self.config), sort_keys=True), dtype=object),
            "calib_a": np.array([float(self.calib_a)], dtype=float),
            "calib_b": np.array([float(self.calib_b)], dtype=float),
            "calib_iso_x": np.asarray(self.calib_iso_x if self.calib_iso_x is not None else np.array([], dtype=float), dtype=float),
            "calib_iso_y": np.asarray(self.calib_iso_y if self.calib_iso_y is not None else np.array([], dtype=float), dtype=float),
        }
        np.savez(artifact_path, **payload)


def load_npz(path: Path) -> ListNetModel:
    z = np.load(path, allow_pickle=True)
    cfg = ListNetConfig(**json.loads(str(z["config_json"].item())))
    return ListNetModel(
        config=cfg,
        feature_columns=[str(c) for c in z["feature_columns"].tolist()],
        w=np.asarray(z["w"], dtype=float),
        feature_means=np.asarray(z["feature_means"], dtype=float),
        feature_stds=np.asarray(z["feature_stds"], dtype=float),
        training_log=[],
        calib_a=float(np.asarray(z.get("calib_a", np.array([1.0]))).ravel()[0]),
        calib_b=float(np.asarray(z.get("calib_b", np.array([0.0]))).ravel()[0]),
        calib_iso_x=np.asarray(z.get("calib_iso_x", np.array([], dtype=float)), dtype=float),
        calib_iso_y=np.asarray(z.get("calib_iso_y", np.array([], dtype=float)), dtype=float),
    )


def _evaluate_dates(frame: pd.DataFrame, w: np.ndarray, tau: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for d, g in frame.groupby("as_of_date", sort=True):
        x = g.drop(columns=["symbol", "as_of_date", "y_excess", "y_gain"]).to_numpy(dtype=float)
        score = x @ w
        rows.extend(
            {
                "as_of_date": d,
                "y_gain": float(yg),
                "y_excess": float(ye),
                "score": float(sc),
            }
            for yg, ye, sc in zip(g["y_gain"].to_numpy(dtype=float), g["y_excess"].to_numpy(dtype=float), score)
        )
    return daily_ranking_metrics(pd.DataFrame(rows), k=30)


def train_listnet_linear(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    config: ListNetConfig | None = None,
    val_dates: set[dt.date] | None = None,
) -> ListNetModel:
    cfg = config or ListNetConfig()
    rng = np.random.default_rng(cfg.seed)

    xf = train_frame.reindex(columns=["symbol", "as_of_date", "y_excess", "y_gain", *feature_columns]).copy()
    feats = xf[feature_columns].astype(float)
    feats_std, means, stds = _standardize_fit_transform(feats)
    work = pd.concat([xf[["symbol", "as_of_date", "y_excess", "y_gain"]], feats_std], axis=1)

    w = np.zeros(len(feature_columns), dtype=float)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    t_step = 0

    date_list = sorted(set(work["as_of_date"].tolist()))
    best_w = w.copy()
    best_val = -np.inf
    bad_epochs = 0
    train_log: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        perm = rng.permutation(len(date_list))
        shuffled_dates = [date_list[i] for i in perm]
        epoch_losses: list[float] = []

        for i in range(0, len(shuffled_dates), cfg.batch_dates):
            batch_dates = set(shuffled_dates[i : i + cfg.batch_dates])
            grads = np.zeros_like(w)
            losses = []
            for d, gdf in work[work["as_of_date"].isin(batch_dates)].groupby("as_of_date", sort=False):
                x = gdf[feature_columns].to_numpy(dtype=float)
                y_gain = gdf["y_gain"].to_numpy(dtype=float)
                loss_t, grad_t, _ = listnet_loss_and_grad(x, y_gain, w, tau=cfg.tau, lambda_l2=0.0)
                grads += grad_t
                losses.append(loss_t)
                _ = d
            if not losses:
                continue
            grads = grads / float(len(losses)) + cfg.lambda_l2 * w
            batch_loss = float(np.mean(losses) + 0.5 * cfg.lambda_l2 * np.sum(w * w))
            epoch_losses.append(batch_loss)

            t_step += 1
            m = cfg.beta1 * m + (1.0 - cfg.beta1) * grads
            v = cfg.beta2 * v + (1.0 - cfg.beta2) * (grads * grads)
            m_hat = m / (1.0 - cfg.beta1 ** t_step)
            v_hat = v / (1.0 - cfg.beta2 ** t_step)
            w = w - cfg.lr * (m_hat / (np.sqrt(v_hat) + cfg.eps))

        metrics_train = _evaluate_dates(work, w, tau=cfg.tau)
        train_ndcg = float(metrics_train["ndcg_at_k"].mean()) if not metrics_train.empty else 0.0

        val_ndcg = train_ndcg
        if val_dates:
            val_df = work[work["as_of_date"].isin(val_dates)]
            if not val_df.empty:
                metrics_val = _evaluate_dates(val_df, w, tau=cfg.tau)
                val_ndcg = float(metrics_val["ndcg_at_k"].mean()) if not metrics_val.empty else 0.0

        train_log.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "train_ndcg_30": train_ndcg,
                "val_ndcg_30": val_ndcg,
            }
        )

        if val_ndcg > (best_val + 1e-12):
            best_val = val_ndcg
            best_w = w.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                break

    split_idx = max(1, int(0.8 * len(date_list))) if date_list else 1
    train_dates_for_cal = set(date_list[:split_idx])
    val_dates_for_cal = set(date_list[split_idx:]) if len(date_list) > split_idx else set(date_list[-1:])

    score_by_row = work[feature_columns].to_numpy(dtype=float) @ best_w
    cal_df = work[["as_of_date", "y_excess"]].copy()
    cal_df["score_raw"] = score_by_row
    cal_df["score_z"] = cal_df.groupby("as_of_date", group_keys=False)["score_raw"].transform(
        lambda s: ((s - float(s.mean())) / max(float(s.std(ddof=0)), 1e-12)) if len(s) > 1 else 0.0
    )
    cal_df["z"] = (cal_df["y_excess"].astype(float) > 0.0).astype(float)

    tr = cal_df[cal_df["as_of_date"].isin(train_dates_for_cal)]
    va = cal_df[cal_df["as_of_date"].isin(val_dates_for_cal)]
    calib = fit_calibrated_probability_model(
        tr["score_z"].to_numpy(dtype=float),
        tr["z"].to_numpy(dtype=float),
        va["score_z"].to_numpy(dtype=float),
        va["z"].to_numpy(dtype=float),
    )

    return ListNetModel(
        config=cfg,
        feature_columns=feature_columns,
        w=best_w,
        feature_means=means,
        feature_stds=stds,
        training_log=train_log,
        calib_a=float(calib.a),
        calib_b=float(calib.b),
        calib_iso_x=np.asarray(calib.iso_x, dtype=float),
        calib_iso_y=np.asarray(calib.iso_y, dtype=float),
    )


def cross_validate_listnet(frame: pd.DataFrame, feature_columns: list[str], config: ListNetConfig | None = None) -> pd.DataFrame:
    cfg = config or ListNetConfig()
    dates = [d for d in frame["as_of_date"].tolist()]
    splitter = PurgedKFoldEmbargo(
        n_splits=cfg.cv_splits,
        purge_horizon_days=cfg.purge_horizon_days,
        embargo_days=cfg.embargo_days,
    )
    splits = splitter.split(dates)
    out_rows: list[dict[str, float]] = []
    for fold_id, (train_dates, test_dates) in enumerate(splits, start=1):
        train_df = frame[frame["as_of_date"].isin(set(train_dates))]
        test_df = frame[frame["as_of_date"].isin(set(test_dates))]
        if train_df.empty or test_df.empty:
            continue
        model = train_listnet_linear(train_df, feature_columns=feature_columns, config=cfg)
        pred = model.predict_frame(test_df[["symbol", "as_of_date", *[c for c in feature_columns if c != "missing_count"]]])
        eval_df = test_df[["as_of_date", "y_gain", "y_excess"]].copy()
        eval_df["score"] = pred["score_raw"].to_numpy(dtype=float)
        metrics = daily_ranking_metrics(eval_df, k=30)
        out_rows.append(
            {
                "fold": float(fold_id),
                "ndcg_at_30": float(metrics["ndcg_at_k"].mean()) if not metrics.empty else 0.0,
                "rank_ic": float(metrics["rank_ic"].mean()) if not metrics.empty else 0.0,
            }
        )
    return pd.DataFrame(out_rows)
