from __future__ import annotations

from itertools import combinations

import numpy as np

from core.quant_stats.sharpe import sharpe_non_annualized

from .schemas import CSCVReport


def _rank_ascending(values: np.ndarray, target_idx: int) -> float:
    v = float(values[target_idx])
    less = int(np.sum(values < v))
    equal = int(np.sum(values == v))
    # Mid-rank deterministic cho các trường hợp đồng hạng
    return float(less + (equal + 1) / 2.0)


def compute_cscv_pbo(matrix_returns, s_segments: int = 8) -> tuple[CSCVReport | None, str | None]:
    m = np.asarray(matrix_returns, dtype=float)
    if m.ndim != 2:
        return None, "ma trận lợi nhuận không hợp lệ"
    t, n = m.shape
    if n > 60:
        return None, "quá nhiều trial"
    if t < s_segments * 30:
        return None, "sample quá ngắn"
    if s_segments != 8:
        return None, "CSCV yêu cầu S=8"

    segments = np.array_split(np.arange(t), s_segments)
    combos = list(combinations(range(s_segments), s_segments // 2))
    lambdas: list[float] = []
    is_metrics: list[float] = []
    oos_metrics: list[float] = []
    oos_losses: list[float] = []

    for c in combos:
        train_idx = np.concatenate([segments[i] for i in c])
        test_idx = np.concatenate([segments[i] for i in range(s_segments) if i not in c])
        is_m = np.asarray([sharpe_non_annualized(m[train_idx, j]) for j in range(n)], dtype=float)
        oos_m = np.asarray([sharpe_non_annualized(m[test_idx, j]) for j in range(n)], dtype=float)
        n_star = int(np.argmax(is_m))
        rank = _rank_ascending(oos_m, n_star)
        omega = rank / float(n + 1)
        lambdas.append(float(np.log(omega / (1.0 - omega))))
        is_metrics.append(float(is_m[n_star]))
        oos_metrics.append(float(oos_m[n_star]))
        oos_losses.append(float(np.prod(1.0 + m[test_idx, n_star]) - 1.0))

    l = np.asarray(lambdas, dtype=float)
    x = np.asarray(is_metrics, dtype=float)
    y = np.asarray(oos_metrics, dtype=float)
    x_mean = float(np.mean(x))
    beta = 0.0 if np.allclose(x, x_mean) else float(np.sum((x - x_mean) * (y - np.mean(y))) / np.sum((x - x_mean) ** 2))
    report = CSCVReport(
        s_segments=s_segments,
        n_trials=n,
        n_combinations=len(combos),
        logits_p05=float(np.quantile(l, 0.05)),
        logits_p50=float(np.quantile(l, 0.50)),
        logits_p95=float(np.quantile(l, 0.95)),
        pbo_phi=float(np.mean(l < 0.0) + 0.5 * np.mean(np.isclose(l, 0.0))),
        perf_decay_beta=float(beta),
        prob_loss_oos=float(np.mean(np.asarray(oos_losses) < 0.0)),
        notes_vi="CSCV/PBO dùng Sharpe không thường niên; chỉ phản ánh độ tin cậy thống kê.",
    )
    return report, None
