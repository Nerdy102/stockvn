from __future__ import annotations

import pandas as pd
from core.db.models import PriceOHLCV
from core.db.session import get_engine
from core.ml.backtest import run_walk_forward
from core.ml.features import build_ml_features, feature_columns
from sqlmodel import Session, select


def main() -> None:
    engine = get_engine("sqlite:///./vn_invest.db")
    with Session(engine) as s:
        rows = list(s.exec(select(PriceOHLCV)).all())
    df = pd.DataFrame([r.model_dump() for r in rows])
    feat = build_ml_features(df).dropna(subset=["y_excess"])
    curve, metrics = run_walk_forward(feat, feature_columns(feat))
    print({"points": int(len(curve)), "metrics": metrics})


if __name__ == "__main__":
    main()
