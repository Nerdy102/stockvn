from __future__ import annotations

import pandas as pd
from core.db.models import PriceOHLCV
from core.db.session import get_engine
from core.ml.features import build_ml_features, feature_columns
from core.ml.models import MlModelBundle
from sqlmodel import Session, select


def main() -> None:
    engine = get_engine("sqlite:///./vn_invest.db")
    with Session(engine) as s:
        rows = list(s.exec(select(PriceOHLCV)).all())
    df = pd.DataFrame([r.model_dump() for r in rows])
    feat = build_ml_features(df)
    feat = feat.dropna(subset=["y_excess"])
    cols = feature_columns(feat)
    model = MlModelBundle().fit(feat[cols], feat["y_excess"])
    pred = model.predict(feat[cols])
    print({"rows": int(len(feat)), "pred_mean": float(pd.Series(pred).mean())})


if __name__ == "__main__":
    main()
