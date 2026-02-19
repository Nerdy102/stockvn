from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", default="data_drop/inbox")
    parser.add_argument("--mapping", default="configs/providers/data_drop_default.yaml")
    parser.add_argument("--out", default="data_demo/prices_demo_1d.csv")
    args = parser.parse_args()

    inbox = Path(args.inbox)
    files = sorted([p for p in inbox.glob("*.csv")])
    if not files:
        print("Không có file trong data_drop/inbox")
        return

    mapping = yaml.safe_load(Path(args.mapping).read_text(encoding="utf-8")) or {}
    fields = mapping.get("fields") or {}

    out_rows = []
    for f in files:
        df = pd.read_csv(f)
        row = pd.DataFrame({k: df[v] for k, v in fields.items() if v in df.columns})
        out_rows.append(row)

    merged = pd.concat(out_rows, ignore_index=True)
    merged["value_vnd"] = merged["close"].astype(float) * merged["volume"].astype(float)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Đã ingest {len(files)} file, ghi {len(merged)} dòng vào {out_path}")


if __name__ == "__main__":
    main()
