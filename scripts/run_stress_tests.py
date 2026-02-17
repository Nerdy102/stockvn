from __future__ import annotations


def main() -> None:
    scenarios = [
        {"name": "cost_x2", "commission_mult": 2.0},
        {"name": "fill_ratio_x0.5", "fill_mult": 0.5},
        {"name": "remove_best_5_days", "drop_best_n": 5},
        {"name": "slippage_base_plus_10", "slippage_plus_bps": 10},
    ]
    print({"scenarios": scenarios})


if __name__ == "__main__":
    main()
