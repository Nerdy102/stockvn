from __future__ import annotations

import itertools

RSI_GRID = [10, 14, 21]
MA_FAST_GRID = [10, 20]
MA_SLOW_GRID = [50, 100]
BREAKOUT_GRID = [20, 55]


def main() -> None:
    variants = list(itertools.product(RSI_GRID, MA_FAST_GRID, MA_SLOW_GRID, BREAKOUT_GRID))
    print({"num_variants": len(variants), "variants_sample": variants[:5]})


if __name__ == "__main__":
    main()
