from .bootstrap import block_bootstrap_samples, bootstrap_ci
from .consistency import check_cost_nonnegativity, check_equity_identity, check_return_identity
from .metrics import compute_cost_attribution, compute_performance_metrics, compute_tail_metrics
from .multiple_testing import (
    benjamini_hochberg,
    dsr,
    format_pvalue,
    min_trl,
    pbo_cscv,
    psr,
    reality_check,
    spa,
)
from .registry import StrategySpec, build_strategy_registry
from .splits import purged_cv_splits, walk_forward_splits

__all__ = [
    "StrategySpec",
    "benjamini_hochberg",
    "block_bootstrap_samples",
    "bootstrap_ci",
    "build_strategy_registry",
    "compute_cost_attribution",
    "check_cost_nonnegativity",
    "check_equity_identity",
    "check_return_identity",
    "compute_performance_metrics",
    "compute_tail_metrics",
    "dsr",
    "format_pvalue",
    "min_trl",
    "pbo_cscv",
    "psr",
    "purged_cv_splits",
    "reality_check",
    "spa",
    "walk_forward_splits",
]
