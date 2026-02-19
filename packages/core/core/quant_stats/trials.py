from __future__ import annotations

import hashlib
import json


def make_trial_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def n_eff_default(n_trials: int) -> int:
    return max(1, int(n_trials))
