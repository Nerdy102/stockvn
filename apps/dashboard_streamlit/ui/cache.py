from __future__ import annotations

import json
from typing import Any

import streamlit as st

from apps.dashboard_streamlit.lib.api import get, post


@st.cache_data(ttl=300)
def _cached_get_json(url: str, params_key: str) -> Any:
    params = json.loads(params_key) if params_key else None
    return get(url, params=params)


@st.cache_data(ttl=60)
def _cached_post_json(url: str, payload_key: str) -> Any:
    payload = json.loads(payload_key) if payload_key else None
    return post(url, json=payload)


def cached_get_json(url: str, params: dict[str, Any] | None, ttl_s: int) -> Any:
    decorated = st.cache_data(ttl=ttl_s)(_cached_get_json.__wrapped__)
    params_key = json.dumps(params or {}, sort_keys=True, ensure_ascii=False)
    return decorated(url, params_key)


def cached_post_json(url: str, payload: dict[str, Any] | None, ttl_s: int = 60) -> Any:
    decorated = st.cache_data(ttl=ttl_s)(_cached_post_json.__wrapped__)
    payload_key = json.dumps(payload or {}, sort_keys=True, ensure_ascii=False)
    return decorated(url, payload_key)
