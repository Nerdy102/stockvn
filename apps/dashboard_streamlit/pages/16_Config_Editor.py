from __future__ import annotations

import yaml
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "config_editor"
PAGE_TITLE = "Config Editor"


def render() -> None:
    st.title("Config Editor")
    name = st.selectbox("config", ["raocmoe", "eval_lab", "execution", "fees_taxes"])
    base = cached_get_json("/configs", {"name": name}, ttl_s=5)
    text = st.text_area("YAML", value=base.get("yaml", ""), height=350)
    if st.button("Validate"):
        try:
            obj = yaml.safe_load(text)
            st.success("Valid YAML")
            st.json({"keys": list((obj or {}).keys())})
        except Exception as exc:
            st.error(str(exc))
    if st.button("Save override"):
        st.json(cached_post_json(f"/configs?name={name}", {"yaml": text}, ttl_s=1))
    st.subheader("Active merged")
    st.json(cached_get_json("/configs/active", {"name": name}, ttl_s=2))
