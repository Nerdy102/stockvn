from apps.dashboard_streamlit.ui.tokens import validate_tokens


def test_ui_tokens_validate() -> None:
    assert validate_tokens() == []
