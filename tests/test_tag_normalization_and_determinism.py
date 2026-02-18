from api_fastapi.routers.watchlists import normalize_tags


def test_tag_normalization_and_determinism() -> None:
    tags = normalize_tags(["KQKD", "kqkd", "policy"])
    assert tags == ["kqkd", "policy"]
