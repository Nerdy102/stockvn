from __future__ import annotations


def test_ui_kiosk_import() -> None:
    import apps.web_kiosk.app as kiosk

    assert hasattr(kiosk, "render")
