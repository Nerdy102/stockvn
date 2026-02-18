from __future__ import annotations

from pathlib import Path

FORBIDDEN = ["guaranteed", "buy now", "win rate", "sure profit", "phím mã", "cam kết"]


def main() -> int:
    base = Path("apps/dashboard_streamlit")
    py_files = list(base.rglob("*.py"))
    errors: list[str] = []

    for path in py_files:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for phrase in FORBIDDEN:
            if phrase in text:
                errors.append(f"Forbidden phrase '{phrase}' found in {path}")

    layout_path = Path("apps/dashboard_streamlit/ui/layout.py")
    layout_text = layout_path.read_text(encoding="utf-8", errors="ignore")
    if "DISCLAIMER_SHORT" not in layout_text:
        errors.append("layout.py must reference DISCLAIMER_SHORT in topbar.")

    if errors:
        for err in errors:
            print(err)
        return 1
    print("UI guardrail check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
