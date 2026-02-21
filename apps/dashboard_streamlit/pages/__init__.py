from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_PAGE_FILES = [
    "0_Tong_quan_hom_nay.py",
    "1_Screener.py",
    "2_Charting.py",
    "3_Heatmap.py",
    "4_Portfolio.py",
    "5_Alerts.py",
    "6_ML_Lab.py",
    "7_Alpha_v2_Lab.py",
    "8_Data_Health.py",
    "9_Settings.py",
    "10_New_Orders.py",
    "11_Quant_Console.py",
    "12_RAOCMOE_Lab.py",
    "13_Eval_Lab.py",
    "14_Run_Manager.py",
    "15_Data_Manager.py",
    "16_Config_Editor.py",
    "17_Realtime_Console.py",
]


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load page: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _page_registry() -> list[ModuleType]:
    base = Path(__file__).resolve().parent
    return [_load(base / name) for name in _PAGE_FILES]
