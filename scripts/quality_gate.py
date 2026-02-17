from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_PREFIXES = (
    "tests/fixtures",
    "data_demo",
    ".git",
    ".venv",
    "__pycache__",
)

WORD_TODO = "TO" + "DO"
WORD_FIXME = "FIX" + "ME"
WORD_HACK = "HA" + "CK"

TEXT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (WORD_TODO, re.compile(rf"\b{re.escape(WORD_TODO)}\b", re.IGNORECASE)),
    (WORD_FIXME, re.compile(rf"\b{re.escape(WORD_FIXME)}\b", re.IGNORECASE)),
    (WORD_HACK, re.compile(rf"\b{re.escape(WORD_HACK)}\b", re.IGNORECASE)),
    ("pass placeholder", re.compile(r"\bpass\s*#", re.IGNORECASE)),
    (
        "return None placeholder",
        re.compile(r"return\s+None\s*#\s*placeholder", re.IGNORECASE),
    ),
]


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    reason: str


def _is_excluded(path: Path) -> bool:
    normalized = path.as_posix()
    return any(normalized == d or normalized.startswith(f"{d}/") for d in EXCLUDED_PREFIXES)


def _scan_text(path: Path, repo_root: Path) -> list[Violation]:
    violations: list[Violation] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return violations

    rel = path.relative_to(repo_root).as_posix()
    for line_no, line in enumerate(text.splitlines(), start=1):
        for reason, pattern in TEXT_PATTERNS:
            if pattern.search(line):
                violations.append(Violation(rel, line_no, reason))
    return violations


def _scan_docstrings(path: Path, repo_root: Path) -> list[Violation]:
    violations: list[Violation] = []
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return violations

    rel = path.relative_to(repo_root).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc and "placeholder" in doc.lower():
                violations.append(Violation(rel, node.lineno, "docstring contains placeholder"))
    return violations


def find_violations(repo_root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root)
        if _is_excluded(rel):
            continue
        violations.extend(_scan_text(path, repo_root))
        if path.suffix == ".py":
            violations.extend(_scan_docstrings(path, repo_root))
    return sorted(violations, key=lambda v: (v.path, v.line, v.reason))


def main() -> int:
    violations = find_violations(Path.cwd())
    if not violations:
        print("quality_gate: OK")
        return 0

    print("quality_gate: violations found")
    for violation in violations:
        print(f"{violation.path}:{violation.line}: {violation.reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
