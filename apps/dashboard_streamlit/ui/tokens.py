from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TypographyTokens:
    sizes: tuple[int, ...] = (12, 14, 16, 20, 24)
    weights: tuple[int, ...] = (400, 600)


@dataclass(frozen=True)
class SpacingTokens:
    scale: tuple[int, ...] = (4, 8, 12, 16, 24, 32)


@dataclass(frozen=True)
class RadiusTokens:
    scale: tuple[int, ...] = (6, 10)


@dataclass(frozen=True)
class ColorTokens:
    risk_ok: str = "#1f9d55"
    risk_warn: str = "#f59e0b"
    risk_bad: str = "#dc2626"
    neutral_bg: str = "#0f172a"
    neutral_text: str = "#e2e8f0"
    muted_text: str = "#94a3b8"


@dataclass(frozen=True)
class DesignTokens:
    typography: TypographyTokens = TypographyTokens()
    spacing: SpacingTokens = SpacingTokens()
    radius: RadiusTokens = RadiusTokens()
    colors: ColorTokens = ColorTokens()


TOKENS = DesignTokens()


def validate_tokens() -> list[str]:
    errors: list[str] = []
    if TOKENS.typography.sizes != (12, 14, 16, 20, 24):
        errors.append("Typography sizes must match [12,14,16,20,24].")
    if TOKENS.typography.weights != (400, 600):
        errors.append("Typography weights must match [400,600].")
    if TOKENS.spacing.scale != (4, 8, 12, 16, 24, 32):
        errors.append("Spacing scale must match [4,8,12,16,24,32].")
    if TOKENS.radius.scale != (6, 10):
        errors.append("Radius scale must match [6,10].")
    for field_name in (
        "risk_ok",
        "risk_warn",
        "risk_bad",
        "neutral_bg",
        "neutral_text",
        "muted_text",
    ):
        val = getattr(TOKENS.colors, field_name)
        if not (isinstance(val, str) and val.startswith("#") and len(val) in (4, 7)):
            errors.append(f"Color token {field_name} must be a hex string.")
    return errors
