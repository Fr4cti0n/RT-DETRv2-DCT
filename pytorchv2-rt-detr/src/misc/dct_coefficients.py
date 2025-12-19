from __future__ import annotations

import math
from typing import Iterable, Sequence

__all__ = [
    "ZIGZAG_ORDER",
    "validate_coeff_count",
    "validate_coeff_window",
    "window_to_count",
    "count_to_window",
    "resolve_coefficient_counts",
    "resolve_coefficient_config",
    "build_active_indices",
    "build_active_mask",
]

ZIGZAG_ORDER: Sequence[int] = (
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
)


def validate_coeff_count(value: int, *, name: str) -> int:
    count = int(value)
    if not 0 <= count <= 64:
        raise ValueError(f"{name} must be within [0, 64]; received {count}.")
    return count


def validate_coeff_window(value: int, *, name: str) -> int:
    window = int(value)
    if not 1 <= window <= 8:
        raise ValueError(f"{name} must be within [1, 8]; received {window}.")
    return window


def window_to_count(window: int) -> int:
    window_validated = validate_coeff_window(window, name="coeff_window")
    return window_validated * window_validated


def count_to_window(count: int) -> int | None:
    count_validated = validate_coeff_count(count, name="coeff_count")
    if count_validated == 0:
        return None
    root = int(math.isqrt(count_validated))
    if root * root != count_validated:
        return None
    return root


def resolve_coefficient_counts(
    *,
    coeff_window: int | None = None,
    coeff_count: int | None = None,
    coeff_window_luma: int | None = None,
    coeff_count_luma: int | None = None,
    coeff_window_chroma: int | None = None,
    coeff_count_chroma: int | None = None,
    coeff_window_cb: int | None = None,
    coeff_count_cb: int | None = None,
    coeff_window_cr: int | None = None,
    coeff_count_cr: int | None = None,
) -> tuple[int, int, int, int]:
    """Resolve base/luma/chroma coefficient counts from window/count hints."""

    base_count = 64
    if coeff_window is not None:
        base_count = window_to_count(coeff_window)
    if coeff_count is not None:
        base_count = validate_coeff_count(coeff_count, name="coeff_count")

    luma_count = base_count
    if coeff_window_luma is not None:
        luma_count = window_to_count(coeff_window_luma)
    if coeff_count_luma is not None:
        luma_count = validate_coeff_count(coeff_count_luma, name="coeff_count_luma")

    chroma_count = luma_count
    if coeff_window_chroma is not None:
        chroma_count = window_to_count(coeff_window_chroma)
    if coeff_count_chroma is not None:
        chroma_count = validate_coeff_count(coeff_count_chroma, name="coeff_count_chroma")

    cb_count = chroma_count
    if coeff_window_cb is not None:
        cb_count = window_to_count(coeff_window_cb)
    if coeff_count_cb is not None:
        cb_count = validate_coeff_count(coeff_count_cb, name="coeff_count_cb")

    cr_count = chroma_count
    if coeff_window_cr is not None:
        cr_count = window_to_count(coeff_window_cr)
    if coeff_count_cr is not None:
        cr_count = validate_coeff_count(coeff_count_cr, name="coeff_count_cr")

    return base_count, luma_count, cb_count, cr_count


def resolve_coefficient_config(
    *,
    coeff_window: int | None = None,
    coeff_count: int | None = None,
    coeff_window_luma: int | None = None,
    coeff_count_luma: int | None = None,
    coeff_window_chroma: int | None = None,
    coeff_count_chroma: int | None = None,
    coeff_window_cb: int | None = None,
    coeff_count_cb: int | None = None,
    coeff_window_cr: int | None = None,
    coeff_count_cr: int | None = None,
) -> dict[str, int | None]:
    base_count, luma_count, cb_count, cr_count = resolve_coefficient_counts(
        coeff_window=coeff_window,
        coeff_count=coeff_count,
        coeff_window_luma=coeff_window_luma,
        coeff_count_luma=coeff_count_luma,
        coeff_window_chroma=coeff_window_chroma,
        coeff_count_chroma=coeff_count_chroma,
        coeff_window_cb=coeff_window_cb,
        coeff_count_cb=coeff_count_cb,
        coeff_window_cr=coeff_window_cr,
        coeff_count_cr=coeff_count_cr,
    )
    window = count_to_window(base_count)
    window_luma = count_to_window(luma_count)
    window_cb = count_to_window(cb_count)
    window_cr = count_to_window(cr_count)
    window_chroma = window_cb if window_cb == window_cr else None
    chroma_count = cb_count if cb_count == cr_count else max(cb_count, cr_count)
    return {
        "coeff_count": base_count,
        "coeff_count_luma": luma_count,
        "coeff_count_chroma": chroma_count,
        "coeff_count_cb": cb_count,
        "coeff_count_cr": cr_count,
        "coeff_window": window,
        "coeff_window_luma": window_luma,
        "coeff_window_chroma": window_chroma,
        "coeff_window_cb": window_cb,
        "coeff_window_cr": window_cr,
    }


def build_active_indices(count: int) -> list[int]:
    count_validated = validate_coeff_count(count, name="coeff_count")
    if count_validated <= 0:
        return []
    return list(ZIGZAG_ORDER[:count_validated]) if count_validated < 64 else list(range(64))


def build_active_mask(count: int, *, value_active: float = 1.0, value_inactive: float = 0.0) -> list[float]:
    indices = build_active_indices(count)
    mask = [value_inactive] * 64
    for idx in indices:
        mask[idx] = value_active
    if len(indices) >= 64 and count >= 64:
        return [value_active] * 64
    return mask
