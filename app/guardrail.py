"""Deterministic guardrail validation.

This module enforces the structural and semantic constraints on the
interpreter's output before it is fed into the optimizer. It is intentionally
strict: malformed interpretations are rejected with `GuardrailError`, not
silently repaired. The optimizer assumes the data it receives is clean.
"""

from __future__ import annotations

from typing import Any, Dict, List

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


class GuardrailError(ValueError):
    """Raised when an interpreter payload fails deterministic validation."""


def _check_hours(value: Any) -> List[int]:
    if not isinstance(value, list) or not value:
        raise GuardrailError("structured_adjustment.hours must be a non-empty list")
    out: List[int] = []
    for h in value:
        if not isinstance(h, int) or isinstance(h, bool):
            raise GuardrailError(f"hour entries must be ints, got {h!r}")
        if h < 0 or h > 23:
            raise GuardrailError(f"hour entries must be in 0..23, got {h}")
        out.append(h)
    if len(set(out)) != len(out):
        raise GuardrailError("hours must be unique")
    if out != sorted(out):
        raise GuardrailError("hours must be in ascending order")
    return out


def validate_interpretations(
    payload: List[Dict[str, Any]], n_notes: int, capacity_kwh: float
) -> List[Dict[str, Any]]:
    """Validate a list of directive_interpretation dicts.

    Returns the same list (with structured_adjustment pruned of None values)
    so the optimizer never has to deal with absent keys.
    """

    if not isinstance(payload, list):
        raise GuardrailError("directive_interpretation must be a list")

    if len(payload) != n_notes:
        raise GuardrailError(
            f"directive_interpretation length ({len(payload)}) "
            f"does not match operator_notes length ({n_notes})"
        )

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise GuardrailError(f"entry {i} is not an object")
        idx = item.get("note_index")
        if idx != i:
            raise GuardrailError(
                f"entry {i} has note_index={idx!r}, expected {i}"
            )
        dtype = item.get("directive_type")
        if dtype not in ALLOWED_TYPES:
            raise GuardrailError(f"entry {i}: unknown directive_type {dtype!r}")
        applies = item.get("applies")
        sa = item.get("structured_adjustment")

        if dtype == "no_op":
            if applies is not False:
                raise GuardrailError(f"entry {i}: no_op must have applies=false")
            if sa is not None:
                raise GuardrailError(f"entry {i}: no_op must have structured_adjustment=null")
        else:
            if applies is not True:
                raise GuardrailError(f"entry {i}: {dtype} must have applies=true")
            if not isinstance(sa, dict):
                raise GuardrailError(f"entry {i}: structured_adjustment must be an object")

            cleaned: Dict[str, Any] = {"hours": _check_hours(sa.get("hours"))}

            if dtype == "solar_reduction":
                f = sa.get("factor")
                if not isinstance(f, (int, float)) or isinstance(f, bool):
                    raise GuardrailError(f"entry {i}: factor must be numeric")
                if not (0.0 <= float(f) <= 1.0):
                    raise GuardrailError(f"entry {i}: factor must be in [0,1]")
                cleaned["factor"] = float(f)

            elif dtype == "minimum_battery_reserve":
                mn = sa.get("minimum_energy_kwh")
                if not isinstance(mn, (int, float)) or isinstance(mn, bool):
                    raise GuardrailError(f"entry {i}: minimum_energy_kwh must be numeric")
                mn = float(mn)
                if mn < 0 or mn > capacity_kwh:
                    raise GuardrailError(
                        f"entry {i}: minimum_energy_kwh {mn} outside [0,{capacity_kwh}]"
                    )
                cleaned["minimum_energy_kwh"] = mn

            elif dtype == "max_grid_window":
                cap = sa.get("max_grid_kwh")
                if not isinstance(cap, (int, float)) or isinstance(cap, bool):
                    raise GuardrailError(f"entry {i}: max_grid_kwh must be numeric")
                cap = float(cap)
                if cap < 0:
                    raise GuardrailError(f"entry {i}: max_grid_kwh must be >= 0")
                cleaned["max_grid_kwh"] = cap

            # no_charge_window / no_discharge_window need no extra fields.

            out.append(
                {
                    "note_index": idx,
                    "applies": True,
                    "directive_type": dtype,
                    "structured_adjustment": cleaned,
                    "explanation": str(item.get("explanation", ""))[:500],
                }
            )
            continue

        out.append(
            {
                "note_index": idx,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": str(item.get("explanation", ""))[:500],
            }
        )

    return out
