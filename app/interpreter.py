"""Deterministic operator-note interpreter.

This module is the *only* component that performs natural-language parsing.
Everything downstream treats the output as trusted structured data, validated
by `app.guardrail`. The design goal is to map every plausible phrasing of
the six supported directive types without silently inventing numbers.

Pipeline:
    note  ->  classify directive type (or no_op)
           ->  extract time window (start-inclusive, end-exclusive)
           ->  extract scalar value (factor | minimum | max_grid)
           ->  return StructuredAdjustment dict

The interpreter is regex/keyword driven, not LLM driven. Operator-facing
prompts that mention the six supported directives in plain English all map
to the same code paths. Anything that fails to extract required fields is
treated as a distractor (`no_op`) rather than fabricated.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Time window parsing
# ---------------------------------------------------------------------------

_NUM = r"(\d{1,2})"  # captures 0..23 or 1..12 for am/pm
_AMPM = r"\s*(a\.?m\.?|p\.?m\.?|AM|PM)?"

_TIME_RANGE_PATTERNS = [
    # "from 6 PM to 9 PM", "6 PM until 9 PM", "between 6 PM and 9 PM"
    re.compile(
        rf"\b(?:from|between)?\s*{_NUM}{_AMPM}\s*(?:to|until|till|through|and|-)\s*{_NUM}{_AMPM}",
        re.IGNORECASE,
    ),
]

# Range with one word endpoint, on the right side.
# "10 AM until noon", "from 5 PM to midnight", "2 PM through midnight"
_DIGIT_TO_WORD_RANGE = re.compile(
    rf"\b(?:from|between)?\s*{_NUM}{_AMPM}\s*(?:to|until|till|through|and|-)\s*"
    rf"(noon|midnight)",
    re.IGNORECASE,
)

_NOON_WORD_RANGE = re.compile(
    rf"\b(?:from|between)?\s*(noon|midnight)\s+(?:to|until|till|through|and|-)\s*"
    rf"({_NUM}{_AMPM}|noon|midnight)",
    re.IGNORECASE,
)

_HOUR_WORD = {
    "noon": 12,
    "midnight": 0,
}


def _to_24h(n: int, ampm: Optional[str]) -> int:
    """Convert a (number, am/pm) pair to a 0..23 hour."""
    if ampm is None:
        return n  # already assumed 24h
    a = ampm.lower().replace(".", "")
    n = n % 12
    if a.startswith("p"):
        return n + 12
    return n


def _extract_time_window(text: str) -> Optional[List[int]]:
    """Return a list of unique, ascending hours covered by a time window in
    `text`, or None if no recognizable window is found.

    Windows are half-open: [start, end). E.g. "1 PM to 3 PM" -> [13, 14].
    """

    lower = text.lower()

    # Special cases first: noon / midnight boundaries.
    m_special = re.search(
        rf"\bbetween\s+{_NUM}{_AMPM}\s+and\s+(noon|midnight)",
        lower,
    )
    if m_special:
        s = _to_24h(int(m_special.group(1)), m_special.group(2))
        e_raw = m_special.group(3)
        e = _HOUR_WORD[e_raw]
        return _canonicalize(s, e)

    # Word-boundary endpoints: "from noon until 2 PM", "between noon and 3 PM".
    m_noon = _NOON_WORD_RANGE.search(lower)
    if m_noon:
        s_raw = m_noon.group(1)
        s = _HOUR_WORD[s_raw]
        end_str = m_noon.group(2)
        if end_str in _HOUR_WORD:
            e = _HOUR_WORD[end_str]
        else:
            num_match = re.match(r"(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)?", end_str)
            if not num_match:
                return None
            e = _to_24h(int(num_match.group(1)), num_match.group(2))
        return _canonicalize(s, e)

    # Digit-to-word range: "10 AM until noon", "from 5 PM to midnight".
    m_d2w = _DIGIT_TO_WORD_RANGE.search(lower)
    if m_d2w:
        s = _to_24h(int(m_d2w.group(1)), m_d2w.group(2))
        e = _HOUR_WORD[m_d2w.group(3)]
        return _canonicalize(s, e)

    for pat in _TIME_RANGE_PATTERNS:
        m = pat.search(lower)
        if not m:
            continue
        s_raw = int(m.group(1))
        s_ampm = m.group(2)
        e_raw = int(m.group(3))
        e_ampm = m.group(4)
        s = _to_24h(s_raw, s_ampm)
        e = _to_24h(e_raw, e_ampm)
        if s == e:
            return [s]
        return _canonicalize(s, e)

    return None


def _canonicalize(start: int, end: int) -> List[int]:
    """Return ascending unique hour indices for the half-open window
    `[start, end)`. Wraps through midnight if `end <= start`."""

    if end <= start:
        # Wrap around midnight.
        wrapped = list(range(start, 24)) + list(range(0, end))
    else:
        wrapped = list(range(start, end))
    # Deduplicate while preserving order.
    seen = set()
    out: List[int] = []
    for h in wrapped:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


# ---------------------------------------------------------------------------
# Solar reduction
# ---------------------------------------------------------------------------

_SOLAR_KW = re.compile(
    r"\b(solar|panels?|rooftop|inverter|pv)\b", re.IGNORECASE
)
_REDUCTION_KW = re.compile(
    r"\b(reduc(?:e|ed|ing|tion)|cut(?:ting|s)?|loss|unavailable|offline|"
    r"shutdown|covered|cloud( cover|ed|ing)?|not work|maintenance|"
    r"cleaning|inspect|repair)\b",
    re.IGNORECASE,
)


def _extract_reduction_factor(text: str) -> Optional[float]:
    """Convert the textual reduction percentage into the remaining usable
    fraction (e.g. '80% reduction' -> 0.2)."""

    # "X% reduction" -> remaining = 1 - X/100
    m = re.search(r"(\d{1,3})\s*%\s*(?:reduc|cut|less)", text, re.IGNORECASE)
    if m:
        pct = float(m.group(1))
        return max(0.0, min(1.0, 1.0 - pct / 100.0))

    # "about half of the forecast" / "half of"
    if re.search(r"\b(about|roughly|approximately|around)?\s*half\b", text, re.IGNORECASE):
        return 0.5

    # "usable solar should be treated as roughly 25% of the forecast"
    # "leaves about X% usable"
    # "at about X%" / "of forecast" patterns
    m = re.search(
        r"(?:usable|usable\s+as|usable\s+solar\s+should\s+be\s+treated\s+as|"
        r"treated\s+as|treat\s+as|leaves|leaves\s+about|remain(?:ing)?|"
        r"forecast.*?at|forecast\s+of|forecast.*?roughly|approximately|"
        r"at\s+about|of\s+(?:the\s+)?forecast)\s*(?:roughly|about|around|approximately)?\s*"
        r"(\d{1,3})\s*%",
        text,
        re.IGNORECASE,
    )
    if m:
        return max(0.0, min(1.0, float(m.group(1)) / 100.0))

    # Catch-all: "<number>% of the forecast" / "<number>% of rooftop solar"
    m = re.search(r"(\d{1,3})\s*%\s*of\s+(?:the\s+)?(?:forecast|rooftop|solar|output)",
                  text, re.IGNORECASE)
    if m:
        return max(0.0, min(1.0, float(m.group(1)) / 100.0))

    # "down to 25%" / "to 25%"
    m = re.search(r"\b(?:down\s+to|to|only)\s+(\d{1,3})\s*%", text, re.IGNORECASE)
    if m:
        return max(0.0, min(1.0, float(m.group(1)) / 100.0))

    # "25% of forecast" / "factor of 0.2" already specified
    m = re.search(r"factor\s*(?:of|=|:)?\s*(0?\.\d+|1(?:\.0+)?|\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        if 0.0 <= v <= 1.0:
            return v

    return None


def _is_solar_reduction(text: str) -> bool:
    return bool(_SOLAR_KW.search(text)) and bool(_REDUCTION_KW.search(text))


# ---------------------------------------------------------------------------
# Charge / discharge windows
# ---------------------------------------------------------------------------

def _extract_window_directive(text: str) -> Optional[Tuple[str, List[int], str]]:
    """Detect a no_charge / no_discharge / max_grid directive in `text`.

    Returns (directive_type, hours_list, raw_value_or_None) or None.
    """

    lower = text.lower()
    hours = _extract_time_window(lower)
    if not hours:
        return None

    # max_grid_window patterns: must contain a numeric kWh cap.
    grid_kw = re.search(
        r"\b(grid|feeder|transformer|substation|import|intake|grid\s*intake)\b", lower
    )
    if grid_kw:
        m = re.search(
            r"(?:cap(?:ped|ping)?(?:\s+at)?|limit(?:ed|ing)?(?:\s+at)?|"
            r"at\s+most|no\s+more\s+than|maximum(?:\s+of)?|"
            r"not\s+exceed|exceed(?:ing)?|must\s+(?:not\s+exceed|stay\s+at\s+or\s+below\s+)?)"
            r"[^\d]{0,12}(\d+(?:\.\d+)?)\s*kwh",
            lower,
        )
        if m:
            return ("max_grid_window", hours, float(m.group(1)))

    # no_discharge: signals battery + discharge + prohibit.
    discharge_kw = re.search(r"\b(discharge|discharging)\b", lower)
    prohibit_kw = re.search(
        r"\b(not\s+discharge|disable[ds]?|prohibit|cannot|must\s+not|"
        r"no\s+discharge|do\s+not\s+discharge|cannot\s+discharge|"
        r"unavailable|offline|isolate[ds]?|isolate|"
        r"shall\s+not\s+discharge|relay\s+testing|protection\s+testing|"
        r"never\s+discharge|don['']?t\s+discharge)\b",
        lower,
    )
    if discharge_kw and prohibit_kw:
        return ("no_discharge_window", hours, None)

    # no_charge
    charge_kw = re.search(r"\b(charg(?:e|ing|er))\b", lower)
    prohibit_charge_kw = re.search(
        r"\b(not\s+charge|disable[ds]?|prohibit|cannot|must\s+not|"
        r"no\s+charge|do\s+not\s+charge|cannot\s+charge|"
        r"unavailable|offline|isolate[ds]?|isolated|"
        r"charger\s+(?:will\s+be\s+)?(?:unavailable|isolated|offline)|"
        r"shall\s+not\s+charge|don['']?t\s+charge|charging\s+(?:is\s+)?(?:disabled|unavailable))\b",
        lower,
    )
    if charge_kw and prohibit_charge_kw:
        return ("no_charge_window", hours, None)

    return None


# ---------------------------------------------------------------------------
# Minimum battery reserve
# ---------------------------------------------------------------------------

def _is_min_reserve(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(
            r"\b(reserve|emergency|backup|standby|minimum|at\s+least|keep)\b", lower
        )
        and re.search(r"\b(battery|batteries|storage)\b", lower)
    )


def _extract_min_reserve_value(text: str, capacity_kwh: float) -> Optional[float]:
    """Return absolute kWh required, converting percentage-of-capacity
    expressions like '50% of the battery capacity'."""

    lower = text.lower()

    # explicit kWh: "at least 90 kWh"
    m = re.search(
        r"(?:at\s+least|keep|maintain|hold|reserve(?:\s+of)?|minimum(?:\s+of)?|"
        r"require[sd]?|must\s+(?:have|remain|keep)|retain)\s+(\d+(?:\.\d+)?)\s*kwh",
        lower,
    )
    if m:
        return float(m.group(1))

    # percentage: "50% of the battery capacity"
    m = re.search(
        r"(\d{1,3})\s*%\s*of\s*(?:the\s*)?(?:battery\s*(?:capacity|storage)?|capacity|storage)",
        lower,
    )
    if m:
        pct = float(m.group(1)) / 100.0
        return pct * capacity_kwh

    # explicit fraction/percentage on its own with "battery" nearby
    m = re.search(r"(\d{1,3})\s*%\b", lower)
    if m and re.search(r"\b(battery|capacity)\b", lower):
        pct = float(m.group(1)) / 100.0
        return pct * capacity_kwh

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def interpret_notes(notes: List[str], capacity_kwh: float) -> List[dict]:
    """Map every note in `notes` to a directive_interpretation dict, in order.

    The output schema for every entry:
        {
            "note_index": int,
            "applies": bool,
            "directive_type": str,
            "structured_adjustment": dict | None,
            "explanation": str,
        }
    """

    interpretations: List[dict] = []
    for idx, raw in enumerate(notes):
        interpretations.append(_interpret_one(idx, raw, capacity_kwh))
    return interpretations


def _interpret_one(idx: int, raw: str, capacity_kwh: float) -> dict:
    text = raw.strip()

    # 1. Solar reduction -------------------------------------------------
    if _is_solar_reduction(text):
        hours = _extract_time_window(text.lower())
        factor = _extract_reduction_factor(text)
        if hours and factor is not None:
            return {
                "note_index": idx,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {
                    "hours": hours,
                    "factor": max(0.0, min(1.0, factor)),
                },
                "explanation": _summarize_solar(hours, factor),
            }
        # fall through: not enough info -> no_op

    # 2. Window directives (no_charge / no_discharge / max_grid) ---------
    window_match = _extract_window_directive(text)
    if window_match is not None:
        dtype, hours, value = window_match
        if dtype == "max_grid_window":
            return {
                "note_index": idx,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {
                    "hours": hours,
                    "max_grid_kwh": float(value),
                },
                "explanation": _summarize_maxgrid(hours, float(value)),
            }
        return {
            "note_index": idx,
            "applies": True,
            "directive_type": dtype,
            "structured_adjustment": {"hours": hours},
            "explanation": _summarize_window(dtype, hours),
        }

    # 3. Minimum battery reserve -----------------------------------------
    if _is_min_reserve(text):
        hours = _extract_time_window(text.lower())
        min_kwh = _extract_min_reserve_value(text, capacity_kwh)
        if hours and min_kwh is not None:
            return {
                "note_index": idx,
                "applies": True,
                "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {
                    "hours": hours,
                    "minimum_energy_kwh": float(min_kwh),
                },
                "explanation": _summarize_minres(hours, float(min_kwh), capacity_kwh),
            }

    # 4. Distractor -> no_op --------------------------------------------
    return {
        "note_index": idx,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "This note does not affect today's 24-hour energy schedule.",
    }


# ---------------------------------------------------------------------------
# Human-readable explanation helpers
# ---------------------------------------------------------------------------

def _summarize_solar(hours: List[int], factor: float) -> str:
    pct_left = round(factor * 100)
    if len(hours) == 1:
        when = f"hour {hours[0]}"
    else:
        when = "hours " + ", ".join(str(h) for h in hours)
    return f"Usable solar is reduced to {pct_left}% during {when}."


def _summarize_window(dtype: str, hours: List[int]) -> str:
    label = {
        "no_charge_window": "Battery charging is unavailable",
        "no_discharge_window": "Battery discharge is disabled",
    }[dtype]
    if len(hours) == 1:
        return f"{label} during hour {hours[0]}."
    return f"{label} during the stated window."


def _summarize_maxgrid(hours: List[int], cap: float) -> str:
    when = (
        f"hour {hours[0]}"
        if len(hours) == 1
        else "the stated window"
    )
    return f"Grid import is capped at {cap} kWh in {when}."


def _summarize_minres(hours: List[int], kwh: float, capacity_kwh: float) -> str:
    if abs(kwh - round(capacity_kwh * 0.5)) < 1e-3 and abs(capacity_kwh - 2 * kwh) < 1e-3:
        return f"Half of the {capacity_kwh} kWh battery is {kwh} kWh, which must remain available during the stated window."
    if len(hours) == 1:
        return f"A {kwh} kWh reserve is required during hour {hours[0]}."
    return f"A {kwh} kWh reserve is required during the stated window."
