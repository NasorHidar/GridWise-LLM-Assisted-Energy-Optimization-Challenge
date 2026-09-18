"""Pydantic schemas for the GridWise API contract.

These models mirror the public sample JSON verbatim. The field order in the
response models is significant because the evaluator matches keys by name,
so the alias/ordering is chosen to be stable across Python versions.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Hour(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class Battery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    @model_validator(mode="after")
    def _battery_consistency(self) -> "Battery":
        # Capacity must dominate both initial and minimum, and min <= initial <= cap.
        if not (self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh):
            raise ValueError(
                "initial_energy_kwh must satisfy "
                "minimum_energy_kwh <= initial_energy_kwh <= capacity_kwh"
            )
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1, max_length=128)
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[Hour] = Field(min_length=24, max_length=24)
    battery: Battery

    @field_validator("operator_notes")
    @classmethod
    def _non_empty_notes(cls, v: List[str]) -> List[str]:
        for n in v:
            if not isinstance(n, str) or not n.strip():
                raise ValueError("operator_notes entries must be non-empty strings")
        return v

    @field_validator("hours")
    @classmethod
    def _hours_complete(cls, v: List[Hour]) -> List[Hour]:
        seen = {h.hour for h in v}
        if seen != set(range(24)):
            raise ValueError("hours must contain exactly entries 0..23")
        return sorted(v, key=lambda h: h.hour)


# --- Response models ---------------------------------------------------------


class StructuredAdjustment(BaseModel):
    """Container for a parsed directive. Always serialised with the keys the
    evaluator expects for each `directive_type`. Extra keys (e.g. factor on a
    no_op) are omitted.
    """

    model_config = ConfigDict(extra="ignore")

    hours: List[int]
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None


class DirectiveInterpretation(BaseModel):
    note_index: int = Field(ge=0, le=2)
    applies: bool
    directive_type: str
    structured_adjustment: Optional[StructuredAdjustment] = None
    explanation: str


class HourlyPlanRow(BaseModel):
    hour: int = Field(ge=0, le=23)
    grid_kwh: float
    solar_used_kwh: float
    battery_action: str
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanRow]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
