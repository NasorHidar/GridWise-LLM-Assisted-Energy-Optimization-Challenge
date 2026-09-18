"""FastAPI entrypoint for the GridWise service."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .guardrail import GuardrailError, validate_interpretations
from .interpreter import interpret_notes
from .optimizer import InfeasibleError, optimize
from .schemas import (
    DirectiveInterpretation,
    HourlyPlanRow,
    OptimizeRequest,
    OptimizeResponse,
)

log = logging.getLogger("gridwise")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="GridWise Energy Optimizer",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Error handlers — never leak internals or stack traces.
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def _validation_exc(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_request",
            "message": "Request body failed schema validation.",
        },
    )


@app.exception_handler(json.JSONDecodeError)
async def _json_exc(_: Request, exc: json.JSONDecodeError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_json",
            "message": "Request body is not valid JSON.",
        },
    )


@app.exception_handler(GuardrailError)
async def _guardrail_exc(_: Request, exc: GuardrailError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_directive",
            "message": str(exc),
        },
    )


@app.exception_handler(InfeasibleError)
async def _infeasible_exc(_: Request, exc: InfeasibleError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "infeasible_schedule",
            "message": "The applied directives have no feasible schedule.",
        },
    )


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error in optimizer")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An internal error occurred. Please retry.",
        },
    )


# ---------------------------------------------------------------------------
# /optimize-energy
# ---------------------------------------------------------------------------

@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(request: Request) -> OptimizeResponse:
    """Orchestrate the LLM -> guardrail -> optimizer -> validator pipeline."""

    t0 = time.perf_counter()

    # 1. Parse + validate the incoming payload ourselves so we can return
    #    a uniform 400 for both malformed JSON and schema mismatches.
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid_json")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_request")

    try:
        payload = OptimizeRequest.model_validate(raw)
    except ValidationError as e:
        log.info("schema rejected: %s", e.errors()[0].get("msg", ""))
        raise HTTPException(status_code=400, detail="invalid_request")

    capacity = payload.battery.capacity_kwh
    hours = payload.hours
    demand = [h.demand_kwh for h in hours]
    solar = [h.solar_kwh for h in hours]
    tariff = [h.tariff_bdt_per_kwh for h in hours]
    battery = {
        "capacity_kwh": capacity,
        "initial_energy_kwh": payload.battery.initial_energy_kwh,
        "minimum_energy_kwh": payload.battery.minimum_energy_kwh,
        "max_charge_kwh_per_hour": payload.battery.max_charge_kwh_per_hour,
        "max_discharge_kwh_per_hour": payload.battery.max_discharge_kwh_per_hour,
    }

    # 2. LLM-equivalent: deterministic note interpretation.
    raw_interps = interpret_notes(payload.operator_notes, capacity_kwh=capacity)

    # 3. Guardrail: validate the structured interpretation.
    interps = validate_interpretations(
        raw_interps,
        n_notes=len(payload.operator_notes),
        capacity_kwh=capacity,
    )

    # 4. Mathematical optimizer (PuLP/CBC).
    solution = optimize(
        scenario_id=payload.scenario_id,
        demand=demand,
        solar=solar,
        tariff=tariff,
        battery=battery,
        interpretations=interps,
    )

    # 5. Final validator — recompute totals and check tolerances.
    hourly_plan = _finalize_plan(solution["hourly_plan"])
    total_grid = sum(row.grid_kwh for row in hourly_plan)
    total_cost = sum(row.grid_kwh * tariff[h] for h, row in enumerate(hourly_plan))
    peak_grid = max(row.grid_kwh for row in hourly_plan)

    if abs(total_grid - solution["total_grid_kwh"]) > 0.05:
        raise RuntimeError("total_grid_kwh mismatch between solver and recompute")
    if abs(total_cost - solution["total_cost_bdt"]) > 0.05:
        raise RuntimeError("total_cost_bdt mismatch between solver and recompute")

    # 6. Compose response.
    directive_interp = [
        DirectiveInterpretation.model_validate(d) for d in interps
    ]

    response = OptimizeResponse(
        scenario_id=payload.scenario_id,
        directive_interpretation=directive_interp,
        hourly_plan=hourly_plan,
        total_grid_kwh=round(total_grid, 4),
        total_cost_bdt=round(total_cost, 4),
        peak_grid_kwh=round(peak_grid, 4),
        plan_summary=solution["plan_summary"],
    )

    log.info(
        "scenario=%s notes=%d total_cost=%.2f total_grid=%.2f latency_ms=%.1f",
        payload.scenario_id,
        len(payload.operator_notes),
        total_cost,
        total_grid,
        (time.perf_counter() - t0) * 1000.0,
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finalize_plan(rows: list) -> list:
    """Convert the optimizer dicts into strict `HourlyPlanRow` objects with
    end-of-day neutrality snapped."""

    out: list = []
    for r in rows:
        out.append(
            HourlyPlanRow(
                hour=r["hour"],
                grid_kwh=round(float(r["grid_kwh"]), 4),
                solar_used_kwh=round(float(r["solar_used_kwh"]), 4),
                battery_action=r["battery_action"],
                battery_kwh=round(float(r["battery_kwh"]), 4),
                battery_energy_after_kwh=round(float(r["battery_energy_after_kwh"]), 4),
            )
        )
    return out
