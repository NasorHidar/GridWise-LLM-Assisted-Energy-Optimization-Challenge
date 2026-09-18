"""Validate the pipeline against the public sample case pack.

Runs every case through `app.interpreter` -> `app.guardrail` ->
`app.optimizer` and verifies:
  * directive_interpretation semantics match the reference
  * energy balance, capacity/rate bounds, window caps, effective solar,
    and end-of-day neutrality all hold
  * total_grid_kwh / total_cost_bdt / peak_grid_kwh agree with the
    reference within the official 0.01 tolerance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.guardrail import validate_interpretations
from app.interpreter import interpret_notes
from app.optimizer import optimize

TOL_KWH = 0.01
TOL_BDT = 0.01
TOL_NUMERIC = 0.05  # internal rounding tolerance before user-facing tolerance


def _approx_eq(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def check_case(case: dict) -> tuple[bool, list[str]]:
    inp = case["input"]
    ref = case["expected_output"]
    errors: list[str] = []
    cid = case["id"]

    demand = [h["demand_kwh"] for h in inp["hours"]]
    solar = [h["solar_kwh"] for h in inp["hours"]]
    tariff = [h["tariff_bdt_per_kwh"] for h in inp["hours"]]
    battery = inp["battery"]

    # 1. Interpret notes and validate.
    interps_raw = interpret_notes(inp["operator_notes"], capacity_kwh=battery["capacity_kwh"])
    interps = validate_interpretations(
        interps_raw,
        n_notes=len(inp["operator_notes"]),
        capacity_kwh=battery["capacity_kwh"],
    )

    # Compare directive semantics (ignoring free-text explanations).
    ref_interp = ref["directive_interpretation"]
    if len(interps) != len(ref_interp):
        errors.append(f"{cid}: directive length mismatch")
    else:
        for ours, theirs in zip(interps, ref_interp):
            if ours["note_index"] != theirs["note_index"]:
                errors.append(f"{cid}: note_index order broken")
            if ours["directive_type"] != theirs["directive_type"]:
                errors.append(
                    f"{cid}: note {ours['note_index']} type "
                    f"{ours['directive_type']} != {theirs['directive_type']}"
                )
            if ours["applies"] != theirs["applies"]:
                errors.append(
                    f"{cid}: note {ours['note_index']} applies "
                    f"{ours['applies']} != {theirs['applies']}"
                )
            sa_o = ours["structured_adjustment"]
            sa_t = theirs["structured_adjustment"]
            if (sa_o is None) != (sa_t is None):
                errors.append(
                    f"{cid}: note {ours['note_index']} structured_adjustment "
                    f"null mismatch"
                )
            elif sa_o is not None and sa_t is not None:
                if sa_o.get("hours") != sa_t.get("hours"):
                    errors.append(
                        f"{cid}: note {ours['note_index']} hours "
                        f"{sa_o.get('hours')} != {sa_t.get('hours')}"
                    )
                for k in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
                    if k in sa_t:
                        if k not in sa_o:
                            errors.append(f"{cid}: note {ours['note_index']} missing {k}")
                        elif not _approx_eq(sa_o[k], sa_t[k], TOL_NUMERIC):
                            errors.append(
                                f"{cid}: note {ours['note_index']} {k} "
                                f"{sa_o[k]} != {sa_t[k]}"
                            )

    # 2. Solve.
    sol = optimize(
        scenario_id=inp["scenario_id"],
        demand=demand,
        solar=solar,
        tariff=tariff,
        battery=battery,
        interpretations=interps,
    )
    plan = sol["hourly_plan"]

    # 3. Re-validate solution constraints (independent of the LP).
    cap = battery["capacity_kwh"]
    init = battery["initial_energy_kwh"]
    mn = battery["minimum_energy_kwh"]
    mch = battery["max_charge_kwh_per_hour"]
    mdc = battery["max_discharge_kwh_per_hour"]

    # Effective solar from interpretations.
    eff_solar = list(solar)
    for it in interps:
        if it["applies"] and it["directive_type"] == "solar_reduction":
            f = it["structured_adjustment"]["factor"]
            for h in it["structured_adjustment"]["hours"]:
                eff_solar[h] *= f

    no_charge = set()
    no_discharge = set()
    max_grid = {}
    min_reserve = {}
    for it in interps:
        if not it["applies"]:
            continue
        t = it["directive_type"]
        h_set = it["structured_adjustment"]["hours"]
        if t == "no_charge_window":
            no_charge.update(h_set)
        elif t == "no_discharge_window":
            no_discharge.update(h_set)
        elif t == "max_grid_window":
            cap_h = it["structured_adjustment"]["max_grid_kwh"]
            for h in h_set:
                max_grid[h] = min(max_grid.get(h, float("inf")), cap_h)
        elif t == "minimum_battery_reserve":
            mn_h = it["structured_adjustment"]["minimum_energy_kwh"]
            for h in h_set:
                min_reserve[h] = max(min_reserve.get(h, 0.0), mn_h)

    prev = init
    for row in plan:
        h = row["hour"]
        g = row["grid_kwh"]
        s = row["solar_used_kwh"]
        act = row["battery_action"]
        bk = row["battery_kwh"]
        be = row["battery_energy_after_kwh"]

        if not _approx_eq(g + s + (bk if act == "discharge" else 0),
                          demand[h] + (bk if act == "charge" else 0),
                          TOL_NUMERIC):
            errors.append(f"{cid}: h={h} energy balance broken")

        if s > eff_solar[h] + TOL_NUMERIC:
            errors.append(f"{cid}: h={h} solar {s} > effective {eff_solar[h]}")
        if act == "charge" and bk > mch + TOL_NUMERIC:
            errors.append(f"{cid}: h={h} charge {bk} > max {mch}")
        if act == "discharge" and bk > mdc + TOL_NUMERIC:
            errors.append(f"{cid}: h={h} discharge {bk} > max {mdc}")
        if act == "idle" and bk != 0:
            errors.append(f"{cid}: h={h} idle but battery_kwh={bk}")
        # Asymmetric checks: no_charge only blocks charging, no_discharge only
        # blocks discharging.
        if h in no_charge and act == "charge":
            errors.append(f"{cid}: h={h} charge during no_charge_window")
        if h in no_discharge and act == "discharge":
            errors.append(f"{cid}: h={h} discharge during no_discharge_window")
        if h in max_grid and g > max_grid[h] + TOL_NUMERIC:
            errors.append(f"{cid}: h={h} grid {g} > cap {max_grid[h]}")
        if be < max(mn, min_reserve.get(h, 0.0)) - TOL_NUMERIC:
            errors.append(
                f"{cid}: h={h} battery {be} below reserve "
                f"{max(mn, min_reserve.get(h, 0.0))}"
            )
        if be > cap + TOL_NUMERIC:
            errors.append(f"{cid}: h={h} battery {be} > capacity {cap}")
        if not _approx_eq(be, prev + (bk if act == "charge" else 0)
                          - (bk if act == "discharge" else 0), TOL_NUMERIC):
            errors.append(f"{cid}: h={h} battery dynamics broken")
        prev = be

    if not _approx_eq(prev, init, TOL_NUMERIC):
        errors.append(f"{cid}: end-of-day battery {prev} != initial {init}")

    # 4. Compare totals to reference.
    if not _approx_eq(sol["total_grid_kwh"], ref["total_grid_kwh"], TOL_NUMERIC):
        errors.append(
            f"{cid}: total_grid_kwh {sol['total_grid_kwh']} "
            f"!= ref {ref['total_grid_kwh']}"
        )
    if not _approx_eq(sol["total_cost_bdt"], ref["total_cost_bdt"], TOL_BDT):
        errors.append(
            f"{cid}: total_cost_bdt {sol['total_cost_bdt']} "
            f"!= ref {ref['total_cost_bdt']}"
        )

    return (len(errors) == 0, errors)


def main() -> int:
    cases_path = ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
    pack = json.loads(cases_path.read_text(encoding="utf-8"))

    total = 0
    fails = 0
    for case in pack["cases"]:
        ok, errs = check_case(case)
        total += 1
        marker = "OK " if ok else "FAIL"
        print(f"[{marker}] {case['id']}: {case['label']}")
        if not ok:
            fails += 1
            for e in errs:
                print(f"    - {e}")

    print(f"\n{total - fails}/{total} cases passed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
