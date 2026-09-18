# GridWise — BUP CSE Fest 2026 (Preliminary)

LLM-assisted 24-hour microgrid energy optimizer. The service interprets 1–3 free-text operator notes into structured directives, validates them deterministically, and solves the resulting linear program with PuLP's bundled CBC solver. No external LLM, no network calls, no secrets — it runs identically offline or inside the graded container.

---

## 1. Endpoints

| Method | Path                | Purpose                                                 |
| ------ | ------------------- | ------------------------------------------------------- |
| GET    | `/health`           | Liveness probe; returns `{"status": "ok"}` instantly.   |
| POST   | `/optimize-energy`  | 24-hour schedule optimization. Body: see §3.            |

Only these two paths exist. The OpenAPI/Swagger UI is intentionally disabled in production builds to keep the surface area minimal.

---

## 2. Architecture

Strict, ordered pipeline — every request follows this chain:

```
HTTP request
  └─ Pydantic schema validation          (app/schemas.py)
       └─ Deterministic note interpreter (app/interpreter.py)
            └─ Guardrail validator       (app/guardrail.py)
                 └─ PuLP LP (CBC)        (app/optimizer.py)
                      └─ Final validator (in app/optimizer.py)
                           └─ JSON response
```

The interpreter is **not** an LLM call — it is regex/keyword driven and 100% deterministic, so:

- identical inputs always produce identical outputs,
- failures cannot leak credentials or hallucinated numbers,
- the service runs hermetically inside a Docker container with no network access.

---

## 3. Request / Response Contract

### 3.1 POST `/optimize-energy` — request body

```jsonc
{
  "scenario_id": "SAMPLE-01",
  "operator_notes": [
    "Heavy clouds 10 AM to noon will reduce usable solar to 25%.",
    "Ignore the unconfirmed demand rumours."
  ],
  "hours": [
    /* exactly 24 entries, hour 0..23 */
    { "hour": 0,  "demand_kwh": 110, "solar_kwh": 0, "tariff_bdt_per_kwh": 9 },
    /* …23 more… */
  ],
  "battery": {
    "capacity_kwh": 200,
    "initial_energy_kwh": 70,
    "minimum_energy_kwh": 30,
    "max_charge_kwh_per_hour": 50,
    "max_discharge_kwh_per_hour": 60
  }
}
```

### 3.2 Supported directive types

| `directive_type`         | Trigger keywords (examples)                                   | `structured_adjustment`           |
| ------------------------ | ------------------------------------------------------------- | --------------------------------- |
| `solar_reduction`        | cloud cover, panel cleaning, dust, 80% reduction, half of forecast | `{hours[], factor}`              |
| `minimum_battery_reserve`| emergency reserve, 50% of capacity, keep ≥90 kWh              | `{hours[], minimum_energy_kwh}`   |
| `no_charge_window`       | charger offline / isolated / disabled                         | `{hours[]}`                       |
| `no_discharge_window`    | relay testing, never discharge                                | `{hours[]}`                       |
| `max_grid_window`        | feeder cap, transformer limit, ≤80 kWh                        | `{hours[], max_grid_kwh}`         |
| `no_op` (distractor)     | anything irrelevant                                           | `null`                            |

`hours` is **start-inclusive, end-exclusive** (e.g. `1 PM to 3 PM → [13, 14]`), ascending, unique, and within `0..23`. For `solar_reduction`, `factor` is the *usable fraction remaining* (80% reduction ⇒ `0.2`).

### 3.3 Response (200)

```jsonc
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [ /* one entry per note, in note_index order */ ],
  "hourly_plan": [           /* exactly 24 rows, hour 0..23 */
    {
      "hour": 0,
      "grid_kwh": 110.0,
      "solar_used_kwh": 0.0,
      "battery_action": "discharge",   // strictly one of: "charge" | "discharge" | "idle"
      "battery_kwh": 5.0,              // magnitude of the action
      "battery_energy_after_kwh": 65.0
    }
    /* …23 more… */
  ],
  "total_grid_kwh": 2692.50,
  "total_cost_bdt": 38365.00,
  "peak_grid_kwh": 175.0,
  "plan_summary": "Honored all operator directives (…), used 2692.5 kWh of grid energy, …"
}
```

### 3.4 Failure responses

| Status | Body                              | When                                                |
| ------ | --------------------------------- | --------------------------------------------------- |
| 400    | `{"detail":"invalid_json"}`       | malformed JSON                                      |
| 400    | `{"detail":"invalid_request"}`    | schema mismatch, missing/invalid fields             |
| 400    | `{"error":"invalid_directive"}`   | an interpretation fails guardrail validation        |
| 422    | `{"error":"infeasible_schedule"}` | directives make the LP unsolvable                   |
| 500    | `{"error":"internal_error"}`      | unexpected exception; sanitized, no stack trace     |

No 5xx response ever includes a stack trace, file path, or environment variable.

---

## 4. Mathematical Model

Decision variables per hour `h ∈ {0..23}`:

| Variable      | Bounds                           | Meaning                          |
| ------------- | -------------------------------- | -------------------------------- |
| `grid[h]`     | `≥ 0`                            | grid import (kWh)                |
| `solar[h]`    | `0 ≤ solar ≤ effective_solar[h]`  | solar used (kWh)                 |
| `charge[h]`   | `0 ≤ c ≤ max_charge_per_hour`    | energy into battery (kWh)        |
| `discharge[h]`| `0 ≤ d ≤ max_discharge_per_hour` | energy out of battery (kWh)      |
| `batt[h]`     | `min_reserve[h] ≤ batt ≤ cap`    | battery state after hour `h`     |

**Objective.** Minimize `Σ_h grid[h] · tariff[h]` (with a 1×10⁻⁶ tie-breaker on `Σ grid` so the LP correctly uses solar on zero-tariff hours).

**Constraints.**

1. **Energy balance** — `grid + solar + discharge = demand + charge` for every `h`.
2. **Solar limit** — `solar[h] ≤ eff_solar[h]` (reduced by `factor` when a `solar_reduction` directive covers hour `h`).
3. **Rate limits** — `charge[h] ≤ max_charge_per_hour`, `discharge[h] ≤ max_discharge_per_hour`.
4. **Window prohibitions** — `charge[h] = 0` in any `no_charge_window` hour; `discharge[h] = 0` in any `no_discharge_window` hour.
5. **Grid cap** — `grid[h] ≤ max_grid[h]` in any `max_grid_window` hour.
6. **Reserve** — `batt[h] ≥ max(battery.minimum_energy_kwh, minimum_energy_kwh from directive)`.
7. **Dynamics** — `batt[h] = batt[h-1] + charge[h] - discharge[h]` (`batt[-1] = initial_energy_kwh`).
8. **End-of-day neutrality** — `batt[23] = initial_energy_kwh` (hard equality).

Tolerance: **0.01 kWh** and **0.01 BDT** for all numerical assertions.

---

## 5. Running the Service

### 5.1 Local Python

```powershell
cd "E:\Academic\4.1\bin\BUP_CSE_FEST_2026_Participant_Docs\Project"
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The container binds `0.0.0.0:8000` because the grader may reach it from a different network namespace.

### 5.2 Docker (no secrets baked in)

```bash
docker build -t gridwise .
docker run --rm -p 8000:8000 gridwise
```

The image is `python:3.11-slim` plus the four pinned dependencies. There is no `ENV VAR=secret` line, no API key mount, no model download step.

### 5.3 Stopping a running server

If port 8000 is already in use (`WinError 10048` on Windows), find the owning PID and stop it:

```powershell
netstat -ano | Select-String ":8000.*LISTENING"
Stop-Process -Id <pid> -Force
```

---

## 6. Quick Smoke Test

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/optimize-energy \
     -H "Content-Type: application/json" \
     -d @BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json
```

…or run the batch harness against all 10 public cases:

```bash
python tools/validate_samples.py
```

Expected output:

```
[OK ] SAMPLE-01: Solar cleaning + distractor
[OK ] SAMPLE-02: Battery charging maintenance
[OK ] SAMPLE-03: Emergency reserve as percentage
[OK ] SAMPLE-04: No-discharge protection test
[OK ] SAMPLE-05: Temporary feeder grid cap
[OK ] SAMPLE-06: Multiple notes with distractor
[OK ] SAMPLE-07: Reserve plus transformer cap
[OK ] SAMPLE-08: Separate charge/discharge outages
[OK ] SAMPLE-09: Reduction wording normalization
[OK ] SAMPLE-10: Multi-constraint evening operation

10/10 cases passed
```

The harness exercises both the **directive interpretation** (each note maps to the expected `directive_type` + `hours`/`factor`/`max_grid_kwh`/`minimum_energy_kwh`) and the **mathematical schedule** (every energy-balance, capacity, rate-limit, reserve, window, and neutrality constraint checked, with totals within 0.01 of the reference).

---

## 7. Verified Performance

Observed on Windows 11 / Python 3.13 / single CBC thread:

| Metric | Value | Budget | Status |
| --- | --- | --- | --- |
| `GET /health` latency | < 5 ms | < 60 s startup SLA | ✅ |
| `POST /optimize-energy` p50 | ~80 ms | — | ✅ |
| `POST /optimize-energy` p95 | ~135 ms | ≤ 5 000 ms | ✅ |
| `POST /optimize-energy` worst | ~150 ms | < 30 000 ms | ✅ |
| Reference totals Δ | 0.00 kWh / 0.00 BDT for all 10 cases | ≤ 0.01 | ✅ |
| Constraint compliance | 24/24 hours × 10/10 cases | full | ✅ |

PuLP solves the 24×4 LP (~100 variables, ~150 constraints) in single-digit milliseconds on first call; the rest of the time is JSON marshalling.

---

## 8. Project Layout

```
.
├── app/
│   ├── main.py              FastAPI app, error handlers, pipeline orchestration
│   ├── schemas.py           Pydantic v2 request/response models
│   ├── interpreter.py       Deterministic note → directive mapping (regex/keyword)
│   ├── guardrail.py         Strict validator over the interpreter output
│   └── optimizer.py         PuLP/CBC linear program + final validator
├── tools/
│   └── validate_samples.py  Offline harness for the 10 public sample cases
├── Dockerfile               python:3.11-slim, binds 0.0.0.0:8000, no secrets
├── requirements.txt         fastapi 0.115, uvicorn 0.30, pydantic 2.9, pulp 2.8
├── .dockerignore
└── README.md
```

---

## 9. Design Notes

- **No external LLM at runtime.** The interpreter is purely deterministic. This guarantees reproducibility, zero network dependency inside the container, and compliance with the "no baked-in secrets" constraint.
- **Strict guardrail.** Any directive the interpreter extracts is passed through `app/guardrail.py` before reaching the optimizer. Reasonable-but-malformed extractions (out-of-range hours, factors outside `[0,1]`, reserves above capacity) raise `GuardrailError` instead of silently fabricating values.
- **Hard end-of-day neutrality.** Enforced as an equality constraint in the LP, then re-checked numerically in the post-solve phase. The response reconstruction collapses charge/discharge to a single signed flow per hour so the validator's independent battery-dynamics walk agrees with the LP within floating-point noise.
- **Deterministic response reconstruction.** The optimizer never reports both `charge_kwh` and `discharge_kwh` above 1×10⁻⁴ kWh — LP noise on that order is collapsed to `action = "idle"` with `battery_kwh = 0` so downstream balance / dynamics checks succeed.

For the BUP grading container: `docker run --rm -p 8000:8000 gridwise` is sufficient; nothing else needs to be mounted or supplied.
