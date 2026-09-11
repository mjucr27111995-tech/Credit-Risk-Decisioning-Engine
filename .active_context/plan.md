# Implementation Plan — LF-CRM-1

## Phase 1 — Planner (done)
- [x] Feasibility: model artifact exists, 13-feature contract readable, deps installed.
- [x] System design: layering, bounded context, interfaces -> `spec.md`.
- [x] Architecture review: dependency rule verified (domain has zero outward imports).

## Phase 2 — Generator
1. [x] Scaffold: `pyproject.toml`, pinned `requirements*.txt`, `.gitignore`, theme.
2. [x] Cross-cutting: `errors.py`, `config.py`, `observability.py`.
3. [x] Domain: `domain/models.py` (enums + aggregate + value objects), `domain/scorecard.py`.
4. [x] Scoring: `scoring/artifact.py`, `scoring/features.py`, `scoring/scorer.py`.
5. [x] UI: `ui/theme.py`, `ui/inputs.py`, `ui/charts.py`, `ui/app.py`, root `app.py`.
6. [x] Tests alongside each module (unit) + parity integration test.

## Phase 3 — Validator
- [x] `ruff`/`mypy` clean (where available in env).
- [x] `pytest` green.
- [x] Numeric parity vs legacy prototype proven.
- [x] Headless Streamlit smoke run with no exceptions.

## Risks
| Risk | Mitigation |
|---|---|
| sklearn 1.9 unpickling a 1.3.0 artifact | Warn loudly at load; parity test guards behaviour; pin versions |
| Scaler placeholder hack | Isolated + documented in one module; covered by contract test |
| Suspected target leakage in DPD-derived features | **Investigated and refuted.** Bureau data describes ~3.5 external accounts per customer vs 1 loan held here; dropping both features costs only 0.045 AUC. Evidence encoded in `tests/integration/test_dataset_integrity.py`. Real caveat is synthetic-data provenance, documented in `docs/MODEL_CARD.md`. |
