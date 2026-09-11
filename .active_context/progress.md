# Progress — LF-CRM-1

- **Status**: COMPLETE (pending human review of the UI)
- **Phase**: Validator — all gates passed
- **Ticket**: LF-CRM-1 (SOW Phase 1: model + scorecard + Streamlit UI)

## Validator gate results

| Gate | Result |
|---|---|
| `ruff check` | All checks passed |
| `ruff format --check` | 35 files already formatted |
| `mypy --strict` | No issues in 20 source files |
| `pytest` | 243 passed |
| Coverage | 97% overall; `config`, `errors`, `observability`, `domain/*`, `scoring/*` at **100%** |
| Legacy parity | PD matches the prototype to < 1e-9 across 4 profiles |
| Dataset integrity | 7 structural checks refuting the leakage hypothesis |
| Fairness | 20 checks: disparate impact, proxy, conditional parity, calibration, band saturation |
| Monotonicity | Worse delinquency / LTI never raises the score |
| Live smoke run | Streamlit serves HTTP 200, renders, no exceptions |

## Delivered

- Code limits enforced: every function <= 50 lines, every module <= 300
  lines, cyclomatic complexity <= 10 (checked by `ruff` C90 + a size audit).
- Layered project under `Credit_Risk_Modelling/` with `ui -> scoring -> domain`
  and a framework-free core, replacing the prototype in
  `Project2_StreamlitApp_Resources/app/`.
- Typed settings validated at startup; `CRM-*` error hierarchy; structlog BPE
  logging (L1 `RISK_ASSESSMENT` + three L2 events) with a per-session
  correlation id and no PII.
- `LoanApplication` aggregate enforcing training-range invariants;
  configurable scorecard whose bands rescale with the calibration.
- Artifact loader validating structure, feature contract, estimator linearity,
  coefficient count and scaler interface; version-skew warnings surfaced.
- Exact per-feature explainability (`coefficient x scaled value`) surfaced in
  the UI as a diverging bar chart plus a calculation-detail table.
- README and `docs/MODEL_CARD.md` documenting all seven known limitations.

## Decisions worth remembering

1. **`loan_to_income` is rounded to 2 dp**, matching the training notebook.
   The prototype omitted the rounding, so it fed the model a slightly
   different feature than it was fitted on. This is the only intentional
   behavioural change; it is asserted in `TestTrainingTimeRounding`.
2. **`disallow_any_explicit` is off** in mypy config: `np.ndarray` is generic
   over `Any` and pydantic's `BaseModel` metaclass is untyped, so the flag
   reports library noise rather than defects. `strict = true` remains on.
3. **Scaler placeholder debt retained** but isolated in
   `scoring/features.py` and covered by contract tests. Fixing it properly
   requires refitting the scaler, which would change model inputs.

## Not done, by design (YAGNI — SOW Phase 2)

REST API, `/health/*` endpoints, rate limiting, auth, persistence, retraining
pipeline, drift monitoring. The core is framework-free so Phase 2 can wrap it
without a rewrite.

## Recommended next step

**Fairness audit: COMPLETE** — see `docs/FAIRNESS_AUDIT.md`. No disparate impact
found (all groups clear the four-fifths threshold; gender proxy AUC 0.504).
The audit did surface a **calibration defect**: PD is inflated 1.78× because the
model was fitted on a class-balanced sample, mis-banding 12.4% of applications.
Deliberately **not** corrected — it would break parity with the prototype and
needs business sign-off. Pinned by `TestCalibrationDefect`.

Remaining items, in order of value:

1. **Redesign calibration and band cutoffs together.** The intercept shift alone
   is *not* safe: it promotes applicants defaulting at 23.6% and puts 89.5% of
   the portfolio in `Excellent`. The top band is already saturated (75.7% score
   ≥ 849). This is a credit-policy decision for Lauki Finance.
   *Requires sign-off — deviates from the prototype.*
2. **Qualified compliance review** against the RBI Fair Practices Code. The
   four-fifths rule used in the audit is a US convention, not Indian law.
3. **Confirm bureau vintage** with the data owner: prove snapshots are pulled
   pre-decision and enforce it in the ingestion contract.
4. **Re-serialise the artifact** under scikit-learn 1.9.0 to clear the version
   skew warning.
5. **Refit the scaler** on the model's 13 features only, retiring the
   placeholder workaround in `scoring/features.py`.
