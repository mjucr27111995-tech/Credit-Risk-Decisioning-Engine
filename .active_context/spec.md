# Technical Specification — Lauki Finance Credit Risk Scorecard

- **Ticket**: LF-CRM-1 (Phase 1 deliverable, per `SOW Credit Risk Model.pdf`)
- **Status**: APPROVED (implementation authorised)
- **Author**: Engineering
- **Supersedes**: `Project2_StreamlitApp_Resources/app/` (notebook-grade prototype)

## 1. Problem Statement

Loan officers at Lauki Finance (NBFC) need a real-time assessment of a loan
application that returns (a) probability of default, (b) a CIBIL-like credit
score in the 300–900 range, (c) a rating band of Poor / Average / Good /
Excellent, and (d) an explanation of *why*, because the SOW mandates high
explainability so the business can interpret and tweak model behaviour.

The trained model already exists (`artifacts/model_data.joblib`, logistic
regression, test AUC 0.98 / Gini 0.96). This work is **not** a modelling task —
it is productionising the inference path and UI.

## 2. Bounded Context

Single bounded context: **Credit Risk Assessment**.

- **Aggregate root**: `LoanApplication` — the consistency boundary. All
  validation invariants live here; nothing downstream re-validates.
- **Value objects**: `RiskRating`, `RecommendedAction`, `FeatureContribution`,
  `RiskAssessment` (immutable result).
- **Domain service**: `CreditRiskScorer` — turns an application into an
  assessment. No I/O, no framework types.
- **Infrastructure**: `ModelArtifact` (joblib load + validation),
  `FeatureVectorBuilder` (domain -> model matrix).
- **Presentation**: Streamlit UI. Depends inward only.

Ubiquitous language: PD (probability of default), DPD (days past due),
LTI (loan-to-income), delinquency ratio, credit utilisation, decile rank
ordering, KS statistic.

## 3. Layering & Dependency Rule

```
ui  ->  scoring  ->  domain
              \        ^
               \-> config / errors / observability (cross-cutting)
```

`domain` imports nothing from `scoring` or `ui`. Streamlit is referenced only
inside `credit_risk.ui`, so the core is reusable by the Phase 2 STP service.

## 4. Interfaces

```python
LoanApplication      # pydantic v2, validates at the boundary
CreditRiskScorer.score(application: LoanApplication) -> RiskAssessment
RiskAssessment(probability_of_default, credit_score, rating,
               recommended_action, contributions, model_version)
```

Scorecard transform (must stay numerically identical to the prototype):

```
credit_score = base_score + (1 - PD) * scale_length     # 300 + (1-PD)*600
Poor [300,500)  Average [500,650)  Good [650,750)  Excellent [750,900]
```

## 5. Feature Contract

13 model features, in the order persisted in the artifact:

`age, loan_tenure_months, number_of_open_accounts, credit_utilization_ratio,
loan_to_income, delinquency_ratio, avg_dpd_per_delinquency,
residence_type_Owned, residence_type_Rented, loan_purpose_Education,
loan_purpose_Home, loan_purpose_Personal, loan_type_Unsecured`

The persisted `MinMaxScaler` was fitted on 18 columns, 11 of which the model
never consumes. Those 11 are supplied as explicit, named placeholders and
dropped immediately after scaling. This is inherited technical debt from the
training notebook; it is isolated in one module (`scoring/features.py`) and
documented so it can be removed when the scaler is refitted.

## 6. Error Codes

| Code | Type | Status | Raised when |
|---|---|---|---|
| `CRM-VAL-001` | validation | 400 | Application fails domain invariants |
| `CRM-NOT-001` | not found | 404 | Model artifact file missing |
| `CRM-EXT-001` | external | 503 | Artifact unreadable / deserialisation failed |
| `CRM-CON-001` | conflict | 409 | Artifact schema mismatch (unexpected features) |
| `CRM-INT-001` | internal | 500 | Invalid configuration at startup |
| `CRM-INT-002` | internal | 500 | Scoring failed unexpectedly |

No exception is ever swallowed; every external boundary (disk, joblib, sklearn)
is wrapped and re-raised as a typed error.

## 7. Observability

structlog, JSON in non-dev. Business Process Events:

- **L1** `RISK_ASSESSMENT` — one per user-triggered assessment, never sampled.
- **L2** `MODEL_ARTIFACT_LOAD`, `FEATURE_VECTOR_BUILD`, `SCORECARD_MAPPING`.

Message shape: `credit-risk-scorecard | Event: <NAME> <SUCCESS|FAILURE> | <domainId>`
with `correlationID` bound per session and `domainId` = the assessment id.
PD, score and rating are logged; no PII (no names, no zipcodes) is logged.

## 8. Configuration

All configuration from the environment, prefix `CRM_`, bound to a typed
`Settings` model, validated at startup, crash with `CRM-INT-001` on failure.
Keys: `CRM_ARTIFACT_PATH`, `CRM_APP_ENV`, `CRM_LOG_LEVEL`, `CRM_LOG_JSON`,
`CRM_BASE_SCORE`, `CRM_SCALE_LENGTH`, `CRM_STP_ENABLED`.

## 9. Testing Strategy

- Unit (target >= 90%): scorecard band edges, feature vector contract,
  validation rejections, error codes, config validation. All I/O mocked.
- Integration: load the real artifact and assert **numeric parity** with the
  legacy `prediction_helper.py` (max abs PD delta < 1e-9), plus monotonicity
  (worse inputs never produce a higher score).
- Naming: `should_<action>_with<Condition>_<expectedResult>`.

## 10. Explicitly Out of Scope (YAGNI)

REST API, `/health/*` endpoints, rate limiting, auth, database persistence,
model retraining pipeline, drift monitoring. These belong to SOW Phase 2
(Monitoring / MLOps / STP) and are not built now. The core package is kept
framework-free so Phase 2 can wrap it without a rewrite.
