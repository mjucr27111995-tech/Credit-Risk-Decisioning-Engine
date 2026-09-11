# Model Card — Lauki Finance Credit Risk Scorecard

| | |
|---|---|
| **Model** | Logistic regression (binary classification) |
| **Version** | `lr-e127bcce15da` (SHA-256 prefix of the artifact file) |
| **Artifact** | `artifacts/model_data.joblib` |
| **Owner** | AtliQ AI, Data Science |
| **Intended use** | Decision support for Lauki Finance loan officers |
| **Trained with** | scikit-learn 1.3.0 |
| **Serving with** | scikit-learn 1.9.0 — see *Version skew* below |

## Intended use and users

Assists loan officers in triaging retail loan applications by estimating the
probability of default and mapping it to a 300–900 credit score.

**In scope:** ranking and triaging applications; flagging high-confidence cases
for Straight-Through Processing; explaining a decision to a reviewer.

**Out of scope:** the sole basis for declining an applicant; pricing; collections
strategy; any population unlike the training data (non-retail, non-India,
commercial lending).

## Training data

Three joined tables, 50,000 loans, one row per loan:

| Source | Contents |
|---|---|
| `customers.csv` | Age, gender, marital status, employment, income, dependants, residence, address tenure, city/state/zip |
| `loans.csv` | Purpose, type, sanction/loan amount, fees, GST, disbursement, tenure, outstanding, bank balance, dates, `default` |
| `bureau_data.csv` | Open/closed accounts, total loan months, delinquent months, total DPD, enquiry count, credit utilisation |

The target `default` is class-imbalanced.

## Preprocessing

1. Stratified 75/25 train/test split **before** EDA, to prevent leakage into
   feature decisions.
2. `residence_type` nulls (47 rows) imputed with the mode (`Owned`).
3. Outliers removed by business rule: `processing_fee / loan_amount >= 3%`.
   Validation rules confirmed GST <= 20% and `net_disbursement <= loan_amount`.
4. Typo corrected: `Personaal` → `Personal`.
5. Engineered ratios: `loan_to_income`, `delinquency_ratio`,
   `avg_dpd_per_delinquency`.
6. Identifiers and business-excluded raw columns dropped.
7. MinMax scaling, VIF screening for multicollinearity, WOE/IV selection at
   `IV > 0.02`, then one-hot encoding with `drop_first=True`.

## Model selection

Four attempts: baseline LR/RF/XGB; RandomUnderSampler; SMOTETomek + Optuna-tuned
LR; SMOTETomek + Optuna-tuned XGB.

Logistic regression was chosen over XGBoost. Performance was comparable, and the
SOW mandates high explainability so the business can interpret and adjust model
behaviour — a linear model makes every decision decomposable.

Hyperparameters: `C=9.3734`, `solver='saga'`, `tol=0.0178`, `penalty='l2'`.

## Features and coefficients

Positive raises the log-odds of default. Coefficients apply to **MinMax-scaled**
inputs, so magnitudes are comparable.

| Feature | Coefficient | Direction |
|---|---:|---|
| `loan_to_income` | +18.1017 | Higher borrowing vs income → riskier |
| `credit_utilization_ratio` | +16.1752 | Maxed-out credit → riskier |
| `delinquency_ratio` | +13.9319 | More delinquent months → riskier |
| `loan_purpose_Home` | −3.7006 | Home loans safest |
| `avg_dpd_per_delinquency` | +2.0790 | Deeper delinquency → riskier |
| `residence_type_Rented` | +1.8945 | Renting → riskier |
| `residence_type_Owned` | −1.8319 | Owning → safer |
| `number_of_open_accounts` | +1.1719 | More open accounts → riskier |
| `loan_purpose_Personal` | +1.0859 | |
| `loan_type_Unsecured` | +1.0859 | No collateral → riskier |
| `loan_purpose_Education` | +0.9551 | |
| `loan_tenure_months` | +0.6423 | Longer tenure → riskier |
| `age` | +0.0574 | Negligible |

Intercept: **−21.2835**. Reference (all-zero) categories: `Mortgage`, `Auto`,
`Secured`.

## Reported performance

| Metric | Test set |
|---|---|
| AUC | 0.98 |
| Gini | 0.96 |
| Rank ordering | Monotonic across deciles |
| KS statistic | Computed per decile |

**These numbers reflect a synthetic teaching dataset and will not transfer to a
real portfolio. See limitation 1.**

## Scorecard

```
credit_score = 300 + (1 − PD) × 600
```

| Rating | Score | Action |
|---|---|---|
| Excellent | 750–900 | Auto-approve (STP eligible) |
| Good | 650–749 | Approve with manual review |
| Average | 500–649 | Refer to underwriter |
| Poor | 300–499 | Decline (STP eligible) |

## Limitations and risks

### 1. Synthetic data — the most important caveat

**Investigated and corrected.** An earlier draft of this model card asserted
target leakage via `delinquency_ratio` and `avg_dpd_per_delinquency`. The data
does not support that claim, and it has been withdrawn. Evidence:

| Check | Result | Reading |
|---|---:|---|
| `total_loan_months` mean vs current `loan_tenure_months` mean | 76 vs 26 | Bureau months span far more than this loan |
| Rows where `total_loan_months == loan_tenure_months` | 0.6% | Bureau is not describing this loan |
| Mean open + closed accounts per customer | 3.5 | Customer holds ~3.5 accounts... |
| Loans held with Lauki Finance per customer | 1.0 | ...so bureau reports **other lenders'** accounts |
| Univariate AUC of `delinquency_ratio` | 0.726 | Predictive, nowhere near deterministic |
| Univariate AUC of `avg_dpd_per_delinquency` | 0.654 | Same |
| Default rate where `delinquency_ratio > 75` | 57.6% | A leaked target would sit near 100% |
| AUC dropping both features | 0.983 → 0.938 | They carry 0.045 AUC, not the bulk |

The features describe the applicant's **prior repayment history with other
lenders** — precisely what a CIBIL report provides, and precisely what is
available when an application is submitted. They are legitimate inputs.

The strongest single predictor is in fact `credit_utilization_ratio`
(univariate AUC 0.877), an unambiguously application-time bureau field.

**The real caveat is provenance, not leakage.** This is a synthetic teaching
dataset with a clean generative relationship between the bureau fields and the
target. Real portfolios are noisier, so **AUC 0.98 / Gini 0.96 will not
transfer to production**. Treat them as a property of this dataset, not as a
forecast of live performance.

### 1a. Unverifiable bureau vintage

`bureau_data.csv` carries no as-of date, so it cannot be proven from the data
alone that each snapshot was pulled **before** the corresponding loan was
disbursed. The structural evidence above indicates it reflects external
history, but a production deployment should confirm with the data owner that
bureau pulls are timestamped and strictly pre-decision, then enforce that in
the ingestion contract.

### 2. Distributional validity

Coefficients apply to MinMax-scaled inputs, so the model extrapolates poorly
outside the training range. Input bounds are enforced by `LoanApplication`
(age 18–100, tenure 1–360, ratios 0–100, open accounts 1–20) and out-of-range
values are rejected rather than silently extrapolated.

### 3. Scaler placeholder debt

The persisted `MinMaxScaler` was fitted on 18 columns; the model consumes 13.
The 11 unused columns are supplied as named placeholders (constant `1.0`) purely
to satisfy `scaler.transform`, then dropped before prediction. They cannot
influence the output.

Confined to `src/credit_risk/scoring/features.py` and covered by contract tests.
**Fix:** refit the scaler on the model's feature set only.

### 4. Version skew

Pickled with scikit-learn 1.3.0, served under 1.9.0. The loader surfaces this as
a warning on every load; integration parity tests guard the behaviour. **Fix:**
re-serialise under the pinned serving version.

### 5. Fairness

`gender` and `marital_status` were available in the source data and are **not**
model features. However, no formal fairness audit (disparate impact across
protected groups) has been performed. Proxy discrimination via `zipcode`-adjacent
signals has not been ruled out. Recommended before production.

### 6. Pickle as a trust boundary

`joblib.load` executes arbitrary code. The artifact is trusted first-party code
shipped with the application. **Never** point `CRM_ARTIFACT_PATH` at a
user-supplied file.

### 7. No monitoring

No drift detection, no PSI tracking, no scheduled recalibration. Deferred to SOW
Phase 2.

## Deliberate deviation from the prototype

`Project2_StreamlitApp_Resources/app/prediction_helper.py` computed
`loan_to_income` as `loan_amount / income` **unrounded**, while the training
notebook used `round(..., 2)`. The prototype therefore fed the model a slightly
different feature than the one it was fitted on.

This project follows the **training-time** definition. Parity with the prototype
is otherwise exact to within `1e-9`, verified in
`tests/integration/test_real_model_parity.py`.

## Maintenance

| Trigger | Action |
|---|---|
| Any artifact change | Re-run the parity and monotonicity suites |
| scikit-learn upgrade | Re-serialise the artifact; re-run all tests |
| Phase 2 kickoff | Confirm bureau snapshots are timestamped pre-decision; add drift monitoring |
| Quarterly | Review rank ordering and KS on recent production data |
