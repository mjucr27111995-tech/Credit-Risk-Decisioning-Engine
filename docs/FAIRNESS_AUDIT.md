# Fairness Audit — Credit Risk Scorecard

**Model version:** `lr-e127bcce15da`
**Population:** 49,985 applications (full dataset, post-cleaning)
**Method:** every application scored through the shipped artifact. The vectorised
replication used here was verified against the production scoring path on a
random sample — maximum deviation **7.77e-16**, so these numbers describe the
model that actually ships.
**Reproduce:** `python3 scripts/fairness_audit.py`

> **No model, feature, or scoring change was made as a result of this audit.**
> Numeric parity with `Project2_StreamlitApp_Resources` remains exact. Every
> recommendation below is a proposal awaiting sign-off, not an applied change.

---

## Verdict

| Question | Answer |
|---|---|
| Material disparate impact on gender? | **No** — impact ratio 0.996, and the features carry no gender signal at all |
| Material disparate impact on marital / employment status? | **No** — 0.983 / 0.980 |
| Age disparity? | **Present but risk-justified** — see below |
| Geographic disparity? | **No** — worst city ratio 0.990 |
| **Any defect found?** | **Yes — but it is calibration, not bias.** See [the calibration defect](#the-real-defect-probabilities-are-inflated-178x) |

The audit's headline is not a fairness problem. It is that the model
**overstates probability of default by 1.78×**, which causes the scorecard to
reject applicants it should approve.

---

## 1. Disparate impact

Policy modelled: **approve at credit score ≥ 650** (the "Good" band boundary).
Impact ratio = group approval rate ÷ best group's approval rate.

### Gender

| Group | n | Approval | Actual default | Mean predicted PD | AUC | Impact ratio |
|---|---:|---:|---:|---:|---:|---:|
| M | 29,980 | 84.50% | 8.47% | 15.23% | 0.9829 | 1.000 |
| F | 20,005 | 84.19% | 8.79% | 15.41% | 0.9833 | **0.996** |

A 0.31 percentage-point approval gap, against a 0.32 point gap in *actual*
default rates. Predictive accuracy is equal across groups (AUC differs by
0.0004).

### Other attributes

| Attribute | Worst group | Impact ratio | Verdict |
|---|---|---:|---|
| Marital status | Single | 0.983 | Pass |
| Employment status | Salaried | 0.980 | Pass |
| Dependants (0–4) | 1 dependant | 0.978 | Pass |
| City (10 cities) | Chennai | 0.990 | Pass |
| **Age** | 18–25 | **0.865** | Pass, and risk-justified |

All groups clear the four-fifths (80%) threshold.

> **Jurisdiction caveat.** The four-fifths rule is a US EEOC/Reg-B convention,
> not Indian law. India has no codified disparate-impact test for lending; the
> RBI Fair Practices Code requires non-discrimination without prescribing a
> statistical threshold. The 80% rule is used here as a recognised industry
> benchmark, not a compliance certification. **A qualified compliance review is
> still required before production lending.**

## 2. Proxy discrimination

Protected attributes are correctly excluded from the feature set — but excluded
attributes can still be reconstructed from correlated features. Test: train a
gradient-boosted classifier to predict each protected attribute *from the 13
model features*. AUC 0.50 means no signal; > 0.60 means a usable proxy.

| Attribute | Proxy AUC | Reading |
|---|---:|---|
| Gender | **0.504** | No signal whatsoever. Gender is genuinely absent. |
| Marital status | 0.668 | Partially reconstructible |
| Employment status | 0.673 | Partially reconstructible |

Marital and employment status *are* partially inferable from the features.
This matters less than it appears, because their realised impact ratios are
0.983 and 0.980 — the proxy signal exists but does not translate into unequal
approval. Worth re-testing if features change.

## 3. Conditional parity

Within a score band, do groups default at the same rate? If yes, the score
means the same thing for everyone.

| Band | Gender gap | Marital gap | Employment gap |
|---|---:|---:|---:|
| 300–499 | 2.24 pp | 0.46 pp | 0.19 pp |
| 500–649 | 1.29 pp | 2.23 pp | 0.41 pp |
| 650–749 | 0.82 pp | 0.02 pp | 1.31 pp |
| 750–900 | 0.04 pp | 0.01 pp | 0.01 pp |

Largest gap 2.24 pp, in the smallest and highest-risk band. A given score
carries essentially the same meaning across groups.

## 4. Age: disparity without bias

`age` is a direct model feature (coefficient +0.0574 — the *weakest* of the 13).
Younger applicants are approved less often:

| Age band | n | Approval | Actual default | Risk vs. best | Approval vs. best |
|---|---:|---:|---:|---:|---:|
| 56+ | 2,727 | 91.31% | 4.47% | 1.00× | 1.000 |
| 46–55 | 10,986 | 87.78% | 6.56% | 1.47× | 0.961 |
| 36–45 | 19,074 | 84.32% | 8.58% | 1.92× | 0.923 |
| 26–35 | 13,190 | 81.83% | 10.09% | 2.26× | 0.896 |
| 18–25 | 4,008 | 78.99% | 12.13% | **2.71×** | **0.865** |

Applicants aged 18–25 carry **2.71× the default risk** of the 56+ group but
receive only **13.5% lower approval**. The model is *less* discriminating than
the underlying risk would justify — the disparity tracks real, measured
repayment behaviour rather than amplifying it.

This is legitimate risk-based differentiation, not bias. It is still worth
disclosing: age-based differentiation is restricted in some jurisdictions
(ECOA in the US), and a 0.865 ratio is the closest any group comes to the
threshold.

---

## The real defect: probabilities are inflated 1.78×

The audit's most consequential finding is not about fairness.

| Metric | Value |
|---|---:|
| Mean predicted PD | **15.30%** |
| Actual default rate | **8.60%** |
| Overprediction | **1.78×** |

The gap is present in every group and every risk decile:

| Decile | n | Actual | Predicted | Ratio |
|---:|---:|---:|---:|---:|
| 5 | 4,998 | 0.02% | 0.19% | 9.5× |
| 6 | 4,998 | 0.26% | 1.21% | 4.7× |
| 7 | 4,999 | 0.94% | 8.65% | 9.2× |
| 8 | 4,998 | 13.15% | 47.68% | 3.6× |
| 9 | 4,999 | 71.59% | 95.24% | 1.3× |

### Cause — confirmed from the training notebook

The training notebook rebalanced classes before fitting. The chain is verified
end to end in `Project2_KSS_ME_Resources/credit_risk_model_codebasics.ipynb`:

| Cell | What it shows |
|---|---|
| 11 | Original balance: 45,703 non-default / 4,297 default = **8.59%** |
| 117 | `SMOTETomek` resamples to 34,195 / 34,195 = **exactly 50/50** |
| 121 | Best Optuna params `C=9.373445810448587, solver=saga, tol=0.01780541278077913, class_weight=None`, fitted on `X_train_smt` — the 50/50 data |
| 144 | `final_model = best_model_logistic` |
| 150 | Dumped to `artifacts/model_data.joblib` |

Those hyperparameters match the shipped artifact exactly, so the model in
production is confirmed to be the one fitted on the 50/50 sample. `class_weight`
was `None`, so no further reweighting offsets it.

A logistic regression fitted on a 50/50 sample learns the *sample's* base rate,
not the population's 8.6%. Rank ordering survives — which is why AUC is 0.983
and the model is still a good discriminator — but the absolute probabilities are
shifted upward.

The notebook's own output already hinted at this: at the default 0.5 threshold
it reported **precision 0.57 with recall 0.94** for the default class — the
signature of a model that flags far more defaults than actually occur.

### Why it matters

The scorecard maps PD directly to a 300–900 score, so inflated PDs mean
systematically depressed scores. **12.4%** of applications (6,205 of 49,985) sit
in a different band than they would if calibrated, and the approval rate at a
≥ 650 cutoff would move from **84.4%** to **91.5%**.

### But the correction is *not* a clean win — investigated further

An earlier draft of this document claimed the correction would rescue
"≈ 3,575 creditworthy applicants wrongly rejected." **That framing was wrong and
is withdrawn.** Checking who actually moves:

| Group | n | Actual default rate |
|---|---:|---:|
| Currently rejected (< 650) who would be approved | 3,577 | **23.60%** |
| Portfolio average | 49,985 | 8.60% |

The applicants the correction promotes default at **2.7× the portfolio average**.
They are not wrongly rejected — they are genuinely higher risk. Approving them
would raise realised losses.

Worse, the correction **destroys the rating bands**:

| | Poor | Average | Good | Excellent |
|---|---:|---:|---:|---:|
| Now | 12.1% | 3.5% | 3.0% | 81.4% |
| Corrected | 6.2% | 2.2% | 2.0% | **89.5%** |

Nearly 90% of applicants would land in `Excellent`, and `Average`/`Good` become
nearly vacant. Concrete example — three real applicants, all currently scoring
649, one with a 40% delinquency ratio and 83% credit utilisation, jump to a
score of **862** ("auto-approve, STP eligible") under the correction. That is
plainly wrong regardless of what the calibration arithmetic says.

### Root cause is the score mapping, not the intercept

The scorecard is a *linear* map of a *heavily skewed* PD distribution, so it
inherits that skew:

| Percentile | PD now |
|---|---:|
| p50 | 0.07% |
| p75 | 7.46% |
| p90 | 81.17% |

The median applicant already has a near-zero PD. Consequently **75.7% of all
applicants already score ≥ 849** — the `Excellent` band is saturated *before*
any correction. Fixing the intercept makes that saturation worse, not better.

### Recommendation: do not apply the intercept shift alone

Calibration and banding must be addressed together. A better direction is to
keep PD as-is and set band cutoffs by **actual risk** rather than fixed score
arithmetic:

| Proposed band | n | Share | Actual default rate |
|---|---:|---:|---:|
| Excellent (PD < 2%) | 39,744 | 79.5% | 0.15% |
| Good (2–6%) | 2,321 | 4.6% | 4.74% |
| Average (6–15%) | 1,755 | 3.5% | 13.90% |
| Poor (> 15%) | 6,165 | 12.3% | 62.98% |

These bands are monotonic and meaningfully separated, and each label carries an
honest risk interpretation.

**Nothing here has been applied.** Both options change scores and would break
numeric parity with `Project2_StreamlitApp_Resources`. This is a credit-policy
decision for Lauki Finance, not an engineering one.

### Proposed fix (not applied, and not recommended in isolation)

King & Zeng prior correction — a single intercept shift, applied post-hoc:

```
logit_corrected = logit(p) - ln[ ((1 - τ) / τ) × (ȳ / (1 - ȳ)) ]
```

with population rate τ = 0.086 and training rate ȳ = 0.5 — **both confirmed**,
not assumed: τ from the raw class counts and ȳ from `SMOTETomek`'s exact 50/50
output (cells 11 and 117). This gives a shift of **−2.3639**. Measured effect:

| Metric | Before | After | Change |
|---|---:|---:|---|
| Mean predicted PD | 15.30% | **8.46%** | vs. 8.60% actual |
| Brier score | 0.04976 | **0.02810** | 44% better |
| AUC | 0.983081 | 0.983081 | **unchanged** |

AUC is identical to six decimal places, confirming this is purely a calibration
correction: it re-levels probabilities without touching ranking or any
coefficient.

**Do not apply this on its own.** Despite the improved Brier score, the section
above shows it promotes applicants who default at 23.6% and collapses 89.5% of
the portfolio into a single rating band. Calibration and band cutoffs have to be
redesigned together, as one credit-policy decision.

---

## Recommendations

| Priority | Action | Deviates from prototype? |
|---|---|---|
| 1 | **Redesign calibration + band cutoffs together** (see above). Do *not* apply the intercept shift alone. Credit-policy decision. | **Yes** — needs sign-off |
| 2 | Obtain qualified compliance review against RBI Fair Practices Code | No |
| 3 | Re-run this audit whenever features or the artifact change | No |
| 4 | Confirm bureau snapshots are timestamped pre-decision | No |
| 5 | Monitor approval-rate parity by group in production; alert below 0.80 | No |

## What this audit does not cover

- **Intersectional subgroups** (e.g. young female applicants) — not tested.
- **Proxy strength of `city`** for caste, religion, or income geography — the
  dataset has 10 metros with near-identical distributions, so this cannot be
  assessed here and would need real geographic data.
- **Applicants absent from the data.** This measures only those who applied and
  were recorded. Anyone deterred before applying is invisible to it.
- **Label quality.** `default` is taken as ground truth. If historical
  collections or write-off practices were themselves biased, that bias is baked
  into the target and no audit of the model can surface it.
