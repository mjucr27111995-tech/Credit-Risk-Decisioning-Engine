"""Fairness audit: measure disparate impact of the shipped model.

Read-only analysis. Does not modify the model, features, or scoring path.
Replicates the shipped artifact pipeline vectorised for speed, and verifies
that replication against the project's own scorer before trusting it.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = ROOT.parent / "Project2_DataCollection_Resources"
DATA = Path(os.environ.get("CRM_AUDIT_DATA_DIR") or _DEFAULT_DATA).expanduser()
sys.path.insert(0, str(ROOT / "src"))

_REQUIRED_CSVS = ("customers.csv", "loans.csv", "bureau_data.csv")

PLACEHOLDERS = [
    "number_of_dependants",
    "years_at_current_address",
    "zipcode",
    "sanction_amount",
    "processing_fee",
    "gst",
    "net_disbursement",
    "principal_outstanding",
    "bank_balance_at_application",
    "number_of_closed_accounts",
    "enquiry_count",
]


def _require_data() -> None:
    """Fail fast with an actionable message when the source CSVs are absent."""
    missing = [name for name in _REQUIRED_CSVS if not (DATA / name).is_file()]
    if not missing:
        return
    raise SystemExit(
        f"Audit data not found in {DATA}\n"
        f"  missing: {', '.join(missing)}\n\n"
        "These CSVs come from the Codebasics course data-collection resources and\n"
        "are not redistributed with this repository. Point CRM_AUDIT_DATA_DIR at a\n"
        "directory containing them, e.g.\n"
        "  CRM_AUDIT_DATA_DIR=/path/to/data python3 scripts/fairness_audit.py\n\n"
        "The application itself does not need them; it ships the trained artifact."
    )


def load_scored() -> pd.DataFrame:
    """Join the source tables, engineer features, and score with the real artifact."""
    _require_data()
    customers = pd.read_csv(DATA / "customers.csv")
    loans = pd.read_csv(DATA / "loans.csv")
    bureau = pd.read_csv(DATA / "bureau_data.csv")
    df = customers.merge(loans, on="cust_id").merge(bureau, on="cust_id")

    df["default"] = df["default"].astype(int)
    df["loan_purpose"] = df["loan_purpose"].replace("Personaal", "Personal")
    df["residence_type"] = df["residence_type"].fillna("Owned")
    df = df[df["processing_fee"] / df["loan_amount"] < 0.03].copy()

    df["loan_to_income"] = (df["loan_amount"] / df["income"]).round(2)
    df["delinquency_ratio"] = (df["delinquent_months"] * 100 / df["total_loan_months"]).round(1)
    df["avg_dpd_per_delinquency"] = np.where(
        df["delinquent_months"] != 0,
        (df["total_dpd"] / df["delinquent_months"]).round(1),
        0,
    )

    artifact = joblib.load(ROOT / "artifacts" / "model_data.joblib")
    model, features = artifact["model"], list(artifact["features"])
    scaler, cols = artifact["scaler"], list(artifact["cols_to_scale"])

    frame = pd.DataFrame(index=df.index)
    for col in [
        "age",
        "loan_tenure_months",
        "number_of_open_accounts",
        "credit_utilization_ratio",
        "loan_to_income",
        "delinquency_ratio",
        "avg_dpd_per_delinquency",
    ]:
        frame[col] = df[col].to_numpy()
    frame["residence_type_Owned"] = (df["residence_type"] == "Owned").astype(int).to_numpy()
    frame["residence_type_Rented"] = (df["residence_type"] == "Rented").astype(int).to_numpy()
    for purpose in ["Education", "Home", "Personal"]:
        frame[f"loan_purpose_{purpose}"] = (df["loan_purpose"] == purpose).astype(int).to_numpy()
    frame["loan_type_Unsecured"] = (df["loan_type"] == "Unsecured").astype(int).to_numpy()
    for col in PLACEHOLDERS:
        frame[col] = 1.0

    frame[cols] = scaler.transform(frame[cols])
    df["pd"] = model.predict_proba(frame[features])[:, 1]
    df["score"] = (300 + (1 - df["pd"]) * 600).round().astype(int)
    return df


def verify_against_project_scorer(df: pd.DataFrame) -> None:
    """Confirm the vectorised replication matches the shipped scoring path."""
    from credit_risk.config import get_settings
    from credit_risk.domain.models import LoanApplication
    from credit_risk.scoring.scorer import build_scorer

    scorer = build_scorer(get_settings())
    sample = df.sample(25, random_state=7)
    deltas = []
    for _, row in sample.iterrows():
        try:
            app = LoanApplication(
                age=int(row["age"]),
                income=int(row["income"]),
                loan_amount=int(row["loan_amount"]),
                loan_tenure_months=int(row["loan_tenure_months"]),
                credit_utilization_ratio=float(row["credit_utilization_ratio"]),
                number_of_open_accounts=int(row["number_of_open_accounts"]),
                delinquency_ratio=float(row["delinquency_ratio"]),
                avg_dpd_per_delinquency=float(row["avg_dpd_per_delinquency"]),
                residence_type=row["residence_type"],
                loan_purpose=row["loan_purpose"],
                loan_type=row["loan_type"],
            )
        except Exception:
            continue
        deltas.append(abs(scorer.score(app).probability_of_default - row["pd"]))
    print(f"  replication verified on {len(deltas)} rows, max delta = {max(deltas):.2e}")
    assert max(deltas) < 1e-9, "replication does not match the shipped scorer"


def disparate_impact(df: pd.DataFrame, attr: str, cutoff: int = 650) -> pd.DataFrame:
    """Approval rate, outcome rates, and calibration per group."""
    df = df.copy()
    df["approved"] = df["score"] >= cutoff
    rows = []
    for group, g in df.groupby(attr):
        defaulters, payers = g[g["default"] == 1], g[g["default"] == 0]
        rows.append(
            {
                "group": group,
                "n": len(g),
                "approval_rate": g["approved"].mean(),
                "actual_default_rate": g["default"].mean(),
                "mean_predicted_pd": g["pd"].mean(),
                "calibration_gap": g["pd"].mean() - g["default"].mean(),
                "auc": roc_auc_score(g["default"], g["pd"])
                if g["default"].nunique() > 1
                else np.nan,
                "fnr_approved_defaulters": defaulters["approved"].mean()
                if len(defaulters)
                else np.nan,
                "fpr_rejected_payers": (~payers["approved"]).mean() if len(payers) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("approval_rate", ascending=False)
    best = out["approval_rate"].max()
    out["impact_ratio"] = (out["approval_rate"] / best).round(4)
    return out


def proxy_strength(df: pd.DataFrame, attr: str) -> float:
    """Can the 13 model features reconstruct a protected attribute?"""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    feats = [
        "age",
        "loan_tenure_months",
        "number_of_open_accounts",
        "credit_utilization_ratio",
        "loan_to_income",
        "delinquency_ratio",
        "avg_dpd_per_delinquency",
    ]
    X = pd.get_dummies(df[feats + ["residence_type", "loan_purpose", "loan_type"]], drop_first=True)
    y = (df[attr] == df[attr].value_counts().index[0]).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = GradientBoostingClassifier(random_state=42, n_estimators=120).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def main() -> None:
    df = load_scored()
    print(f"Scored {len(df):,} applications with the shipped artifact.")
    verify_against_project_scorer(df)

    df["age_band"] = pd.cut(
        df["age"], [17, 25, 35, 45, 55, 100], labels=["18-25", "26-35", "36-45", "46-55", "56+"]
    )
    df["dependants_band"] = df["number_of_dependants"].clip(upper=4)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    for attr in [
        "gender",
        "marital_status",
        "employment_status",
        "age_band",
        "dependants_band",
        "city",
    ]:
        print(f"\n{'=' * 96}\n{attr.upper()}  (approve at score >= 650)\n{'=' * 96}")
        print(disparate_impact(df, attr).to_string(index=False, float_format="%.4f"))

    print(
        f"\n{'=' * 96}\nPROXY TEST: can the 13 features predict a protected attribute?"
        f"\n(AUC 0.50 = no signal; > 0.60 = meaningful proxy)\n{'=' * 96}"
    )
    for attr in ["gender", "marital_status", "employment_status"]:
        print(f"  {attr:<22} AUC = {proxy_strength(df, attr):.4f}")


if __name__ == "__main__":
    main()
