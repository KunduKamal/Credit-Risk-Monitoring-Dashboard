"""
risk_engine.py
--------------
Layer 2 of the monitoring system: convert raw customer data into a composite
risk score with explainable flags.

Design principles (these are the talking points for the interview):

1. EXPLAINABLE BY DESIGN.
   Every account ends up Red/Amber/Green for a reason. The reason is shown to
   the analyst (and ultimately to sales/account-management when defending a
   credit-limit change). A black-box ML model would fail this test.

2. WEIGHTED COMPOSITE.
   Four input streams, each scored 0-100 (higher = more risky), then weighted:
       40%  Creditreform score & trajectory  (the strongest single predictor)
       25%  Financial health ratios          (current ratio, leverage, OCF)
       25%  Payment behaviour                (late + failed direct debits)
       10%  Spend anomaly                    (sudden surge = "loading up" signal)

3. FLAGS ARE SEPARATE FROM SCORE.
   A customer can be Green overall but still trigger a specific flag (e.g.
   one-off spend spike). Flags are the analyst's queue; score is the
   prioritisation order.

4. THRESHOLDS ARE CONFIGURABLE.
   In production these come from a config file, calibrated to the actual
   default rate of the book. Numbers here are sensible defaults.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configurable thresholds
# ----------------------------------------------------------------------------
CREDITREFORM_AT_RISK_THRESHOLD = 300       # score above this = elevated risk
CREDITREFORM_DOWNGRADE_FLAG    = 25        # 3-month worsening that triggers flag

CURRENT_RATIO_DISTRESS         = 1.0       # below this = liquidity stress
DEBT_TO_EBITDA_DISTRESS        = 5.0       # above this = over-leveraged
OCF_MARGIN_DISTRESS            = 0.03      # below this = poor cash generation

PAYMENT_LATE_FLAG              = 3         # late payments in trailing 3 months
PAYMENT_FAILED_FLAG            = 1         # failed direct debits in trailing 3m

SPEND_ZSCORE_FLAG              = 2.0       # z-score on latest vs trailing 6m

# Composite scoring weights (sum to 1.0)
WEIGHTS = {
    "bureau":    0.40,
    "financial": 0.25,
    "payment":   0.25,
    "spend":     0.10,
}

# Tier cutoffs on composite score (0-100)
TIER_AMBER_CUTOFF = 35
TIER_RED_CUTOFF   = 60


# ----------------------------------------------------------------------------
# Component scoring functions
# ----------------------------------------------------------------------------
def _score_bureau(behaviour_df):
    """
    Score Creditreform component, per customer.
    Combines absolute level (latest score) and trajectory (3m delta).
    """
    latest = behaviour_df.sort_values(["customer_id", "month"]).groupby("customer_id").tail(1)[
        ["customer_id", "creditreform_score"]
    ].rename(columns={"creditreform_score": "creditreform_latest"})

    three_months_ago = behaviour_df.sort_values(["customer_id", "month"]).groupby("customer_id").nth(-4)[
        ["customer_id", "creditreform_score"]
    ].rename(columns={"creditreform_score": "creditreform_3m_ago"})

    merged = latest.merge(three_months_ago, on="customer_id", how="left")
    merged["creditreform_delta_3m"] = merged["creditreform_latest"] - merged["creditreform_3m_ago"]

    # Map latest score (100-600) into a 0-100 risk component.
    # Below 200 ~ minimal risk; above 400 ~ very high risk.
    merged["bureau_score"] = np.clip((merged["creditreform_latest"] - 150) / (450 - 150) * 100, 0, 100)

    # Add a kicker for sharp downgrades
    merged.loc[merged["creditreform_delta_3m"] >= 50, "bureau_score"] = np.clip(
        merged["bureau_score"] + 15, 0, 100
    )

    merged["flag_score_downgrade"] = merged["creditreform_delta_3m"] >= CREDITREFORM_DOWNGRADE_FLAG

    return merged[["customer_id", "creditreform_latest", "creditreform_delta_3m",
                   "bureau_score", "flag_score_downgrade"]]


def _score_financial(financials_df):
    """
    Score financial-statement health, per customer.
    Three ratios: current ratio, debt/EBITDA, OCF margin (with revenue proxy).
    """
    f = financials_df.copy()

    f["current_ratio"]   = f["current_assets_eur"] / f["current_liabilities_eur"].replace(0, np.nan)
    f["debt_to_ebitda"]  = f["total_debt_eur"]    / f["ebitda_eur"].replace(0, np.nan)
    f["ocf_margin"]      = f["operating_cashflow_eur"] / f["revenue_eur"].replace(0, np.nan)

    # Build risk components on 0-100 scale (higher = riskier)
    # Current ratio: 2.0+ healthy -> 0,  0.5 -> 100
    f["risk_current"] = np.clip((2.0 - f["current_ratio"]) / 1.5 * 100, 0, 100)
    # Debt/EBITDA: 1x healthy -> 0,  >8x -> 100
    f["risk_leverage"] = np.clip(f["debt_to_ebitda"].fillna(15) / 8 * 100, 0, 100)
    # OCF margin: 12%+ healthy -> 0,  negative -> 100
    f["risk_ocf"] = np.clip((0.12 - f["ocf_margin"]) / 0.20 * 100, 0, 100)

    f["financial_score"] = (f["risk_current"] + f["risk_leverage"] + f["risk_ocf"]) / 3

    f["flag_financial_distress"] = (
        (f["current_ratio"] < CURRENT_RATIO_DISTRESS) |
        (f["debt_to_ebitda"] > DEBT_TO_EBITDA_DISTRESS) |
        (f["ocf_margin"] < OCF_MARGIN_DISTRESS)
    )

    return f[["customer_id", "current_ratio", "debt_to_ebitda", "ocf_margin",
              "financial_score", "flag_financial_distress"]]


def _score_payment(behaviour_df):
    """
    Score payment behaviour over the last 3 months.
    Late and failed payments are weighted differently.
    """
    sorted_df = behaviour_df.sort_values(["customer_id", "month"])
    last_3m = sorted_df.groupby("customer_id").tail(3)

    agg = last_3m.groupby("customer_id").agg(
        payments_on_time_3m=("payments_on_time", "sum"),
        payments_late_3m=("payments_late", "sum"),
        payments_failed_3m=("payments_failed", "sum"),
    ).reset_index()

    agg["total_invoices_3m"] = (
        agg["payments_on_time_3m"] + agg["payments_late_3m"] + agg["payments_failed_3m"]
    )
    agg["late_rate"]   = agg["payments_late_3m"]   / agg["total_invoices_3m"].replace(0, np.nan)
    agg["failed_rate"] = agg["payments_failed_3m"] / agg["total_invoices_3m"].replace(0, np.nan)

    # Score: late counts 1x, failed counts 3x (a failed DD is a much stronger signal)
    raw = agg["late_rate"].fillna(0) * 100 + agg["failed_rate"].fillna(0) * 300
    agg["payment_score"] = np.clip(raw, 0, 100)

    agg["flag_payment_distress"] = (
        (agg["payments_late_3m"]   >= PAYMENT_LATE_FLAG) |
        (agg["payments_failed_3m"] >= PAYMENT_FAILED_FLAG)
    )

    return agg[["customer_id", "payments_late_3m", "payments_failed_3m",
                "late_rate", "failed_rate", "payment_score", "flag_payment_distress"]]


def _score_spend_anomaly(behaviour_df):
    """
    Score spend anomalies using a z-score on the latest month's spend
    versus the prior 6-month baseline. Distress signal: a sudden surge
    can mean the customer is "loading up" before going dark.
    """
    sorted_df = behaviour_df.sort_values(["customer_id", "month"])
    results = []

    for customer_id, group in sorted_df.groupby("customer_id"):
        spends = group["fuel_spend_eur"].values
        latest = spends[-1]
        baseline = spends[-7:-1]  # 6 months ending one month before latest
        mu  = baseline.mean()
        sig = baseline.std(ddof=1) if len(baseline) > 1 else 1.0
        sig = max(sig, 1.0)  # avoid divide-by-zero

        z = (latest - mu) / sig
        # Risk component: positive z-scores worsen the risk (negative spend drop also bad,
        # but a different signal -- handled separately if needed).
        risk = np.clip(z * 25, 0, 100)

        results.append({
            "customer_id": customer_id,
            "latest_spend_eur": int(latest),
            "baseline_avg_spend_eur": int(round(mu)),
            "spend_zscore": round(z, 2),
            "spend_score": float(round(risk, 1)),
            "flag_spend_anomaly": z >= SPEND_ZSCORE_FLAG,
        })

    return pd.DataFrame(results)


# ----------------------------------------------------------------------------
# Composite scoring
# ----------------------------------------------------------------------------
def score_portfolio(customers_df, financials_df, behaviour_df):
    """
    Run all four component scorers and combine into a composite risk score.
    Returns one row per customer with all sub-scores, flags, and tier.
    """
    bureau    = _score_bureau(behaviour_df)
    financial = _score_financial(financials_df)
    payment   = _score_payment(behaviour_df)
    spend     = _score_spend_anomaly(behaviour_df)

    out = (
        customers_df
        .merge(bureau,    on="customer_id", how="left")
        .merge(financial, on="customer_id", how="left")
        .merge(payment,   on="customer_id", how="left")
        .merge(spend,     on="customer_id", how="left")
    )

    out["composite_risk_score"] = (
        WEIGHTS["bureau"]    * out["bureau_score"]    +
        WEIGHTS["financial"] * out["financial_score"] +
        WEIGHTS["payment"]   * out["payment_score"]   +
        WEIGHTS["spend"]     * out["spend_score"]
    ).round(1)

    # Use left-closed intervals [a, b) via right=False so the tier boundaries
    # match the documented cutoffs exactly:
    #   Green: [0, 35)    Amber: [35, 60)    Red: [60, 100]
    # (pd.cut's default right=True would put a score of exactly 35 into Green,
    # contradicting "Amber starts at 35".)
    out["risk_tier"] = pd.cut(
        out["composite_risk_score"],
        bins=[0, TIER_AMBER_CUTOFF, TIER_RED_CUTOFF, 100.01],
        labels=["Green", "Amber", "Red"],
        right=False,
        include_lowest=True,
    )

    # Primary flag (the human-readable reason this customer is on the queue)
    def _primary_flag(row):
        flags = []
        if row["flag_financial_distress"]: flags.append("Financial distress")
        if row["flag_payment_distress"]:   flags.append("Payment issues")
        if row["flag_score_downgrade"]:    flags.append("Bureau downgrade")
        if row["flag_spend_anomaly"]:      flags.append("Spend anomaly")
        return " + ".join(flags) if flags else "—"

    out["primary_flag"] = out.apply(_primary_flag, axis=1)
    out["flag_count"] = (
        out["flag_financial_distress"].astype(int) +
        out["flag_payment_distress"].astype(int) +
        out["flag_score_downgrade"].astype(int) +
        out["flag_spend_anomaly"].astype(int)
    )

    return out.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_generator import generate_portfolio
    customers, financials, behaviour = generate_portfolio()
    scored = score_portfolio(customers, financials, behaviour)

    print(f"Scored {len(scored)} customers")
    print("\nTier breakdown:")
    print(scored["risk_tier"].value_counts())
    print("\nTop 5 highest-risk accounts:")
    cols = ["customer_id", "company_name", "industry", "composite_risk_score", "risk_tier", "primary_flag"]
    print(scored[cols].head(5).to_string(index=False))
