"""
expected_loss.py
----------------
Implements the foundational credit-risk quantification framework used in
Basel III IRB and IFRS 9 expected credit loss (ECL) calculations:

    Expected Loss (EL) = PD × LGD × EAD

Where:
    PD   = Probability of Default                  (12-month forward, point-in-time)
    LGD  = Loss Given Default     (% of EAD lost net of recoveries)
    EAD  = Exposure at Default    (€ at risk when default occurs)
    EL   = Expected Loss          (€ loss expected over the horizon)

Why this matters for Radius (the interview narrative):
    - The composite risk score tells the analyst WHO to review.
    - EL tells management HOW MUCH money is at stake.
    - These are complementary: the score drives operational triage,
      EL drives portfolio-level decisions on capital, pricing, and provisioning.

Calibration sources:
    - PD anchor points are aligned with Creditreform's published
      Bonitätsindex default-rate bands for German SMEs.
    - LGD baseline (~75%) is typical for unsecured trade credit / fuel cards.
    - EAD uses a CCF (Credit Conversion Factor) approach borrowed from Basel:
      EAD = current outstanding + CCF × undrawn limit.

In production these calibrations would be derived from Radius's own
default history, not anchor points. Below they're documented and defensible.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# PD calibration
# ----------------------------------------------------------------------------
# Anchor points: Creditreform Bonitätsindex score -> 12m default probability.
# Lower score = better credit. These bands approximate Creditreform's
# published default rates for German SMEs across their 7 risk bands.
_PD_ANCHOR_SCORES = np.array([125, 175, 225, 275, 325, 425, 525, 600])
_PD_ANCHOR_PROBS  = np.array([0.003, 0.005, 0.010, 0.020, 0.050, 0.150, 0.400, 0.950])

# Industry PD multipliers. Reflects observed sector-level default rate variation
# in European SME fleet customers. Haulage has historically had the highest
# default rate in this segment.
_INDUSTRY_PD_MULT = {
    "Haulage & Logistics": 1.30,
    "Construction":        1.25,
    "Courier & Delivery":  1.10,
    "Trades & Services":   1.05,
    "Retail Distribution": 1.00,
    "Public Sector":       0.60,
}

# Trajectory kicker: if Creditreform score has worsened sharply in the last
# 3 months, the forward-looking PD should reflect that drift, not just the
# current level. We extrapolate the trend forward by 6 months.
TRAJECTORY_EXTRAPOLATION_MONTHS = 6


def calculate_pd(creditreform_latest, creditreform_delta_3m, industry):
    """
    Calculate 12-month PD for a single customer.

    Combines:
      1. Base PD from current Creditreform score (interpolated from anchors)
      2. Industry-specific multiplier
      3. Trajectory adjustment (extrapolate score drift forward 6m)

    Returns: PD as a float in [0, 1]
    """
    # Base PD from current score
    base_pd = float(np.interp(creditreform_latest, _PD_ANCHOR_SCORES, _PD_ANCHOR_PROBS))

    # Industry adjustment
    industry_mult = _INDUSTRY_PD_MULT.get(industry, 1.0)

    # Trajectory adjustment: extrapolate the 3-month score drift forward by
    # 6 months, then re-interpolate the PD at that hypothetical future score.
    # Use the higher of base or trajectory-adjusted PD (forward-looking IFRS 9 logic).
    if pd.notna(creditreform_delta_3m):
        monthly_drift = creditreform_delta_3m / 3.0
        projected_score = creditreform_latest + monthly_drift * TRAJECTORY_EXTRAPOLATION_MONTHS
        projected_score = np.clip(projected_score, 100, 600)
        projected_pd = float(np.interp(projected_score, _PD_ANCHOR_SCORES, _PD_ANCHOR_PROBS))
        forward_pd = max(base_pd, projected_pd)
    else:
        forward_pd = base_pd

    pd_value = forward_pd * industry_mult
    return float(np.clip(pd_value, 0.0001, 0.99))


# ----------------------------------------------------------------------------
# LGD calibration
# ----------------------------------------------------------------------------
# Baseline LGD for unsecured B2B trade credit (fuel cards): 75%.
# This means on average, when a fuel-card customer defaults, Radius recovers
# 25 cents on the euro after debt collection costs.
#
# Adjustments:
#   - Larger SME customers: more likely to have personal guarantees, traceable
#     assets, and serious enough exposure to pursue legally → lower LGD.
#   - Public sector: very high recovery → much lower LGD.
#   - Haulage/Construction: hard assets (vehicles, equipment) can sometimes
#     be recovered → slightly lower LGD.

_LGD_BASE = 0.75

_LGD_SIZE_ADJUSTMENT = {
    "1-5":     0.00,   # smallest fleets — limited recovery
    "6-20":   -0.03,
    "21-50":  -0.07,
    "51-100": -0.12,
    "100+":   -0.15,
}

_LGD_INDUSTRY_ADJUSTMENT = {
    "Haulage & Logistics": -0.03,  # vehicles repossessable
    "Construction":        -0.05,  # equipment repossessable
    "Courier & Delivery":  -0.02,
    "Trades & Services":    0.00,
    "Retail Distribution":  0.00,
    "Public Sector":       -0.30,  # extremely high recovery
}


def calculate_lgd(industry, size_band):
    """
    Calculate LGD for a single customer based on industry and size band.

    Returns: LGD as a float in [0.40, 0.90]
    """
    lgd = _LGD_BASE
    lgd += _LGD_SIZE_ADJUSTMENT.get(size_band, 0.0)
    lgd += _LGD_INDUSTRY_ADJUSTMENT.get(industry, 0.0)
    return float(np.clip(lgd, 0.40, 0.90))


# ----------------------------------------------------------------------------
# EAD calibration
# ----------------------------------------------------------------------------
# EAD = current drawn balance + CCF × (credit_limit - drawn)
#
# For Radius fuel cards:
#   - "Drawn" is approximated as the last 1.5 months of fuel spend, since
#     payment terms are typically 7-14 days, meaning ~1.5 months of usage
#     sits unpaid between fuel-up and collection.
#   - CCF (Credit Conversion Factor) reflects the share of unused credit limit
#     that customers tend to draw before default is detected. For revolving
#     trade credit, Basel II/III suggests CCFs in the 50-80% range. We use 60%.
#
# This captures the "load up before default" phenomenon: a customer who
# realises they're going under will run their credit line up to the limit
# while they still can. EAD captures that future exposure, not just today's.

_CCF = 0.60           # Credit Conversion Factor on undrawn portion
_DRAWN_MONTHS = 0.7   # Outstanding (billed-but-unpaid) balance, in months of spend.
                      # With weekly billing and 7-14 day payment terms, roughly
                      # 2-3 weeks of fuel sits unpaid at any time -> ~0.7 months.
                      # The CCF then captures ADDITIONAL drawdown on the remaining
                      # headroom before default is detected and the card stopped.


def calculate_ead(latest_spend_eur, credit_limit_eur):
    """
    Calculate EAD for a single customer using the Basel CCF approach:

        EAD = current outstanding balance + CCF x undrawn headroom

    - "Outstanding" ~ 0.7 months of recent spend (billed but not yet paid).
    - "Undrawn headroom" = remaining credit limit, of which a defaulting
      customer typically draws ~60% (CCF) before the card is stopped.

    Returns: EAD in EUR (float), capped at the credit limit.
    """
    drawn = latest_spend_eur * _DRAWN_MONTHS
    drawn = min(drawn, credit_limit_eur)  # can't owe more than the limit
    undrawn = max(credit_limit_eur - drawn, 0)
    ead = drawn + _CCF * undrawn
    return float(min(ead, credit_limit_eur))


# ----------------------------------------------------------------------------
# Combined Expected Loss
# ----------------------------------------------------------------------------
def calculate_expected_loss(scored_df):
    """
    Add PD, LGD, EAD, and EL columns to the scored portfolio DataFrame.

    The input is the output of risk_engine.score_portfolio() — which already
    contains all the inputs we need (creditreform score, latest spend,
    credit limit, industry, size band, etc.).
    """
    df = scored_df.copy()

    df["PD"] = df.apply(
        lambda r: calculate_pd(
            r["creditreform_latest"],
            r["creditreform_delta_3m"],
            r["industry"],
        ),
        axis=1,
    )

    df["LGD"] = df.apply(
        lambda r: calculate_lgd(r["industry"], r["size_band"]),
        axis=1,
    )

    df["EAD_eur"] = df.apply(
        lambda r: calculate_ead(r["latest_spend_eur"], r["credit_limit_eur"]),
        axis=1,
    )

    df["EL_eur"] = df["PD"] * df["LGD"] * df["EAD_eur"]

    # Also compute a simplified Unexpected Loss (UL) estimate.
    # Real Basel IRB formula uses asset correlation; this is the standalone
    # standard deviation, useful for ranking volatility contributors.
    df["UL_eur"] = np.sqrt(df["PD"] * (1 - df["PD"])) * df["LGD"] * df["EAD_eur"]

    # EL as % of EAD (the "loss rate") — useful for benchmarking against industry
    df["EL_rate_pct"] = (df["EL_eur"] / df["EAD_eur"].replace(0, np.nan) * 100).round(2)

    return df


# ----------------------------------------------------------------------------
# Portfolio-level summary
# ----------------------------------------------------------------------------
def portfolio_summary(df_with_el):
    """
    Roll up to portfolio totals — the management-level view.
    """
    total_ead = df_with_el["EAD_eur"].sum()
    total_el  = df_with_el["EL_eur"].sum()
    total_ul  = df_with_el["UL_eur"].sum()
    weighted_pd  = (df_with_el["PD"]  * df_with_el["EAD_eur"]).sum() / total_ead if total_ead else 0
    weighted_lgd = (df_with_el["LGD"] * df_with_el["EAD_eur"]).sum() / total_ead if total_ead else 0

    return {
        "total_exposure_eur":     total_ead,
        "expected_loss_eur":      total_el,
        "unexpected_loss_eur":    total_ul,
        "portfolio_loss_rate_pct": (total_el / total_ead * 100) if total_ead else 0,
        "exposure_weighted_pd":   weighted_pd,
        "exposure_weighted_lgd":  weighted_lgd,
        "n_customers":            len(df_with_el),
    }


# ----------------------------------------------------------------------------
# Stress scenarios
# ----------------------------------------------------------------------------
def stress_test(df_with_el, pd_shock_multiplier=1.5, lgd_shock_add=0.0):
    """
    Apply a stress scenario to the portfolio and return new EL.

    pd_shock_multiplier:  multiply every customer's PD by this factor
                          (capped at 1.0). 1.5 = mild recession; 2.0 = severe.
    lgd_shock_add:        additive shock to LGD (e.g. 0.05 = +5pp).
    """
    stressed = df_with_el.copy()
    stressed["PD_stressed"]  = np.clip(stressed["PD"] * pd_shock_multiplier, 0, 0.99)
    stressed["LGD_stressed"] = np.clip(stressed["LGD"] + lgd_shock_add, 0.40, 0.95)
    stressed["EL_stressed_eur"] = stressed["PD_stressed"] * stressed["LGD_stressed"] * stressed["EAD_eur"]
    return stressed


if __name__ == "__main__":
    from data_generator import generate_portfolio
    from risk_engine import score_portfolio

    customers, financials, behaviour = generate_portfolio()
    scored = score_portfolio(customers, financials, behaviour)
    scored_el = calculate_expected_loss(scored)

    summary = portfolio_summary(scored_el)
    print("=" * 60)
    print("PORTFOLIO EXPECTED LOSS SUMMARY")
    print("=" * 60)
    print(f"Customers:                 {summary['n_customers']:>15,}")
    print(f"Total exposure (EAD):      €{summary['total_exposure_eur']:>14,.0f}")
    print(f"Expected Loss (12m):       €{summary['expected_loss_eur']:>14,.0f}")
    print(f"Unexpected Loss (1σ):      €{summary['unexpected_loss_eur']:>14,.0f}")
    print(f"Portfolio loss rate:        {summary['portfolio_loss_rate_pct']:>14.2f}%")
    print(f"Exposure-weighted PD:       {summary['exposure_weighted_pd'] * 100:>14.2f}%")
    print(f"Exposure-weighted LGD:      {summary['exposure_weighted_lgd'] * 100:>14.2f}%")
    print()
    print("Top 5 by Expected Loss:")
    cols = ["customer_id", "company_name", "industry", "PD", "LGD", "EAD_eur", "EL_eur"]
    top = scored_el.sort_values("EL_eur", ascending=False).head(5)[cols]
    for _, r in top.iterrows():
        print(f"  {r['customer_id']}  {r['company_name'][:30]:<30}  "
              f"PD={r['PD']*100:5.1f}%  LGD={r['LGD']*100:4.0f}%  "
              f"EAD=€{r['EAD_eur']:>7,.0f}  EL=€{r['EL_eur']:>7,.0f}")

    # Stress test
    print()
    stressed = stress_test(scored_el, pd_shock_multiplier=1.5)
    stressed_total = stressed["EL_stressed_eur"].sum()
    print(f"Mild recession scenario (PD × 1.5):  EL = €{stressed_total:,.0f} "
          f"(+{(stressed_total / summary['expected_loss_eur'] - 1) * 100:.1f}%)")
    stressed_severe = stress_test(scored_el, pd_shock_multiplier=2.0, lgd_shock_add=0.05)
    stressed_severe_total = stressed_severe["EL_stressed_eur"].sum()
    print(f"Severe recession scenario (PD × 2.0, LGD +5pp):  EL = €{stressed_severe_total:,.0f} "
          f"(+{(stressed_severe_total / summary['expected_loss_eur'] - 1) * 100:.1f}%)")
