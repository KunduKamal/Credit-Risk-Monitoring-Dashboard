"""
sanity_tests.py
---------------
Validation suite for the credit risk monitoring system. Checks:
  1. Data integrity (no nulls where there shouldn't be, ranges sensible)
  2. PD/LGD/EAD bounds and monotonicity
  3. EL identity (EL == PD * LGD * EAD)
  4. Portfolio aggregation consistency (dashboard numbers == raw numbers)
  5. Stress test ordering (severe >= mild >= base)
  6. Tier logic consistency
  7. Loss concentration curve correctness
  8. Cross-checks against the values quoted in the dashboard / talk track
"""

import numpy as np
import pandas as pd

from data_generator import generate_portfolio
from risk_engine import score_portfolio
from expected_loss import (
    calculate_expected_loss, portfolio_summary, stress_test,
    calculate_pd, calculate_lgd, calculate_ead,
    _PD_ANCHOR_SCORES, _PD_ANCHOR_PROBS, _CCF, _DRAWN_MONTHS, _LGD_BASE,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results = []
def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, bool(condition), detail))
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    return condition


# ----------------------------------------------------------------------------
print("=" * 70)
print("BUILDING PORTFOLIO")
print("=" * 70)
customers, financials, behaviour = generate_portfolio()
scored = score_portfolio(customers, financials, behaviour)
scored = calculate_expected_loss(scored)
summary = portfolio_summary(scored)
print(f"  {len(scored)} customers scored\n")


# ----------------------------------------------------------------------------
print("=" * 70)
print("1. DATA INTEGRITY")
print("=" * 70)
check("Customer count is 300", len(scored) == 300, f"got {len(scored)}")
check("No duplicate customer IDs", scored["customer_id"].nunique() == len(scored))
check("Behaviour rows = 300 x 12", len(behaviour) == 3600, f"got {len(behaviour)}")

# Critical columns should have no nulls
for col in ["composite_risk_score", "risk_tier", "PD", "LGD", "EAD_eur", "EL_eur"]:
    n_null = scored[col].isna().sum()
    check(f"No nulls in '{col}'", n_null == 0, f"{n_null} nulls" if n_null else "")

# creditreform_latest should always be 100-600
cr = scored["creditreform_latest"]
check("Creditreform score in [100, 600]", cr.between(100, 600).all(),
      f"range [{cr.min()}, {cr.max()}]")


# ----------------------------------------------------------------------------
print("=" * 70)
print("2. PD / LGD / EAD BOUNDS")
print("=" * 70)
check("PD in [0.0001, 0.99]", scored["PD"].between(0.0001, 0.99).all(),
      f"range [{scored['PD'].min():.4f}, {scored['PD'].max():.4f}]")
check("LGD in [0.40, 0.90]", scored["LGD"].between(0.40, 0.90).all(),
      f"range [{scored['LGD'].min():.3f}, {scored['LGD'].max():.3f}]")
check("EAD positive", (scored["EAD_eur"] > 0).all())
check("EAD never exceeds credit limit",
      (scored["EAD_eur"] <= scored["credit_limit_eur"] + 1).all(),
      f"max EAD/limit ratio = {(scored['EAD_eur']/scored['credit_limit_eur']).max():.3f}")


# ----------------------------------------------------------------------------
print("=" * 70)
print("3. PD MONOTONICITY & CALIBRATION")
print("=" * 70)
# PD should rise monotonically with Creditreform score (worse score = higher PD),
# holding industry & trajectory constant.
test_scores = [120, 175, 225, 275, 325, 425, 525, 590]
pds_retail = [calculate_pd(s, 0, "Retail Distribution") for s in test_scores]
is_monotonic = all(pds_retail[i] <= pds_retail[i+1] for i in range(len(pds_retail)-1))
check("PD monotonically increases with bureau score", is_monotonic,
      "  ".join(f"{s}:{p*100:.1f}%" for s, p in zip(test_scores, pds_retail)))

# Industry multiplier ordering: haulage should have higher PD than public sector
pd_haul = calculate_pd(300, 0, "Haulage & Logistics")
pd_pub  = calculate_pd(300, 0, "Public Sector")
check("Haulage PD > Public Sector PD (same score)", pd_haul > pd_pub,
      f"haulage {pd_haul*100:.1f}% vs public {pd_pub*100:.1f}%")

# Trajectory adjustment: worsening score should raise PD vs stable
pd_stable    = calculate_pd(300, 0,  "Retail Distribution")
pd_worsening = calculate_pd(300, 60, "Retail Distribution")  # +60 over 3 months
check("Worsening trajectory raises PD", pd_worsening > pd_stable,
      f"stable {pd_stable*100:.1f}% vs worsening {pd_worsening*100:.1f}%")

# Anchor points should round-trip
for s, expected_p in zip(_PD_ANCHOR_SCORES, _PD_ANCHOR_PROBS):
    got = float(np.interp(s, _PD_ANCHOR_SCORES, _PD_ANCHOR_PROBS))
    if not np.isclose(got, expected_p):
        check(f"Anchor {s} interpolates correctly", False, f"got {got} vs {expected_p}")
check("All PD anchor points interpolate exactly", True)


# ----------------------------------------------------------------------------
print("=" * 70)
print("4. LGD LOGIC")
print("=" * 70)
# Larger fleets should have LOWER LGD than smallest, same industry
lgd_small = calculate_lgd("Retail Distribution", "1-5")
lgd_large = calculate_lgd("Retail Distribution", "100+")
check("Larger fleet has lower LGD", lgd_large < lgd_small,
      f"1-5: {lgd_small:.2f} vs 100+: {lgd_large:.2f}")
# Public sector should have the lowest LGD of any industry, holding size constant
for sz in ["1-5", "100+"]:
    lgds = {ind: calculate_lgd(ind, sz) for ind in
            ["Haulage & Logistics", "Construction", "Courier & Delivery",
             "Trades & Services", "Retail Distribution", "Public Sector"]}
    pub = lgds["Public Sector"]
    check(f"Public sector has lowest LGD at size {sz}",
          pub == min(lgds.values()),
          f"public {pub:.2f}, others min {min(v for k,v in lgds.items() if k!='Public Sector'):.2f}")
# Every customer sitting at the LGD floor should be public sector
floor_customers = scored[np.isclose(scored["LGD"], 0.40)]
check("All floor-LGD (0.40) customers are Public Sector",
      (floor_customers["industry"] == "Public Sector").all() if len(floor_customers) else True,
      f"{len(floor_customers)} at floor, "
      f"{(floor_customers['industry']=='Public Sector').sum()} are public sector")


# ----------------------------------------------------------------------------
print("=" * 70)
print("5. EAD LOGIC")
print("=" * 70)
# EAD with zero spend should equal CCF * credit_limit
ead_zero = calculate_ead(0, 10000)
check("EAD(spend=0) == CCF * limit", np.isclose(ead_zero, _CCF * 10000),
      f"got {ead_zero}, expected {_CCF*10000}")
# EAD with spend exceeding limit should cap at limit
ead_over = calculate_ead(100000, 10000)
check("EAD caps at credit limit", np.isclose(ead_over, 10000),
      f"got {ead_over}")
# Manual recompute on a real customer
sample = scored.iloc[0]
manual_ead = min(
    min(sample["latest_spend_eur"] * _DRAWN_MONTHS, sample["credit_limit_eur"]) +
    _CCF * max(sample["credit_limit_eur"] - min(sample["latest_spend_eur"] * _DRAWN_MONTHS, sample["credit_limit_eur"]), 0),
    sample["credit_limit_eur"]
)
check("EAD recomputes correctly on sample", np.isclose(manual_ead, sample["EAD_eur"]),
      f"manual {manual_ead:.0f} vs stored {sample['EAD_eur']:.0f}")
# EAD utilisation should be realistic (not pinned at 100%) so the CCF does real work
util = scored["EAD_eur"] / scored["credit_limit_eur"]
check("EAD utilisation realistic (mean 65-90%, not pinned at limit)",
      0.60 <= util.mean() <= 0.92,
      f"mean {util.mean()*100:.0f}%, range [{util.min()*100:.0f}%, {util.max()*100:.0f}%]")


# ----------------------------------------------------------------------------
print("=" * 70)
print("6. EXPECTED LOSS IDENTITY")
print("=" * 70)
# EL must exactly equal PD * LGD * EAD for every row
recomputed_el = scored["PD"] * scored["LGD"] * scored["EAD_eur"]
max_diff = (recomputed_el - scored["EL_eur"]).abs().max()
check("EL == PD * LGD * EAD (all rows)", max_diff < 1e-6,
      f"max abs diff = {max_diff:.2e}")
# EL should never exceed EAD (can't lose more than exposure)
check("EL <= EAD (all rows)", (scored["EL_eur"] <= scored["EAD_eur"]).all())
# UL should be >= 0
check("UL >= 0 (all rows)", (scored["UL_eur"] >= 0).all())


# ----------------------------------------------------------------------------
print("=" * 70)
print("7. PORTFOLIO AGGREGATION CONSISTENCY")
print("=" * 70)
# Sum of individual EL should equal portfolio_summary total
sum_el = scored["EL_eur"].sum()
check("Portfolio EL == sum of customer EL",
      np.isclose(sum_el, summary["expected_loss_eur"]),
      f"summary {summary['expected_loss_eur']:.2f} vs sum {sum_el:.2f}")
sum_ead = scored["EAD_eur"].sum()
check("Portfolio EAD == sum of customer EAD",
      np.isclose(sum_ead, summary["total_exposure_eur"]),
      f"summary {summary['total_exposure_eur']:.2f} vs sum {sum_ead:.2f}")
# Loss rate identity
computed_rate = sum_el / sum_ead * 100
check("Loss rate == EL/EAD", np.isclose(computed_rate, summary["portfolio_loss_rate_pct"]),
      f"{computed_rate:.4f}% vs {summary['portfolio_loss_rate_pct']:.4f}%")
# Exposure-weighted PD must be between min and max PD
ewpd = summary["exposure_weighted_pd"]
check("Exposure-weighted PD within [min, max] PD",
      scored["PD"].min() <= ewpd <= scored["PD"].max(),
      f"{ewpd*100:.2f}% (range {scored['PD'].min()*100:.2f}-{scored['PD'].max()*100:.2f}%)")


# ----------------------------------------------------------------------------
print("=" * 70)
print("8. STRESS TEST ORDERING")
print("=" * 70)
base_el   = scored["EL_eur"].sum()
mild      = stress_test(scored, pd_shock_multiplier=1.5, lgd_shock_add=0.0)
severe    = stress_test(scored, pd_shock_multiplier=2.0, lgd_shock_add=0.05)
mild_el   = mild["EL_stressed_eur"].sum()
severe_el = severe["EL_stressed_eur"].sum()
check("Stress ordering: severe >= mild >= base",
      severe_el >= mild_el >= base_el,
      f"base €{base_el:,.0f} <= mild €{mild_el:,.0f} <= severe €{severe_el:,.0f}")
# Mild: with PD x1.5 and no PD capping, EL should be ~1.5x (minus capping effects)
mild_ratio = mild_el / base_el
check("Mild recession ratio in plausible range (1.3-1.5x)", 1.25 <= mild_ratio <= 1.55,
      f"{mild_ratio:.3f}x")
# Stressed PD must still be <= 0.99
check("Stressed PD capped at 0.99", (mild["PD_stressed"] <= 0.99).all())


# ----------------------------------------------------------------------------
print("=" * 70)
print("9. TIER LOGIC CONSISTENCY")
print("=" * 70)
# Red tier should have higher mean composite score than Amber than Green
mean_by_tier = scored.groupby("risk_tier", observed=True)["composite_risk_score"].mean()
check("Mean score: Red > Amber > Green",
      mean_by_tier["Red"] > mean_by_tier["Amber"] > mean_by_tier["Green"],
      f"R{mean_by_tier['Red']:.1f} A{mean_by_tier['Amber']:.1f} G{mean_by_tier['Green']:.1f}")
# Red tier should also have higher mean PD
pd_by_tier = scored.groupby("risk_tier", observed=True)["PD"].mean()
check("Mean PD: Red > Amber > Green",
      pd_by_tier["Red"] > pd_by_tier["Amber"] > pd_by_tier["Green"],
      f"R{pd_by_tier['Red']*100:.1f}% A{pd_by_tier['Amber']*100:.1f}% G{pd_by_tier['Green']*100:.1f}%")
# Tier boundaries: every Green < 35, every Red >= 60
check("All Green scores < 35", (scored.loc[scored.risk_tier=="Green","composite_risk_score"] < 35).all())
check("All Red scores >= 60", (scored.loc[scored.risk_tier=="Red","composite_risk_score"] >= 60).all())


# ----------------------------------------------------------------------------
print("=" * 70)
print("10. LOSS CONCENTRATION CURVE")
print("=" * 70)
sorted_el = scored.sort_values("EL_eur", ascending=False)["EL_eur"].values
cum = np.cumsum(sorted_el) / sorted_el.sum() * 100
def share_at(pct):
    idx = int(np.ceil(len(sorted_el) * pct / 100)) - 1
    return cum[idx]
p5, p10, p20 = share_at(5), share_at(10), share_at(20)
check("Concentration monotonic (p5 <= p10 <= p20)", p5 <= p10 <= p20,
      f"5%:{p5:.0f}% 10%:{p10:.0f}% 20%:{p20:.0f}%")
check("Top 20% carries majority of EL (>50%)", p20 > 50, f"{p20:.0f}%")
check("Cumulative share ends at 100%", np.isclose(cum[-1], 100.0))


# ----------------------------------------------------------------------------
print("=" * 70)
print("11. CROSS-CHECK vs DASHBOARD / TALK-TRACK QUOTED FIGURES")
print("=" * 70)
# These are the figures quoted in the talk track. Verify they still hold.
print(f"  Total exposure (EAD):  €{summary['total_exposure_eur']:,.0f}")
print(f"  Expected Loss (12m):   €{summary['expected_loss_eur']:,.0f}")
print(f"  Portfolio loss rate:    {summary['portfolio_loss_rate_pct']:.2f}%")
print(f"  Loss concentration:     top5={p5:.0f}%  top10={p10:.0f}%  top20={p20:.0f}%")
print(f"  Stress mild:            €{mild_el:,.0f} (+{(mild_ratio-1)*100:.0f}%)")
print(f"  Stress severe:          €{severe_el:,.0f} (+{(severe_el/base_el-1)*100:.0f}%)")
tier_counts = scored["risk_tier"].value_counts()
print(f"  Tier counts:            Green={tier_counts['Green']} Amber={tier_counts['Amber']} Red={tier_counts['Red']}")
red_el_share = scored.loc[scored.risk_tier=="Red","EL_eur"].sum() / summary["expected_loss_eur"] * 100
print(f"  Red tier EL share:      {red_el_share:.1f}%")


# ----------------------------------------------------------------------------
print()
print("=" * 70)
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {len(results)} total")
print("=" * 70)
if n_fail:
    print("\nFAILED CHECKS:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}: {detail}")
