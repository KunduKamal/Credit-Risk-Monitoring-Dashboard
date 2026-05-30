"""
data_generator.py
-----------------
Generates a synthetic portfolio of European fleet SMEs that mirrors the
characteristics of a typical fuel-card / trade-credit customer base.

In a real Radius implementation, these tables would be sourced from:
    - customer_profile     -> internal CRM / Salesforce
    - financial_snapshot   -> Bundesanzeiger / Handelsregister / Companies House
    - credit_bureau        -> Creditreform / Schufa API
    - spend_history        -> internal billing system
    - payment_history      -> internal A/R ledger

The schema and joins below are intentionally identical to that real-world setup,
so swapping synthetic data for production data is a configuration change, not
a rewrite.
"""

import numpy as np
import pandas as pd
from datetime import date

# Reproducibility: same seed -> same portfolio every run.
RNG = np.random.default_rng(seed=42)

# ----------------------------------------------------------------------------
# Portfolio constants
# ----------------------------------------------------------------------------
N_CUSTOMERS = 300
N_MONTHS = 12  # rolling 12-month behavioural window

INDUSTRIES = {
    # Industry -> (portfolio weight, baseline risk multiplier).
    # Haulage and construction are historically the highest-default fleet sectors.
    "Haulage & Logistics": (0.30, 1.30),
    "Construction":        (0.22, 1.25),
    "Courier & Delivery":  (0.18, 1.10),
    "Trades & Services":   (0.15, 1.05),
    "Retail Distribution": (0.10, 1.00),
    "Public Sector":       (0.05, 0.70),
}

REGIONS = ["Berlin", "Hamburg", "Frankfurt", "Munich", "Köln", "Stuttgart", "Düsseldorf"]

SIZE_BANDS = {
    # vehicles -> (weight, avg monthly fuel spend EUR per vehicle)
    "1-5":     (0.45, 850),
    "6-20":    (0.32, 950),
    "21-50":   (0.15, 1050),
    "51-100":  (0.06, 1150),
    "100+":    (0.02, 1300),
}

# Synthetic but plausible German SME company names
COMPANY_SUFFIXES = ["GmbH", "GmbH & Co. KG", "AG", "UG", "OHG"]
NAME_FIRST = ["Berliner", "Hanseatic", "Rhein", "Alpen", "Norddeutsche", "Bayern",
              "Schwarzwald", "Elbe", "Donau", "Spree", "Mainzer", "Frankfurt",
              "Münchner", "Hamburger", "Kölner", "Sächsische", "Thüringer",
              "Hessen", "Westfalen", "Nordsee"]
NAME_SECOND = ["Transport", "Logistik", "Spedition", "Bau", "Express", "Kurier",
               "Handwerk", "Service", "Fracht", "Mobilität", "Versand", "Verkehr",
               "Bauunternehmen", "Trans", "Cargo", "Lieferdienst"]


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------
def _weighted_choice(options_dict):
    """Draw from a dict of {key: (weight, ...)}; returns the key."""
    keys = list(options_dict.keys())
    weights = np.array([v[0] for v in options_dict.values()])
    weights = weights / weights.sum()
    return RNG.choice(keys, p=weights)


def _generate_customer_profile():
    """Generate one customer profile row."""
    industry = _weighted_choice(INDUSTRIES)
    size_band = _weighted_choice(SIZE_BANDS)
    region = RNG.choice(REGIONS)

    name = f"{RNG.choice(NAME_FIRST)} {RNG.choice(NAME_SECOND)} {RNG.choice(COMPANY_SUFFIXES)}"
    onboarding_year = int(RNG.integers(2014, 2025))

    # Credit limit roughly tracks size band * 1.5 months of expected fuel spend.
    avg_spend_per_vehicle = SIZE_BANDS[size_band][1]
    midpoint_vehicles = {"1-5": 3, "6-20": 12, "21-50": 35, "51-100": 75, "100+": 150}[size_band]
    expected_monthly_spend = avg_spend_per_vehicle * midpoint_vehicles
    credit_limit = int(round(expected_monthly_spend * 1.5, -2))  # round to nearest 100

    return {
        "industry": industry,
        "region": region,
        "size_band": size_band,
        "company_name": name,
        "onboarding_year": onboarding_year,
        "credit_limit_eur": credit_limit,
        "expected_monthly_spend_eur": expected_monthly_spend,
    }


def _generate_financial_snapshot(profile, latent_health):
    """
    Generate the latest annual financial snapshot.

    `latent_health` is a value in [0, 1]; 1 = strong, 0 = distressed.
    Used to keep financials, payment behaviour, and credit bureau scores
    correlated -- which is how the real world looks.
    """
    industry_mult = INDUSTRIES[profile["industry"]][1]
    expected_spend = profile["expected_monthly_spend_eur"]

    # Revenue scales with size; fleet customers' fuel cost is roughly 8-15% of revenue.
    fuel_as_pct_of_revenue = RNG.uniform(0.08, 0.15)
    revenue = (expected_spend * 12) / fuel_as_pct_of_revenue
    revenue *= RNG.uniform(0.85, 1.15)  # noise

    # EBITDA margin: healthy companies 8-18%, weak companies -2% to 8%.
    ebitda_margin = RNG.uniform(0.08, 0.18) * latent_health + RNG.uniform(-0.02, 0.08) * (1 - latent_health)
    ebitda = revenue * ebitda_margin

    # Operating cashflow ~ 70-110% of EBITDA, worse when distressed
    ocf_conversion = RNG.uniform(0.85, 1.10) * latent_health + RNG.uniform(0.55, 0.85) * (1 - latent_health)
    operating_cashflow = ebitda * ocf_conversion

    # Balance sheet items
    current_assets = revenue * RNG.uniform(0.18, 0.35)
    current_liabilities = current_assets / (RNG.uniform(1.25, 2.0) * latent_health + RNG.uniform(0.7, 1.2) * (1 - latent_health))
    total_debt = revenue * RNG.uniform(0.10, 0.25) * (1 + (1 - latent_health) * industry_mult * 0.5)
    cash = current_assets * RNG.uniform(0.1, 0.4) * latent_health + current_assets * RNG.uniform(0.02, 0.15) * (1 - latent_health)

    return {
        "revenue_eur":             int(round(revenue, -3)),
        "ebitda_eur":              int(round(ebitda, -3)),
        "operating_cashflow_eur":  int(round(operating_cashflow, -3)),
        "current_assets_eur":      int(round(current_assets, -3)),
        "current_liabilities_eur": int(round(current_liabilities, -3)),
        "total_debt_eur":          int(round(total_debt, -3)),
        "cash_eur":                int(round(cash, -3)),
    }


def _generate_behavioural_history(customer_id, profile, latent_health):
    """
    Generate 12 months of behavioural data:
        - monthly fuel spend
        - payment outcomes (on time / late / failed)
        - Creditreform Bonitätsindex (refreshed monthly)

    Creditreform scale: 100 = excellent, 600 = insolvent. Lower is better.
    """
    expected_spend = profile["expected_monthly_spend_eur"]
    industry_mult = INDUSTRIES[profile["industry"]][1]

    # Starting Creditreform score: healthy ~150-220, distressed 300-500+
    starting_score = RNG.uniform(140, 220) * latent_health + RNG.uniform(280, 480) * (1 - latent_health)
    score = float(starting_score)

    # Drift direction over the year. Distressed companies drift up (worse).
    monthly_drift = RNG.normal(0, 2) + (1 - latent_health) * RNG.uniform(2, 8) * industry_mult

    rows = []
    for month_offset in range(N_MONTHS, 0, -1):
        # date label: months ago from today
        month_label = pd.Timestamp(date.today()) - pd.DateOffset(months=month_offset - 1)
        month_str = month_label.strftime("%Y-%m")

        # Fuel spend: usually steady, occasional anomaly
        spend_noise = RNG.normal(0, 0.12)
        spend = expected_spend * (1 + spend_noise)

        # Distressed customers sometimes "load up" before going dark -> spike in latest 1-2 months
        if latent_health < 0.35 and month_offset <= 2 and RNG.random() < 0.40:
            spend *= RNG.uniform(1.6, 2.4)

        # Payment behaviour: number of weekly invoices in the month ~4, outcome depends on health
        invoices_due = RNG.integers(3, 6)
        on_time_prob = 0.96 * latent_health + 0.55 * (1 - latent_health)
        late_prob = (1 - on_time_prob) * 0.75
        failed_prob = (1 - on_time_prob) * 0.25

        outcomes = RNG.choice(
            ["on_time", "late", "failed"],
            size=invoices_due,
            p=[on_time_prob, late_prob, failed_prob],
        )
        on_time = int(np.sum(outcomes == "on_time"))
        late = int(np.sum(outcomes == "late"))
        failed = int(np.sum(outcomes == "failed"))

        # Update Creditreform score with drift + payment-driven adjustment
        score += monthly_drift + (late * 2 + failed * 6) - (on_time * 0.3)
        score = float(np.clip(score, 100, 600))

        rows.append({
            "customer_id": customer_id,
            "month": month_str,
            "fuel_spend_eur": int(round(spend)),
            "payments_on_time": on_time,
            "payments_late": late,
            "payments_failed": failed,
            "creditreform_score": int(round(score)),
        })

    return rows


def generate_portfolio():
    """
    Build the full portfolio: customer profiles + latest financials + 12-month behaviour.
    Returns three DataFrames mimicking the three source tables in a real warehouse.
    """
    customers, financials, behaviour = [], [], []

    for i in range(1, N_CUSTOMERS + 1):
        customer_id = f"RAD{i:05d}"
        profile = _generate_customer_profile()

        # Latent health drives correlated signal across financials, payments, and scores.
        # 70% of the book is healthy, 20% borderline, 10% distressed -- realistic for SME books.
        roll = RNG.random()
        if roll < 0.70:
            latent_health = RNG.uniform(0.65, 0.95)
        elif roll < 0.90:
            latent_health = RNG.uniform(0.35, 0.65)
        else:
            latent_health = RNG.uniform(0.05, 0.35)

        financials_row = _generate_financial_snapshot(profile, latent_health)

        customer_row = {"customer_id": customer_id, **profile, "latent_health_truth": round(latent_health, 3)}
        customers.append(customer_row)

        financials.append({"customer_id": customer_id, **financials_row})
        behaviour.extend(_generate_behavioural_history(customer_id, profile, latent_health))

    return (
        pd.DataFrame(customers),
        pd.DataFrame(financials),
        pd.DataFrame(behaviour),
    )


if __name__ == "__main__":
    customers, financials, behaviour = generate_portfolio()
    print(f"Generated {len(customers)} customers")
    print(f"Generated {len(financials)} financial snapshots")
    print(f"Generated {len(behaviour)} monthly behaviour rows")
    print("\nSample customer:")
    print(customers.head(1).T)
    print("\nSample financials:")
    print(financials.head(1).T)
    print("\nSample behaviour (3 months for first customer):")
    print(behaviour[behaviour.customer_id == customers.iloc[0].customer_id].tail(3))
