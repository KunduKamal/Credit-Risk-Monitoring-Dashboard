"""
main.py
-------
Orchestrates the full pipeline:
    1. Generate portfolio data
    2. Run risk scoring engine (composite score + tiering)
    3. Calculate Expected Loss (PD × LGD × EAD)
    4. Build interactive HTML dashboard
    5. Export flagged-accounts CSV

Run: python main.py
Outputs land in ./output/
"""

import os
import pandas as pd

from data_generator import generate_portfolio
from risk_engine import score_portfolio
from expected_loss import calculate_expected_loss, portfolio_summary
from dashboard_builder import build_dashboard


def main():
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    print("[1/5] Generating portfolio data...")
    customers, financials, behaviour = generate_portfolio()
    print(f"      -> {len(customers)} customers, {len(behaviour)} monthly behaviour rows")

    print("[2/5] Running composite risk scoring (tier assignment)...")
    scored = score_portfolio(customers, financials, behaviour)
    tier_counts = scored["risk_tier"].value_counts().reindex(["Green", "Amber", "Red"])
    print(f"      -> Green: {tier_counts['Green']}  Amber: {tier_counts['Amber']}  Red: {tier_counts['Red']}")

    print("[3/5] Calculating Expected Loss (PD × LGD × EAD)...")
    scored = calculate_expected_loss(scored)
    summary = portfolio_summary(scored)
    print(f"      -> Total exposure (EAD):  €{summary['total_exposure_eur']:>12,.0f}")
    print(f"      -> Expected Loss (12m):   €{summary['expected_loss_eur']:>12,.0f}")
    print(f"      -> Portfolio loss rate:    {summary['portfolio_loss_rate_pct']:>12.2f}%")

    print("[4/5] Building interactive dashboard...")
    dashboard_path = build_dashboard(
        customers, financials, behaviour, scored,
        out_path=f"{out_dir}/dashboard.html",
    )
    print(f"      -> {dashboard_path}")

    print("[5/5] Exporting flagged-accounts CSV (analyst working file)...")
    flagged = scored[scored["risk_tier"].isin(["Red", "Amber"])].copy()
    export_cols = [
        "customer_id", "company_name", "industry", "region", "size_band",
        "credit_limit_eur", "composite_risk_score", "risk_tier", "primary_flag",
        "creditreform_latest", "creditreform_delta_3m",
        "current_ratio", "debt_to_ebitda", "ocf_margin",
        "payments_late_3m", "payments_failed_3m",
        "latest_spend_eur", "baseline_avg_spend_eur", "spend_zscore",
        "PD", "LGD", "EAD_eur", "EL_eur", "EL_rate_pct",
    ]
    flagged_csv_path = f"{out_dir}/flagged_accounts.csv"
    flagged.sort_values("EL_eur", ascending=False)[export_cols].to_csv(flagged_csv_path, index=False)
    print(f"      -> {flagged_csv_path} ({len(flagged)} accounts, sorted by EL)")

    # Also export the full scored portfolio for reference
    full_csv_path = f"{out_dir}/portfolio_scored.csv"
    scored[export_cols + ["latent_health_truth"]].to_csv(full_csv_path, index=False)
    print(f"      -> {full_csv_path} ({len(scored)} accounts)")

    print("\nDone. Open dashboard.html in your browser.")


if __name__ == "__main__":
    main()
