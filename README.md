# Credit Risk Monitor — Demo Prototype

> A three-layer credit risk monitoring system for SME trade-credit portfolios.
> [Click here to view the webpage!](https://github.com/KunduKamal/Credit-Risk-Monitoring-Dashboard/blob/main/dashboard.html)

---

## What this is

A working prototype of the credit risk monitoring system.

> *"Layer one is data ingestion. Layer two is a scoring layer that flags
> customers crossing pre-defined thresholds. Layer three is the analyst-facing
> dashboard — surface the flagged accounts at the top so we spend time on the
> ~5% that need attention, not the 95% that don't."*

This codebase implements exactly that, end-to-end.

## What it does

The system operates at two altitudes simultaneously:

**Operational (the analyst view) — *who needs attention today?***
1. **Ingests** a portfolio of 300 SME fleet customers with 12 months of
   behavioural data (financials, fuel spend, payment history, Creditreform
   bureau scores)
2. **Scores** each customer on four dimensions (bureau, financials, payments,
   spend anomaly) and assigns a Red/Amber/Green tier
3. **Surfaces** the top-flagged accounts in an interactive HTML dashboard

**Actuarial (the management view) — *how much money is at stake?***
4. **Quantifies** every customer's Probability of Default (PD), Loss Given
   Default (LGD), and Exposure at Default (EAD)
5. **Computes** Expected Loss: `EL = PD × LGD × EAD` (Basel III / IFRS 9 framework)
6. **Stress tests** the portfolio under mild and severe recession scenarios

Open `output/dashboard.html` in any browser to see the result.

## Project structure

```
risk_monitor/
├── data_generator.py        Layer 1: data ingestion (simulated source tables)
├── risk_engine.py           Layer 2a: composite operational scoring
├── expected_loss.py         Layer 2b: PD/LGD/EAD/EL actuarial framework
├── dashboard_builder.py     Layer 3: interactive HTML dashboard
├── main.py                  Pipeline orchestrator
├── README.md                This file
├── INTERVIEW_TALK_TRACK.md  How to walk through this in the interview
└── output/
    ├── dashboard.html       Self-contained interactive dashboard
    ├── dashboard_preview.png Screenshot for quick reference
    ├── flagged_accounts.csv  Red + Amber tier customers (sorted by EL)
    └── portfolio_scored.csv  Full scored portfolio with PD/LGD/EAD/EL
```

## How to run

```bash
pip install pandas numpy plotly
python main.py
```

Open `output/dashboard.html` in your browser. That's it.

## The three layers, explained

### Layer 1 — Data ingestion (`data_generator.py`)

In this prototype, three "source tables" are simulated to mirror the real
warehouse structure at Radius:

| Source table       | Real-world source                              |
|--------------------|------------------------------------------------|
| customer_profile   | Internal CRM / Salesforce                      |
| financial_snapshot | Bundesanzeiger / Handelsregister / Companies House |
| credit_bureau      | Creditreform / Schufa API                      |
| spend_history      | Internal billing system                        |
| payment_history    | Internal A/R ledger                            |

The data simulation includes correlated "latent health" — financially weak
customers also tend to have worse payment behaviour and rising bureau scores.
This mimics real-world signal correlation and lets the scoring engine
demonstrate convergent risk detection.

### Layer 2 — Risk scoring engine (`risk_engine.py`)

Composite score = weighted combination of four sub-scores, each on a 0–100 scale:

| Component         | Weight | What it measures                                        |
|-------------------|--------|---------------------------------------------------------|
| Bureau            | 40%    | Creditreform score level + 3-month trajectory           |
| Financial health  | 25%    | Current ratio, debt/EBITDA, OCF margin                  |
| Payment behaviour | 25%    | Late + failed direct debits over trailing 3 months      |
| Spend anomaly     | 10%    | Z-score of latest month spend vs 6-month baseline       |

Each customer also gets up to four binary **flags** explaining *why* they're
on the queue. This separation between numeric score (for ranking) and
explainable flags (for analyst action) is intentional — risk decisions get
challenged by sales and account management, so the reason must be visible.

### Layer 3 — Analyst dashboard (`dashboard_builder.py`)

Interactive HTML dashboard with two sections:

**Operational view:**
- KPI strip — at-a-glance portfolio health including total EL
- Tier mix donut — proportion of book at each risk level
- Industry concentration — where the bad book is concentrated
- Bureau score trend — 12-month direction of travel
- Spend anomaly example — drill-down on one flagged customer

**Actuarial view:**
- Loss concentration (Lorenz) curve — what share of customers drives what share of EL
- Stress test scenarios — portfolio EL under base case, mild recession, severe recession

**Action queue:**
- Top 20 highest-risk accounts with full breakdown (composite score, PD, LGD, EAD, EL, flags)

Built with Plotly; outputs a single self-contained HTML file (no server needed).

## Layer 2b — Expected Loss framework (`expected_loss.py`)

The actuarial layer that converts operational signals into euro-denominated loss expectations.

### Probability of Default (PD)
- Base PD interpolated from Creditreform Bonitätsindex score (anchor points calibrated
  to published German SME default-rate bands).
- Industry multiplier (haulage 1.30×, public sector 0.60×, etc.).
- Forward-looking trajectory adjustment: if score has worsened sharply over 3 months,
  extrapolate the drift 6 months forward and use the higher of base vs projected PD.
  This implements an IFRS 9-style forward-looking ECL adjustment.

### Loss Given Default (LGD)
- Baseline 75% (typical for unsecured B2B trade credit).
- Adjustments for size band (larger customers = lower LGD due to personal guarantees
  and traceable assets) and industry (haulage/construction have recoverable assets).
- Floor 40%, ceiling 90%.

### Exposure at Default (EAD)
- `EAD = drawn balance + CCF × undrawn limit` (Basel III revolving-credit formulation).
- Drawn balance ≈ 1.5 months of recent spend (reflects 7–14 day payment terms).
- CCF (Credit Conversion Factor) = 60% — captures the "load up before default"
  phenomenon where customers run up their credit line as default approaches.

### Expected Loss
- `EL = PD × LGD × EAD` per customer, summed to portfolio level.
- Also computes Unexpected Loss (UL) as the standalone standard deviation —
  the volatility component used for capital allocation under IRB.

### Stress testing
- Mild recession: PD × 1.5 across the book.
- Severe recession: PD × 2.0 and LGD + 5pp.
- Mirrors the forward-looking macro overlay used in IFRS 9 ECL calculations.

## Methodology notes

- **Synthetic data, real architecture.** I'm using simulated data because real
  Radius customer data is proprietary and Creditreform/Schufa data is paid.
  The architecture, scoring logic, and dashboard are identical to what would
  ship to production.
- **Thresholds are sensible defaults, not calibrated.** In production these
  would be tuned to the actual default rate of the book — typically by running
  the scorer historically and adjusting cutoffs so that, say, the top X% of
  flagged accounts capture Y% of actual defaults.
- **Explainability over accuracy.** A more accurate black-box ML model
  (gradient boosting on dozens of features) could probably beat this composite,
  but at the cost of being unable to justify a credit limit reduction to a
  long-standing customer's account manager. For trade credit at this scale,
  explainability wins.

## What's missing

- **Real data connectors** — Creditreform API client, Bundesanzeiger scraper,
  internal database queries
- **Backtesting framework** — replay historical data through the scorer to
  calibrate thresholds against actual defaults
- **Alerting** — auto-email / Teams notification when a Green account crosses
  into Amber, so the team isn't checking a dashboard manually
- **Stakeholder report templates** — auto-generated PowerPoint / PDF for the
  account-management conversation when a credit limit needs to come down
- **Sector benchmarks** — comparing each customer's financials to a sector
  median, not just absolute thresholds (a 1.1 current ratio is fine for
  retail, weak for construction)

