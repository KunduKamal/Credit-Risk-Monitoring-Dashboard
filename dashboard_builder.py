"""
dashboard_builder.py
--------------------
Layer 3 of the monitoring system: the analyst-facing dashboard.

Outputs a single self-contained HTML file (uses Plotly's offline mode) so it
can be opened in any browser without a server. This is the same architectural
pattern Radius could deploy internally on top of Power BI or a Streamlit/Dash
app -- the visual logic and KPIs are identical.

The dashboard now operates at two altitudes:

  OPERATIONAL (analyst view) - "who do I review today?"
    - Tier mix, industry concentration, bureau trend
    - Spend anomaly drill-down
    - Top accounts queue with explainable flags

  ACTUARIAL (management view) - "how much money is at stake?"
    - Total Expected Loss (EL = PD x LGD x EAD)
    - Loss concentration curve
    - Stress test scenarios

Both views share the same underlying data and scoring -- one drives daily
action, the other drives portfolio-level decisions on capital and pricing.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.io import to_html

from expected_loss import portfolio_summary, stress_test

# Radius brand-ish colour palette (subtle, no logos)
COLOR_RED   = "#D32F2F"
COLOR_AMBER = "#F9A825"
COLOR_GREEN = "#388E3C"
COLOR_BLUE  = "#1565C0"
COLOR_GREY  = "#666666"
COLOR_BG    = "#F5F6F8"

TIER_COLORS = {"Green": COLOR_GREEN, "Amber": COLOR_AMBER, "Red": COLOR_RED}


def _fig_tier_donut(scored):
    """Portfolio tier breakdown."""
    counts = scored["risk_tier"].value_counts().reindex(["Green", "Amber", "Red"])
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.55,
        marker=dict(colors=[TIER_COLORS[t] for t in counts.index]),
        textinfo="label+percent",
        textfont=dict(size=14),
    ))
    total_exposure = scored["EAD_eur"].sum()
    red_exposure = scored.loc[scored["risk_tier"] == "Red", "EAD_eur"].sum()
    pct = red_exposure / total_exposure * 100 if total_exposure else 0

    fig.add_annotation(
        text=f"<b>{len(scored)}</b><br>customers",
        x=0.5, y=0.5, font_size=20, showarrow=False,
    )
    fig.update_layout(
        title=dict(text=f"Risk Tier Mix<br><sub>Red represents €{red_exposure:,.0f} ({pct:.1f}%) of total exposure at default (EAD)</sub>",
                   x=0.5, xanchor="center"),
        showlegend=True, height=400, margin=dict(t=80, b=20, l=20, r=20),
    )
    return fig


def _fig_industry_concentration(scored):
    """Industry x tier stacked bar -- where is the bad book concentrated?"""
    pivot = (
        scored.groupby(["industry", "risk_tier"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Green", "Amber", "Red"], fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True)

    fig = go.Figure()
    for tier in ["Green", "Amber", "Red"]:
        fig.add_trace(go.Bar(
            name=tier, y=pivot.index, x=pivot[tier],
            orientation="h", marker_color=TIER_COLORS[tier],
        ))
    fig.update_layout(
        title=dict(text="Risk by Industry<br><sub>Where is the at-risk exposure concentrated?</sub>",
                   x=0.5, xanchor="center"),
        barmode="stack", height=400,
        xaxis_title="Number of customers",
        yaxis_title=None,
        margin=dict(t=80, b=40, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def _fig_portfolio_trend(behaviour):
    """Average Creditreform score across the book over 12 months."""
    monthly = behaviour.groupby("month")["creditreform_score"].agg(["mean", "median"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["mean"],
        mode="lines+markers", name="Portfolio average",
        line=dict(color=COLOR_BLUE, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["median"],
        mode="lines+markers", name="Portfolio median",
        line=dict(color=COLOR_GREY, width=2, dash="dot"),
    ))
    fig.add_hline(y=300, line_dash="dash", line_color=COLOR_AMBER,
                  annotation_text="Watch threshold (300)", annotation_position="top left")

    fig.update_layout(
        title=dict(text="Portfolio Bureau Score Trend (12 months)<br><sub>Lower is better. Rising line = book deteriorating.</sub>",
                   x=0.5, xanchor="center"),
        xaxis_title="Month", yaxis_title="Creditreform score",
        height=400, margin=dict(t=80, b=40, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def _fig_anomaly_example(behaviour, scored):
    """Pick a real anomaly-flagged customer and show their 12-month spend pattern."""
    flagged = scored[scored["flag_spend_anomaly"] & (scored["risk_tier"] == "Red")]
    if len(flagged) == 0:
        flagged = scored[scored["flag_spend_anomaly"]]
    if len(flagged) == 0:
        return None
    pick = flagged.iloc[0]
    customer_id = pick["customer_id"]
    company = pick["company_name"]

    cust_hist = behaviour[behaviour["customer_id"] == customer_id].sort_values("month")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cust_hist["month"], y=cust_hist["fuel_spend_eur"],
        marker_color=[COLOR_RED if i == len(cust_hist) - 1 else COLOR_BLUE
                      for i in range(len(cust_hist))],
        name="Monthly fuel spend",
    ))
    baseline = cust_hist["fuel_spend_eur"].iloc[-7:-1].mean()
    fig.add_hline(y=baseline, line_dash="dash", line_color=COLOR_GREY,
                  annotation_text=f"6-month baseline (€{baseline:,.0f})",
                  annotation_position="left")

    fig.update_layout(
        title=dict(text=f"Spend Anomaly Example: {company} ({customer_id})<br><sub>Latest month spend is {pick['spend_zscore']:.1f}σ above baseline — possible 'load-up before default'</sub>",
                   x=0.5, xanchor="center"),
        xaxis_title="Month", yaxis_title="Fuel spend (EUR)",
        height=400, margin=dict(t=80, b=40, l=20, r=20),
        showlegend=False,
    )
    return fig


def _fig_top_flagged_table(scored, top_n=20):
    """The actionable queue: top N highest-risk accounts."""
    cols_in  = ["customer_id", "company_name", "industry", "credit_limit_eur",
                "composite_risk_score", "risk_tier", "creditreform_latest",
                "PD", "LGD", "EAD_eur", "EL_eur", "primary_flag"]
    top = scored[cols_in].head(top_n).copy()

    # Format display values
    top["credit_limit_eur"]     = top["credit_limit_eur"].apply(lambda x: f"€{x:,.0f}")
    top["composite_risk_score"] = top["composite_risk_score"].apply(lambda x: f"{x:.1f}")
    top["PD"]                   = top["PD"].apply(lambda x: f"{x*100:.1f}%")
    top["LGD"]                  = top["LGD"].apply(lambda x: f"{x*100:.0f}%")
    top["EAD_eur"]              = top["EAD_eur"].apply(lambda x: f"€{x:,.0f}")
    top["EL_eur"]               = top["EL_eur"].apply(lambda x: f"€{x:,.0f}")

    headers = ["Customer ID", "Company", "Industry", "Credit limit", "Risk score",
               "Tier", "Bureau", "PD", "LGD", "EAD", "EL", "Primary flags"]
    cell_values = [top[c].tolist() for c in cols_in]

    tier_to_color = {"Red": "#FFCDD2", "Amber": "#FFE0B2", "Green": "#C8E6C9"}
    row_colors = [tier_to_color.get(str(t), "white") for t in top["risk_tier"]]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color=COLOR_BLUE, font=dict(color="white", size=11),
            align="left", height=32,
        ),
        cells=dict(
            values=cell_values,
            fill_color=[row_colors] * len(cols_in),
            align="left", height=28, font=dict(size=10),
        ),
    ))
    fig.update_layout(
        title=dict(text=f"Top {top_n} Highest-Risk Accounts (analyst queue)<br><sub>Composite score for triage; PD/LGD/EAD/EL for the actuarial picture</sub>",
                   x=0.5, xanchor="center"),
        height=640, margin=dict(t=80, b=20, l=10, r=10),
    )
    return fig


def _fig_loss_concentration(scored):
    """
    Lorenz-style curve: what share of customers drives what share of EL?
    Steeper curves = more concentrated risk = fewer accounts driving most loss.

    For trade credit portfolios, this is one of the most important risk
    visualisations -- it tells the team how much triage discipline matters.
    """
    sorted_el = scored.sort_values("EL_eur", ascending=False)["EL_eur"].values
    cum_el = np.cumsum(sorted_el)
    total_el = cum_el[-1] if len(cum_el) > 0 else 1
    cum_share_customers = np.arange(1, len(sorted_el) + 1) / len(sorted_el) * 100
    cum_share_el = cum_el / total_el * 100

    # Find threshold points: where do top 5%, 10%, 20% land?
    def _share_at(pct):
        idx = int(np.ceil(len(sorted_el) * pct / 100)) - 1
        return cum_share_el[idx] if idx >= 0 else 0

    p5  = _share_at(5)
    p10 = _share_at(10)
    p20 = _share_at(20)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_share_customers, y=cum_share_el,
        mode="lines", name="Actual loss concentration",
        line=dict(color=COLOR_RED, width=3),
        fill="tozeroy", fillcolor="rgba(211, 47, 47, 0.15)",
    ))
    # Equality line (perfectly uniform book)
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines", name="Equal distribution (theoretical)",
        line=dict(color=COLOR_GREY, dash="dash", width=1.5),
    ))

    # Reference markers
    for pct, val, label in [(5, p5, "5%"), (10, p10, "10%"), (20, p20, "20%")]:
        fig.add_trace(go.Scatter(
            x=[pct], y=[val], mode="markers+text",
            marker=dict(size=10, color=COLOR_BLUE),
            text=[f"<b>Top {label}: {val:.0f}%</b>"],
            textposition="top right" if pct < 15 else "bottom right",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=f"Loss Concentration Curve<br><sub>Top 5% of customers carry {p5:.0f}% of expected loss · Top 10% carry {p10:.0f}% · Top 20% carry {p20:.0f}%</sub>",
                   x=0.5, xanchor="center"),
        xaxis=dict(title="Cumulative share of customers (%)", range=[0, 100]),
        yaxis=dict(title="Cumulative share of Expected Loss (%)", range=[0, 105]),
        height=420, margin=dict(t=80, b=40, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def _fig_stress_test(scored):
    """
    Compare portfolio EL under three scenarios: base, mild recession, severe.
    This is the management-facing chart -- "what's our downside if conditions worsen?"
    """
    base_el = scored["EL_eur"].sum()

    mild = stress_test(scored, pd_shock_multiplier=1.5, lgd_shock_add=0.0)
    severe = stress_test(scored, pd_shock_multiplier=2.0, lgd_shock_add=0.05)

    mild_el = mild["EL_stressed_eur"].sum()
    severe_el = severe["EL_stressed_eur"].sum()

    scenarios = ["Base case<br>(current PD/LGD)",
                 "Mild recession<br>(PD × 1.5)",
                 "Severe recession<br>(PD × 2.0, LGD +5pp)"]
    values = [base_el, mild_el, severe_el]
    colors = [COLOR_GREEN, COLOR_AMBER, COLOR_RED]
    deltas = ["", f"+{(mild_el/base_el - 1)*100:.0f}%", f"+{(severe_el/base_el - 1)*100:.0f}%"]

    fig = go.Figure(go.Bar(
        x=scenarios, y=values,
        marker_color=colors,
        text=[f"€{v/1000:,.0f}k<br><b>{d}</b>" for v, d in zip(values, deltas)],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Stress Test: Portfolio Expected Loss Under Different Macro Scenarios<br><sub>How much would EL increase if European SME default rates rose? Mirrors Basel III/IFRS 9 stress-testing logic.</sub>",
                   x=0.5, xanchor="center"),
        yaxis=dict(title="Expected Loss (EUR)", range=[0, max(values) * 1.25]),
        showlegend=False, height=420, margin=dict(t=80, b=40, l=20, r=20),
    )
    return fig


def _kpi_strip(scored):
    """Top-line KPI banner as HTML (not Plotly). Now includes EL metrics."""
    total = len(scored)
    red   = (scored["risk_tier"] == "Red").sum()
    amber = (scored["risk_tier"] == "Amber").sum()
    green = (scored["risk_tier"] == "Green").sum()
    total_exposure = scored["EAD_eur"].sum()
    total_el       = scored["EL_eur"].sum()
    loss_rate      = (total_el / total_exposure * 100) if total_exposure else 0
    red_el         = scored.loc[scored["risk_tier"] == "Red", "EL_eur"].sum()
    red_el_share   = (red_el / total_el * 100) if total_el else 0

    return f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:14px; margin:20px 0 28px 0;">
      <div style="padding:18px; background:white; border-left:5px solid {COLOR_BLUE}; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.5px;">Customers monitored</div>
        <div style="font-size:28px; font-weight:600; color:#222; margin-top:6px;">{total:,}</div>
      </div>
      <div style="padding:18px; background:white; border-left:5px solid {COLOR_RED}; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.5px;">Red / Amber / Green</div>
        <div style="font-size:22px; font-weight:600; color:#222; margin-top:6px;">
          <span style="color:{COLOR_RED}">{red}</span> /
          <span style="color:{COLOR_AMBER}">{amber}</span> /
          <span style="color:{COLOR_GREEN}">{green}</span>
        </div>
        <div style="font-size:12px; color:#888; margin-top:2px;">tier distribution</div>
      </div>
      <div style="padding:18px; background:white; border-left:5px solid {COLOR_GREY}; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.5px;">Total exposure (EAD)</div>
        <div style="font-size:28px; font-weight:600; color:#222; margin-top:6px;">€{total_exposure/1e6:.2f}M</div>
        <div style="font-size:12px; color:#888;">at-risk capital across the book</div>
      </div>
      <div style="padding:18px; background:white; border-left:5px solid {COLOR_RED}; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.5px;">Expected Loss (12m)</div>
        <div style="font-size:28px; font-weight:600; color:{COLOR_RED}; margin-top:6px;">€{total_el/1000:,.0f}k</div>
        <div style="font-size:12px; color:#888;">PD × LGD × EAD, summed</div>
      </div>
      <div style="padding:18px; background:white; border-left:5px solid {COLOR_AMBER}; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.5px;">Portfolio loss rate</div>
        <div style="font-size:28px; font-weight:600; color:#222; margin-top:6px;">{loss_rate:.2f}%</div>
        <div style="font-size:12px; color:#888;">EL ÷ EAD</div>
      </div>
      <div style="padding:18px; background:white; border-left:5px solid {COLOR_RED}; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.5px;">Red tier EL share</div>
        <div style="font-size:28px; font-weight:600; color:{COLOR_RED}; margin-top:6px;">{red_el_share:.1f}%</div>
        <div style="font-size:12px; color:#888;">{red} customers drive {red_el_share:.1f}% of expected loss</div>
      </div>
    </div>
    """


def build_dashboard(customers, financials, behaviour, scored, out_path):
    """Build the complete HTML dashboard.

    Two altitudes in the layout:
      1. KPI banner (mixed operational + actuarial)
      2. Operational view (tier mix, industry, bureau trend, spend anomaly)
      3. Actuarial view (loss concentration, stress tests)
      4. Top flagged accounts table (both views merged into the queue)
    """
    figs = {
        "tier_donut":         _fig_tier_donut(scored),
        "industry":           _fig_industry_concentration(scored),
        "bureau_trend":       _fig_portfolio_trend(behaviour),
        "spend_anomaly":      _fig_anomaly_example(behaviour, scored),
        "loss_concentration": _fig_loss_concentration(scored),
        "stress_test":        _fig_stress_test(scored),
        "top_table":          _fig_top_flagged_table(scored, top_n=20),
    }

    # Render to HTML blocks. Inline plotly.js once (first chart),
    # then reuse the loaded library for all subsequent charts.
    html_blocks = {}
    first = True
    for name, fig in figs.items():
        if fig is None:
            continue
        include_js = "inline" if first else False
        html_blocks[name] = to_html(fig, include_plotlyjs=include_js, full_html=False)
        first = False

    kpi_html = _kpi_strip(scored)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Radius Credit Risk Monitor — Portfolio Dashboard</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: {COLOR_BG};
    color: #222;
    margin: 0;
    padding: 24px;
  }}
  .container {{
    max-width: 1280px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 26px;
    margin-bottom: 4px;
  }}
  h2.section {{
    font-size: 16px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 30px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #ddd;
  }}
  .subtitle {{
    color: #666;
    font-size: 14px;
    margin-bottom: 8px;
  }}
  .chart {{
    background: white;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .row {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }}
  .row > .chart {{
    flex: 1;
    min-width: 480px;
  }}
  .footer {{
    color: #888;
    font-size: 12px;
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Credit Risk Monitoring — Portfolio Dashboard</h1>
  <p class="subtitle">
    Operational triage + actuarial loss quantification.
    Three-layer architecture: data ingestion → composite scoring + PD/LGD/EAD → analyst queue.
  </p>
  {kpi_html}

  <h2 class="section">Operational view — who needs attention today</h2>

  <div class="row">
    <div class="chart">{html_blocks["tier_donut"]}</div>
    <div class="chart">{html_blocks["industry"]}</div>
  </div>

  <div class="chart">{html_blocks["bureau_trend"]}</div>

  <div class="chart">{html_blocks["spend_anomaly"]}</div>

  <h2 class="section">Actuarial view — how much money is at stake</h2>

  <div class="chart">{html_blocks["loss_concentration"]}</div>

  <div class="chart">{html_blocks["stress_test"]}</div>

  <h2 class="section">Action queue</h2>

  <div class="chart">{html_blocks["top_table"]}</div>

  <div class="footer">
    Demonstration prototype — synthetic portfolio of 300 SME fleet customers.
    EL framework (PD × LGD × EAD) follows Basel III IRB / IFRS 9 ECL conventions,
    calibrated to indicative Creditreform default-rate bands.
    Production version would connect Creditreform / Schufa APIs, Bundesanzeiger filings,
    and internal billing/payment ledgers, with PD/LGD calibrated from Radius default history.
  </div>
</div>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


if __name__ == "__main__":
    from data_generator import generate_portfolio
    from risk_engine import score_portfolio

    customers, financials, behaviour = generate_portfolio()
    scored = score_portfolio(customers, financials, behaviour)

    out = build_dashboard(customers, financials, behaviour, scored,
                          out_path="output/dashboard.html")
    print(f"Built dashboard: {out}")
