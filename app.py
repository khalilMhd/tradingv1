import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import RISK_RULES
from agent_core import run_autonomous_risk_agent
from broker_data import execute_paper_bracket_order
from database import fetch_recent_reports

st.set_page_config(page_title="Risk Guard AI Agent", page_icon="🛡️", layout="wide")

st.title("🛡️ Risk Guard AI Agent — Autonomous Risk & Paper Trading Lab")
st.caption("Independent quantitative agent for backtesting equity strategies, computing execution confidence & profit forecasts, enforcing 1% risk guardrails, and executing protected paper trades.")

# Sidebar: Experiment Settings, Capital Slider & Hard Guardrails
with st.sidebar:
    st.header("⚙️ Experiment Settings")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    period_input = st.selectbox("Backtest Lookback Period", ["1y", "2y", "5y"], index=1)
    desired_capital = st.slider(
        "Desired Capital to Allocate ($)",
        min_value=500,
        max_value=100000,
        value=10000,
        step=500,
        help="Enter the dollar amount you consider putting into this trade. The agent compares it against the 1% account risk cap."
    )
    auto_save = st.checkbox("Auto-save report to Supabase", value=True)
    run_btn = st.button("🚀 Run Risk Agent Audit", type="primary", use_container_width=True)

    st.divider()
    st.subheader("🔒 Active Hard Guardrails")
    st.markdown(f"""
    - **Max Risk Per Trade:** `{RISK_RULES['MAX_RISK_PER_TRADE_PCT']}%` of equity
    - **Max Allowed Drawdown:** `{RISK_RULES['MAX_ALLOWED_DRAWDOWN_PCT']}%`
    - **Minimum Sharpe Ratio:** `{RISK_RULES['MIN_SHARPE_RATIO']}`
    - **Max Daily Volatility (ATR):** `{RISK_RULES['MAX_DAILY_VOLATILITY_PCT']}%`
    - **Dynamic Stop-Loss:** `{RISK_RULES['ATR_STOP_LOSS_MULTIPLIER']}x ATR`
    - **Target Risk/Reward:** `1:{RISK_RULES['RISK_REWARD_RATIO']}`
    """)

# Main Navigation Tabs
tab1, tab2 = st.tabs(["🔬 Live Strategy Lab & Execution", "🗄️ Saved Reports Archive (Supabase)"])

with tab1:
    if run_btn and ticker_input:
        with st.spinner(f"Running backtests, computing execution odds, profit forecast & 1% capital cap for {ticker_input}..."):
            result = run_autonomous_risk_agent(
                ticker_input,
                period=period_input,
                auto_save=auto_save,
                desired_capital=float(desired_capital)
            )
            st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        if "error" in res:
            st.error(res["error"])
        else:
            lab = res["lab_results"]
            pos = lab["position_sizing"]
            pf = lab.get("profit_forecast", {})
            cg = lab.get("capital_guidance", {})
            status = lab["risk_status"]
            exec_conf = lab.get("execution_confidence_pct", 0.0)

            # Database save notification
            if res["save_status"]["saved"]:
                st.toast("Report saved successfully to Supabase!", icon="✅")
            else:
                st.info(f"Storage Notice: {res['save_status']['message']}")

            # Top Row: Verdict, Execution Chance %, Current Price, Stop-Loss
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Agent Verdict",
                status,
                delta="Safe to Execute" if status == "APPROVED" else "Rejected (High Risk)",
                delta_color="normal" if status == "APPROVED" else "inverse"
            )
            col2.metric(
                "Execution Confidence",
                f"{exec_conf}%",
                delta=f"Win Rate: {pf.get('historical_win_rate_pct', 0)}%"
            )
            col3.metric("Current Price / Volatility", f"${lab['current_price']}", f"ATR: {lab['current_daily_volatility_pct']}%")
            col4.metric("Mandatory Stop-Loss", f"${pos['stop_loss_price']}", f"-{pos['stop_loss_pct']}%", delta_color="inverse")

            st.progress(min(max(exec_conf / 100.0, 0.0), 1.0), text=f"Execution Readiness & Signal Quality Score: {exec_conf}%")

            # Capital Allocation Advisor Box (Desired vs. Optimal Safe Capital under 1% Rule)
            st.subheader("🎯 Capital Allocation Advisor (1% Account Risk Rule)")
            c_adv1, c_adv2, c_adv3 = st.columns(3)
            c_adv1.metric("Your Desired Amount", f"${cg.get('desired_capital_usd', 0):,.2f}", delta=f"{cg.get('desired_shares', 0)} shares")
            c_adv2.metric(
                "Optimal Safe Capital (≤1% Risk)",
                f"${cg.get('optimal_safe_capital_usd', 0):,.2f}",
                delta=f"Safe cap: {cg.get('safe_cap_shares', 0)} shares"
            )
            c_adv3.metric(
                "Est. Loss on Desired Amount",
                f"-${cg.get('desired_potential_loss_usd', 0):,.2f}",
                delta=f"Max 1% budget: ${pos['max_risk_dollars']:,.2f}",
                delta_color="inverse"
            )

            if cg.get("exceeds_1pct_risk"):
                st.warning(
                    f"⚠️ Putting **${cg.get('desired_capital_usd', 0):,.2f}** risks **-${cg.get('desired_potential_loss_usd', 0):,.2f}** if Stop-Loss hits, "
                    f"which exceeds your **1% account risk limit (${pos['max_risk_dollars']:,.2f})**. "
                    f"Recommended safe capital: **${cg.get('optimal_safe_capital_usd', 0):,.2f}** (`{cg.get('safe_cap_shares', 0)}` shares)."
                )
            else:
                st.success(
                    f"✅ Your desired amount (**${cg.get('desired_capital_usd', 0):,.2f}**, est. loss **-${cg.get('desired_potential_loss_usd', 0):,.2f}**, "
                    f"est. target profit **+${cg.get('desired_potential_profit_usd', 0):,.2f}**) stays within your **1% risk limit (${pos['max_risk_dollars']:,.2f})**."
                )

            # Expected Profit & Trade Forecast Cards
            st.subheader("💰 Expected Profit & Trade Forecast (1% Guarded Position)")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric(
                "Target Profit (Take-Profit)",
                f"+${pf.get('expected_gross_profit_usd', 0):,.2f}",
                delta=f"+{pf.get('expected_profit_pct', 0)}% per share"
            )
            p2.metric(
                "Expected Value (EV)",
                f"${pf.get('expected_value_usd', 0):+,.2f}",
                delta="Probability-weighted"
            )
            p3.metric(
                "Est. Annualized Return",
                f"{pf.get('expected_annual_return_pct', 0):+.2f}%",
                delta=f"~${pf.get('expected_annual_profit_usd', 0):+,.2f} on position"
            )
            p4.metric(
                "Capped Max Loss (1% Rule)",
                f"-${pf.get('expected_max_loss_usd', pos['max_risk_dollars']):,.2f}",
                delta=f"{pos['recommended_shares']} shares",
                delta_color="inverse"
            )

            st.divider()

            # Cumulative Equity Curves Chart
            st.subheader("📈 Strategy Backtest Equity Curves")
            curves = lab["equity_curves"]
            fig = go.Figure()
            dates = curves["Dates"]
            for strat_name, values in curves.items():
                if strat_name != "Dates":
                    fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", name=strat_name))
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Cumulative Return (1.0 = Initial Capital)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Strategy Risk Comparison Table
            st.subheader("📊 Quantitative Risk & Return Comparison")
            comp_df = pd.DataFrame(lab["strategies_comparison"]).T
            comp_df = comp_df.rename(columns={
                "total_return_pct": "Total Return (%)",
                "annual_volatility_pct": "Annual Volatility (%)",
                "sharpe_ratio": "Sharpe Ratio",
                "max_drawdown_pct": "Max Drawdown (%)",
                "win_rate_pct": "Win Rate (%)",
                "risk_score": "Risk Score",
                "passed_strict_guardrails": "Passed Guardrails?"
            })
            st.dataframe(comp_df, use_container_width=True)

            # AI CRO Report & Paper Execution Panel
            c_ai, c_exec = st.columns([2, 1])
            with c_ai:
                st.subheader("🧠 Chief Risk Officer (AI Report)")
                st.markdown(res["ai_summary"])

                with st.expander("📰 Recent Company News Scanned"):
                    for n in res["news"]:
                        st.write(n)

            with c_exec:
                st.subheader("⚡ Paper Trading Execution")
                st.write(f"**Paper Account Balance:** `${pos['account_balance']:,.2f}`")
                st.write(f"**Optimal Safe Capital (1%):** `${cg.get('optimal_safe_capital_usd', 0):,.2f}`")
                st.write(f"**Guarded Position Size:** `{pos['recommended_shares']}` shares (`${pf.get('expected_position_value_usd', 0):,.2f}`)")
                st.write(f"**Max Dollar Risk (1% Rule):** `${pos['max_risk_dollars']:,.2f}`")
                st.write(f"**Target Profit at TP (`${pos['take_profit_price']}`):** `+${pf.get('expected_gross_profit_usd', 0):,.2f}`")

                if status == "APPROVED" and pos["recommended_shares"] > 0:
                    if exec_conf < 70.0:
                        st.caption(f"ℹ️ Confidence is {exec_conf}% (below your 70% reference mark — manual execution still available).")
                    if st.button("🟢 Send Protected Bracket Order to Alpaca", type="primary", use_container_width=True):
                        exec_res = execute_paper_bracket_order(
                            ticker=lab["ticker"],
                            qty=pos["recommended_shares"],
                            stop_loss_price=pos["stop_loss_price"],
                            take_profit_price=pos["take_profit_price"]
                        )
                        st.success("Execution Result:")
                        st.json(exec_res)
                else:
                    st.warning("🚫 Execution button locked: Ticker or strategies failed strict risk guardrails.")

with tab2:
    st.subheader("🗄️ Saved Risk Audits & Experiment Archive")
    if st.button("🔄 Refresh Archive"):
        st.rerun()

    reports = fetch_recent_reports(limit=20)
    if not reports:
        st.info("No saved reports found yet (or verify Supabase credentials in Secrets).")
    else:
        summary_table = []
        for r in reports:
            summary_table.append({
                "Timestamp": str(r.get("created_at", ""))[:16].replace("T", " "),
                "Ticker": r.get("ticker"),
                "Verdict": r.get("risk_status"),
                "Winning Strategy": r.get("winning_strategy"),
                "Sharpe": r.get("sharpe_ratio"),
                "Max DD (%)": r.get("max_drawdown"),
                "Stop Loss (%)": r.get("stop_loss_pct")
            })
        st.dataframe(pd.DataFrame(summary_table), use_container_width=True)

        st.divider()
        selected_idx = st.selectbox(
            "Select an audit report to view full CRO analysis:",
            range(len(reports)),
            format_func=lambda i: f"{reports[i]['ticker']} — {str(reports[i].get('created_at', ''))[:10]} ({reports[i]['risk_status']})"
        )
        st.markdown(reports[selected_idx].get("ai_summary", ""))
