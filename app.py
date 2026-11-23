import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import RISK_RULES
from agent_core import run_autonomous_risk_agent
from broker_data import execute_paper_bracket_order
from database import fetch_recent_reports

st.set_page_config(page_title="Risk Guard AI Agent", page_icon="🛡️", layout="wide")

st.title("🛡️ Risk Guard AI Agent — Autonomous Risk & Paper Trading Lab")
st.caption("Independent quantitative agent for backtesting equity strategies, enforcing strict risk guardrails, and executing protected paper trades.")

# Sidebar: Experiment Settings & Hard Guardrails
with st.sidebar:
    st.header("⚙️ Experiment Settings")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    period_input = st.selectbox("Backtest Lookback Period", ["1y", "2y", "5y"], index=1)
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
        with st.spinner(f"Running backtests, scanning {ticker_input} news, and enforcing risk guardrails..."):
            result = run_autonomous_risk_agent(ticker_input, period=period_input, auto_save=auto_save)
            st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        if "error" in res:
            st.error(res["error"])
        else:
            lab = res["lab_results"]
            pos = lab["position_sizing"]
            status = lab["risk_status"]

            # Database save notification
            if res["save_status"]["saved"]:
                st.toast("Report saved successfully to Supabase!", icon="✅")
            else:
                st.info(f"Storage Notice: {res['save_status']['message']}")

            # Key Metrics Bar
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Agent Verdict",
                status,
                delta="Safe to Execute" if status == "APPROVED" else "Rejected (High Risk)",
                delta_color="normal" if status == "APPROVED" else "inverse"
            )
            col2.metric("Winning Strategy", lab["winning_strategy"].split()[0])
            col3.metric("Current Price / Volatility", f"${lab['current_price']}", f"ATR: {lab['current_daily_volatility_pct']}%")
            col4.metric("Mandatory Stop-Loss", f"${pos['stop_loss_price']}", f"-{pos['stop_loss_pct']}%", delta_color="inverse")

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
            comp_df.columns = [
                "Total Return (%)",
                "Annual Volatility (%)",
                "Sharpe Ratio",
                "Max Drawdown (%)",
                "Passed Guardrails?"
            ]
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
                st.write(f"**Max Dollar Risk (1% Rule):** `${pos['max_risk_dollars']:,.2f}`")
                st.write(f"**Calculated Position Size:** `{pos['recommended_shares']}` shares")
                st.write(f"**Take-Profit Target (1:2):** `${pos['take_profit_price']}`")

                if status == "APPROVED" and pos["recommended_shares"] > 0:
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
                "Timestamp": r.get("created_at", "")[:16].replace("T", " "),
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
            format_func=lambda i: f"{reports[i]['ticker']} — {reports[i]['created_at'][:10]} ({reports[i]['risk_status']})"
        )
        st.markdown(reports[selected_idx].get("ai_summary", ""))
