import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import RISK_RULES
from agent_core import run_autonomous_risk_agent
from broker_data import execute_paper_bracket_order
from database import fetch_recent_reports

st.set_page_config(page_title="Risk Guard AI Agent", page_icon="🛡️", layout="wide")

st.title("🛡️ Risk Guard AI Agent — مختبر إدارة المخاطر والتداول التجريبي")
st.caption("نظام مستقل لاختبار استراتيجيات الأسهم، فرض قواعد الحماية الصارمة، وتنفيذ صفقات Paper Trading محمية.")

# الشريط الجانبي: القواعد الصارمة وإعدادات التجربة
with st.sidebar:
    st.header("⚙️ إعدادات التجربة")
    ticker_input = st.text_input("رمز السهم (Ticker)", value="AAPL").upper().strip()
    period_input = st.selectbox("فترة الاختبار التاريخي", ["1y", "2y", "5y"], index=1)
    auto_save = st.checkbox("حفظ التقرير تلقائياً في Supabase", value=True)
    run_btn = st.button("🚀 إطلاق الـ Agent للفحص والتجربة", type="primary", use_container_width=True)

    st.divider()
    st.subheader("🔒 القواعد الصارمة المفعلة")
    st.markdown(f"""
    - **أقصى مخاطرة للصفقة:** `{RISK_RULES['MAX_RISK_PER_TRADE_PCT']}%` من الرصيد
    - **أقصى هبوط مسموح (Max DD):** `{RISK_RULES['MAX_ALLOWED_DRAWDOWN_PCT']}%`
    - **أدنى معامل أمان (Sharpe):** `{RISK_RULES['MIN_SHARPE_RATIO']}`
    - **أقصى تذبذب يومي:** `{RISK_RULES['MAX_DAILY_VOLATILITY_PCT']}%`
    - **مضاعف وقف الخسارة:** `{RISK_RULES['ATR_STOP_LOSS_MULTIPLIER']}x ATR`
    """)

# التبويبات الرئيسية
tab1, tab2 = st.tabs(["🔬 مختبر التجارب الحية والتنفيذ", "🗄️ سجل التقارير المحفوظة (Supabase)"])

with tab1:
    if run_btn and ticker_input:
        with st.spinner(f"جاري تشغيل التجارب، تحليل أخبار {ticker_input}، وتطبيق فلاتر المخاطر..."):
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

            # حالة الحفظ في قاعدة البيانات
            if res["save_status"]["saved"]:
                st.toast(res["save_status"]["message"], icon="✅")
            else:
                st.info(f"ملاحظة الحفظ: {res['save_status']['message']}")

            # شريط المؤشرات الرئيسية
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("قرار الـ Agent", status, delta="آمن للتنفيذ" if status == "APPROVED" else "مرفوض لخطورته", delta_color="normal" if status == "APPROVED" else "inverse")
            col2.metric("الأداة الفائزة", lab["winning_strategy"].split()[0])
            col3.metric("السعر الحالي / التقلب", f"${lab['current_price']}", f"ATR: {lab['current_daily_volatility_pct']}%")
            col4.metric("وقف الخسارة الإجباري", f"${pos['stop_loss_price']}", f"-{pos['stop_loss_pct']}%", delta_color="inverse")

            st.divider()

            # الرسم البياني لمقارنة أداء الاستراتيجيات
            st.subheader("📈 مقارنة المنحنى التراكمي للأدوات (Backtest Equity Curves)")
            curves = lab["equity_curves"]
            fig = go.Figure()
            dates = curves["Dates"]
            for strat_name, values in curves.items():
                if strat_name != "Dates":
                    fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", name=strat_name))
            fig.update_layout(height=400, xaxis_title="التاريخ", yaxis_title="نمو رأس المال (1.0 = البداية)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # جدول مقارنة مقاييس المخاطر
            st.subheader("📊 جدول مقارنة مقاييس المخاطر والعائد")
            comp_df = pd.DataFrame(lab["strategies_comparison"]).T
            comp_df.columns = ["العائد الإجمالي (%)", "التقلب السنوي (%)", "معامل شارب (Sharpe)", "أقصى هبوط (Max DD %)", "اجتاز شروط الأمان؟"]
            st.dataframe(comp_df, use_container_width=True)

            # قسم التقرير التحليلي للـ AI والأخبار
            c_ai, c_exec = st.columns([2, 1])
            with c_ai:
                st.subheader("🧠 تقرير مدير المخاطر (AI Risk Report)")
                st.markdown(res["ai_summary"])

                with st.expander("📰 أحدث أخبار الشركة التي تم فحصها"):
                    for n in res["news"]:
                        st.write(n)

            with c_exec:
                st.subheader("⚡ التنفيذ في حساب التداول التجريبي")
                st.write(f"**رصيد الحساب التجريبي:** `${pos['account_balance']:,.2f}`")
                st.write(f"**الحد الأقصى للخسارة (1%):** `${pos['max_risk_dollars']:,.2f}`")
                st.write(f"**الكمية المحسوبة بدقة:** `{pos['recommended_shares']}` سهم")
                st.write(f"**هدف جني الأرباح (1:2):** `${pos['take_profit_price']}`")

                if status == "APPROVED" and pos["recommended_shares"] > 0:
                    if st.button("🟢 إرسال الصفقة المحمية إلى موقع Alpaca", type="primary", use_container_width=True):
                        exec_res = execute_paper_bracket_order(
                            ticker=lab["ticker"],
                            qty=pos["recommended_shares"],
                            stop_loss_price=pos["stop_loss_price"],
                            take_profit_price=pos["take_profit_price"]
                        )
                        st.success("نتيجة التنفيذ:")
                        st.json(exec_res)
                else:
                    st.warning("🚫 زر التنفيذ مقفل تلقائياً لأن السهم أو الاستراتيجية لم تجتز شروط الأمان الصارمة.")

with tab2:
    st.subheader("🗄️ أرشيف التجارب والتقارير المحفوظة")
    if st.button("🔄 تحديث السجل"):
        st.rerun()

    reports = fetch_recent_reports(limit=20)
    if not reports:
        st.info("لا توجد تقارير محفوظة حالياً (أو تأكد من إضافة مفاتيح Supabase في ملف .env).")
    else:
        summary_table = []
        for r in reports:
            summary_table.append({
                "التاريخ": r.get("created_at", "")[:16].replace("T", " "),
                "السهم": r.get("ticker"),
                "القرار": r.get("risk_status"),
                "الأداة الفائزة": r.get("winning_strategy"),
                "Sharpe": r.get("sharpe_ratio"),
                "Max DD (%)": r.get("max_drawdown"),
                "Stop Loss (%)": r.get("stop_loss_pct")
            })
        st.dataframe(pd.DataFrame(summary_table), use_container_width=True)

        st.divider()
        selected_idx = st.selectbox("اختر تقريراً لقراءة التفاصيل الكاملة:", range(len(reports)), format_func=lambda i: f"{reports[i]['ticker']} — {reports[i]['created_at'][:10]} ({reports[i]['risk_status']})")
        st.markdown(reports[selected_idx].get("ai_summary", ""))
