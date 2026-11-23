import json
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, RISK_RULES
from broker_data import get_company_news
from risk_lab import run_strategy_experiment
from database import save_risk_report

def run_autonomous_risk_agent(ticker: str, period: str = "2y", auto_save: bool = True) -> dict:
    """
    Runs the full Risk Guard Agent:
    1. Executes quantitative strategy backtests + execution probability + profit forecasts.
    2. Fetches live company news.
    3. Passes metrics and news to Gemini to evaluate qualitative/quantitative risk and issue a strict decision.
    4. Saves the report to Supabase.
    """
    # Step 1: Run the quantitative lab
    lab_results = run_strategy_experiment(ticker, period=period)
    if "error" in lab_results:
        return {"error": lab_results["error"]}

    # Step 2: Fetch live company news
    news_list = get_company_news(ticker, limit=5)

    # Prepare data summary for the LLM (excluding large equity curve arrays)
    metrics_for_ai = {k: v for k, v in lab_results.items() if k != "equity_curves"}
    metrics_for_ai["recent_company_news"] = news_list
    metrics_for_ai["strict_guardrails"] = RISK_RULES

    if not GEMINI_API_KEY or "your_gemini" in GEMINI_API_KEY:
        ai_summary = "⚠️ GEMINI_API_KEY is not configured in Secrets. Relying strictly on quantitative guardrails."
    else:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            system_instruction = """
            You are an ultra-strict Chief Risk Officer (CRO) and Quantitative Analyst.
            Your mission is to evaluate strategy backtest metrics, execution confidence %, expected profit projections, and live company news to protect capital and minimize drawdowns.
            Strict Rules:
            1. Explain why the winning strategy succeeded or why the ticker was completely REJECTED based on Sharpe Ratio, Max Drawdown, and Execution Confidence %.
            2. Clearly highlight the Expected Profit Projection (Take-Profit target profit in USD & %, Expected Value EV, and annualized return projection) alongside the maximum capped loss (1% portfolio rule).
            3. Analyze the attached news headlines: flag any severe negative catalysts (earnings misses, lawsuits, regulatory probes, sector headwinds) that warrant caution even if historical metrics look good.
            4. Write the report in clear, direct, executive-level English using structured bullet points.
            """
            prompt = f"Analyze the following experiment results and issue the final Risk Officer Report:\n{json.dumps(metrics_for_ai, indent=2)}"
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.15
                )
            )
            ai_summary = response.text
        except Exception as e:
            ai_summary = f"Error communicating with Gemini API: {str(e)}"

    # Step 3: Save report to Supabase
    save_status = {"saved": False, "message": "Auto-save is disabled."}
    if auto_save:
        save_status = save_risk_report(lab_results, ai_summary)

    return {
        "lab_results": lab_results,
        "news": news_list,
        "ai_summary": ai_summary,
        "save_status": save_status
    }
