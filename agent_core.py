import json
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, RISK_RULES
from broker_data import get_company_news
from risk_lab import run_strategy_experiment
from database import save_risk_report

def run_autonomous_risk_agent(
    ticker: str,
    period: str = "2y",
    auto_save: bool = True,
    desired_capital: float = 10000.0,
    market_type: str = "Stocks & ETFs"
) -> dict:
    """
    Runs the full Risk Guard Agent across Stocks/ETFs, Gold, Oil, or Forex:
    1. Executes quantitative strategy backtests + execution probability + profit forecasts + capital guidance.
    2. Fetches live news/headlines for the asset.
    3. Passes metrics, capital guidance, and news to Gemini to issue a strict risk decision.
    4. Saves the report to Supabase.
    """
    lab_results = run_strategy_experiment(
        ticker,
        period=period,
        desired_capital=desired_capital,
        market_type=market_type
    )
    if "error" in lab_results:
        return {"error": lab_results["error"]}

    news_list = get_company_news(ticker, limit=5)

    metrics_for_ai = {k: v for k, v in lab_results.items() if k != "equity_curves"}
    metrics_for_ai["recent_news"] = news_list
    metrics_for_ai["strict_guardrails"] = RISK_RULES

    if not GEMINI_API_KEY or "your_gemini" in GEMINI_API_KEY:
        ai_summary = "⚠️ GEMINI_API_KEY is not configured in Secrets. Relying strictly on quantitative guardrails."
    else:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            system_instruction = """
            You are an ultra-strict Chief Risk Officer (CRO) and Quantitative Analyst covering Equities, Commodities (Gold & Oil), and Forex.
            Your mission is to evaluate strategy stability, backtest metrics, execution confidence %, expected profit projections, desired vs. optimal safe capital under the 1% portfolio risk rule, and live news/macro catalysts.
            Strict Rules:
            1. Explain why the winning strategy was selected for its stability and risk-adjusted metrics, or why the asset was completely REJECTED based on Sharpe Ratio, Max Drawdown, Volatility, and Execution Confidence %.
            2. Clearly evaluate the user's Desired Capital against the Optimal Safe Capital (1% max account loss rule): state whether the desired amount is safe or exposes the account to >1% loss, and recommend the exact safe capital & unit/share count.
            3. Highlight Expected Profit Projection (Take-Profit profit in USD & %, Expected Value EV, and annualized return projection) alongside capped maximum loss.
            4. Explicitly compare the top strategy's stability (low drawdown, controlled annual volatility, and win rate) against runner-up strategies, accounting for the specific behavior of the chosen market (e.g., Gold safe-haven/rates, Oil supply/geopolitics, Forex central bank/carry dynamics, or Stocks/ETFs).
            5. Analyze the attached headlines for any macro or asset-specific risks.
            6. Write the report in clear, direct, executive-level English using structured bullet points.
            """
            prompt = f"Analyze the following multi-market experiment results and issue the final Risk Officer Report:\n{json.dumps(metrics_for_ai, indent=2)}"

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

    save_status = {"saved": False, "message": "Auto-save is disabled."}
    if auto_save:
        save_status = save_risk_report(lab_results, ai_summary)

    return {
        "lab_results": lab_results,
        "news": news_list,
        "ai_summary": ai_summary,
        "save_status": save_status
    }
