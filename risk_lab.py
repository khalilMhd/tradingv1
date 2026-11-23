import numpy as np
import pandas as pd
from datetime import datetime
from config import RISK_RULES
from broker_data import get_stock_history, get_paper_account_balance

def run_strategy_experiment(ticker: str, period: str = "2y") -> dict:
    """
    يختبر 4 استراتيجيات/أدوات مختلفة على السهم، ويطبق قواعد المخاطر الصارمة لاختيار الأفضل.
    """
    df = get_stock_history(ticker, period=period)
    if df.empty or len(df) < 60:
        return {"error": f"بيانات غير كافية للسهم {ticker}"}

    df["Returns"] = df["Close"].pct_change()

    # حساب مؤشر التقلب الحقيقي (ATR)
    high_low = df["High"] - df["Low"]
    df["ATR"] = high_low.rolling(14).mean()
    df["ATR_Pct"] = (df["ATR"] / df["Close"]) * 100

    # --- 1. تقاطع المتوسطات (SMA 20/50 Trend) ---
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Sig_SMA"] = np.where(df["SMA20"] > df["SMA50"], 1, 0)

    # --- 2. مؤشر القوة النسبية (RSI 14 Mean Reversion) ---
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Sig_RSI"] = np.where(df["RSI"] < 35, 1, np.where(df["RSI"] > 70, 0, np.nan))
    df["Sig_RSI"] = df["Sig_RSI"].ffill().fillna(0)

    # --- 3. قنوات بولينجر (Bollinger Bands Breakout/Reversion) ---
    std20 = df["Close"].rolling(20).std()
    df["BB_Lower"] = df["SMA20"] - (2 * std20)
    df["BB_Upper"] = df["SMA20"] + (2 * std20)
    df["Sig_BB"] = np.where(df["Close"] < df["BB_Lower"], 1, np.where(df["Close"] > df["BB_Upper"], 0, np.nan))
    df["Sig_BB"] = df["Sig_BB"].ffill().fillna(0)

    # --- 4. الاستراتيجية الهجينة المفلترة بالتقلب (Volatility-Guarded Trend) ---
    max_vol = RISK_RULES["MAX_DAILY_VOLATILITY_PCT"]
    df["Sig_Guarded"] = np.where((df["SMA20"] > df["SMA50"]) & (df["ATR_Pct"] < max_vol), 1, 0)

    strategies = {
        "SMA_Trend_20_50": "Sig_SMA",
        "RSI_Mean_Reversion": "Sig_RSI",
        "Bollinger_Bands": "Sig_BB",
        "Volatility_Guarded_Trend": "Sig_Guarded"
    }

    comparison = {}
    equity_curves = {"Dates": df.index.strftime("%Y-%m-%d").tolist()}
    best_strategy = None
    best_sharpe = -999.0

    for name, sig_col in strategies.items():
        strat_ret = (df[sig_col].shift(1) * df["Returns"]).fillna(0)
        cum_ret = (1 + strat_ret).cumprod()
        equity_curves[name] = [round(float(x), 4) for x in cum_ret.tolist()]

        total_ret = (cum_ret.iloc[-1] - 1) * 100
        ann_vol = strat_ret.std() * np.sqrt(252) * 100
        sharpe = (strat_ret.mean() * 252) / (strat_ret.std() * np.sqrt(252) + 1e-9)

        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        # التحقق من اجتياز قواعد المخاطر الصارمة
        passed_rules = (
            max_dd >= RISK_RULES["MAX_ALLOWED_DRAWDOWN_PCT"] and
            sharpe >= RISK_RULES["MIN_SHARPE_RATIO"]
        )

        comparison[name] = {
            "total_return_pct": round(float(total_ret), 2),
            "annual_volatility_pct": round(float(ann_vol), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown_pct": round(float(max_dd), 2),
            "passed_strict_guardrails": bool(passed_rules)
        }

        if passed_rules and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_strategy = name

    # حساب إعدادات الصفقة المحمية بناءً على السعر والتقلب الحالي
    current_price = float(df["Close"].iloc[-1])
    current_atr = float(df["ATR"].iloc[-1])
    current_atr_pct = float(df["ATR_Pct"].iloc[-1])

    stop_loss_distance = current_atr * RISK_RULES["ATR_STOP_LOSS_MULTIPLIER"]
    stop_loss_price = current_price - stop_loss_distance
    take_profit_price = current_price + (stop_loss_distance * RISK_RULES["RISK_REWARD_RATIO"])
    stop_loss_pct = (stop_loss_distance / current_price) * 100

    # حساب حجم المركز بحيث لا تتجاوز الخسارة 1% من الرصيد
    account_balance = get_paper_account_balance()
    max_dollar_risk = account_balance * (RISK_RULES["MAX_RISK_PER_TRADE_PCT"] / 100.0)
    recommended_shares = int(max_dollar_risk // stop_loss_distance) if stop_loss_distance > 0 else 0

    risk_status = "APPROVED" if (best_strategy is not None and current_atr_pct <= max_vol) else "REJECTED"

    return {
        "ticker": ticker.upper(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": round(current_price, 2),
        "current_daily_volatility_pct": round(current_atr_pct, 2),
        "risk_status": risk_status,
        "winning_strategy": best_strategy if best_strategy else "NONE (فشلت كل الأدوات في شروط الأمان)",
        "position_sizing": {
            "account_balance": round(account_balance, 2),
            "max_risk_dollars": round(max_dollar_risk, 2),
            "recommended_shares": recommended_shares,
            "stop_loss_price": round(stop_loss_price, 2),
            "stop_loss_pct": round(stop_loss_pct, 2),
            "take_profit_price": round(take_profit_price, 2)
        },
        "strategies_comparison": comparison,
        "equity_curves": equity_curves
    }
