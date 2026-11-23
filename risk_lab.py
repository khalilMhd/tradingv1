import numpy as np
import pandas as pd
from datetime import datetime
from config import RISK_RULES
from broker_data import get_stock_history, get_paper_account_balance

def run_strategy_experiment(ticker: str, period: str = "2y") -> dict:
    """
    Tests 8 distinct quantitative trading tools/strategies on the target stock,
    ranks them by risk-adjusted performance, and enforces strict capital guardrails.
    """
    df = get_stock_history(ticker, period=period)
    if df.empty or len(df) < 60:
        return {"error": f"Insufficient historical data for {ticker}"}

    df["Returns"] = df["Close"].pct_change()

    # True Range & ATR (Volatility)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_Pct"] = (df["ATR"] / df["Close"]) * 100

    # --- Tool 1: SMA Trend (20/50) ---
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Sig_SMA"] = np.where(df["SMA20"] > df["SMA50"], 1, 0)

    # --- Tool 2: EMA Fast Crossover (9/21) ---
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["Sig_EMA"] = np.where(df["EMA9"] > df["EMA21"], 1, 0)

    # --- Tool 3: MACD Momentum (12, 26, 9) ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    df["Sig_MACD"] = np.where(macd > signal_line, 1, 0)

    # --- Tool 4: RSI Mean Reversion (14) ---
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Sig_RSI"] = np.where(df["RSI"] < 35, 1, np.where(df["RSI"] > 70, 0, np.nan))
    df["Sig_RSI"] = df["Sig_RSI"].ffill().fillna(0)

    # --- Tool 5: Bollinger Bands (20, 2) ---
    std20 = df["Close"].rolling(20).std()
    df["BB_Lower"] = df["SMA20"] - (2 * std20)
    df["BB_Upper"] = df["SMA20"] + (2 * std20)
    df["Sig_BB"] = np.where(df["Close"] < df["BB_Lower"], 1, np.where(df["Close"] > df["BB_Upper"], 0, np.nan))
    df["Sig_BB"] = df["Sig_BB"].ffill().fillna(0)

    # --- Tool 6: Donchian Channel Breakout (20-day High / 10-day Low) ---
    high20 = df["High"].rolling(20).max().shift(1)
    low10 = df["Low"].rolling(10).min().shift(1)
    df["Sig_Donchian"] = np.where(df["Close"] > high20, 1, np.where(df["Close"] < low10, 0, np.nan))
    df["Sig_Donchian"] = df["Sig_Donchian"].ffill().fillna(0)

    # --- Tool 7: ADX Trend Strength Filter (Avoids Choppy Markets) ---
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / (df["ATR"] + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (df["ATR"] + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    df["ADX"] = dx.rolling(14).mean()
    df["Sig_ADX"] = np.where((df["ADX"] > 25) & (plus_di > minus_di), 1, 0)

    # --- Tool 8: Volatility-Guarded Hybrid (EMA + RSI Cap + ATR Shield) ---
    max_vol = RISK_RULES["MAX_DAILY_VOLATILITY_PCT"]
    df["Sig_Hybrid"] = np.where(
        (df["EMA9"] > df["EMA21"]) & (df["RSI"] < 68) & (df["ATR_Pct"] < max_vol),
        1, 0
    )

    strategies = {
        "SMA_Trend_20_50": "Sig_SMA",
        "EMA_Fast_9_21": "Sig_EMA",
        "MACD_Momentum": "Sig_MACD",
        "RSI_Mean_Reversion": "Sig_RSI",
        "Bollinger_Bands": "Sig_BB",
        "Donchian_Breakout": "Sig_Donchian",
        "ADX_Trend_Filter": "Sig_ADX",
        "Volatility_Guarded_Hybrid": "Sig_Hybrid"
    }

    comparison = {}
    equity_curves = {"Dates": df.index.strftime("%Y-%m-%d").tolist()}
    best_strategy = None
    best_score = -999.0

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

        # Composite Risk-Adjusted Score (rewards high Sharpe and penalizes deep Drawdowns)
        risk_score = round(float(sharpe - (abs(max_dd) / 100.0)), 3)

        passed_rules = (
            max_dd >= RISK_RULES["MAX_ALLOWED_DRAWDOWN_PCT"] and
            sharpe >= RISK_RULES["MIN_SHARPE_RATIO"]
        )

        comparison[name] = {
            "total_return_pct": round(float(total_ret), 2),
            "annual_volatility_pct": round(float(ann_vol), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown_pct": round(float(max_dd), 2),
            "risk_score": risk_score,
            "passed_strict_guardrails": bool(passed_rules)
        }

        if passed_rules and risk_score > best_score:
            best_score = risk_score
            best_strategy = name

    # Sort strategies from highest Risk Score to lowest
    comparison = dict(sorted(comparison.items(), key=lambda x: x[1]["risk_score"], reverse=True))

    # Position sizing and mandatory ATR stop-loss
    current_price = float(df["Close"].iloc[-1])
    current_atr = float(df["ATR"].iloc[-1])
    current_atr_pct = float(df["ATR_Pct"].iloc[-1])

    stop_loss_distance = current_atr * RISK_RULES["ATR_STOP_LOSS_MULTIPLIER"]
    stop_loss_price = current_price - stop_loss_distance
    take_profit_price = current_price + (stop_loss_distance * RISK_RULES["RISK_REWARD_RATIO"])
    stop_loss_pct = (stop_loss_distance / current_price) * 100

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
        "winning_strategy": best_strategy if best_strategy else "NONE (All 8 tools failed strict guardrails)",
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
