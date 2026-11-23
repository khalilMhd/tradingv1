import numpy as np
import pandas as pd
from datetime import datetime
from config import RISK_RULES
from broker_data import get_stock_history, get_paper_account_balance


def run_strategy_experiment(
    ticker: str,
    period: str = "2y",
    desired_capital: float = 10000.0,
    market_type: str = "Stocks & ETFs"
) -> dict:
    """
    Tests 8 distinct quantitative trading tools/strategies across Stocks/ETFs, Gold, Oil, or Forex,
    ranks them by stability-aware risk-adjusted performance, computes execution confidence (%),
    forecasts profits, evaluates desired capital against the 1% account risk guardrail,
    and determines the latest actionable trade direction (BUY / SELL / HOLD).
    """
    df = get_stock_history(ticker, period=period)
    if df.empty or len(df) < 60:
        return {"error": f"Insufficient historical data for {ticker} ({market_type})"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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
    best_metrics = {}

    for name, sig_col in strategies.items():
        shifted_sig = df[sig_col].shift(1).fillna(0)
        strat_ret = (shifted_sig * df["Returns"]).fillna(0)
        cum_ret = (1 + strat_ret).cumprod()
        equity_curves[name] = [round(float(x), 4) for x in cum_ret.tolist()]

        total_ret = (cum_ret.iloc[-1] - 1) * 100
        ann_vol = strat_ret.std() * np.sqrt(252) * 100
        sharpe = (strat_ret.mean() * 252) / (strat_ret.std() * np.sqrt(252) + 1e-9)

        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        active_days = strat_ret[shifted_sig > 0]
        win_rate_pct = ((active_days > 0).sum() / len(active_days) * 100) if len(active_days) > 0 else 50.0

        # Stability-aware Risk Score: rewards Sharpe & Win Rate, penalizes Drawdown & Volatility
        stability_penalty = (abs(max_dd) / 100.0) + (ann_vol / 250.0)
        win_bonus = max(0.0, (win_rate_pct - 50.0) / 100.0) * 0.25
        risk_score = round(float(sharpe - stability_penalty + win_bonus), 3)

        passed_rules = (
            max_dd >= RISK_RULES["MAX_ALLOWED_DRAWDOWN_PCT"] and
            sharpe >= RISK_RULES["MIN_SHARPE_RATIO"]
        )

        comparison[name] = {
            "total_return_pct": round(float(total_ret), 2),
            "annual_volatility_pct": round(float(ann_vol), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown_pct": round(float(max_dd), 2),
            "win_rate_pct": round(float(win_rate_pct), 1),
            "risk_score": risk_score,
            "passed_strict_guardrails": bool(passed_rules)
        }

        if passed_rules and risk_score > best_score:
            best_score = risk_score
            best_strategy = name
            best_metrics = comparison[name]

    comparison = dict(sorted(comparison.items(), key=lambda x: x[1]["risk_score"], reverse=True))

    # Price, Volatility, ATR Stop-Loss & Take-Profit
    current_price = float(df["Close"].iloc[-1])
    current_atr = float(df["ATR"].iloc[-1])
    current_atr_pct = float(df["ATR_Pct"].iloc[-1])

    stop_loss_distance = current_atr * RISK_RULES["ATR_STOP_LOSS_MULTIPLIER"]
    stop_loss_price = current_price - stop_loss_distance
    take_profit_price = current_price + (stop_loss_distance * RISK_RULES["RISK_REWARD_RATIO"])
    stop_loss_pct = (stop_loss_distance / current_price) * 100
    take_profit_pct = ((take_profit_price - current_price) / current_price) * 100

    # 1% Account Risk Budget & Optimal Safe Capital
    account_balance = get_paper_account_balance()
    max_dollar_risk = account_balance * (RISK_RULES["MAX_RISK_PER_TRADE_PCT"] / 100.0)
    recommended_shares = int(max_dollar_risk // stop_loss_distance) if stop_loss_distance > 0 else 0

    sl_fraction = max(stop_loss_pct / 100.0, 1e-6)
    optimal_safe_capital_usd = round(min(account_balance, max_dollar_risk / sl_fraction), 2)

    desired_capital_clean = round(float(desired_capital), 2)
    desired_shares_raw = int(desired_capital_clean // current_price) if current_price > 0 else 0
    loss_per_share = max(0.0, current_price - stop_loss_price)
    profit_per_share = max(0.0, take_profit_price - current_price)

    desired_potential_loss_usd = round(desired_shares_raw * loss_per_share, 2)
    desired_potential_profit_usd = round(desired_shares_raw * profit_per_share, 2)
    exceeds_1pct_risk = desired_potential_loss_usd > (max_dollar_risk + 0.01)

    vol_ok = current_atr_pct <= max_vol
    risk_status = "APPROVED" if (best_strategy is not None and vol_ok) else "REJECTED"

    # Execution Confidence Score (0% to 98.5%)
    top_metric = best_metrics if best_metrics else list(comparison.values())[0]
    sharpe_component = np.clip((top_metric["sharpe_ratio"] / 2.0) * 40.0, 0.0, 40.0)
    dd_component = np.clip((1.0 - abs(top_metric["max_drawdown_pct"]) / 40.0) * 25.0, 0.0, 25.0)
    win_component = np.clip((top_metric["win_rate_pct"] / 100.0) * 20.0, 0.0, 20.0)
    vol_component = 15.0 if vol_ok else max(0.0, 15.0 - (current_atr_pct - max_vol) * 5.0)

    raw_confidence = sharpe_component + dd_component + win_component + vol_component
    if risk_status == "REJECTED":
        raw_confidence = min(raw_confidence, 45.0)
    execution_confidence_pct = round(float(np.clip(raw_confidence, 5.0, 98.5)), 1)

    # Latest actionable trade direction (BUY / SELL / HOLD)
    signal_map = {
        "SMA_Trend_20_50": "Sig_SMA",
        "EMA_Fast_9_21": "Sig_EMA",
        "MACD_Momentum": "Sig_MACD",
        "RSI_Mean_Reversion": "Sig_RSI",
        "Bollinger_Bands": "Sig_BB",
        "Donchian_Breakout": "Sig_Donchian",
        "ADX_Trend_Filter": "Sig_ADX",
        "Volatility_Guarded_Hybrid": "Sig_Hybrid"
    }

    rsi_latest = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50.0
    if best_strategy and best_strategy in signal_map and vol_ok:
        latest_sig = int(df[signal_map[best_strategy]].iloc[-1])
        if latest_sig == 1:
            trade_direction = "BUY"
            direction_reason = f"{best_strategy} is currently bullish/active (signal = 1)."
        elif best_strategy == "RSI_Mean_Reversion" and rsi_latest > 70:
            trade_direction = "SELL"
            direction_reason = f"RSI is overbought ({rsi_latest:.1f} > 70), favoring a short/sell setup."
        else:
            trade_direction = "HOLD"
            direction_reason = f"{best_strategy} is currently flat/not signaling a fresh entry (signal = 0)."
    else:
        trade_direction = "HOLD"
        direction_reason = "Guardrails or volatility filter did not approve an active entry."

    # Expected Profit & Loss Forecasts based on Risk-Guarded Recommended Shares
    expected_gross_profit_usd = round(profit_per_share * recommended_shares, 2)
    expected_max_loss_usd = round(loss_per_share * recommended_shares, 2)
    expected_position_value_usd = round(current_price * recommended_shares, 2)

    p_win = np.clip(top_metric["win_rate_pct"] / 100.0, 0.15, 0.85)
    p_loss = 1.0 - p_win
    expected_value_usd = round((p_win * expected_gross_profit_usd) - (p_loss * expected_max_loss_usd), 2)

    years_divisor = 2.0 if period == "2y" else (5.0 if period == "5y" else 1.0)
    expected_annual_return_pct = round(float(top_metric["total_return_pct"] / years_divisor), 2)
    expected_annual_profit_usd = round(float(expected_position_value_usd * (expected_annual_return_pct / 100.0)), 2)

    price_decimals = 4 if "=X" in ticker.upper() or market_type == "Forex" else 2

    return {
        "ticker": ticker.upper(),
        "market_type": market_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": round(current_price, price_decimals),
        "current_daily_volatility_pct": round(current_atr_pct, 2),
        "risk_status": risk_status,
        "trade_direction": trade_direction,
        "direction_reason": direction_reason,
        "execution_confidence_pct": execution_confidence_pct,
        "winning_strategy": best_strategy if best_strategy else "NONE (All 8 tools failed strict guardrails)",
        "capital_guidance": {
            "desired_capital_usd": desired_capital_clean,
            "optimal_safe_capital_usd": optimal_safe_capital_usd,
            "desired_shares": desired_shares_raw,
            "desired_potential_loss_usd": desired_potential_loss_usd,
            "desired_potential_profit_usd": desired_potential_profit_usd,
            "exceeds_1pct_risk": bool(exceeds_1pct_risk),
            "safe_cap_shares": recommended_shares
        },
        "profit_forecast": {
            "expected_position_value_usd": expected_position_value_usd,
            "expected_gross_profit_usd": expected_gross_profit_usd,
            "expected_profit_pct": round(take_profit_pct, 2),
            "expected_max_loss_usd": expected_max_loss_usd,
            "expected_value_usd": expected_value_usd,
            "expected_annual_return_pct": expected_annual_return_pct,
            "expected_annual_profit_usd": expected_annual_profit_usd,
            "historical_win_rate_pct": top_metric["win_rate_pct"]
        },
        "position_sizing": {
            "account_balance": round(account_balance, 2),
            "max_risk_dollars": round(max_dollar_risk, 2),
            "recommended_shares": recommended_shares,
            "stop_loss_price": round(stop_loss_price, price_decimals),
            "stop_loss_pct": round(stop_loss_pct, 2),
            "take_profit_price": round(take_profit_price, price_decimals)
        },
        "strategies_comparison": comparison,
        "equity_curves": equity_curves
    }
