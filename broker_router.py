import requests
import streamlit as st

try:
    from broker_data import execute_paper_bracket_order
except ImportError:
    execute_paper_bracket_order = None


def _to_oanda_instrument(symbol: str) -> str:
    clean = symbol.upper().replace("=X", "").replace("=F", "").strip()
    if len(clean) == 6 and clean.isalpha():
        return f"{clean[:3]}_{clean[3:]}"
    return clean


def calculate_safe_forex_units(
    account_balance: float,
    max_risk_pct: float,
    current_price: float,
    stop_loss_price: float
) -> int:
    """
    Calculates position size (units) so worst-case loss at stop_loss_price
    does not exceed max_risk_pct of account_balance.
    """
    max_dollar_risk = account_balance * (max_risk_pct / 100.0)
    risk_per_unit = abs(current_price - stop_loss_price)
    if risk_per_unit <= 1e-9:
        return 0
    return max(1, int(max_dollar_risk // risk_per_unit))


def execute_oanda_practice_order(
    symbol: str,
    units: int,
    stop_loss_price: float,
    take_profit_price: float
) -> dict:
    """
    Submits a market order with stop-loss and take-profit to OANDA fxTrade Practice (v20)
    if OANDA_API_KEY and OANDA_ACCOUNT_ID are configured in Streamlit Secrets.
    """
    api_key = st.secrets.get("OANDA_API_KEY", "")
    account_id = st.secrets.get("OANDA_ACCOUNT_ID", "")

    if not api_key or not account_id:
        return {
            "status": "SKIPPED_NO_CREDENTIALS",
            "message": "Add OANDA_API_KEY and OANDA_ACCOUNT_ID in Streamlit Secrets to send live practice orders to OANDA."
        }

    instrument = _to_oanda_instrument(symbol)
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{account_id}/orders"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    decimals = 5 if "_" in instrument and not instrument.endswith("_JPY") else (3 if instrument.endswith("_JPY") else 2)

    payload = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "stopLossOnFill": {"price": f"{stop_loss_price:.{decimals}f}"},
            "takeProfitOnFill": {"price": f"{take_profit_price:.{decimals}f}"}
        }
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    try:
        body = resp.json()
    except Exception:
        body = resp.text

    return {
        "status_code": resp.status_code,
        "broker": "OANDA_PRACTICE",
        "instrument": instrument,
        "units": int(units),
        "response": body
    }


def route_trade_execution(
    market_type: str,
    ticker: str,
    qty: int,
    stop_loss_price: float,
    take_profit_price: float,
    current_price: float = 0.0,
    account_balance: float = 100000.0,
    max_risk_pct: float = 1.0
) -> dict:
    """
    Routes execution to Alpaca for Equities/ETFs and to OANDA Practice for Forex.
    """
    if market_type == "Forex" or "=X" in ticker.upper():
        safe_units = calculate_safe_forex_units(
            account_balance=account_balance,
            max_risk_pct=max_risk_pct,
            current_price=current_price if current_price > 0 else abs(qty),
            stop_loss_price=stop_loss_price
        )
        signed_units = -abs(safe_units) if qty < 0 else abs(safe_units)
        return execute_oanda_practice_order(
            symbol=ticker,
            units=signed_units,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )

    if execute_paper_bracket_order is None:
        return {
            "status": "ERROR",
            "message": "execute_paper_bracket_order not found in broker_data.py"
        }

    return execute_paper_bracket_order(
        ticker=ticker,
        qty=qty,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price
    )
