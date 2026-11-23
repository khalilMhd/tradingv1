import os
from dotenv import load_dotenv

load_dotenv()

def get_secret(key_name: str, default: str = "") -> str:
    val = os.getenv(key_name)
    if val:
        return val.strip()
    try:
        import streamlit as st
        if key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    return default

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
ALPACA_API_KEY = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")

# قواعد إدارة المخاطر الصارمة والمتوازنة للأسهم
RISK_RULES = {
    "MAX_RISK_PER_TRADE_PCT": 1.0,      # أقصى خسارة للصفقة (1% من الرصيد)
    "MAX_ALLOWED_DRAWDOWN_PCT": -22.0,  # أقصى هبوط تاريخي مسموح (-22% واقعي للأسهم)
    "MIN_SHARPE_RATIO": 0.75,           # أدنى معامل أمان مقبول (0.75)
    "MAX_DAILY_VOLATILITY_PCT": 4.5,    # أقصى تذبذب يومي مسموح
    "ATR_STOP_LOSS_MULTIPLIER": 1.5,    # وقف الخسارة الديناميكي (1.5x ATR)
    "RISK_REWARD_RATIO": 2.0            # الهدف ضعف وقف الخسارة (1:2)
}
