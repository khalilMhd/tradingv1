import os
from dotenv import load_dotenv

load_dotenv()

# مفاتيح الاتصال
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# قواعد إدارة المخاطر الصارمة (Hard Guardrails - غير قابلة للتفاوض)
RISK_RULES = {
    "MAX_RISK_PER_TRADE_PCT": 1.0,      # أقصى خسارة مسموحة في الصفقة الواحدة (1% من الرصيد)
    "MAX_ALLOWED_DRAWDOWN_PCT": -15.0,  # رفض أي استراتيجية تتجاوز خسارتها التاريخية 15%
    "MIN_SHARPE_RATIO": 1.0,            # رفض أي استراتيجية معامل أمانها أقل من 1.0
    "MAX_DAILY_VOLATILITY_PCT": 4.0,    # منع الدخول إذا كان تذبذب السهم اليومي مفرطاً
    "ATR_STOP_LOSS_MULTIPLIER": 1.5,    # حساب وقف الخسارة ديناميكياً (1.5x ATR)
    "RISK_REWARD_RATIO": 2.0            # الهدف يجب أن يكون ضعف وقف الخسارة على الأقل (1:2)
}
