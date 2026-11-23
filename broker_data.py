import yfinance as yf
import pandas as pd
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

def get_stock_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """جلب البيانات التاريخية للسهم وتنظيفها."""
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_company_news(ticker: str, limit: int = 5) -> list:
    """جلب أحدث الأخبار المرتبطة بالشركة لتقييم مخاطر الأحداث."""
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news[:limit] if stock.news else []
        cleaned_news = []
        for item in news_items:
            content = item.get("content", item)
            title = content.get("title", "بدون عنوان")
            publisher = content.get("provider", {}).get("displayName", content.get("publisher", "مصدر غير معروف"))
            cleaned_news.append(f"- [{publisher}] {title}")
        return cleaned_news if cleaned_news else ["لا توجد أخبار حديثة متاحة حالياً."]
    except Exception as e:
        return [f"تعذر جلب الأخبار: {str(e)}"]

def get_paper_account_balance() -> float:
    """جلب الرصيد المتاح في حساب Alpaca التجريبي (أو 100,000$ افتراضياً)."""
    if not ALPACA_API_KEY or "your_alpaca" in ALPACA_API_KEY:
        return 100000.0
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        account = client.get_account()
        return float(account.cash)
    except Exception:
        return 100000.0

def execute_paper_bracket_order(ticker: str, qty: int, stop_loss_price: float, take_profit_price: float) -> dict:
    """تنفيذ صفقة تجريبية محمية بوقف خسارة وجني أرباح في موقع Alpaca."""
    if qty <= 0:
        return {"status": "SKIPPED", "reason": "الكمية المحسوبة تساوي 0 بسبب صرامة إدارة المخاطر."}

    if not ALPACA_API_KEY or "your_alpaca" in ALPACA_API_KEY:
        return {
            "status": "SIMULATED_PAPER_ORDER",
            "message": f"تمت محاكاة أمر شراء {qty} سهم من {ticker} (أضف مفاتيح Alpaca لتنفيذه حياً في الموقع).",
            "stop_loss": round(stop_loss_price, 2),
            "take_profit": round(take_profit_price, 2)
        }

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        order_data = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class="bracket",
            stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2))
        )
        order = client.submit_order(order_data=order_data)
        return {
            "status": "EXECUTED_LIVE_PAPER",
            "order_id": str(order.id),
            "symbol": ticker.upper(),
            "qty": qty,
            "stop_loss": round(stop_loss_price, 2),
            "take_profit": round(take_profit_price, 2)
        }
    except Exception as e:
        return {"status": "ERROR", "reason": str(e)}
