from config import SUPABASE_URL, SUPABASE_KEY

def get_supabase_client():
    if not SUPABASE_URL or "your_supabase" in SUPABASE_URL:
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

def save_risk_report(experiment_data: dict, ai_summary: str) -> dict:
    """حفظ التقرير الكامل ونتائج المختبر في جدول risk_reports في Supabase."""
    client = get_supabase_client()
    if not client:
        return {"saved": False, "message": "مفاتيح Supabase غير مفعلة في الإعدادات"}

    winning = experiment_data.get("winning_strategy", "NONE")
    comp = experiment_data.get("strategies_comparison", {})
    win_metrics = comp.get(winning, {})

    # إزالة منحنيات الرسم الضخمة قبل الحفظ في JSONB لتخفيف حجم السجل
    clean_json = {k: v for k, v in experiment_data.items() if k != "equity_curves"}

    record = {
        "ticker": experiment_data.get("ticker"),
        "winning_strategy": winning,
        "sharpe_ratio": win_metrics.get("sharpe_ratio", 0.0),
        "max_drawdown": win_metrics.get("max_drawdown_pct", 0.0),
        "risk_status": experiment_data.get("risk_status", "REJECTED"),
        "stop_loss_pct": experiment_data.get("position_sizing", {}).get("stop_loss_pct", 0.0),
        "full_report_json": clean_json,
        "ai_summary": ai_summary
    }

    try:
        res = client.table("risk_reports").insert(record).execute()
        return {"saved": True, "message": "تم حفظ التقرير بنجاح في Supabase!", "data": res.data}
    except Exception as e:
        return {"saved": False, "message": f"خطأ أثناء الحفظ في Supabase: {str(e)}"}

def fetch_recent_reports(limit: int = 15) -> list:
    """جلب أحدث التقارير المحفوظة لعرضها في الواجهة."""
    client = get_supabase_client()
    if not client:
        return []
    try:
        res = client.table("risk_reports").select("*").order("created_at", desc=True).limit(limit).execute()
        return res.data if res.data else []
    except Exception:
        return []
