import json
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, RISK_RULES
from broker_data import get_company_news
from risk_lab import run_strategy_experiment
from database import save_risk_report

def run_autonomous_risk_agent(ticker: str, period: str = "2y", auto_save: bool = True) -> dict:
    """
    يشغل الـ Risk Agent الكامل:
    1. يجري التجارب الرقمية على الاستراتيجيات.
    2. يجمع أخبار الشركة الحية.
    3. يمرر المعطيات لعقل Gemini لتقييم المخاطر النوعية والكمية وإصدار القرار النهائي.
    4. يحفظ التقرير في Supabase.
    """
    # الخطوة 1: تشغيل المختبر الرقمي
    lab_results = run_strategy_experiment(ticker, period=period)
    if "error" in lab_results:
        return {"error": lab_results["error"]}

    # الخطوة 2: جلب أخبار الشركة الحية
    news_list = get_company_news(ticker, limit=5)

    # تجهيز ملخص البيانات للـ LLM (بدون مصفوفة المنحنيات الضخمة)
    metrics_for_ai = {k: v for k, v in lab_results.items() if k != "equity_curves"}
    metrics_for_ai["recent_company_news"] = news_list
    metrics_for_ai["strict_guardrails"] = RISK_RULES

    if not GEMINI_API_KEY or "your_gemini" in GEMINI_API_KEY:
        ai_summary = "⚠️ مفتاح GEMINI_API_KEY غير مضاف في ملف .env. تم الاعتماد على التحليل الرقمي الصارم فقط."
    else:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            system_instruction = """
            أنت مدير مخاطر مالي صارم جداً (Chief Risk Officer) ومحلل كمي.
            مهمتك تقييم نتائج مختبر الاستراتيجيات وأخبار الشركة الحية لحماية رأس المال.
            القواعد الصارمة:
            1. اشرح لماذا فازت الأداة المختارة أو لماذا تم رفض السهم بالكامل بناءً على Sharpe Ratio و Max Drawdown.
            2. حلل الأخبار المرفقة: هل هناك أي خبر سلبي خطير (تراجع أرباح، تحقيقات، أزمة قطاعية) يستدعي الحذر حتى لو كانت الأرقام جيدة؟
            3. أكد على خطة الحماية: السعر الحالي، نقطة وقف الخسارة الإجباري (Stop-Loss)، والكمية المسموحة (قاعدة 1%).
            4. اكتب التقرير باللغة العربية بأسلوب مهني مباشر ومنظم في نقاط واضحة.
            """
            prompt = f"حلل نتائج التجربة التالية وأصدر تقرير المخاطر النهائي:\n{json.dumps(metrics_for_ai, ensure_ascii=False, indent=2)}"
            
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
            ai_summary = f"حدث خطأ أثناء الاتصال بنموذج Gemini: {str(e)}"

    # الخطوة 3: حفظ التقرير في Supabase
    save_status = {"saved": False, "message": "لم يتم تفعيل الحفظ التلقائي."}
    if auto_save:
        save_status = save_risk_report(lab_results, ai_summary)

    return {
        "lab_results": lab_results,
        "news": news_list,
        "ai_summary": ai_summary,
        "save_status": save_status
    }
