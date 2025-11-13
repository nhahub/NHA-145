import json
import os
import time
import random
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("🔥 ERROR: GEMINI_API_KEY not found in .env file. AI responses will be disabled.")
    gemini_model = None
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
        print(" Gemini API Key loaded and Model (gemini-2.5-flash-preview-09-2025) initialized.")
    except Exception as e:
        gemini_model = None
        print(f" ERROR initializing Gemini Model: {e}")

KB_PATH = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
_knowledge_base = {}

def load_knowledge_base():
    global _knowledge_base
    try:
        with open(KB_PATH, 'r', encoding='utf-8') as f:
            _knowledge_base = json.load(f)
        print(" Knowledge Base loaded successfully.")
    except Exception as e:
        print(f" ERROR loading Knowledge Base: {e}")
        _knowledge_base = {}

def _call_gemini_api_with_backoff(system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
    global gemini_model
    if not gemini_model:
        return "Error: Gemini model is not initialized. (Check API Key?)"

    retries = 0
    wait_time = 1.0

    combined_prompt = f"""
    {system_prompt}

    ---
    المهمة الآن:
    {user_prompt}
    """

    while retries < max_retries:
        try:
            response = gemini_model.generate_content(
                contents=[{"parts": [{"text": combined_prompt}]}]
            )
            
            if response and response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Invalid response structure from Gemini API.")
                
        except Exception as e:
            print(f" Gemini API Error: {e}. Retrying ({retries}/{max_retries})...")
            
            if "unexpected keyword argument 'system_instruction'" in str(e):
                 print(" FATAL: Still seeing 'system_instruction' error. This shouldn't happen.")
                 return "Error: Code fix failed."

            retries += 1
            if retries == max_retries:
                print(f" Gemini API call failed after {max_retries} retries.")
                return f"Error: Failed to contact AI agent after {max_retries} attempts."
            
            time.sleep(wait_time + random.uniform(0.1, 0.5))
            wait_time *= 2  

    return "Error: AI Agent call failed after maximum retries."

def get_simple_advice(emotion: str, user_id: int) -> str:
    global _knowledge_base
    
    if not _knowledge_base:
        load_knowledge_base()
        if not _knowledge_base:
            return "Error: Knowledge Base is not loaded." 

    if not emotion:
        emotion_key = 'default'
    else:
        emotion_key = emotion.lower()

    simple_advice = _knowledge_base.get(emotion_key, _knowledge_base.get('default', "Please observe your cat's behavior."))
    
    if not gemini_model:
        print("⚠️ Gemini model not available, returning simple advice.")
        return simple_advice

    system_prompt = (
        "أنت 'مرشد القطط' الآلي، مساعد لطيف ومتعاطف في تطبيق Catto-Lingo."
        "مهمتك هي أخذ 'نصيحة أساسية' وتحويلها إلى رد دافئ وودود ومطمئن للمستخدم."
        "لا تكن آليًا. استخدم إيموجيز القطط 😺."
        "خاطب المستخدم كصديق."
        "خاطب المستخدم باللغة العربية بشكل أساسي، ولكن لا تتردد في استخدام اللغة الإنجليزية إذا كان ذلك يجعل الرد أوضح."
        "ابدأ دائمًا برد فعل على الشعور (مثل 'يا، قطتك سعيدة!') قبل تقديم النصيحة."
        "اجعل الرد قصيرًا ولطيفًا (جملتين أو ثلاث)."
    )
    
    user_prompt = f"النصيحة الأساسية هي: '{simple_advice}'"

    ai_response = _call_gemini_api_with_backoff(system_prompt, user_prompt)
    
    return ai_response

