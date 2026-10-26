import base64
from datetime import date
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from PIL import Image
import requests
import streamlit as st
import sympy as sp
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 頁面配置
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="言昕的高中 AI 學習中樞",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 圖片處理與 SymPy 驗算沙盒
# -----------------------------------------------------------------------------
def optimize_image_base64(img: Image.Image, max_dim: int = 1280) -> tuple[bytes, str]:
    """將圖片等比壓縮並產出 bytes 與 base64 字串"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw_bytes = buf.getvalue()
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    return raw_bytes, b64_str

def execute_sympy_code(code_str: str) -> dict:
    clean_code = re.sub(r"^```python\s*", "", code_str, flags=re.MULTILINE)
    clean_code = re.sub(r"^```\s*$", "", clean_code, flags=re.MULTILINE)
    stdout_capture = io.StringIO()
    exec_globals = {"sp": sp, "sympy": sp, "symbols": sp.symbols, "Eq": sp.Eq, "solve": sp.solve, "sqrt": sp.sqrt}
    exec_locals = {}
    try:
        with redirect_stdout(stdout_capture):
            exec(clean_code, exec_globals, exec_locals)
        return {"success": True, "stdout": stdout_capture.getvalue().strip(), "code": clean_code.strip(), "error": None}
    except Exception as e:
        return {"success": False, "stdout": stdout_capture.getvalue().strip(), "code": clean_code.strip(), "error": str(e)}

# -----------------------------------------------------------------------------
# 跨平台 Multi-LLM 統一呼叫引擎 (Gemini / Grok / OpenAI)
# -----------------------------------------------------------------------------
def call_unified_llm(
    provider: str,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    image_b64: str = None,
    image_bytes: bytes = None,
    response_json: bool = False,
) -> str:
    """統一封裝 Google Gemini、xAI Grok 與 OpenAI 的請求"""
    
    # 1. Google Gemini 模式
    if provider == "Google Gemini":
        client = genai.Client(api_key=api_key)
        contents = []
        if image_bytes:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        contents.append(user_prompt)

        cfg_args = {"temperature": 0.3}
        if system_prompt:
            cfg_args["system_instruction"] = system_prompt
        if response_json:
            cfg_args["response_mime_type"] = "application/json"

        # 優先清單輪詢防 503/404
        candidates = [model_name, "gemini-2.5-flash", "gemini-2.0-flash"]
        last_err = None
        for m in candidates:
            try:
                res = client.models.generate_content(
                    model=m,
                    contents=contents,
                    config=types.GenerateContentConfig(**cfg_args),
                )
                return res.text
            except Exception as e:
                last_err = e
                time.sleep(1)
        raise last_err

    # 2. xAI Grok 模式
    elif provider == "xAI Grok":
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if image_b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model_name or "grok-2-vision-1212",
            "messages": messages,
            "temperature": 0.3,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Grok API 錯誤 ({resp.status_code}): {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    # 3. OpenAI 模式
    elif provider == "OpenAI":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if image_b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model_name or "gpt-4o",
            "messages": messages,
            "temperature": 0.3,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API 錯誤 ({resp.status_code}): {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    else:
        raise ValueError("不支援的模型供應商")

# -----------------------------------------------------------------------------
# 側邊欄設定
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 言昕的高中備考艙")
    active_nav = st.radio(
        "學習專區切換：",
        ["🏠 高中學習中樞 (Landing)", "📐 高中數理蘇格拉底解題", "🧬 高中學測自然探究", "📝 高中英文長難句與文法"],
    )
    st.markdown("---")
    st.subheader("🤖 AI 模型引擎設定")
    
    provider = st.selectbox(
        "選擇 AI 供應商：",
        ["Google Gemini", "xAI Grok", "OpenAI"],
        index=0,
    )

    # 依選定之供應商讀取 Secrets 或預設值
    if provider == "Google Gemini":
        default_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        api_key = st.text_input("Gemini API Key：", value=default_key, type="password")
        model_choice = st.selectbox("模型型號：", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"])
    elif provider == "xAI Grok":
        default_key = st.secrets.get("XAI_API_KEY", os.getenv("XAI_API_KEY", ""))
        api_key = st.text_input("xAI Grok API Key：", value=default_key, type="password")
        model_choice = st.selectbox("模型型號：", ["grok-2-vision-1212", "grok-2-1212"])
    else:
        default_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        api_key = st.text_input("OpenAI API Key：", value=default_key, type="password")
        model_choice = st.selectbox("模型型號：", ["gpt-4o", "gpt-4o-mini"])

    if api_key:
        st.success(f"✅ {provider} 已就緒")

# =============================================================================
# 頁面 1：高中學習中樞 (Landing Page) + 2028 學測倒數計時
# =============================================================================
if active_nav == "🏠 高中學習中樞 (Landing)":
    # 計算 2028 年 1 月第二個週末 (2028/01/08 週六)
    exam_date = date(2028, 1, 8)
    today = date.today()
    days_left = (exam_date - today).days

    head_col1, head_col2 = st.columns([6, 4])
    with head_col1:
        st.title("🎯 言昕的高中學測 / 分科戰情室")
        st.caption("精準聚焦高中升學考科：數A/數B、學測自然、高中英文文法與題型深度引導")

    with head_col2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border: 2px solid #38bdf8; border-radius: 14px; padding: 14px 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <div style="color: #94a3b8; font-size: 13px; font-weight: bold; letter-spacing: 1px;">⏳ 2028 年 1 月學測日倒數 (1/8 週六)</div>
                <div style="font-size: 38px; font-weight: 900; color: #38bdf8; margin: 4px 0;">
                    {days_left} <span style="font-size: 18px; color: #e2e8f0; font-weight: normal;">天</span>
                </div>
                <div style="color: #cbd5e1; font-size: 12px;">穩定每日節奏，紮實攻克每一道難題！</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📚 高中專屬 NotebookLM 知識大腦")
    st.caption("點擊即可一鍵開啟專題筆記本研讀、考點回溯或收聽音訊摘要：")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="border:1px solid #3b82f6; border-radius:12px; padding:18px; background:#18181b; min-height:190px;">
            <h4 style="margin-top:0; color:#60a5fa;">📖 高中文法與寫作精煉</h4>
            <p style="color:#a1a1aa; font-size:13px; line-height:1.5;">高中核心文法體系、倒裝/分詞構句、歷屆學測非選題考點。</p>
            <a href="https://gemini.google.com/notebook/8ed81913-6781-4162-841b-95f5e2ba19b5" target="_blank" style="display:inline-block; margin-top:10px; padding:6px 14px; background:#2563eb; color:white; border-radius:6px; text-decoration:none; font-size:13px; font-weight:bold;">開啟 英文筆記大腦 ↗</a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="border:1px solid #10b981; border-radius:12px; padding:18px; background:#18181b; min-height:190px;">
            <h4 style="margin-top:0; color:#34d399;">📐 高中數學公式與歷屆解法</h4>
            <p style="color:#a1a1aa; font-size:13px; line-height:1.5;">空間向量、圓錐曲線、三角函數與微積分公式推導與錯題彙整。</p>
            <a href="https://notebooklm.google.com" target="_blank" style="display:inline-block; margin-top:10px; padding:6px 14px; background:#059669; color:white; border-radius:6px; text-decoration:none; font-size:13px; font-weight:bold;">開啟 數學專屬筆記 ↗</a>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="border:1px solid #8b5cf6; border-radius:12px; padding:18px; background:#18181b; min-height:190px;">
            <h4 style="margin-top:0; color:#a78bfa;">🧬 學測自然探究與實驗精選</h4>
            <p style="color:#a1a1aa; font-size:13px; line-height:1.5;">高中物化生考綱實驗步驟、控制變因梳理、歷屆跨科圖表整合題。</p>
            <a href="https://notebooklm.google.com" target="_blank" style="display:inline-block; margin-top:10px; padding:6px 14px; background:#7c3aed; color:white; border-radius:6px; text-decoration:none; font-size:13px; font-weight:bold;">開啟 自然探究筆記 ↗</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 高中各科 AI 家教艙")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.subheader("📐 高中數理蘇格拉底解題")
        st.write("• 拍照辨識題目，支援 SymPy 程式驗算")
        st.write("• 一步步提問引導思考，不直接暴雷答案")
        st.caption("👉 左側選單切換至「高中數理蘇格拉底解題」")
    with t2:
        st.subheader("🧬 高中學測自然探究")
        st.write("• 著重大考實驗題：操作/應變/控制變因")
        st.write("• 梳理生物生理反應機制因果鏈")
        st.caption("👉 左側選單切換至「高中學測自然探究」")
    with t3:
        st.subheader("📝 高中英文長難句與文法")
        st.write("• 拆解 4~5 行長難句：S + V + Modifier")
        st.write("• 文法亮點剖析與頂標作文句型改寫")
        st.caption("👉 左側選單切換至「高中英文長難句與文法」")

# =============================================================================
# 頁面 2：高中數理蘇格拉底解題 (支援多模型)
# =============================================================================
elif active_nav == "📐 高中數理蘇格拉底解題":
    st.header(f"📐 高中數理蘇格拉底解題艙 ({provider}: {model_choice})")

    if not api_key:
        st.warning(f"⚠️ 請先在左側填入 **{provider}** 的 API Key！")
        st.stop()

    if "math_messages" not in st.session_state:
        st.session_state.math_messages = []
    if "math_plan" not in st.session_state:
        st.session_state.math_plan = None
    if "math_kit" not in st.session_state:
        st.session_state.math_kit = None
    if "math_image" not in st.session_state:
        st.session_state.math_image = None

    with st.sidebar:
        st.subheader("📸 題目影像輸入")
        input_mode = st.radio("模式：", ["檔案上傳 / 截圖", "即時鏡頭拍照"], key="math_mode")
        img_data = None
        if input_mode == "檔案上傳 / 截圖":
            up = st.file_uploader("上傳題目", type=["png", "jpg", "jpeg"], key="math_up")
            if up:
                img_data = Image.open(up)
        else:
            cam = st.camera_input("拍照", key="math_cam")
            if cam:
                img_data = Image.open(cam)

        if img_data:
            st.session_state.math_image = img_data
            st.image(img_data, caption="已載入題目", use_container_width=True)

        if st.button("🔄 重置數理對話", key="math_reset"):
            st.session_state.math_messages = []
            st.session_state.math_plan = None
            st.session_state.math_kit = None
            st.rerun()

    col_chat, col_tools = st.columns([6, 4])

    with col_tools:
        st.subheader("🛠️ 後台驗算監控與複習套件")
        if st.session_state.math_plan:
            plan = st.session_state.math_plan
            with st.expander("🔍 專家 Python SymPy 驗算紀錄", expanded=False):
                st.markdown(f"**題目識別：** {plan.get('problem_statement')}")
                st.markdown(f"**理論標準解：** `{plan.get('expected_answer')}`")
                st.code(plan.get("python_code", ""), language="python")
                st.info(f"求解輸出：{plan.get('execution_result', {}).get('stdout')}")

            if st.button("✨ 一鍵生成心智圖與觀念記憶卡", use_container_width=True, key="math_kit_btn"):
                with st.spinner("AI 正在提煉考點心智圖與翻轉卡..."):
                    try:
                        kit_prompt = f"""
根據高中數學題目：{plan.get('problem_statement')}：
1. 產出 Mermaid.js 代碼（graph TD 開頭），梳理公式、定理路徑與陷阱。
2. 產出 3 張「考點翻轉學習卡（Flashcards）」。
請直接輸出合法 JSON：
{{"mermaid_code": "graph TD\\n...", "flashcards": [{{"front": "...", "back": "..."}}]}}
"""
                        kit_res = call_unified_llm(
                            provider=provider,
                            api_key=api_key,
                            model_name=model_choice,
                            system_prompt="",
                            user_prompt=kit_prompt,
                            response_json=True,
                        )
                        st.session_state.math_kit = json.loads(kit_res)
                    except Exception as e:
                        st.error(f"生成失敗: {e}")

            if st.session_state.math_kit:
                kit = st.session_state.math_kit
                tab1, tab2 = st.tabs(["🗺️ 核心心智圖", "🗂️ 觀念學習卡"])
                with tab1:
                    st.markdown(f"```mermaid\n{kit.get('mermaid_code', '')}\n```")
                with tab2:
                    for i, card in enumerate(kit.get("flashcards", [])):
                        with st.expander(f"考點 {i+1}：{card.get('front')}"):
                            st.info(f"**解答：** {card.get('back')}")
        else:
            st.caption("👈 上傳高中題目發問後，此處將顯示 Python 後台驗算過程與觀念套件。")

    with col_chat:
        st.subheader("💬 蘇格拉底引導解題")
        for msg in st.session_state.math_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("請輸入你的回答或提問（例如：請引導我 / 這題要怎麼切入？）", key="math_input"):
            if not st.session_state.math_image:
                st.warning("⚠️ 請先在左側上傳或拍照載入題目！")
                st.stop()

            st.session_state.math_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                raw_bytes, b64_str = optimize_image_base64(st.session_state.math_image)
                
                if not st.session_state.math_plan:
                    with st.status(f"🚀 啟動多代理人解題管線 ({provider})...", expanded=True) as status:
                        status.write("📸 辨識高中題目幾何與代數條件...")
                        solver_prompt = f"""
你由兩位高中數學權威專家組成：
1. 【櫃檯專家】：提取題目中的幾何圖形、向量、函數關係與條件數值。使用者指示：{prompt}
2. 【解題專家】：編寫嚴謹的 Python (sympy) 程式碼求解驗算。
請直接輸出合法 JSON：
{{
    "problem_statement": "題目文字描述與條件摘要",
    "python_code": "import sympy as sp\\n...",
    "expected_answer": "最終答案",
    "solution_steps": ["步驟1...", "步驟2...", "步驟3..."]
}}
"""
                        status.write("🧠 編寫 Python 算式並於沙盒驗算...")
                        res_text = call_unified_llm(
                            provider=provider,
                            api_key=api_key,
                            model_name=model_choice,
                            system_prompt="",
                            user_prompt=solver_prompt,
                            image_b64=b64_str,
                            image_bytes=raw_bytes,
                            response_json=True,
                        )
                        # 清理可能夾帶的 markdown 標記
                        clean_json = re.sub(r"^```json\s*", "", res_text.strip(), flags=re.MULTILINE)
                        clean_json = re.sub(r"^```\s*$", "", clean_json.strip(), flags=re.MULTILINE)
                        plan_data = json.loads(clean_json)

                        status.write("⚡ 執行 SymPy 方程求解...")
                        plan_data["execution_result"] = execute_sympy_code(plan_data.get("python_code", ""))
                        st.session_state.math_plan = plan_data
                        status.update(label="✅ 後台題目驗算與步驟拆解完成！", state="complete", expanded=False)

                with st.spinner("🤔 家教正在思考高中教學引導步驟..."):
                    sys_inst = f"""
你是一位親切的高中數學蘇格拉底家教。
【驗算答案】：{st.session_state.math_plan.get('expected_answer')}
【步驟拆解】：{json.dumps(st.session_state.math_plan.get('solution_steps'), ensure_ascii=False)}
【原則】：嚴禁直接給答案；一次只問一個關鍵小問題引導思考；若學生答錯則拆解步驟；數學符號用 LaTeX。
"""
                    conv = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.math_messages])
                    reply = call_unified_llm(
                        provider=provider,
                        api_key=api_key,
                        model_name=model_choice,
                        system_prompt=sys_inst,
                        user_prompt=conv,
                    )

                with st.chat_message("assistant"):
                    st.markdown(reply)
                st.session_state.math_messages.append({"role": "assistant", "content": reply})
                st.rerun()

            except Exception as e:
                st.error(f"⚠️ 發生錯誤：{e}")

# =============================================================================
# 頁面 3：高中學測自然探究
# =============================================================================
elif active_nav == "🧬 高中學測自然探究":
    st.header(f"🧬 高中學測自然探究與生物家教 ({provider}: {model_choice})")
    st.caption("專注於：實驗探究變因、折線/柱狀圖判讀、生理反應步驟鏈條、大考誘答選項破解")

    if not api_key:
        st.warning(f"⚠️ 請先在左側填入 **{provider}** 的 API Key！")
        st.stop()

    if "bio_messages" not in st.session_state:
        st.session_state.bio_messages = []

    for msg in st.session_state.bio_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if bio_prompt := st.chat_input("輸入你想詢問的學測自然題目或觀念（例如：光反應與碳反應的物質轉移 / 動作電位傳導）"):
        st.session_state.bio_messages.append({"role": "user", "content": bio_prompt})
        with st.chat_message("user"):
            st.markdown(bio_prompt)

        with st.spinner("🔬 正在梳理高中課綱機制與因果鏈..."):
            try:
                bio_sys = "你是一位高中學測自然名師：請用清晰步驟鏈 (1 -> 2 -> 3) 說明機制，並標註大考常見陷阱選項。"
                conv = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.bio_messages])
                b_reply = call_unified_llm(
                    provider=provider,
                    api_key=api_key,
                    model_name=model_choice,
                    system_prompt=bio_sys,
                    user_prompt=conv,
                )
                with st.chat_message("assistant"):
                    st.markdown(b_reply)
                st.session_state.bio_messages.append({"role": "assistant", "content": b_reply})
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ 發生錯誤：{e}")

# =============================================================================
# 頁面 4：高中英文長難句與文法
# =============================================================================
elif active_nav == "📝 高中英文長難句與文法":
    st.header(f"📝 高中英文長難句與大考文法家教 ({provider}: {model_choice})")
    st.caption("專門解決高中學測長難句拆解、文法盲點與寫作句型升級")

    if not api_key:
        st.warning(f"⚠️ 請先在左側填入 **{provider}** 的 API Key！")
        st.stop()

    if "eng_messages" not in st.session_state:
        st.session_state.eng_messages = []

    for msg in st.session_state.eng_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if eng_prompt := st.chat_input("貼上一句高中文長難句，或想優化的作文句子："):
        st.session_state.eng_messages.append({"role": "user", "content": eng_prompt})
        with st.chat_message("user"):
            st.markdown(eng_prompt)

        with st.spinner("📖 正在拆解高中句子成分與語法亮點..."):
            try:
                eng_sys = "你是一位高中英文名師：請拆解句子【主詞 S】、【主要動詞 V】、【修飾語 Modifier】，指出核心文法並提供 1~2 個學測頂標句型範例。"
                conv = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.eng_messages])
                e_reply = call_unified_llm(
                    provider=provider,
                    api_key=api_key,
                    model_name=model_choice,
                    system_prompt=eng_sys,
                    user_prompt=conv,
                )
                with st.chat_message("assistant"):
                    st.markdown(e_reply)
                st.session_state.eng_messages.append({"role": "assistant", "content": e_reply})
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ 發生錯誤：{e}")
