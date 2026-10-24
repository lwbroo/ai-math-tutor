import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from PIL import Image
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
# 工具函式
# -----------------------------------------------------------------------------
def optimize_image(img: Image.Image, max_dim: int = 1280) -> bytes:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

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

FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-2.5-flash"]

def generate_content_with_retry(client: genai.Client, primary_model: str, contents, config=None, max_retries=3):
    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
    last_err = None
    for m in models_to_try:
        for attempt in range(max_retries):
            try:
                return client.models.generate_content(model=m, contents=contents, config=config)
            except Exception as e:
                last_err = e
                err_msg = str(e)
                if "503" in err_msg or "UNAVAILABLE" in err_msg or "high demand" in err_msg.lower():
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
    raise last_err

# -----------------------------------------------------------------------------
# 系統設定
# -----------------------------------------------------------------------------
env_key = ""
try:
    env_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
except Exception:
    env_key = os.getenv("GEMINI_API_KEY", "")

with st.sidebar:
    st.title("🎓 言昕的高中備考艙")
    active_nav = st.radio(
        "學習專區切換：",
        ["🏠 高中學習中樞 (Landing)", "📐 高中數理蘇格拉底解題", "🧬 高中學測自然探究", "📝 高中英文長難句與文法"],
    )
    st.markdown("---")
    api_key = st.text_input("Gemini API Key", value=env_key, type="password")
    if api_key:
        st.caption("✅ 引擎就緒 (Gemini 3.8 Flash)")
    st.markdown("---")
    st.info("💡 提示：在 iPad Safari 點擊「分享 ⎋」➜「加入主畫面」，即可獲得全螢幕 App 體驗。")

if not api_key:
    st.warning("⚠️ 請先在左側確認或填入 Gemini API Key！")
    st.stop()

try:
    client = genai.Client(api_key=api_key)
    active_model = "gemini-3.8-flash"
except Exception as e:
    st.error(f"Client 初始化失敗: {e}")
    st.stop()

# =============================================================================
# 頁面 1：高中學習中樞 (Landing Page)
# =============================================================================
if active_nav == "🏠 高中學習中樞 (Landing)":
    st.title("🎯 言昕的高中學測 / 分科戰情室")
    st.caption("精準聚焦高中升學考科：數A/數B、學測自然、高中英文文法與題型深度引導")

    st.markdown("### 📚 高中專屬 NotebookLM 知識大腦")
    st.caption("專屬大考筆記庫，點擊即可開啟專題研讀、來源回溯或生成音訊摘要：")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="border:1px solid #3b82f6; border-radius:12px; padding:18px; background:#18181b; min-height:190px;">
            <h4 style="margin-top:0; color:#60a5fa;">📖 高中文法與寫作精煉</h4>
            <p style="color:#a1a1aa; font-size:13px; line-height:1.5;">高中核心文法體系、常考倒裝/分詞構句、歷屆學測非選題考點。</p>
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
        st.write("• 拍下講義/模考幾何與代數難題")
        st.write("• 後台 Python SymPy 方程求解防幻覺")
        st.write("• 一次拋出一問，引導思考輔助線與定理")
        st.caption("👉 左側選單切換至「高中數理蘇格拉底解題」")

    with t2:
        st.subheader("🧬 高中學測自然探究")
        st.write("• 著重大考實驗題：操作/應變/控制變因")
        st.write("• 跨章節機制因果步驟鏈條梳理")
        st.write("• 精確破除學測選項常見誘答陷阱")
        st.caption("👉 左側選單切換至「高中學測自然探究」")

    with t3:
        st.subheader("📝 高中英文長難句與文法")
        st.write("• 3 秒拆解高中學測 4~5 行長難句成分")
        st.write("• 標註主詞(S)、主要動詞(V)與分詞修飾")
        st.write("• 作文句子升級為學測頂標句型")
        st.caption("👉 左側選單切換至「高中英文長難句與文法」")

# =============================================================================
# 頁面 2：高中數理蘇格拉底解題
# =============================================================================
elif active_nav == "📐 高中數理蘇格拉底解題":
    st.header("📐 高中數理蘇格拉底引導解題艙")
    st.caption("適用高一~高三數學 (數A/數B)、物理，堅持「思考優先於公式」")

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
        st.subheader("🛠️ 後台運算監控與複習套件")
        if st.session_state.math_plan:
            plan = st.session_state.math_plan
            with st.expander("🔍 專家 Python SymPy 驗算紀錄", expanded=False):
                st.markdown(f"**題目識別：** {plan.get('problem_statement')}")
                st.markdown(f"**理論標準解：** `{plan.get('expected_answer')}`")
                st.code(plan.get("python_code", ""), language="python")
                st.info(f"求解輸出：{plan.get('execution_result', {}).get('stdout')}")

            if st.button("✨ 一鍵生成心智圖與觀念記憶卡", use_container_width=True, key="math_kit_btn"):
                with st.spinner("AI 正在提煉高中考點心智圖與翻轉學習卡..."):
                    try:
                        kit_prompt = f"""
根據高中數學題目：{plan.get('problem_statement')}：
1. 產出 Mermaid.js 代碼（graph TD 開頭），梳理此題考查的高中核心公式、解題定理路徑與易錯陷阱。
2. 產出 3 張「考點翻轉學習卡（Flashcards）」，每張包含正面問題（Front）與背面解答（Back）。
請直接輸出合法 JSON：
{{"mermaid_code": "graph TD\\n...", "flashcards": [{{"front": "...", "back": "..."}}]}}
"""
                        res = generate_content_with_retry(
                            client, active_model, [kit_prompt],
                            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3)
                        )
                        st.session_state.math_kit = json.loads(res.text)
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
                            st.info(f"**觀念核心：** {card.get('back')}")
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
                if not st.session_state.math_plan:
                    with st.status("🚀 啟動多代理人解題管線...", expanded=True) as status:
                        status.write("📸 壓縮與辨識高中題目圖形...")
                        img_bytes = optimize_image(st.session_state.math_image)
                        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

                        solver_prompt = f"""
你由兩位高中數學權威專家組成：
1. 【櫃檯專家】：提取題目中的幾何圖形、向量、函數關係與條件數值。
2. 【解題專家】：編寫嚴謹的 Python (sympy) 程式碼求解本題。
請輸出合法 JSON：
{{
    "problem_statement": "題目文字描述與條件摘要",
    "python_code": "import sympy as sp...",
    "expected_answer": "最終答案",
    "solution_steps": ["步驟1...", "步驟2...", "步驟3..."]
}}
"""
                        status.write("🧠 編寫 Python 算式並於沙盒驗算...")
                        res = generate_content_with_retry(
                            client, active_model, [image_part, solver_prompt],
                            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
                        )
                        plan_data = json.loads(res.text)
                        status.write("⚡ 執行 SymPy 代數運算...")
                        plan_data["execution_result"] = execute_sympy_code(plan_data.get("python_code", ""))
                        st.session_state.math_plan = plan_data
                        status.update(label="✅ 後台題目驗算與步驟拆解完成！", state="complete", expanded=False)

                with st.spinner("🤔 家教正在思考高中教學引導步驟..."):
                    sys_inst = f"""
你是一位親切、循循善誘的頂尖高中數學「蘇格拉底家教」。
【後台驗證之標準解法資訊】：
- 題目：{st.session_state.math_plan.get('problem_statement')}
- 驗算最終答案：{st.session_state.math_plan.get('expected_answer')}
- 步驟拆解：{json.dumps(st.session_state.math_plan.get('solution_steps'), ensure_ascii=False)}

【教學準則】：
1. 嚴禁直接透露答案或整串算式！
2. 根據對話進度，一次只問一個關鍵問題（例如：是否需要做輔助線、該用正弦還是餘弦定理）。
3. 學生回答正確時給予鼓勵並推進；學生答錯或卡住時，將問題拆得更直觀。
4. 數學符號使用標準 LaTeX（如 $\\vec{{AB}}$, $\\sin\\theta$, $\\sqrt{{3}}$）。
"""
                    chat_history = [f"{m['role']}: {m['content']}" for m in st.session_state.math_messages[:-1]]
                    chat_history.append(f"user: {prompt}")
                    reply_res = generate_content_with_retry(
                        client, active_model, chat_history,
                        config=types.GenerateContentConfig(system_instruction=sys_inst, temperature=0.4)
                    )
                    reply = reply_res.text

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
    st.header("🧬 高中學測自然探究與生物家教")
    st.caption("專注於：實驗探究變因、折線/柱狀圖判讀、生理反應步驟鏈條、大考誘答選項破解")

    if "bio_messages" not in st.session_state:
        st.session_state.bio_messages = []

    for msg in st.session_state.bio_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if bio_prompt := st.chat_input("輸入你想詢問的學測自然題目或觀念（例如：光反應與碳反應的物質轉移 / 動作電位去極化與再極化離子通道變化）"):
        st.session_state.bio_messages.append({"role": "user", "content": bio_prompt})
        with st.chat_message("user"):
            st.markdown(bio_prompt)

        with st.spinner("🔬 正在梳理高中課綱機制與因果鏈..."):
            bio_sys = """
你是一位頂尖高中學測自然科（生物/物理/化學）名師：
1. 嚴格對齊高中 108 課綱範圍，不使用超出課綱的大學專有名詞。
2. 遇到生理機能或化學反應，使用因果步驟鏈 (步驟 1 ➔ 步驟 2 ➔ 步驟 3) 清晰呈現。
3. 遇到易混淆觀念，製作對照 Markdown 表格。
4. 特別標註「學測最常見陷阱選項與關鍵誘答字眼」。
"""
            chat_hist = [f"{m['role']}: {m['content']}" for m in st.session_state.bio_messages[:-1]]
            chat_hist.append(f"user: {bio_prompt}")
            bio_res = generate_content_with_retry(
                client, active_model, chat_hist,
                config=types.GenerateContentConfig(system_instruction=bio_sys, temperature=0.3)
            )
            b_reply = bio_res.text

        with st.chat_message("assistant"):
            st.markdown(b_reply)
        st.session_state.bio_messages.append({"role": "assistant", "content": b_reply})
        st.rerun()

# =============================================================================
# 頁面 4：高中英文長難句與文法
# =============================================================================
elif active_nav == "📝 高中英文長難句與文法":
    st.header("📝 高中英文長難句與大考文法家教")
    st.caption("專門解決高中學測閱讀測驗長難句、克漏字時態與英文作文句型升級")

    if "eng_messages" not in st.session_state:
        st.session_state.eng_messages = []

    for msg in st.session_state.eng_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if eng_prompt := st.chat_input("貼上一句高中文長難句，或貼上想優化的英文作文段落："):
        st.session_state.eng_messages.append({"role": "user", "content": eng_prompt})
        with st.chat_message("user"):
            st.markdown(eng_prompt)

        with st.spinner("📖 正在拆解高中句子成分與語法亮點..."):
            eng_sys = """
你是一位高中學測英文金牌導師：
1. 【句子結構拆解】：標明【主詞 Subject】、【主要動詞 Verb】、【修飾語/副詞子句/分詞構句 Modifier】。
2. 【高中關鍵語法】：指出句中包含的高中重點文法（例如：關係代名詞、分詞構句、倒裝句、虛擬語氣）。
3. 【大考寫作應用】：提供 1~2 個學測頂標作文句型改寫範例。
"""
            chat_hist = [f"{m['role']}: {m['content']}" for m in st.session_state.eng_messages[:-1]]
            chat_hist.append(f"user: {eng_prompt}")
            eng_res = generate_content_with_retry(
                client, active_model, chat_hist,
                config=types.GenerateContentConfig(system_instruction=eng_sys, temperature=0.3)
            )
            e_reply = eng_res.text

        with st.chat_message("assistant"):
            st.markdown(e_reply)
        st.session_state.eng_messages.append({"role": "assistant", "content": e_reply})
        st.rerun()
