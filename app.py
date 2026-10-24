import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from PIL import Image
import streamlit as st
import sympy as sp
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 頁面基本配置
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI 數理蘇格拉底引導家教",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 優先使用通用穩定模型
PRIMARY_MODEL = "gemini-2.0-flash"

# -----------------------------------------------------------------------------
# 安全執行 Python 驗算程式碼
# -----------------------------------------------------------------------------
def execute_sympy_code(code_str: str) -> dict:
    clean_code = re.sub(r"^```python\s*", "", code_str, flags=re.MULTILINE)
    clean_code = re.sub(r"^```\s*$", "", clean_code, flags=re.MULTILINE)

    stdout_capture = io.StringIO()
    exec_globals = {
        "sp": sp,
        "sympy": sp,
        "symbols": sp.symbols,
        "Eq": sp.Eq,
        "solve": sp.solve,
        "sqrt": sp.sqrt,
    }
    exec_locals = {}

    try:
        with redirect_stdout(stdout_capture):
            exec(clean_code, exec_globals, exec_locals)
        output = stdout_capture.getvalue()
        return {
            "success": True,
            "stdout": output.strip(),
            "code": clean_code.strip(),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue().strip(),
            "code": clean_code.strip(),
            "error": str(e),
        }

# -----------------------------------------------------------------------------
# Multi-Agent 核心邏輯
# -----------------------------------------------------------------------------
def run_front_desk_and_solver(client: genai.Client, image: Image.Image, user_query: str):
    solver_prompt = f"""
你是由兩位專家組成的後台核心：
1. 【櫃檯專家】：請仔細辨識題目圖片與使用者的指示「{user_query}」，提取所有幾何條件、已知數據與欲求目標。
2. 【解題專家】：嚴禁擅自靠直覺心算！你必須編寫一段簡短的 Python (使用 sympy) 程式碼來驗算並求解本題。

請務必輸出合法的 JSON 格式，包含以下欄位：
{{
    "problem_statement": "題目文字描述與條件摘要",
    "python_code": "# 完整的 Python SymPy 程式碼\\nimport sympy as sp\\n...",
    "expected_answer": "最終運算結果（例如 16*sqrt(5)）",
    "solution_steps": [
        "第 1 步：計算底面積 ABC，公式...",
        "第 2 步：尋求高 EF，利用畢氏定理...",
        "第 3 步：帶入錐體體積公式 V = (1/3) * B * h..."
    ]
}}
重要：請直接輸出 JSON 物件，不要包在任何額外的解釋文字中。
"""
    if image.mode != "RGB":
        image = image.convert("RGB")

    response = client.models.generate_content(
        model=PRIMARY_MODEL,
        contents=[image, solver_prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    data = json.loads(response.text)
    execution_result = execute_sympy_code(data.get("python_code", ""))
    data["execution_result"] = execution_result
    return data

def socratic_tutor_step(client: genai.Client, history: list, solved_plan: dict, latest_input: str):
    system_instruction = f"""
你是一位親切、有耐心的頂尖高中數學「蘇格拉底家教」。
【後台驗證之標準解法資訊】：
- 題目：{solved_plan.get('problem_statement')}
- 驗算最終答案：{solved_plan.get('expected_answer')}
- 完整拆解步驟：{json.dumps(solved_plan.get('solution_steps'), ensure_ascii=False)}

【教學準則】：
1. 嚴禁直接透露最終答案或完整算式。
2. 根據目前對話進度，一次只問一個關鍵問題或引導一個小步驟（例如：先問底面積怎麼算、是否需要做輔助線）。
3. 若學生回答正確，給予讚許並推進至下一步。
4. 若學生回答錯誤或說「不知道/不會算」，請把當前問題拆成更簡單的直求提問。
5. 數學符號一律使用 LaTeX 格式（例如 $x^2$, $\\sqrt{{5}}$）。
"""
    chat_contents = []
    for h in history:
        chat_contents.append(f"{h['role']}: {h['content']}")
    chat_contents.append(f"user: {latest_input}")

    response = client.models.generate_content(
        model=PRIMARY_MODEL,
        contents=chat_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,
        ),
    )
    return response.text

def generate_study_kit(client: genai.Client, solved_plan: dict):
    kit_prompt = f"""
根據本題題型與幾何觀念（題目：{solved_plan.get('problem_statement')}）：
1. 產出一段 Mermaid.js 心智圖代碼（以 graph TD 開頭），梳理此題涵蓋的核心公式、解題路徑與易錯陷阱。
2. 產出 3 張「考點翻轉學習卡（Flashcards）」，每張包含正面問題（Front）與背面解答（Back）。

請輸出 JSON 格式：
{{
    "mermaid_code": "graph TD\\n...",
    "flashcards": [
        {{"front": "...", "back": "..."}},
        {{"front": "...", "back": "..."}}
    ]
}}
"""
    response = client.models.generate_content(
        model=PRIMARY_MODEL,
        contents=[kit_prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    return json.loads(response.text)

# -----------------------------------------------------------------------------
# 前端介面與狀態管理
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "solved_plan" not in st.session_state:
    st.session_state.solved_plan = None
if "study_kit" not in st.session_state:
    st.session_state.study_kit = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None

env_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

with st.sidebar:
    st.title("⚙️ 系統設定")
    api_key = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="自動載入自 Secrets 或環境變數",
    )
    if api_key:
        st.success("✅ API Key 已就緒")
    st.markdown("---")
    
    st.header("📸 題目影像輸入")
    input_mode = st.radio("選擇題目上傳方式：", ["檔案上傳 / 截圖", "即時相機拍照"])
    
    img_data = None
    if input_mode == "檔案上傳 / 截圖":
        uploaded_file = st.file_uploader("上傳題目圖片 (PNG, JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            img_data = Image.open(uploaded_file)
    else:
        camera_img = st.camera_input("使用鏡頭拍下題目")
        if camera_img:
            img_data = Image.open(camera_img)

    if img_data:
        st.session_state.current_image = img_data
        st.image(img_data, caption="已載入之題目", use_container_width=True)

    if st.button("🔄 重置新對話"):
        st.session_state.messages = []
        st.session_state.solved_plan = None
        st.session_state.study_kit = None
        st.rerun()

# -----------------------------------------------------------------------------
# 主畫面佈局
# -----------------------------------------------------------------------------
st.header("📐 數理蘇格拉底 AI 引導家教")

if not api_key:
    st.warning("⚠️ 請在左側輸入或在 Streamlit Secrets 設定你的 **Gemini API Key**！")
    st.stop()

client = genai.Client(api_key=api_key)

col_chat, col_tools = st.columns([6, 4])

with col_tools:
    st.subheader("🛠️ 後台多代理人監控與學習套件")
    
    if st.session_state.solved_plan:
        with st.expander("🔍 後台驗算記錄 (Python SymPy 執行軌跡)", expanded=False):
            plan = st.session_state.solved_plan
            st.markdown(f"**識別題目：** {plan.get('problem_statement')}")
            st.markdown(f"**理論解：** `{plan.get('expected_answer')}`")
            st.markdown("**專家驗算程式碼：**")
            st.code(plan.get("python_code", ""), language="python")
            
            exec_res = plan.get("execution_result", {})
            if exec_res.get("success"):
                st.success(f"程式執行成功！輸出：\n{exec_res.get('stdout')}")
            else:
                st.error(f"執行出錯：{exec_res.get('error')}")

        st.markdown("---")
        if st.button("✨ 一鍵生成心智圖與觀念記憶卡", use_container_width=True):
            with st.spinner("AI 正在繪製 Mermaid 心智圖與生成記憶卡..."):
                st.session_state.study_kit = generate_study_kit(client, st.session_state.solved_plan)

        if st.session_state.study_kit:
            kit = st.session_state.study_kit
            tab1, tab2 = st.tabs(["🗺️ 核心心智圖", "🗂️ 觀念學習卡"])
            
            with tab1:
                st.markdown("#### 單元架構與解題路徑")
                mermaid_code = kit.get("mermaid_code", "")
                st.markdown(f"```mermaid\n{mermaid_code}\n```")
                
            with tab2:
                st.markdown("#### 核心考點翻轉卡片")
                for i, card in enumerate(kit.get("flashcards", [])):
                    with st.expander(f"卡片 {i+1}：{card.get('front')}"):
                        st.info(f"**觀念解答：** {card.get('back')}")
    else:
        st.caption("👈 上傳題目並發送第一則訊息後，此處將顯示後台驗算與生成之教材模組。")

with col_chat:
    st.subheader("💬 蘇格拉底解題對話")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("請輸入你的回覆或提問（例如：請教我算這題 / 我不知道怎麼算...）"):
        if not st.session_state.current_image:
            st.warning("⚠️ 請先在左側上傳或拍下題目圖片！")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.solved_plan:
            with st.spinner("🕵️ 後台解題專家正在驗算題意並執行 Python 算式中..."):
                st.session_state.solved_plan = run_front_desk_and_solver(
                    client=client,
                    image=st.session_state.current_image,
                    user_query=prompt,
                )

        with st.spinner("🤔 家教正在思考引導步驟..."):
            reply = socratic_tutor_step(
                client=client,
                history=st.session_state.messages[:-1],
                solved_plan=st.session_state.solved_plan,
                latest_input=prompt,
            )

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
