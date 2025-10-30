# ==========================================
# 📄 Automated Quiz Generator Agent (Refined & Fixed)
# ==========================================

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from openai import OpenAI
import pandas as pd
import os
import tempfile
import requests
import json
from datetime import datetime
import base64
from dotenv import load_dotenv
import random

# -----------------------------
# 1️⃣ Load environment variables & initialize OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# -----------------------------
# 2️⃣ Document Loaders
@st.cache_data(show_spinner=False)
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

# -----------------------------
# 3️⃣ Generate MCQs (randomize correct option)
def generate_content_and_mcq(doc_content, difficulty, num_questions, skill_name="General"):
    prompt = f"""
You are an AI quiz generator.

Generate exactly {num_questions} multiple-choice questions (MCQs) based on the topic below.

Topic/Content:
\"\"\"{doc_content}\"\"\"

Output Format:
Skills\tDifficulty\tName\tScore\tOptions\tCorrect Option

Guidelines:
• Skills = {skill_name}
• Difficulty = {difficulty}
• Name = question text
• Score = 1
• Options = "A. <opt1>; B. <opt2>; C. <opt3>; D. <opt4>"
• Correct Option should be randomly assigned among A/B/C/D
• Do NOT repeat header or explanations. Output CSV rows only.
• Ensure options are plausible distractors, and correct options are not always "A"
"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a precise AI that outputs well-formatted tab-separated CSV lines with randomized correct options."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# -----------------------------
# 4️⃣ Parse AI Output into DataFrame (Refined)
def parse_mcqs_to_platform_csv(ai_text):
    lines = [line.strip() for line in ai_text.splitlines() if line.strip()]
    mcq_rows = []

    for line in lines:
        # Skip header
        if line.startswith("Skills") or not line:
            continue
        parts = line.split("\t")
        if len(parts) == 6:
            skills, difficulty, name, score, options_str, correct_tag = parts
            # Convert correct tag to full text
            option_map = {}
            for opt in options_str.split(";"):
                opt_clean = opt.strip()
                if len(opt_clean) >= 2 and opt_clean[1] == ".":
                    key = opt_clean[0]  # 'A', 'B', 'C', 'D'
                    option_map[key] = opt_clean
            correct_text = option_map.get(correct_tag.strip(), correct_tag.strip())

            mcq_rows.append([
                skills, difficulty, name, score, options_str, correct_text
            ])

    df = pd.DataFrame(mcq_rows, columns=["Skills", "Difficulty", "Name", "Score", "Options", "Correct Option"])
    return df

# -----------------------------
# 5️⃣ Refined HireIT Upload
def upload_to_hireit(df_mcq):
    url = "https://api-hireit.grazitti.com/question-mgmt/upload-json-questions"

    # Embed key
    today_str = datetime.now().strftime("%Y-%m-%d")
    actual_key = "OSfJIyzeRBfp007zqcYD7KBf4"  # Replace with your real key
    sec_api_key = f"{today_str}:{actual_key}"
    base64_api_key = base64.b64encode(sec_api_key.encode("utf-8")).decode("utf-8")

    headers = {
        "Authorization": f"Basic {base64_api_key}",
        "Content-Type": "application/json"
    }

    # Build JSON array exactly as HireIT expects
    json_array = []
    for _, row in df_mcq.iterrows():
        json_array.append({
            "skills": row["Skills"],                       # string, e.g., "General"
            "difficulty": row["Difficulty"].lower(),       # "easy", "medium", "hard"
            "name": row["Name"],
            "score": str(row["Score"]),                     # must be string
            "options": [opt.strip() for opt in row["Options"].split(";")],
            "correctOption": row["Correct Option"],        # full text now
            "type": "MCQ"
        })

    try:
        response = requests.post(url, headers=headers, json=json_array)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# -----------------------------
# 6️⃣ Streamlit UI
st.set_page_config(page_title="Automated Quiz Generator", layout="wide")
st.title("📄 Automated Quiz Generator Agent")

# Inputs
input_type = st.radio("Input Type:", ["Upload Document (PDF/TXT/DOCX)", "Topic Name"])
num_mcqs = st.number_input("Number of MCQs:", min_value=5, max_value=100, value=20, step=1)
difficulty = st.selectbox("Select Difficulty:", ["Easy", "Medium", "Hard"])
skill_name = st.text_input("Skill Name (for CSV/HireIT):", "General")
output_type = st.radio("Output Type:", ["Generate CSV", "Generate & Upload JSON to HireIT"])

uploaded_file, topic_name = None, None
if input_type == "Upload Document (PDF/TXT/DOCX)":
    uploaded_file = st.file_uploader("Upload file:", type=["pdf","txt","docx"])
else:
    topic_name = st.text_input("Enter topic name:")

# -----------------------------
# 7️⃣ Generate Quiz
if st.button("Generate Quiz"):
    with st.spinner("⏳ Processing your input..."):
        if uploaded_file:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name
            try:
                docs = load_document(temp_file_path)
                doc_text = "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                st.error(f"❌ Error loading document: {e}")
                st.stop()
            finally:
                os.remove(temp_file_path)
        elif topic_name:
            doc_text = topic_name
        else:
            st.error("⚠️ Please provide a file or topic name.")
            st.stop()

        # Generate MCQs
        ai_output = generate_content_and_mcq(doc_text, difficulty, num_mcqs, skill_name)
        df_mcq = parse_mcqs_to_platform_csv(ai_output)

        if df_mcq.empty:
            st.warning("⚠️ No valid MCQs generated. Try a simpler topic or fewer questions.")
        else:
            st.subheader("✅ Generated MCQs")
            st.dataframe(df_mcq.reset_index(drop=True), use_container_width=True)

            if output_type == "Generate CSV":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = f"generated_quiz_{timestamp}.csv"
                csv_content = df_mcq.to_csv(sep="\t", index=False).encode('utf-8')
                st.download_button(label = "⬇️ Download CSV", data = csv_content,
                                   file_name = csv_file, mime= "text/csv")

            elif output_type == "Generate & Upload JSON to HireIT":
                hireit_response = upload_to_hireit(df_mcq)
                if "error" in hireit_response:
                    st.error(f"❌ Upload failed: {hireit_response['error']}")
                else:
                    st.success("✅ Successfully uploaded JSON to HireIT!")
                    st.json(hireit_response)


