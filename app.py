# ✅ Streamlit Setup
import streamlit as st
st.set_page_config(page_title="🌍 Multilingual QA", layout="centered")

# 🚀 Imports
from transformers import pipeline
from openai import OpenAI
import os

# 🔐 API Keys (use environment variables or secrets)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Optional: also secure this
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# 🌍 Supported Output Languages
language_map = {
    "English": "English",
    "Hindi": "Hindi",
    "Spanish": "Spanish",
    "Arabic": "Arabic",
    "German": "German",
    "Vietnamese": "Vietnamese",
    "Chinese (Simplified)": "Chinese"
}

# ✅ Load your fine-tuned model from Hugging Face
@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="Anirudh2857/multilingual-qa-model",  # Replace with your HF model repo
        tokenizer="Anirudh2857/multilingual-qa-model",
        use_auth_token=HUGGINGFACE_TOKEN
    )

qa_pipeline = load_qa_model()

# 🔁 GPT Translation
def gpt_translate(text, target_language):
    prompt = f"Translate the following answer into {target_language}:\n\n'{text}'"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful multilingual assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT Translation Error: {str(e)}")
        return text

# 🖼️ Streamlit Interface
st.title(":globe_with_meridians: Multilingual QA")
st.markdown("Ask a question and get the answer in your preferred language.")

question = st.text_input("❓ Enter your **question**:")
context = st.text_area("📄 Enter the **context passage**:")
qa_lang = st.selectbox("🌐 Language for QA output:", list(language_map.keys()), key="qa_lang")
submit = st.button("Get Answer")

# 🚀 Inference
if submit:
    if not question.strip() or not context.strip():
        st.warning("❗ Please enter both a question and a context.")
    else:
        with st.spinner("Thinking... 🧠"):
            try:
                result = qa_pipeline(question=question, context=context)
                original_answer = result["answer"]
                confidence = round(result["score"] * 100, 2)

                translated_answer = gpt_translate(original_answer, language_map[qa_lang])

                st.markdown("### ✅ Original Answer:")
                st.success(original_answer)
                st.caption(f"📊 Confidence Score: {confidence}%")

                st.markdown(f"### 🌐 Translated Answer ({qa_lang}):")
                st.info(translated_answer)

            except Exception as e:
                st.error(f"🚫 Error: {str(e)}")
