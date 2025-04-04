# ✅ Must be the first Streamlit command
import streamlit as st
st.set_page_config(page_title="🌍 Multilingual QA", layout="centered")

# 🚀 Imports
from transformers import pipeline
import openai
import os

# 🔐 Securely load OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🌍 Language options
language_map = {
    "English": "English",
    "Hindi": "Hindi",
    "Spanish": "Spanish",
    "Arabic": "Arabic",
    "German": "German",
    "Vietnamese": "Vietnamese",
    "Chinese (Simplified)": "Chinese"
}

# ✅ Load fine-tuned QA model from Hugging Face
@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="Anirudh2857/Multilingual_QA",  # 🔁 Replace with your actual Hugging Face repo
        tokenizer="Anirudh2857/Multilingual_QA"
    )

qa_pipeline = load_qa_model()

# 🔁 GPT-based translation using OpenAI API
def gpt_translate(text, target_language):
    prompt = f"Translate the following answer into {target_language}:\n\n'{text}'"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful multilingual assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        st.error(f"GPT Translation Error: {str(e)}")
        return text

# 🖼️ Streamlit UI
st.title(":globe_with_meridians: Multilingual QA")
st.markdown("Ask a question and get the answer in your preferred language.")

question = st.text_input("❓ Enter your **question**:")
context = st.text_area("📄 Enter the **context passage**:")
qa_lang = st.selectbox("🌐 Language for QA output:", list(language_map.keys()), key="qa_lang")
submit = st.button("Get Answer")

# 🚀 Run QA + Translation
if submit:
    if not question.strip() or not context.strip():
        st.warning("❗ Please enter both a question and a context.")
    else:
        with st.spinner("Thinking... 🧠"):
            try:
                result = qa_pipeline(question=question, context=context)
                original_answer = result["answer"]
                confidence = round(result["score"] * 100, 2)

                qa_translated = gpt_translate(original_answer, language_map[qa_lang])

                # Display results
                st.markdown("### ✅ Original Answer (same as context language):")
                st.success(original_answer)
                st.caption(f"📊 Confidence Score: {confidence}%")

                st.markdown(f"### 🌐 Translated Answer ({qa_lang}):")
                st.info(qa_translated)

            except Exception as e:
                st.error(f"🚫 Error: {str(e)}")
