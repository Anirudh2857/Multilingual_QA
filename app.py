import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import openai
import os

# 🔧 Streamlit Page Config
st.set_page_config(page_title="🌍 Multilingual QA System", layout="centered")
st.title("🌍 Multilingual QA with Translation")
st.markdown("Ask a question and get the answer in your preferred language.")

# 🔐 API Keys from Streamlit secrets
HUGGINGFACE_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

# 🌐 Supported Languages
language_map = {
    "English": "English",
    "Hindi": "Hindi",
    "Spanish": "Spanish",
    "Arabic": "Arabic",
    "German": "German",
    "Vietnamese": "Vietnamese",
    "Chinese (Simplified)": "Chinese"
}

# 🧠 Load Multilingual QA Model
@st.cache_resource(show_spinner="Loading multilingual QA model...")
def load_qa_pipeline():
    try:
        model_name = "Anirudh2857/multilingual-qa-model"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=HUGGINGFACE_TOKEN,
            trust_remote_code=True,
            use_fast=False
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            use_auth_token=HUGGINGFACE_TOKEN,
            trust_remote_code=True
        )
        return pipeline("question-answering", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

qa_pipeline = load_qa_pipeline()

# 🔁 Translate Answer using GPT (Updated for openai>=1.0.0)
def gpt_translate(text, target_language):
    prompt = f"Translate the following answer into {target_language}:\n\n'{text}'"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Using GPT-4 for translation
            messages=[
                {"role": "system", "content": "You are a helpful multilingual assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

# 🧾 User Inputs
question = st.text_input("❓ Enter your question:")
context = st.text_area("📄 Enter the context passage:")
selected_language = st.selectbox("🌐 Translate answer into:", list(language_map.keys()), index=0)
submit = st.button("Get Answer")

# 🚀 Run Inference
if submit:
    if not question.strip() or not context.strip():
        st.warning("Please enter both the context and the question.")
    elif qa_pipeline is None:
        st.error("QA model not available.")
    else:
        with st.spinner("Processing..."):
            try:
                # Perform question answering
                result = qa_pipeline(question=question, context=context)
                answer = result["answer"]
                confidence = round(result["score"] * 100, 2)

                # Translate the answer into the selected language
                translated = gpt_translate(answer, language_map[selected_language])

                # Display results
                st.markdown("### ✅ Original Answer:")
                st.success(answer)
                st.caption(f"Confidence Score: {confidence}%")

                st.markdown(f"### 🌐 Translated Answer ({selected_language}):")
                st.info(translated)

            except Exception as e:
                st.error(f"Error during QA or translation: {str(e)}")
