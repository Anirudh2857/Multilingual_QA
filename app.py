import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from openai import OpenAI
import os

# 🔧 Streamlit Config
st.set_page_config(page_title="🌍 Multilingual QA System", layout="centered")
st.title("🌍 Multilingual QA with Translation")
st.markdown("Ask a question and get the answer in your preferred language.")

# 🔐 API Keys
HUGGINGFACE_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🌍 Language Options
language_map = {
    "English": "English",
    "Hindi": "Hindi",
    "Spanish": "Spanish",
    "Arabic": "Arabic",
    "German": "German",
    "Vietnamese": "Vietnamese",
    "Chinese (Simplified)": "Chinese"
}

# 🤖 Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# 🧠 Load QA Model
@st.cache_resource(show_spinner="Loading model and tokenizer...")
def load_model_and_tokenizer():
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
        st.error(f"Model loading error: {e}")
        return None

qa_pipeline = load_model_and_tokenizer()

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

# 🧾 Inputs
question = st.text_input("❓ Enter your question:")
context = st.text_area("📄 Enter the context passage:")
qa_lang = st.selectbox("🌐 Translate answer into:", list(language_map.keys()), index=0)
submit = st.button("Get Answer")

# 🚀 Inference
if submit:
    if not context.strip() or not question.strip():
        st.warning("Please enter both context and question.")
    elif qa_pipeline is None:
        st.error("QA pipeline not available.")
    else:
        with st.spinner("Generating answer..."):
            try:
                result = qa_pipeline(question=question, context=context)
                answer = result["answer"]
                confidence = round(result["score"] * 100, 2)

                translated = gpt_translate(answer, language_map[qa_lang])

                st.markdown("### ✅ Original Answer:")
                st.success(answer)
                st.caption(f"Confidence Score: {confidence}%")

                st.markdown(f"### 🌐 Translated Answer ({qa_lang}):")
                st.info(translated)

            except Exception as e:
                st.error(f"Error during QA pipeline execution: {str(e)}")
