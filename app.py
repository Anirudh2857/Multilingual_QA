import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# Set up Streamlit page
st.set_page_config(page_title="Multilingual QA System", layout="wide")
st.title("🌍 Multilingual Question Answering System")

# Load Hugging Face token from Streamlit secrets
HUGGING_FACE_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]

@st.cache_resource(show_spinner="Loading model and tokenizer...")
def load_model_and_tokenizer():
    try:
        model_name = "Anirudh2857/multilingual-qa-model"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=HUGGING_FACE_TOKEN,
            trust_remote_code=True,
            use_fast=False  # safer with custom models
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            use_auth_token=HUGGING_FACE_TOKEN,
            trust_remote_code=True
        )
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None

qa_pipeline = load_model_and_tokenizer()

# UI Input
context = st.text_area("📝 Enter context passage:", height=200)
question = st.text_input("❓ Enter your question:")

if st.button("Get Answer"):
    if not context or not question:
        st.warning("Please enter both context and question.")
    elif qa_pipeline is None:
        st.error("QA pipeline not available.")
    else:
        try:
            result = qa_pipeline(question=question, context=context)
            st.success("✅ Answer:")
            st.markdown(f"**{result['answer']}**")
            st.caption(f"Confidence: {result['score']:.2f}")
        except Exception as e:
            st.error(f"Error during QA pipeline execution: {e}")
