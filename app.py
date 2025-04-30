import streamlit as st
from transformers import XLMRobertaForQuestionAnswering, XLMRobertaTokenizer, pipeline
import os

# Ensure the Hugging Face token is set in Streamlit secrets
HUGGING_FACE_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]

# Set up Streamlit page configuration
st.set_page_config(page_title="Multilingual QA System", layout="wide")

# Define the function to load the QA pipeline
@st.cache_resource
def load_qa_model():
    """
    This function loads the multilingual question answering model and tokenizer
    with Hugging Face token for authentication.
    """
    model_name = "Anirudh2857/multilingual-qa-model"  # Replace with your model

    try:
        # Check if Hugging Face token is available
        if not HUGGING_FACE_TOKEN:
            raise ValueError("Hugging Face token is not set. Please set it in your Streamlit secrets.")

        # Load the model and tokenizer from Hugging Face using the provided token
        model = XLMRobertaForQuestionAnswering.from_pretrained(model_name, use_auth_token=HUGGING_FACE_TOKEN)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, use_auth_token=HUGGING_FACE_TOKEN)

        # Return the question answering pipeline
        return pipeline("question-answering", model=model, tokenizer=tokenizer)

    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None

# Load the QA pipeline
qa_pipeline = load_qa_model()

# Title and Instructions for the app
st.title("Multilingual Question Answering System")
st.markdown("""
    ## Instructions:
    - Type a question in the 'Ask a question' field.
    - Provide some text context in the 'Provide context for the question' field.
    - The model will attempt to find an answer from the provided context.
""")

# Input fields for question and context
question = st.text_input("Ask a question:")
context = st.text_area("Provide context for the question:")

# Display the answer when both question and context are provided
if question and context:
    if qa_pipeline:
        try:
            # Get the answer using the QA pipeline
            result = qa_pipeline({
                "context": context,
                "question": question
            })
            st.subheader("Answer")
            st.write(result["answer"])  # Display the answer
        except Exception as e:
            st.error(f"Error during QA pipeline execution: {e}")
    else:
        st.error("Failed to load QA pipeline.")
else:
    st.warning("Please enter both a question and context.")
