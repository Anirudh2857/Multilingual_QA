import os
import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load the OpenAI API key and Hugging Face token from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Initialize the question answering pipeline
@st.cache_resource
def load_qa_model():
    """
    Loads the multilingual question answering model and tokenizer with Hugging Face token for authentication.
    """
    model_name = "Anirudh2857/multilingual-qa-model"  # Update with your model name
    try:
        # Ensure the Hugging Face token is available for authentication
        if not HUGGING_FACE_TOKEN:
            raise ValueError("Hugging Face token is not set. Please set it in your environment variables.")
        
        # Load model and tokenizer using Auto classes with Hugging Face token
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, use_auth_token=HUGGING_FACE_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGING_FACE_TOKEN)
        
        # Return the pipeline with model and tokenizer
        return pipeline("question-answering", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None

# Load the question answering pipeline
qa_pipeline = load_qa_model()

# Set page config
st.set_page_config(page_title="Multilingual QA System", layout="wide")

# Streamlit app header
st.title("Multilingual Question Answering System")

# Input field for the user to ask questions
question = st.text_input("Ask a question:")

# Input field for the context (could be any text, like a paragraph, for answering)
context = st.text_area("Provide context for the question:")

# Display the answer when both question and context are provided
if question and context:
    if qa_pipeline:
        try:
            # Get the answer from the pipeline
            result = qa_pipeline({
                "context": context,
                "question": question
            })
            st.subheader("Answer")
            st.write(result["answer"])  # Show the answer
        except Exception as e:
            st.error(f"Error during QA pipeline execution: {e}")
    else:
        st.error("Failed to load QA pipeline.")
else:
    st.warning("Please enter both a question and context.")

# Optional: Add additional features or instructions here
st.markdown(
    """
    ## Instructions:
    - Type a question in the 'Ask a question' field.
    - Provide some text context in the 'Provide context for the question' field.
    - The model will attempt to find an answer from the provided context.
    """
)

# Handle environment and caching
@st.cache_data
def load_environment_variables():
    """
    Loads environment variables for Streamlit Cloud.
    """
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API Key is not set.")
        if not HUGGING_FACE_TOKEN:
            raise ValueError("Hugging Face token is not set.")
        return OPENAI_API_KEY, HUGGING_FACE_TOKEN
    except Exception as e:
        st.error(f"Error loading environment variables: {e}")
        return None, None

# Ensure the OpenAI API key and Hugging Face token are loaded correctly
api_key, hf_token = load_environment_variables()
if api_key and hf_token:
    st.success("OpenAI API Key and Hugging Face Token are loaded.")
else:
    st.warning("API keys are not set properly.")
