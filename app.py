import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import os

# Ensure that Hugging Face token is set
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')  # Load Hugging Face token from environment variables

if not HUGGING_FACE_TOKEN:
    st.error("Hugging Face token is not set. Please set it in your environment variables.")
    st.stop()  # Stop execution if token is not set

# Define a function to load the QA model and tokenizer
def load_qa_model():
    try:
        model_name = "Anirudh2857/multilingual-qa-model"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGING_FACE_TOKEN)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, use_auth_token=HUGGING_FACE_TOKEN)
        
        # Create the QA pipeline
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        
        return qa_pipeline
    
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None

# Load the QA model and tokenizer
qa_pipeline = load_qa_model()

if not qa_pipeline:
    st.stop()  # Stop execution if model or tokenizer could not be loaded

# Streamlit app header
st.set_page_config(page_title="Multilingual QA System", layout="wide")
st.title("Multilingual Question Answering System")

# Get user input for context and question
context = st.text_area("Enter context (passage to be used for QA)", height=300)
question = st.text_input("Enter your question")

# Ensure that both question and context are provided before executing the QA
if st.button("Get Answer") and context and question:
    try:
        # Debug: Print the inputs to make sure they are correct
        st.write(f"Question: {question}")
        st.write(f"Context: {context[:500]}...")  # Show a truncated context for readability
        
        # Run the QA pipeline
        result = qa_pipeline({
            'context': context,
            'question': question
        })
        
        # Display the answer
        st.subheader("Answer")
        st.write(result['answer'])

    except Exception as e:
        st.error(f"Error during QA pipeline execution: {e}")
else:
    if not context:
        st.warning("Please enter the context for the question answering.")
    if not question:
        st.warning("Please enter a question to get an answer.")
