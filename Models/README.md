Multilingual Question Answering Model

This project trains a multilingual question-answering (QA) model using the mlqa dataset, leveraging the xlm-roberta-base model from Hugging Face's transformer library. The model is designed to handle questions and answers in multiple languages, including English, Arabic, German, Spanish, Hindi, Vietnamese, and Chinese.

Installation

To get started, you will need to install the required libraries. You can do so by running the following command:

pip install transformers datasets streamlit --quiet
Setup and Training

The training process involves:

Loading the multilingual question-answering dataset (mlqa) in various language pairs.
Combining the test splits from each language pair into a single training dataset.
Tokenizing the dataset and preparing it for training using the xlm-roberta-base model.
Fine-tuning the model with the tokenized dataset.
Saving the trained model and tokenizer.
Zipping the model directory for easy sharing or uploading to platforms like Hugging Face.
Dataset
The mlqa dataset is a multilingual dataset for machine reading comprehension. It contains question-answer pairs in multiple languages, and the languages used for this specific project include:

English to Arabic
English to German
English to Spanish
English to Hindi
English to Vietnamese
English to Chinese
Tokenization
The tokenizer used in this project is xlm-roberta-base, which is well-suited for multilingual text. The dataset is tokenized, and answer spans are aligned with tokenized sequences. This ensures that the model can correctly learn to predict the start and end positions of answers in the context.

Model Training
The model is trained using the Trainer API from Hugging Face's transformers library. Key training parameters include:

Learning Rate: 3e-5
Batch Size: 8
Epochs: 20
Weight Decay: 0.01
The training process ensures that the model is optimized for multilingual question answering.

Saving and Zipping the Model
After training, the model and tokenizer are saved in the multilingual-qa-model directory. This directory is then zipped into a file named multilingual-qa-model.zip, ready for uploading or sharing.

How to Use

Running the Script
Run the script in a Python environment to start the training process. After training, the model will be saved as a .zip file, which you can upload to platforms like Hugging Face or load for inference.

Example of Using the Trained Model for Inference
Once the model is trained, you can load it for inference in your own applications. Below is an example of how to load and use the trained model for question answering.

from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="path_to_your_saved_model",
    tokenizer="path_to_your_saved_tokenizer"
)

context = "Your context here"
question = "Your question here"

answer = qa_pipeline(question=question, context=context)
print(answer)
Model and Tokenizer Files
The model and tokenizer are saved in the multilingual-qa-model.zip file.
To use the model, unzip the file and load it as shown in the example above.
Notes

The training process can take significant time depending on your hardware setup. For best performance, it is recommended to use a GPU-enabled environment.
You can modify the training_args to suit your specific needs, such as adjusting the number of epochs or batch size.
