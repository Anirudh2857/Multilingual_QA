# 🌍 Multilingual Question Answering App

This is a Streamlit web application that allows users to ask questions in English, provide a context passage, and receive an answer that is translated into their selected language using GPT-4. It combines a fine-tuned multilingual QA model (hosted on Hugging Face) with OpenAI's GPT for translation.

---

## 🚀 Features

- 🔍 **Question Answering** using a fine-tuned multilingual transformer model.
- 🌐 **Language Translation** of answers into Hindi, Spanish, Arabic, German, Vietnamese, or Simplified Chinese using GPT-4.
- 🧠 Built with `transformers`, `Streamlit`, and `OpenAI`.

---


## 🔗 Try the App

👉 [Click here to try the app](https://multilingualapp-keftxtknxesuckmg7hdebi.streamlit.app)  
<!-- Replace the above with your deployed app URL -->

---

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/multilingual-qa-app.git
   cd multilingual-qa-app
   ```

2. **Create a virtual environment (optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key**
   - Create a `.env` file (optional but recommended) or set the environment variable directly.
   ```env
   OPENAI_API_KEY=your-openai-api-key
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Model Info

This app uses a fine-tuned `question-answering` transformer model hosted on Hugging Face:  
👉 [https://huggingface.co/your-username/multilingual-qa-model](https://huggingface.co/your-username/multilingual-qa-model)

---

## 🛠️ Tech Stack

- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers)
- [Streamlit](https://streamlit.io/)
- [OpenAI API](https://platform.openai.com/)
- [Python](https://www.python.org/)

---

## ✨ Future Improvements

- Add voice input and output
- Improve answer confidence visualization
- Cache GPT translations to reduce costs

---

## 📄 License

MIT License. Feel free to fork and build upon it!

