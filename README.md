# QueryMind-AI-Powered-Web-Content-QA-Bot
# QueryMind 🧠 - AI-Powered Web Content QA Bot

QueryMind is a powerful Streamlit application that lets you query information from any set of webpages using state-of-the-art language models and vector embeddings. Just enter up to 3 URLs, and QueryMind will extract and chunk the content, embed it using HuggingFace models, and enable you to ask questions about that content directly.

---

## 🔍 Features

- 🌐 Fetch and process web content from multiple URLs.
- 🧠 Use `ChatMistralAI` for intelligent question-answering.
- 🔍 Chunk and index content with `RecursiveCharacterTextSplitter` and FAISS.
- 🧩 Generate embeddings using HuggingFace (`all-MiniLM-L6-v2`).
- 🗃️ Save and reload vector index using `pickle`.
- 🖼️ Streamlit interface with sidebar controls and live status updates.

---

## 📦 Tech Stack

- **Frontend**: Streamlit
- **LLM**: Mistral AI (`ChatMistralAI`)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Vector DB**: FAISS
- **Environment Management**: `dotenv`
- **Backend Processing**: LangChain

---

## 🚀 How to Run Locally

### 1. Clone the repository

git clone https://github.com/yourusername/querymind.git
cd querymind
pip install -r requirements.txt
OPENAI_API_KEY=your_openai_or_mistral_api_key
streamlit run app.py
\
---

🛠 Requirements
Make sure you have the following installed:

Python 3.8+

streamlit

langchain

faiss-cpu

huggingface-hub

python-dotenv

mistralai or required credentials for ChatMistralAI

---

📷 Screenshots
![image](https://github.com/user-attachments/assets/b8405ff2-96fb-4888-9fa7-4a8d8f6175db)


---


🙌 Acknowledgements
LangChain

Mistral AI

FAISS

HuggingFace Embeddings





---


