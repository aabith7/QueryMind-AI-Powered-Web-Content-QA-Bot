import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_mistralai.chat_models import ChatMistralAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import os
import requests
import streamlit as st

def download_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, allow_redirects=True)
    with open(destination, "wb") as f:
        f.write(response.content)

# Check and download the file if not present
file_path = "faiss_store_openai.pkl"
if not os.path.exists(file_path):
    st.info("Downloading FAISS store from Google Drive...")
    file_id = "1-9kLVeo7PepdXtphpcc7sn3knd-PWKGo"
    download_from_google_drive(file_id, file_path)
    st.success("Download complete!")





load_dotenv()  # take environment variables from .env (especially openai api key)
st.image("ChatGPT Image Jun 18, 2025, 09_50_57 AM.png", width=150) 
st.title("QueryMind 🧠")
st.sidebar.title("Website URLs 🕸️")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

llm = ChatMistralAI(
    model="mistral-medium",  # or mistral-small or mistral-tiny
    temperature=0.7,
    max_tokens=500
)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)