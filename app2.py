import streamlit as st
from PyPDF2 import PdfReader
import docx
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import os


def parse_txt(file):
    return file.read().decode("utf-8")

def parse_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def parse_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_file(file):
    if file.name.endswith(".pdf"):
        return parse_pdf(file)
    elif file.name.endswith(".docx"):
        return parse_docx(file)
    elif file.name.endswith(".txt"):
        return parse_txt(file)
    else:
        return ""

def create_vectorstore_from_documents(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = splitter.create_documents(texts)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore

def get_gemini_response(context: str, question: str, api_key: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""Use the following document excerpts to answer the question:

    Context:
    {context}
    Question:
    {question}
    Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

st.set_page_config(page_title="Chat with Documents", layout="wide")
st.title("📄 Chat With Your Documents (Gemini RAG)")

api_key = 'Your_API_Key'

uploaded_files = st.file_uploader("Upload your documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        texts = [parse_file(file) for file in uploaded_files]
        vectorstore = create_vectorstore_from_documents(texts)
    st.success("Documents successfully processed! Ask a question below.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        speaker, message = chat
        with st.chat_message(speaker.lower()):
            st.markdown(message)

    if prompt := st.chat_input("💬 Ask a question about your documents:"):
        st.session_state.chat_history.append(("You", prompt))
        with st.spinner("Generating answer..."):
            docs = vectorstore.similarity_search(prompt, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = get_gemini_response(context, prompt, api_key)
            st.session_state.chat_history.append(("Gemini", answer))

        with st.chat_message("you"):
            st.markdown(prompt)
        with st.chat_message("gemini"):
            st.markdown(answer)