# 📄 Chat With Your Documents (Gemini RAG)

This is a **document-aware AI chatbot** that allows users to upload PDF, DOCX, and TXT files and ask natural language questions. The system uses **Google Gemini** for response generation and a **FAISS-based vector store** with **Hugging Face embeddings** for document chunk retrieval.

---

## 🚀 Features

- 🧠 Gemini 1.5 Flash as the language model
- 📚 Accepts multiple document types: PDF, DOCX, TXT
- 🪄 Converts uploaded files into chunked vector documents
- 🔍 Uses FAISS for similarity search across document content
- 💬 Conversational UI with memory of chat history (via Streamlit)
- 📦 Local, secure file processing

---

## 🧠 How It Works

1. **File Upload**: User uploads one or more documents (PDF, DOCX, TXT).
2. **Parsing & Splitting**: Documents are parsed and split into text chunks.
3. **Embedding & Indexing**: Text chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index.
4. **Context Retrieval**: When a user asks a question, the top 3 most relevant document chunks are retrieved.
5. **Response Generation**: Gemini generates a response based on the retrieved context and the user's query.
6. **Chat UI**: The system displays the conversation using a chat-style interface in Streamlit.

---

## 🛠️ Tech Stack

- `Python 3.9+`
- `Streamlit`
- `LangChain`
- `FAISS`
- `sentence-transformers`
- `PyPDF2`
- `python-docx`
- `Google Generative AI SDK (Gemini)`

---

## 📂 Project Structure

```
📦 project-root/
├── 📄 app2.py # Main Streamlit application
├── 📄 requirements.txt # required libraries to run the app
└── 📄 README.md # Project documentation
```

---

## ▶️ Run the App

```
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app2.py
```

---

## 📌 Use Cases

- Legal document summarization
- Financial report analysis
- Academic paper Q&A
- HR policy understanding

---

## 📜 License

This project is open-source under the MIT License. Feel free to use, share, and build on it!
