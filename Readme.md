# Company Knowledge-Base Chatbot

A document question-answering application built with Python, Streamlit, OpenAI and ChromaDB using a custom Retrieval-Augmented Generation (RAG) pipeline.

## Features

- Upload company documents in PDF, TXT and DOCX formats.
- Extract text and split documents into overlapping chunks.
- Generate embeddings using OpenAI's embedding model.
- Store and search document vectors locally with ChromaDB.
- Ask natural-language questions about uploaded documents.
- Generate answers grounded in retrieved document passages.
- Display source documents, excerpts and page references when available.
- Maintain chat history during the current Streamlit session.

## Technology Stack

- Python
- Streamlit
- OpenAI API
- OpenAI embeddings (`text-embedding-3-small`)
- ChromaDB
- PyPDF
- python-docx
- python-dotenv

## Project Structure

```text
company-knowledge-chatbot/
├── app.py
├── rag.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd company-knowledge-chatbot
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a local `.env` file using `.env.example` as a template.

Add your own OpenAI API key:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Never commit your real `.env` file or publish your API key.

### 5. Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## How It Works

1. Upload documents through the Streamlit interface.
2. Extract text from each document.
3. Split the text into overlapping chunks.
4. Generate embeddings for the chunks.
5. Store embeddings, text and metadata in ChromaDB.
6. Embed each user question and retrieve relevant passages.
7. Send the retrieved context and question to the OpenAI response model.
8. Display the answer and retrieved source passages.

## Limitations

- Answers depend on the quality and relevance of retrieved document passages.
- Scanned PDFs may require OCR before indexing.
- OpenAI API usage may incur charges.
- The starter application is designed for local, single-user use and does not implement authentication or document-level access control.
- Only upload documents you are authorized to process.

## Future Improvements

- Incremental document indexing and deletion.
- Improved citation accuracy and retrieval evaluation.
- Authentication and document-level permissions.
- Deployment with secure configuration and monitoring.

## Security

API keys, local databases, virtual environments and confidential documents should remain outside version control. Use synthetic or public sample documents when demonstrating the project.
