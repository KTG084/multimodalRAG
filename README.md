# ChaterBox - Multimodal RAG Chatbot API

A FastAPI-based conversational AI system that processes PDFs and images, stores them in a vector database, and answers questions using retrieval-augmented generation (RAG).

## What It Does

This chatbot can:
- Upload and process PDF documents
- Analyze images using GPT-4 Vision
- Answer questions based on your uploaded content
- Maintain conversation history across sessions
- Search through documents intelligently using semantic search

## Tech Stack

- **FastAPI** - API framework
- **LangChain** - RAG orchestration
- **Pinecone** - Vector database for embeddings
- **OpenAI GPT-4** - Language model and vision analysis
- **PyPDF2 & PyMuPDF** - PDF text extraction
- **Tesseract OCR** - Image text extraction fallback

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key
- Tesseract OCR installed

### Installation

```bash
pip install fastapi uvicorn langchain langchain-openai langchain-pinecone
pip install python-multipart PyPDF2 pymupdf pillow pytesseract python-dotenv openai pinecone
```

### Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PORT=8000
```

### Run the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Create Session
```
POST /session/create
```
Returns a new session ID for tracking conversations.

### Upload File
```
POST /upload
```
Upload a PDF or image (JPEG, PNG, WebP). The file gets processed and stored in the vector database.

### Chat
```
POST /chat
Body: {
  "question": "your question",
  "session_id": "optional_session_id"
}
```
Ask questions about uploaded documents. Maintains conversation context.

### Combined Upload + Ask
```
POST /ask
Form data:
  - file: PDF or image (optional)
  - question: your question (optional)
```
Upload a file and ask a question in one request.

### Get Chat History
```
GET /session/{session_id}/history
```
Retrieve all messages from a session.

### Clear Session
```
DELETE /session/{session_id}
```
Delete a conversation session.

## How It Works

### PDF Processing
1. Extracts text using PyPDF2
2. Falls back to PyMuPDF if needed
3. Splits text into chunks
4. Creates embeddings and stores in Pinecone

### Image Processing
1. Preprocesses and optimizes the image
2. Sends to GPT-4 Vision for detailed description
3. Falls back to OCR if vision analysis is limited
4. Stores the description as searchable content

### Question Answering
1. Converts your question to embeddings
2. Retrieves relevant document chunks from Pinecone
3. Passes context to GPT-4 with conversation history
4. Returns answer with source references

## Example Usage

```python
import requests

# Create session
session = requests.post("http://localhost:8000/session/create").json()
session_id = session["session_id"]

# Upload document
files = {"file": open("document.pdf", "rb")}
requests.post("http://localhost:8000/upload", files=files)

# Ask question
response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "What is this document about?", "session_id": session_id}
).json()

print(response["answer"])
```

## Features

- **Multimodal Support** - Works with text documents and images
- **Context-Aware** - Remembers conversation history
- **Smart Retrieval** - Uses semantic search to find relevant information
- **OCR Fallback** - Extracts text from images when needed
- **Session Management** - Multiple concurrent conversations

## Notes

- Images are analyzed using GPT-4 Vision for detailed descriptions
- PDFs are chunked into 1000-character segments with 200-character overlap
- The system uses OpenAI's `text-embedding-3-large` model for embeddings
- Maximum 5 relevant chunks are retrieved per query

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`