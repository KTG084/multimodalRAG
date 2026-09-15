# ChaterBox RAG Service

A stateless FastAPI microservice that handles retrieval-augmented generation for the ChaterBox app: it ingests PDFs and images into Pinecone, and answers questions by retrieving relevant chunks and streaming an LLM response.

It owns Pinecone only — no user table, no session table, no in-memory chat history. Every request carries everything it needs (query, history, user/document scope). A separate Next.js backend owns all persistent state (Postgres, auth, sessions) and is the only expected caller of this service, authenticated via a shared `X-Internal-Key` header.

## Tech Stack

- **FastAPI** — API framework
- **LangChain** — RAG orchestration
- **Pinecone** — vector database for embeddings
- **Groq** (`openai/gpt-oss-120b` by default) — chat generation
- **Google Gemini** — embeddings (`gemini-embedding-001`) and image descriptions (`gemini-3.5-flash-lite`)
- **PyPDF2 & PyMuPDF** — PDF text extraction (PyMuPDF as a fallback for text PyPDF2 can't extract)
- **Tesseract OCR** — image text extraction fallback

## Setup

### Prerequisites

- Python 3.11+
- Groq API key
- Google (Gemini) API key
- Pinecone API key + an existing index
- Tesseract OCR installed locally (`tesseract-ocr` on apt, or the Windows/macOS installer) — only needed for the OCR fallback path on image ingestion

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=chaterbox-embedding-index

# Shared secret with the Next.js backend — every request must send this
# back as the X-Internal-Key header.
INTERNAL_API_KEY=change_me_to_a_random_secret

# Optional: base URL of the Next.js app, used to push "document ready"
# webhooks after ingestion. If unset, /ingest still works, it just can't
# notify Next.js of completion.
NEXTJS_INTERNAL_URL=http://localhost:3000

PORT=8000
```

See [main.py](main.py) for the full list of tunable env vars (models, reasoning effort, embedding dimensions, history length, etc.) — all have sane defaults.

### Run the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## API Endpoints

All endpoints below (except `/`, `/health`) require an `X-Internal-Key` header matching `INTERNAL_API_KEY`.

### Ingest a Document
```
POST /ingest
Body: {
  "document_id": "...",
  "user_id": "...",
  "file_url": "https://...",      // publicly reachable URL, downloaded server-side
  "filename": "report.pdf",
  "content_type": "application/pdf"  // or image/jpeg, image/png, image/jpg, image/webp
}
```
Downloads the file, extracts/describes its content, chunks it, embeds it, and stores it in Pinecone. Best-effort notifies `NEXTJS_INTERNAL_URL` of the resulting status (`ready` / `failed`).

### Query (streaming)
```
POST /query
Body: {
  "query": "your question",
  "user_id": "...",
  "history": [{"role": "user" | "assistant", "content": "..."}],
  "document_ids": ["..."] | null,   // null = search the user's whole library; [] = no documents in scope
  "generate_title": false            // true only on the first message of a new session
}
```
Returns a Server-Sent Events stream: a `sources` event, an optional `title` event, then `token` events, then `done` (or `error` if something fails mid-stream).

### Delete a Document
```
DELETE /documents/{document_id}?user_id=...
```
Deletes every vector belonging to that document for that user.

### Health check
```
GET /health
```

## How It Works

### PDF Ingestion
1. Extracts text per page with PyPDF2; falls back to PyMuPDF if no page yields text.
2. Splits each page into ~1000-character chunks (200-character overlap), keeping chunks scoped to a single page so answers can cite a page number.

### Image Ingestion
1. Preprocesses the image (flattens transparency, downscales to max 2048px, re-encodes as JPEG).
2. Sends it to Gemini for a structured description.
3. Falls back to Tesseract OCR if the description looks too short/unusable, appending any extracted text.

### Query Answering
1. If there's chat history, rewrites the query into a standalone question with a small/fast Groq model.
2. Embeds it and retrieves the top 5 matching chunks from Pinecone, scoped to the user (and optionally specific documents).
3. Streams a Groq-generated answer, grounded in the retrieved context when relevant, falling back to the model's own knowledge otherwise.

## Deploying (Railway)

This repo is set up to deploy on Railway without Docker:

- **`nixpacks.toml`** tells Railway's Nixpacks builder to `apt-get install tesseract-ocr` during the build, since `pytesseract` needs the system binary, not just the Python package.
- **`Procfile`** defines the start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **`railway.toml`** points Railway at the Nixpacks builder and configures a `/health` health check.

Steps:
1. Push this repo to GitHub and create a new Railway project from it (or `railway up` from the CLI).
2. Set the environment variables listed above in the Railway dashboard (`.env` is not deployed).
3. Deploy. Railway builds via Nixpacks (installing Tesseract per `nixpacks.toml`), then runs the Procfile's start command.

If this service and its Next.js caller both run on Railway, put them in the same project so they can reach each other over Railway's private network instead of the public internet.

## Local Smoke Test

[test_system.py](test_system.py) exercises the full ingest → query → delete flow against a running local instance:

```bash
python main.py            # in one terminal
python test_system.py     # in another
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
