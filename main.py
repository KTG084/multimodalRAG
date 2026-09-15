"""RAG microservice. Stateless — owns Pinecone only, no user/session tables.
Called only by the Next.js backend, authenticated via X-Internal-Key."""

from datetime import datetime
from typing import Dict, List, Optional
import asyncio
import hmac
import io
import json
import logging
import os

import base64
import fitz
import httpx
import PyPDF2
import pytesseract
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from PIL import Image
from pinecone import Pinecone as PineconeClient
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag_service")

logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
# Empty string disables reasoning_effort for non-reasoning models, which reject it.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
# Must match the dimension the Pinecone index was created with.
GEMINI_EMBEDDING_DIMENSIONS = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "3072"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chaterbox-embedding-index")

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
NEXTJS_INTERNAL_URL = os.getenv("NEXTJS_INTERNAL_URL")

MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_HISTORY_TURNS = 10
IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


async def verify_internal_key(x_internal_key: Optional[str] = Header(None)):
    if not INTERNAL_API_KEY or not hmac.compare_digest(x_internal_key or "", INTERNAL_API_KEY):
        raise HTTPException(401, "Invalid or missing internal API key")


class IngestRequest(BaseModel):
    document_id: str
    user_id: str
    file_url: str
    filename: str
    content_type: str


class IngestResponse(BaseModel):
    status: str
    document_id: str
    chunk_count: int


class HistoryTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    user_id: str
    history: List[HistoryTurn] = []
    document_ids: Optional[List[str]] = None  # None => search the user's whole library
    generate_title: bool = False  # true only on the first message of a session


class DeleteResponse(BaseModel):
    status: str
    document_id: str
    deleted_count: int


def sse_event(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


def extract_text(content) -> str:
    """Gemini can return content as a list of blocks (text/signature/etc.)
    instead of a plain string when thinking is enabled."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


def user_facing_error(e: Exception) -> str:
    text = str(e)
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "rate_limit_exceeded" in text:
        return "The AI service's rate limit or daily quota was reached. Please try again later."
    if "invalid_api_key" in text:
        return "The AI service rejected its API key. Check the RAG service configuration."
    if isinstance(e, ValueError):
        return text
    return "Something went wrong while processing your request. Please try again."


def require_groq_api_key() -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set on the RAG service.")
    return GROQ_API_KEY


async def download_file(url: str) -> bytes:
    chunks: List[bytes] = []
    total = 0

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"File exceeds {MAX_DOWNLOAD_BYTES} byte limit")
                chunks.append(chunk)

    return b"".join(chunks)


async def notify_ingest_status(
    document_id: str, status: str, chunk_count: Optional[int] = None, error: Optional[str] = None
):
    if not NEXTJS_INTERNAL_URL:
        return

    payload: Dict = {"status": status}
    if chunk_count is not None:
        payload["chunk_count"] = chunk_count
    if error is not None:
        payload["error"] = error

    url = f"{NEXTJS_INTERNAL_URL}/api/internal/documents/{document_id}/status"
    headers = {"X-Internal-Key": INTERNAL_API_KEY or ""}
    attempts = 3

    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            return
        except httpx.HTTPError as e:
            logger.warning(
                "Ingest-status callback attempt %d/%d failed for document %s: %s",
                attempt, attempts, document_id, e,
            )
            if attempt < attempts:
                await asyncio.sleep(2 ** (attempt - 1))

    logger.error(
        "Ingest-status callback permanently failed for document %s after %d attempts - "
        "Next.js will never see this '%s' status unless it polls separately.",
        document_id, attempts, status,
    )


class RagService:
    def __init__(self):
        self._embeddings = None
        self._vectorStore = None
        self._llm = None
        self._fast_llm = None
        self._vision_llm = None
        self._pinecone_index = None
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=GEMINI_EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY,
                output_dimensionality=GEMINI_EMBEDDING_DIMENSIONS,
            )
        return self._embeddings

    @property
    def vectorStore(self):
        if self._vectorStore is None:
            self._vectorStore = PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME, embedding=self.embeddings
            )
        return self._vectorStore

    @property
    def pinecone_index(self):
        if self._pinecone_index is None:
            pc = PineconeClient(api_key=PINECONE_API_KEY)
            self._pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        return self._pinecone_index

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatGroq(
                model_name=GROQ_CHAT_MODEL,
                groq_api_key=require_groq_api_key(),
                temperature=0.5,
                max_tokens=4096,
                reasoning_effort=GROQ_REASONING_EFFORT or None,
                streaming=True,
                max_retries=2,
            )
        return self._llm

    @property
    def fast_llm(self):
        if self._fast_llm is None:
            self._fast_llm = ChatGroq(
                model_name=GROQ_FAST_MODEL,
                groq_api_key=require_groq_api_key(),
                temperature=0,
                max_tokens=256,
                reasoning_effort=GROQ_REASONING_EFFORT or None,
                max_retries=2,
            )
        return self._fast_llm

    @property
    def vision_llm(self):
        if self._vision_llm is None:
            self._vision_llm = ChatGoogleGenerativeAI(
                model=GEMINI_VISION_MODEL,
                google_api_key=GOOGLE_API_KEY,
                thinking_level="low",
                max_output_tokens=4096,
                max_retries=2,
            )
        return self._vision_llm

    def preprocess_image(self, image_bytes: bytes) -> tuple[bytes, dict]:
        try:
            image = Image.open(io.BytesIO(image_bytes))

            metadata = {
                "original_size": image.size,
                "format": image.format,
                "mode": image.mode,
            }

            if image.mode in ("RGBA", "LA", "P"):
                # JPEG has no alpha channel: flatten onto an RGB white background.
                image = image.convert("RGBA")
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            max_dimension = 2048
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                metadata["resized"] = True
                metadata["new_size"] = new_size

            output = io.BytesIO()
            image.save(output, format="JPEG", quality=95, optimize=True)
            processed_bytes = output.getvalue()

            return processed_bytes, metadata
        except Exception as e:
            logger.warning("Image preprocessing failed, using original bytes: %s", e)
            return image_bytes, {"error": str(e)}

    def describe_image_with_gemini(self, image_bytes: bytes, filename: str) -> tuple[str, dict]:
        try:
            processed_bytes, img_metadata = self.preprocess_image(image_bytes)
            img_base64 = base64.b64encode(processed_bytes).decode("utf-8")

            enhanced_prompt = """You are an expert at analyzing images for a document search and retrieval system.

Analyze this image comprehensively and provide:

1. Document Type
2. Main Content
3. Text Content
4. Visual Elements
5. Context and Purpose
6. Key Information

Be detailed and structured."""

            response = self.vision_llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": enhanced_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                        ]
                    )
                ]
            )

            description = extract_text(response.content)
            usage = getattr(response, "usage_metadata", None) or {}
            metadata = {
                "filename": filename,
                "model": GEMINI_VISION_MODEL,
                "tokens_used": usage.get("total_tokens", 0),
                "image_metadata": img_metadata,
                "timestamp": datetime.now().isoformat(),
            }

            return description, metadata
        except Exception:
            logger.exception("Gemini vision description failed for %s", filename)
            return f"Image: {filename} (description unavailable)", {}

    def extract_text_with_ocr_fallback(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.warning("OCR fallback failed: %s", e)
            return ""

    def process_image(self, image_bytes: bytes, filename: str) -> List[Document]:
        try:
            description, metadata = self.describe_image_with_gemini(image_bytes, filename)

            if len(description) < 100 or "unable to view" in description.lower():
                ocr_text = self.extract_text_with_ocr_fallback(image_bytes)
                if ocr_text:
                    description += f"\n\n[OCR Extracted Text]:\n{ocr_text}"

            document = Document(
                page_content=description,
                metadata={
                    "source": filename,
                    "type": "image",
                    "content_preview": description[:200],
                    "timestamp": datetime.now().isoformat(),
                    "model": metadata.get("model", GEMINI_VISION_MODEL),
                    "tokens_used": metadata.get("tokens_used", 0),
                    "image_format": str(metadata.get("image_metadata", {}).get("format", "unknown")),
                    "image_mode": str(metadata.get("image_metadata", {}).get("mode", "unknown")),
                    "description_length": len(description),
                },
            )

            return [document]
        except Exception as e:
            raise ValueError(f"Error processing image {filename}: {str(e)}")

    def extract_pages_from_pdf(self, pdf_stream: io.BytesIO) -> List[str]:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            pages = [page.extract_text() or "" for page in pdf_reader.pages]

            if not any(page_text.strip() for page_text in pages):
                pdf_stream.seek(0)
                pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
                pages = [pdf_document[i].get_text() for i in range(pdf_document.page_count)]
                pdf_document.close()

            return pages
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")

    def process_pdf(self, pdf_stream: io.BytesIO, filename: str) -> List[Document]:
        pages = self.extract_pages_from_pdf(pdf_stream)
        if not any(page_text.strip() for page_text in pages):
            raise ValueError("No text extracted from PDF")

        documents = []
        for page_num, page_text in enumerate(pages):
            if not page_text.strip():
                continue
            for i, chunk in enumerate(self._text_splitter.split_text(page_text)):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "type": "pdf",
                            "page": page_num + 1,
                            "chunk_index": i,
                        },
                    )
                )
        return documents

    def delete_document(self, document_id: str, user_id: str) -> int:
        # Serverless Pinecone indexes don't support delete(filter=...), so
        # find matching ids via a filtered query first.
        index = self.pinecone_index
        dimension = index.describe_index_stats().dimension
        zero_vector = [0.0] * dimension

        results = index.query(
            vector=zero_vector,
            filter={"document_id": document_id, "user_id": user_id},
            top_k=10000,
            include_values=False,
        )
        ids = [match.id for match in results.matches]
        if ids:
            index.delete(ids=ids)
        return len(ids)

    @staticmethod
    def make_title(query: str, max_length: int = 50) -> str:
        title = " ".join(query.split())
        if len(title) <= max_length:
            return title or "New chat"
        cut = title[:max_length].rsplit(" ", 1)[0] or title[:max_length]
        return cut.rstrip(" ,.;:-") + "..."

    async def stream_query(
        self,
        query: str,
        user_id: str,
        history: List[HistoryTurn],
        document_ids: Optional[List[str]],
        generate_title: bool = False,
    ):
        """Yields SSE events: sources, optional title, token(s), then done/error."""
        try:
            lc_history = [
                HumanMessage(content=turn.content) if turn.role == "user" else AIMessage(content=turn.content)
                for turn in history[-MAX_HISTORY_TURNS:]
            ]

            # An empty (but non-None) document_ids means "this chat has no ready
            # documents yet" — that must return no context, not fall through to
            # searching the user's whole library across every other chat.
            if document_ids is not None and len(document_ids) == 0:
                docs = []
            else:
                filter_: Dict = {"user_id": user_id}
                if document_ids:
                    filter_["document_id"] = {"$in": document_ids}

                retriever = self.vectorStore.as_retriever(search_kwargs={"k": 5, "filter": filter_})

                search_query = query
                if lc_history:
                    contextualize_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "Given the chat history and a follow-up question, rephrase the "
                                "follow-up as a standalone question. Return only the question.",
                            ),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    rewritten = await (contextualize_prompt | self.fast_llm).ainvoke(
                        {"input": query, "chat_history": lc_history}
                    )
                    # Must be a plain str: LangChain's text parsers return a str subclass
                    # (TextAccessor) that the Gemini embeddings API rejects with a 500.
                    search_query = str(extract_text(rewritten.content)).strip() or query

                docs = await retriever.ainvoke(search_query)

            sources = [
                {
                    "content": doc.page_content[:300],
                    "source": doc.metadata.get("source", "unknown"),
                    "type": doc.metadata.get("type", "unknown"),
                    "document_id": doc.metadata.get("document_id"),
                    # Pinecone returns stored numbers as floats (5.0)
                    "page": int(doc.metadata["page"]) if doc.metadata.get("page") is not None else None,
                }
                for doc in docs
            ]
            yield sse_event("sources", {"sources": sources})

            if generate_title:
                yield sse_event("title", {"title": self.make_title(query)})

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful, knowledgeable assistant. If document context is "
                        "given below, use it to answer when it's relevant and prefer it over "
                        "your own knowledge if the two conflict. If the context doesn't contain "
                        "the answer — or none is given, because this chat has no documents — "
                        "answer normally from your own knowledge instead. Never refuse just "
                        "because nothing was retrieved.\n\n"
                        "Context:\n{context}",
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            chain = qa_prompt | self.llm
            context_text = "\n\n".join(doc.page_content for doc in docs)

            async for chunk in chain.astream(
                {"input": query, "chat_history": lc_history, "context": context_text}
            ):
                text = extract_text(chunk.content)
                if text:
                    yield sse_event("token", {"content": text})

            yield sse_event("done", {})
        except Exception as e:
            logger.exception("stream_query failed for user %s", user_id)
            yield sse_event("error", {"message": user_facing_error(e)})


rag_service = RagService()

app = FastAPI(
    title="RAG Service",
    version="2.0",
    description="Stateless multimodal retrieval/ingestion microservice — called only by the Next.js backend.",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/")
async def root():
    return {"message": "RAG service is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_internal_key)])
async def ingest(request: IngestRequest):
    try:
        content = await download_file(request.file_url)

        if request.content_type == "application/pdf":
            documents = await asyncio.to_thread(
                rag_service.process_pdf, io.BytesIO(content), request.filename
            )
        elif request.content_type in IMAGE_CONTENT_TYPES:
            documents = await asyncio.to_thread(
                rag_service.process_image, content, request.filename
            )
        else:
            raise HTTPException(400, f"Unsupported content type: {request.content_type}")

        for document in documents:
            document.metadata["user_id"] = request.user_id
            document.metadata["document_id"] = request.document_id

        await asyncio.to_thread(rag_service.vectorStore.add_documents, documents)

        logger.info(
            "Ingested document %s (%d chunks) for user %s",
            request.document_id, len(documents), request.user_id,
        )
        await notify_ingest_status(request.document_id, "ready", chunk_count=len(documents))
        return IngestResponse(status="ready", document_id=request.document_id, chunk_count=len(documents))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion failed for document %s", request.document_id)
        message = user_facing_error(e)
        await notify_ingest_status(request.document_id, "failed", error=message)
        raise HTTPException(500, message)


@app.post("/query", dependencies=[Depends(verify_internal_key)])
async def query(request: QueryRequest):
    return StreamingResponse(
        rag_service.stream_query(
            request.query,
            request.user_id,
            request.history,
            request.document_ids,
            request.generate_title,
        ),
        media_type="text/event-stream",
    )


@app.delete(
    "/documents/{document_id}", response_model=DeleteResponse, dependencies=[Depends(verify_internal_key)]
)
async def delete_document(document_id: str, user_id: str):
    deleted_count = rag_service.delete_document(document_id, user_id)
    return DeleteResponse(status="deleted", document_id=document_id, deleted_count=deleted_count)


if __name__ == "__main__":
    import uvicorn

    PORT = int(os.getenv("PORT", 8000))
    RELOAD = os.getenv("RELOAD", "false").lower() == "true"
    print(f"\nRAG service starting on http://0.0.0.0:{PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=RELOAD)
