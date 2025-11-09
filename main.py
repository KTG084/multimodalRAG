from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from datetime import datetime
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import base64
import fitz
import PyPDF2
import io
import os
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import pytesseract

load_dotenv()


class UploadResponse(BaseModel):
    status: str
    message: str
    doc_count: int
    filename: str
    description: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    session_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]


class ChatMessage(BaseModel):
    role: str
    content: str
    timestemp: Optional[str] = None


class ChatHistoryManager:
    """Manages chat sessions"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def create_session(self) -> str:
        """Create New Session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "uploaded_files": [],
        }
        return session_id

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Add message to history"""
        if session_id not in self.sessions:
            self.create_session()

        message = ChatMessage(
            role=role, content=content, timestemp=datetime.now().isoformat()
        )

        self.sessions[session_id]["messages"].append(message)

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Get session history"""
        return self.sessions.get(session_id, {}).get("messages", [])

    def clear_session(self, session_id: str):
        """Clear session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def to_langchain_messages(self, session_id: str):
        """Convert to LangChain format"""
        from langchain_core.messages import HumanMessage, AIMessage

        messages = []
        history = self.get_history(session_id)

        for msg in history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))

        return messages


class Chater:
    def __init__(self):
        self._embeddings = None
        self._vectorStore = None
        self._llm = None
        self._vision_client = None
        self.chat_history_manager = ChatHistoryManager()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        return self._embeddings

    @property
    def vectorStore(self):
        if self._vectorStore is None:
            pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = "chaterbox-embedding-index"
            self._vectorStore = PineconeVectorStore(
                index_name=index_name, embedding=self.embeddings
            )
        return self._vectorStore

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                temperature=0.5,
                model="gpt-4o",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        return self._llm

    @property
    def vision_client(self):
        """Separate client for vision tasks"""
        if self._vision_client is None:
            self._vision_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._vision_client

    def preprocess_image(self, image_bytes: bytes) -> tuple[bytes, dict]:
        """Preprocess image for optimal vision model performance"""
        try:
            image = Image.open(io.BytesIO(image_bytes))

            metadata = {
                "original_size": image.size,
                "format": image.format,
                "mode": image.mode,
            }

            if image.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGBA", image.size, (255, 255, 255))
                if image.mode == "P":
                    image = image.convert("RGBA")
                background.paste(
                    image, mask=image.split()[-1] if image.mode == "RGBA" else None
                )
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
            return image_bytes, {"error": str(e)}

    def describe_image_with_gpt4v(self, image_bytes: bytes, filename: str) -> str:
        """Use GPT-4v to generate detailed image description"""
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

            response = self.vision_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": enhanced_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
                temperature=0.2,
            )

            description = response.choices[0].message.content
            metadata = {
                "filename": filename,
                "model": "gpt-4o",
                "tokens_used": response.usage.total_tokens,
                "image_metadata": img_metadata,
                "timestamp": datetime.now().isoformat(),
            }

            return description, metadata
        except Exception:
            return f"Image: {filename} (description unavailable)", {}

    def extract_text_with_ocr_fallback(self, image_bytes: bytes) -> str:
        """Fallback OCR extraction if GPT-4v fails"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception:
            return ""

    def process_image(self, image_bytes: bytes, filename: str) -> list:
        """Process image with GPT-4V and OCR fallback"""
        try:
            description, metadata = self.describe_image_with_gpt4v(image_bytes, filename)

            if len(description) < 100 or "unable to view" in description.lower():
                ocr_text = self.extract_text_with_ocr_fallback(image_bytes)
                if ocr_text:
                    description += f"\n\n[OCR Extracted Text]:\n{ocr_text}"

            image = Image.open(io.BytesIO(image_bytes))
            if max(image.size) > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                thumb_buffer = io.BytesIO()
                image.save(thumb_buffer, format="JPEG", quality=85)
                img_base64 = base64.b64encode(thumb_buffer.getvalue()).decode("utf-8")
            else:
                img_base64 = base64.b64encode(image_bytes).decode("utf-8")

            document = Document(
                page_content=description,
                metadata={
                    "source": filename,
                    "type": "image",
                    "content_preview": description[:200],
                    "timestamp": datetime.now().isoformat(),
                    # Flatten metadata - Pinecone compatible
                    "model": metadata.get("model", "gpt-4o"),
                    "tokens_used": metadata.get("tokens_used", 0),
                    "image_format": str(metadata.get("image_metadata", {}).get("format", "unknown")),
                    "image_mode": str(metadata.get("image_metadata", {}).get("mode", "unknown")),
                    "description_length": len(description),
                }
            )

            return [document], description
        except Exception as e:
            raise ValueError(f"Error processing image {filename}: {str(e)}")

    def extract_text_from_pdf(self, pdf_stream: io.BytesIO) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            if not text.strip():
                pdf_stream.seek(0)
                pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text += page.get_text() + "\n\n"
                pdf_document.close()

            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")

    def process_pdf(self, pdf_stream: io.BytesIO, filename: str) -> list:
        """Process PDF into chunks"""
        text = self.extract_text_from_pdf(pdf_stream)
        if not text:
            raise ValueError("No text extracted from PDF")

        chunks = self._text_splitter.split_text(text)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "type": "pdf",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        return documents

    def query_rag(self, question: str) -> dict:
        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 5})

        system_prompt = """You are a helpful AI assistant. Use the following context to answer the question.
If you don't know the answer based on the context, say so.

Context: {context}"""

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        result = rag_chain.invoke({"input": question})

        sources = []
        if "context" in result:
            for doc in result["context"]:
                sources.append(
                    {
                        "content": doc.page_content[:200],
                        "source": doc.metadata.get("source", "unknown"),
                        "type": doc.metadata.get("type", "unknown"),
                    }
                )

        return {"answer": result["answer"], "sources": sources}

    def create_rag_chain_with_history(self):
        """Create history-aware RAG Chain"""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given chat history and new question, reformulate as standalone question.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.vectorStore.as_retriever(search_kwargs={"k": 5}),
            contextualize_q_prompt,
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Use context to answer. Be concise.\n\nContext: {context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    def query_with_history(self, question: str, session_id: str) -> dict:
        """Query with chat history"""
        lc_history = self.chat_history_manager.to_langchain_messages(session_id)
        rag_chain = self.create_rag_chain_with_history()

        result = rag_chain.invoke({"input": question, "chat_history": lc_history})
        self.chat_history_manager.add_message(session_id, "user", question)
        self.chat_history_manager.add_message(session_id, "assistant", result["answer"])

        sources = []
        if "context" in result:
            for doc in result["context"]:
                source_info = {
                    "content": doc.page_content[:300],
                    "source": doc.metadata.get("source", "unknown"),
                }

                if doc.metadata.get("type") == "image" and "image_thumbnail" in doc.metadata:
                    source_info["has_image"] = True
                    source_info["content_preview"] = doc.metadata.get(
                        "content_preview", ""
                    )

                sources.append(source_info)

        return {"answer": result["answer"], "sources": sources}


chater = Chater()

app = FastAPI(
    title="ChatBOT",
    version="1.0",
    description="Advanced Chatbot",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "RAG System is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/session/create")
async def create_session():
    session_id = chater.chat_history_manager.create_session()
    return {"session_id": session_id}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    history = chater.chat_history_manager.get_history(session_id)
    return {"session_id": session_id, "history": history}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    chater.chat_history_manager.clear_session(session_id)
    return {"message": "Session cleared", "session_id": session_id}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf_or_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_type = file.content_type
        description = None

        if file_type == "application/pdf":
            pdf_stream = io.BytesIO(content)
            documents = chater.process_pdf(pdf_stream, file.filename)
        elif file_type in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            documents, description = chater.process_image(content, file.filename)
        else:
            raise HTTPException(400, f"Unsupported file: {file.content_type}")

        chater.vectorStore.add_documents(documents)

        return UploadResponse(
            status="success",
            message=f"Processed {file.filename}",
            doc_count=len(documents),
            filename=file.filename,
            description=description[:500] if description else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    try:
        if not request.session_id:
            request.session_id = chater.chat_history_manager.create_session()

        result = chater.query_with_history(request.question, request.session_id)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ask")
async def ask(file: Optional[UploadFile] = File(None), question: Optional[str] = Form(None)):
    try:
        if not file and not question:
            raise HTTPException(400, "Provide file or question")

        session_id = chater.chat_history_manager.create_session()

        if file:
            content = await file.read()
            file_type = file.content_type

            if file_type == "application/pdf":
                pdf_stream = io.BytesIO(content)
                documents = chater.process_pdf(pdf_stream, file.filename)
            elif file_type in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
                documents, description = chater.process_image(content, file.filename)
            else:
                raise HTTPException(400, f"Unsupported file: {file.content_type}")

            chater.vectorStore.add_documents(documents)

        if question:
            result = chater.query_with_history(question, session_id)
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "session_id": session_id,
            }

        return {"message": "File uploaded", "session_id": session_id}

    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn

    PORT = int(os.getenv("PORT", 8000))
    print(f"\nServer starting on http://0.0.0.0:{PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
