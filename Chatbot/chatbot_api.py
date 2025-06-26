import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
#include key
import os
from .data import documents
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage

# Load Hugging Face API key from environment variable
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
if not HUGGINGFACE_API_KEY:
    raise RuntimeError('HUGGINGFACE_API_KEY environment variable is not set. Please set it to your Hugging Face API token.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PharaohGuide Chatbot API",
    description="API for PharaohGuide RAG Chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals for chatbot
faiss_db = None
chat_model = None

class ChatRequest(BaseModel):
    query: str
    k: int = 5

@app.on_event("startup")
async def startup_event():
    global faiss_db, chat_model
    try:
        logger.info("Initializing RAG components...")
        docs = [Document(page_content=text) for text in documents]
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_db = FAISS.from_documents(docs, embedding_model)
        llm = HuggingFaceEndpoint(repo_id="microsoft/Phi-3.5-mini-instruct", task="text_generation")
        chat_model = ChatHuggingFace(llm=llm)
        logger.info("RAG components initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        raise e

@app.get("/")
async def root():
    return {
        "message": "PharaohGuide Chatbot API",
        "status": "running",
        "endpoints": {
            "/chat": "Chat with the PharaohGuide RAG chatbot",
            "/health": "Check API health",
            "/config": "Get current configuration"
        }
    }

@app.get("/health")
async def health_check():
    if faiss_db is None or chat_model is None:
        return {"status": "error", "message": "Chatbot not initialized"}
    return {"status": "healthy", "message": "Chatbot ready"}

@app.get("/config")
async def get_config():
    return {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "microsoft/Phi-3.5-mini-instruct",
        "num_documents": len(documents)
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if faiss_db is None or chat_model is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    query = request.query
    k = request.k
    try:
        context_docs = faiss_db.similarity_search(query, k)
        context_text = "\n".join([doc.page_content for doc in context_docs])
        messages = [
            SystemMessage(content="You are a scientist"),
            HumanMessage(content=f"Answer the question: {query}\nBased on the context:\n{context_text}")
        ]
        response = chat_model.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot_api:app", host="0.0.0.0", port=8080, reload=True, log_level="info") 