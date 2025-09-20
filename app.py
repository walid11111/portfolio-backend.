from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_utils import get_rag_chain
import asyncio
import logging
from functools import lru_cache
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio Chatbot API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://portfolio-frontend-5t25.vercel.app"],  # Updated with Vercel frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    query: str

# Initialize RAG chain
try:
    logger.debug("Initializing RAG chain...")
    rag_chain = get_rag_chain()
    logger.debug("RAG chain initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG chain: {str(e)}")
    raise

def preprocess_query(query: str) -> str:
    """Normalize and correct typos in the query."""
    # Convert to lowercase
    query = query.lower().strip()
    
    # Common typo corrections
    typo_corrections = {
        r'\bwhats\b': 'what is',
        r'\bhows\b': 'how is',
        r'\bwheres\b': 'where is',
        r'\bwhos\b': 'who is',
        r'\bcannt\b': 'cannot',
        r'\bdont\b': 'do not',
        r'\baint\b': "ain't"
    }
    
    for pattern, replacement in typo_corrections.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    return query

@lru_cache(maxsize=100)
def cached_rag_invoke(query: str) -> str:
    """Invoke RAG chain with caching for repeated queries."""
    return rag_chain.invoke({"question": query})["answer"]

async def stream_response(query: str):
    """Async generator that yields tokens progressively."""
    try:
        logger.debug(f"Processing query: {query}")
        processed_query = preprocess_query(query)
        result = cached_rag_invoke(processed_query)
        word_count = len(result.split())
        sleep_time = 0.03 if word_count > 50 else 0.05
        for word in result.split():
            yield word + " "
            await asyncio.sleep(sleep_time)  # Adjustable typing speed
        logger.debug("Query processed successfully")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        yield f"Error: {str(e)} "

@app.post("/chat")
async def chat(request: ChatRequest):
    """Endpoint to process user queries with RAG, streaming word-by-word."""
    if not request.query.strip():
        logger.warning("Received empty query")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.debug(f"Received chat request: {request.query}")
    return StreamingResponse(
        stream_response(request.query),
        media_type="text/plain"
    )

@app.get("/test")
async def test():
    """Test endpoint to verify backend is running."""
    logger.debug("Test endpoint called")
    return {"message": "Backend is running"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)