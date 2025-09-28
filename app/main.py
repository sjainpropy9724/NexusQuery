from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid      # Used to generate unique IDs
import os
from .celery_worker import process_files_task
from . import processing, database
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# In-memory storage for the results (in a real app, this would be a persistent database)
# It's shared because both the celery worker and the main app import it
from .shared_db import get_kb, set_kb

# --- Pydantic Data Models ---
# These models define the structure of our API requests and responses.

class Message(BaseModel):
    role: str
    content: str

class IngestRequest(BaseModel):
    urls: Optional[List[str]] = Field(default_factory=list)

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = Field(default_factory=list)

class Source(BaseModel):
    # Will be used to show the source of the answer
    file_name: str
    content: str
    score: Optional[float] = None  # Making score as optional

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


# --------------- API Application ----------------

# Creating an instance of the FastAPI application
# This 'app' object is the main point of interaction for the API.
app = FastAPI(title="NexusQuery API")
database.initialize_db()

# Defining the temporary directory
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# ----------------- API Endpints ------------------

# The decorator '@app.get("/")' tells FastAPI that the function below
# should handle requests that come to the root URL ("/").

@app.get("/")
def read_root():
    """
    The function returns a JSON response
    FastAPI automatically converts Python dictionaries to JSON
    """
    return {"message": "Welcome to NexusQuery API!"}

@app.post("/knowledge-bases", status_code=201)
def create_knowledge_base():
    """
    Creates a knowledge base and returns a knowledge base (kb_id)
    """
    kb_id = str(uuid.uuid4())
    new_kb = {
        "id": kb_id, 
        "files":[], 
        "chunks": [],
        "status": "created"
    }
    set_kb(kb_id, new_kb)   # Created in Redis
    return {"knowledge_base_id": kb_id, 
            "message": "Knowledge base created successfully"}

@app.post("/knowledge-bases/{kb_id}/upload", status_code=202)
async def upload_sources(kb_id: str, files: List[UploadFile] = File(...), urls: List[str] = Form(default=[])):
    """
    Uploads multiples files, URLs to a specific knowledge base and then process
    them in background using Celery and Redis to be asynchronous
    """
    if not get_kb(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge Base not found.")
    
    sources_to_process = []

    # Handle URLs passed in the form
    for url in urls:
        if url:
            sources_to_process.append(url)
    
    # Handle uploaded files
    for file in files:
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        sources_to_process.append(file_path)

    if not sources_to_process:
        raise HTTPException(status_code=400, detail="No sources provided.")
    
    process_files_task.delay(kb_id, sources_to_process)

    return {
        "message": f"{len(sources_to_process)} sources(s) received. Processing has started in the background.",
        "knowledge_base_id": kb_id
    }


@app.post("/knowledge-bases/{kb_id}/query", response_model=QueryResponse)
def query_knowledge_base(kb_id: str, request: QueryRequest):
    """
    Queries a specific knowledge base
    (Logic for Hybrid Search and LLM Generation is here and in related modules)
    """
    current_kb = get_kb(kb_id)
    if not current_kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    if current_kb.get("status") != 'ready':
        raise HTTPException(status_code=400, detail=f"KB not ready. Status: {current_kb.get('status')}")
    
    try:
        database.add_message_to_history(kb_id, "user", request.query)
        # Get paths from Redis
        faiss_path = current_kb.get('faiss_path')
        bm25_path = current_kb.get('bm25_path')

        if not faiss_path or not bm25_path:
            raise HTTPException(status_code=500, detail="Index paths not found in knowledge base.")
        
        # Load the indices from the disk
        # For loading the index, we can use a special "dummy" embedding object that doesn't actually load the big model,
        # but still satisfies the FAISS.load_local requirement. LangChain provides FakeEmbeddings for exactly this purpose.
        dummy_embedder = FakeEmbeddings(size=384) # size must match our 'all-MiniLM-L6-v2' model
        vector_db = FAISS.load_local(
            faiss_path,
            dummy_embedder, 
            allow_dangerous_deserialization=True
        )
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)

        answer, sources_data = processing.query_and_generate(
            query = request.query,
            history=request.history,
            vector_db=vector_db,
            bm25_index=bm25_index,
            chunks=current_kb["chunks"]
        )

        database.add_message_to_history(kb_id, "assistant", answer)

        response_sources = [Source(**s) for s in sources_data]

        return QueryResponse(answer=answer, sources=response_sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")

@app.get("/knowledge-bases/{kb_id}/history", response_model=List[Message])
def get_chat_history(kb_id: str):
    """Retrieves the full chat history for a given knowledge base."""
    if not get_kb(kb_id): # Check if KB exists in Redis
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return database.get_history(kb_id)

@app.get("/knowledge-bases/{kb_id}/status")
def get_kb_status(kb_id: str):
    """
    Checks the processing status of a knowledge base
    """
    kb_data = get_kb(kb_id)  # Get kb_id from Redis
    if not kb_data:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    status = kb_data.get("status", "pending")  
    # It tries to get the value associated with the key "status". If that key doesn't exist for some reason,
    # instead of crashing, it will return the default value "pending"

    file_count = len(kb_data.get("files", []))
    # It tries to get the value associated with the key "files". If that key doesn't exist for some reason,
    # instead of crashing, it will return the default value [] (means empty list)

    return {
        "knowledge_base_id": kb_id,
        "status": status,
        "file_count": file_count
    }

