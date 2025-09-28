from celery import Celery
import os
from . import processing
import pickle   # This is thing which is giving us "Persistance" by having a shared
                # storage of indices where indices are serialized. They are being deserialized
                # whenever required.

# In-memory storage for the results (in a real app, this would be a persistent database)
# It's shared because both the celery worker and the main app import it
from .shared_db import get_kb, set_kb

# Defining the Redis URL for Celery to use as a message broker
# This assumes Redis is running on the default localhost port
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')

# Create a Celery Application instance
celery_app = Celery('tasks', broker=CELERY_BROKER_URL)

@celery_app.task
def process_files_task(kb_id: str, sources: list[str]):
    """
    A celery task to process uploaded files in the background.
    """
    print(f"Celery worker recieved task for kb_id: {kb_id}")

    current_kb = get_kb(kb_id)   # Get the knowledge base from Redis
    if not current_kb:
        print(f"Error: Knowledge Base {kb_id} not found.")
        return
    
    current_kb['status'] = 'processing'
    set_kb(kb_id, current_kb)   # Updates status in Redis

    try:
        for source in sources:
            try:
                raw_text = processing.get_text(source)
                new_chunks = processing.chunk_text(raw_text)
                
                current_kb["chunks"].extend(new_chunks)
                current_kb["files"].append(os.path.basename(source))
            except Exception as e:
                current_kb['status'] = 'failed'
                print(f"Error processing file {source}: {e}. Task failed for {kb_id}")
            finally:
                if os.path.exists(source):
                    os.remove(source)
            
        # After processing all files, indices will be built
        if current_kb["chunks"]:
            # Build the indices
            vector_db, bm25_index = processing.build_hybrid_indices(current_kb["chunks"])
            
            # Define paths to save the indices
            index_folder = os.path.join("indices", kb_id)
            os.makedirs(index_folder, exist_ok=True)
            faiss_path = os.path.join(index_folder, "faiss_index")
            bm25_path = os.path.join(index_folder, "bm25_index.pkl")

            # Save the indices to disk
            vector_db.save_local(faiss_path)
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25_index, f)
            
            # Store the PATHS in Redis, not the objects
            current_kb["faiss_path"] = faiss_path
            current_kb["bm25_path"] = bm25_path

        current_kb['status'] = 'ready'
        print(f"Task for kb_id: {kb_id} completed successfully. Total chunks: {len(current_kb['chunks'])}")

    except Exception as e:
        # Mark the job as failed.
        current_kb['status'] = 'failed'
        print(f"Task for kb_id: {kb_id} failed. Error: {e}")
        return False
    
    set_kb(kb_id, current_kb)   # Save the final state to Redis
    return True
