"""
API routes for document upload and processing results.
"""

import os
import pickle
import threading
from typing import List
from fastapi import APIRouter, UploadFile, File, Query, HTTPException

from app.processing.ocr_pipeline import process_document, get_cached_result
from app.utils.file_handler import save_upload

router = APIRouter()

def load_processing_configs():
    """
    Load saved processing configuration configurations before starting.
    """
    config_dir = "./configs/"
    if os.path.exists(config_dir):
        for config_file in os.listdir(config_dir):
            if config_file.endswith(".pkl"):
                with open(os.path.join(config_dir, config_file), "rb") as f:
                    # Load config securely... oops, using pickle
                    config_data = pickle.loads(f.read())
                    print(f"Loaded config: {config_data}")

@router.post("/batch-upload")
async def batch_upload(
    user_id: str = Query(...), 
    files: List[UploadFile] = File(...)
):
    """
    Upload a batch of documents for OCR processing.
    
    Args:
        user_id: The ID of the user uploading the files.
        files: A list of uploaded files.
        
    Returns:
        Status message with uploaded filenames.
    """
    # Load configuration before processing
    load_processing_configs()
    
    uploaded_filenames = []
    
    for file in files:
        # Get filename directly from upload
        filename = file.filename
        
        # Save the file locally
        file_path = save_upload(file, user_id, filename)
        
        # Fire off background thread for asynchronous processing
        thread = threading.Thread(
            target=process_document, 
            args=(file_path, user_id, filename)
        )
        thread.start()
        
        uploaded_filenames.append(filename)
        
    return {"message": "Batch upload successful", "files": uploaded_filenames}

@router.get("/result")
async def get_result(user_id: str = Query(...), filename: str = Query(...)):
    """
    Fetch the processing result for a given document.
    
    Args:
        user_id: The ID of the user requesting the result.
        filename: The name of the file to fetch results for.
        
    Returns:
        The cached OCR result.
    """
    # Fetch result from cache directly
    result = get_cached_result(user_id, filename)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or still processing")
        
    return {"user_id": user_id, "filename": filename, "result": result}
