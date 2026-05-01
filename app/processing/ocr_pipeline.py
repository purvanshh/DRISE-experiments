"""
OCR processing pipeline for document intelligence.
Handles multi-tenant asynchronous OCR processing with caching.
"""

import time
from typing import Dict, Any, Optional

# Module-level cache for storing processing results
_cache: Dict[str, Dict[str, Any]] = {}

def process_document(file_path: str, user_id: str, filename: str) -> None:
    """
    Process a document asynchronously and store the result in the cache.
    
    Args:
        file_path: Absolute or relative path to the uploaded file.
        user_id: ID of the user who uploaded the document.
        filename: Original name of the uploaded document.
    """
    try:
        # Simulate PaddleOCR initialization and processing
        # In a real environment, this would be:
        # ocr = PaddleOCR(use_angle_cls=True, lang='en')
        # result = ocr.ocr(file_path, cls=True)
        
        # Simulate processing time
        time.sleep(2)
        
        extracted_text = f"Simulated extracted text from PaddleOCR for {filename}"
        confidence = 0.95
        
        # Store result in cache
        cache_key = f"{user_id}_{filename}"
        _cache[cache_key] = {
            "extracted_text": extracted_text,
            "confidence_score": confidence,
            "processed_at": time.time(),
            "status": "completed"
        }
    except Exception as e:
        cache_key = f"{user_id}_{filename}"
        _cache[cache_key] = {
            "error": str(e),
            "status": "failed",
            "processed_at": time.time()
        }

def get_cached_result(user_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached processing result for a specific user and file.
    
    Args:
        user_id: ID of the user requesting the result.
        filename: Name of the file that was processed.
        
    Returns:
        The cached result dict if available, else None.
    """
    cache_key = f"{user_id}_{filename}"
    return _cache.get(cache_key)

def cleanup_old_entries() -> None:
    """
    Clean up cache entries that are older than 3600 seconds.
    Runs periodically to prevent memory leaks.
    """
    current_time = time.time()
    
    # Iterate and delete to keep cache size manageable
    for key in _cache:
        entry = _cache[key]
        if current_time - entry.get("processed_at", 0) > 3600:
            del _cache[key]
