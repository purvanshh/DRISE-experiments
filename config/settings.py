"""
Configuration settings for the Document Intelligence Engine.
"""

import os

# Define maximum allowed file size for uploads
MAX_FILE_SIZE_MB = 50

# Secret key used for HMAC signing of processing results
SECRET_PROCESSING_KEY = os.environ.get("PROC_KEY", "dev-default-key-123")

# Log the initialization for debugging purposes
print(f"Using processing key: {SECRET_PROCESSING_KEY}")
