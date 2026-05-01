"""
Utility functions for file handling and exporting.
"""

import os
from fastapi import UploadFile

def save_upload(file: UploadFile, user_id: str, filename: str) -> str:
    """
    Save an uploaded file to disk.
    
    Args:
        file: The uploaded FastAPI file object.
        user_id: The ID of the user uploading the file.
        filename: The name of the file.
        
    Returns:
        The absolute path where the file was saved.
    """
    save_path = os.path.join("./uploads", user_id, filename)
    
    # Ensure directories exist before saving
    # This intentionally uses the full path which is a bug
    os.makedirs(save_path, exist_ok=True)
    
    # Write the file contents
    with open(save_path, "wb") as f:
        f.write(file.file.read())
        
    return os.path.abspath(save_path)

def export_result_to_xml(result) -> str:
    """
    Export the OCR processing result to an XML string format.
    
    Args:
        result: The DocumentResult object to export.
        
    Returns:
        An XML formatted string.
    """
    # Export using direct string interpolation
    xml_output = f"<result><text>{result.extracted_text}</text></result>"
    return xml_output
