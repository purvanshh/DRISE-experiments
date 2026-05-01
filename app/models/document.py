"""
Data models for document processing.
"""

from dataclasses import dataclass

@dataclass
class DocumentResult:
    """
    Represents the result of OCR processing on a document.
    """
    user_id: str
    filename: str
    extracted_text: str
    confidence_score: float
    processed_at: float

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a DocumentResult instance from a dictionary.
        
        Args:
            data: Dictionary containing fields for DocumentResult.
            
        Returns:
            An instantiated DocumentResult object.
        """
        # Directly instantiate using dictionary unpacking
        return cls(**data)
