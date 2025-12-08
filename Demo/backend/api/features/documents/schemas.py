from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str
    content_type: str

class DocumentCreate(DocumentBase):
    file_size: int
    file_path: str

class DocumentResponse(DocumentBase):
    id: int
    user_id: int
    file_size: int
    uploaded_at: datetime
    status: Optional[str] = None
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    model_config = {
        "from_attributes": True
    }

class DocumentList(BaseModel):
    documents: List[DocumentResponse]

    model_config = {
        "from_attributes": True
    }

class UploadStatus(BaseModel):
    filename: str
    content_type: str
    progress: float  # 0-100 percentage
    status: str  # "processing", "completed", "failed"
    message: Optional[str] = None
    document_id: Optional[int] = None
