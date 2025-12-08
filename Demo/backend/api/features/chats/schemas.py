from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Thread schemas
class ThreadBase(BaseModel):
    title: Optional[str] = None

class ThreadCreate(ThreadBase):
    pass

class ThreadResponse(ThreadBase):
    id: int
    uuid: str
    user_id: int
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }

class ThreadList(BaseModel):
    threads: List[ThreadResponse]

    model_config = {
        "from_attributes": True
    }

# Chat schemas
class ChatBase(BaseModel):
    message: str
    role: str = "user"

class ChatCreate(ChatBase):
    thread_id: Optional[int] = None
    search_context: Optional[str] = None
    thread_uuid: Optional[str] = None

class ChatResponse(ChatBase):
    id: int
    user_id: int
    thread_id: int
    created_at: datetime

    model_config = {
        "from_attributes": True
    }

class ChatList(BaseModel):
    chats: List[ChatResponse]

    model_config = {
        "from_attributes": True
    }

class ThreadWithChats(ThreadResponse):
    chats: List[ChatResponse] = []

    model_config = {
        "from_attributes": True
    }
