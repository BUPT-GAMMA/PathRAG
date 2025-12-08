from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserProfile(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    theme: Optional[str] = None

    class Config:
        orm_mode = True

class UserList(BaseModel):
    users: List[UserProfile]

    class Config:
        orm_mode = True

class ThemeUpdate(BaseModel):
    theme: str
