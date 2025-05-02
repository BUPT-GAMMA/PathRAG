from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# Token schema
class Token(BaseModel):
    access_token: str
    token_type: str

# Token data schema
class TokenData(BaseModel):
    username: Optional[str] = None

# User base schema
class UserBase(BaseModel):
    username: str
    email: EmailStr

# User creation schema
class UserCreate(UserBase):
    password: str

# User schema
class User(UserBase):
    id: int
    created_at: datetime
    theme: Optional[str] = None

    class Config:
        orm_mode = True

# User in DB schema
class UserInDB(User):
    hashed_password: str

    class Config:
        orm_mode = True

# Login schema
class Login(BaseModel):
    username: str
    password: str
