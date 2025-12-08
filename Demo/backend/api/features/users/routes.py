from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from models.database import get_db, User
from api.auth.jwt_handler import get_current_active_user
from .schemas import UserProfile, UserList, ThemeUpdate

router = APIRouter(
    prefix="/users",
    tags=["Users"],
    dependencies=[Depends(get_current_active_user)]
)

@router.get("/", response_model=UserList)
async def get_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    users = db.query(User).all()
    return {"users": users}

@router.get("/{user_id}", response_model=UserProfile)
async def get_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/theme", response_model=UserProfile)
async def update_theme(theme_data: ThemeUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Update the user's theme preference"""
    # Validate theme
    allowed_themes = ["blue", "red", "violet"]
    if theme_data.theme not in allowed_themes:
        raise HTTPException(status_code=400, detail=f"Invalid theme. Allowed themes: {', '.join(allowed_themes)}")

    # Update user theme
    user = db.query(User).filter(User.id == current_user.id).first()
    user.theme = theme_data.theme
    db.commit()
    db.refresh(user)

    return user
