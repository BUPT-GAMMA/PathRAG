from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid as uuid_pkg
import datetime
from typing import List, Optional

from models.database import get_db, Chat, User, Thread
from api.auth.jwt_handler import get_current_active_user
from .schemas import ChatCreate, ChatResponse, ChatList, ThreadCreate, ThreadResponse, ThreadList, ThreadWithChats
from PathRAG import QueryParam
from api.features.rag_manager import get_rag_instance

# Get the PathRAG instance from the manager
rag = get_rag_instance()

router = APIRouter(
    prefix="/chats",
    tags=["Chats"],
    dependencies=[Depends(get_current_active_user)]
)

# Thread endpoints
@router.get("/threads", response_model=ThreadList)
async def get_threads(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Get all chat threads for the current user"""
    threads = db.query(Thread).filter(
        Thread.user_id == current_user.id,
        Thread.is_deleted == False
    ).order_by(Thread.updated_at.desc()).all()
    return {"threads": threads}

@router.post("/threads", response_model=ThreadResponse)
async def create_thread(
    thread: ThreadCreate = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new chat thread"""
    # Generate a unique UUID for the thread
    thread_uuid = str(uuid_pkg.uuid4())

    # Create thread record
    db_thread = Thread(
        uuid=thread_uuid,
        user_id=current_user.id,
        title=thread.title if thread and thread.title else "New Chat"
    )
    db.add(db_thread)
    db.commit()
    db.refresh(db_thread)
    return db_thread

@router.get("/threads/{thread_uuid}", response_model=ThreadWithChats)
async def get_thread(
    thread_uuid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific thread with all its chats"""
    thread = db.query(Thread).filter(
        Thread.uuid == thread_uuid,
        Thread.user_id == current_user.id,
        Thread.is_deleted == False
    ).first()

    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Get all chats for this thread
    chats = db.query(Chat).filter(
        Chat.thread_id == thread.id
    ).order_by(Chat.created_at.asc()).all()

    # Create response with thread and chats
    result = ThreadWithChats.model_validate(thread)
    result.chats = chats

    return result

@router.delete("/threads/{thread_uuid}", response_model=dict)
async def delete_thread(
    thread_uuid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Mark a thread as deleted"""
    thread = db.query(Thread).filter(
        Thread.uuid == thread_uuid,
        Thread.user_id == current_user.id
    ).first()

    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Mark as deleted instead of actually deleting
    thread.is_deleted = True
    db.commit()

    return {"success": True, "message": "Thread deleted successfully"}

@router.put("/threads/{thread_uuid}", response_model=ThreadResponse)
async def update_thread(
    thread_uuid: str,
    thread_data: ThreadCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a thread's title"""
    thread = db.query(Thread).filter(
        Thread.uuid == thread_uuid,
        Thread.user_id == current_user.id,
        Thread.is_deleted == False
    ).first()

    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Update title
    if thread_data.title is not None:
        thread.title = thread_data.title

    db.commit()
    db.refresh(thread)

    return thread

# Chat endpoints
@router.get("/", response_model=ChatList)
async def get_chats(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Get all chats for the current user"""
    chats = db.query(Chat).filter(Chat.user_id == current_user.id).all()
    return {"chats": chats}

@router.get("/recent", response_model=ThreadList)
async def get_recent_threads(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Get the 5 most recent chat threads for the current user"""
    recent_threads = db.query(Thread).filter(
        Thread.user_id == current_user.id,
        Thread.is_deleted == False
    ).order_by(Thread.updated_at.desc()).limit(5).all()

    return {"threads": recent_threads}

@router.post("/chat/{thread_uuid}", response_model=ChatResponse)
async def create_chat(
    thread_uuid: str,
    chat: ChatCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new chat message and get a response"""
    # Find or create thread
    thread_id = None
    if thread_uuid:
        # Use existing thread ID
        thread = db.query(Thread).filter(
            Thread.uuid == thread_uuid,
            Thread.user_id == current_user.id,
            Thread.is_deleted == False
        ).first()

        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")

        thread_id = thread.id

    # Query PathRAG
    response_text = await rag.aquery(chat.message, param=QueryParam(mode=chat.search_context))

    # Create user message record
    user_chat = Chat(
        user_id=current_user.id,
        thread_id=thread_id,
        role="user",
        message=chat.message
    )
    db.add(user_chat)

    # Create system response record
    system_chat = Chat(
        user_id=current_user.id,
        thread_id=thread_id,
        role="system",
        message=response_text
    )
    db.add(system_chat)

    # Update thread title and timestamp
    thread.title = chat.message[:50] if len(chat.message) > 50 else chat.message
    thread.updated_at = datetime.datetime.now(datetime.timezone.utc)

    db.commit()
    db.refresh(user_chat)

    # Return the user message
    return user_chat

