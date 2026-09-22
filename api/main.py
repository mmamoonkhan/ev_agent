"""
=============================================================
  GreenEnergy — Smart EV Assistant
  FastAPI Backend
=============================================================
  Endpoints:
    GET  /          → Health check
    POST /ask       → Ask a question
    POST /chat      → Multi-turn conversation
    DELETE /chat    → Clear conversation history
=============================================================
  How to run:
    uvicorn api.main:app --reload --port 8000
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import ask_greenenergy, ConversationMemory

# ── FastAPI app ─────────────────────────────────────────────
app = FastAPI(
    title="GreenEnergy — Smart EV Assistant",
    description="An AI agent specialized in Electric Vehicle knowledge",
    version="1.0.0"
)

# ── CORS — allow frontend to talk to API ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session storage ──────────────────────────────
sessions: Dict[str, ConversationMemory] = {}


# ── Request/Response models ────────────────────────────────
class AskRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


class AskResponse(BaseModel):
    answer: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    turn: int


# ── Endpoints ──────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "agent": "GreenEnergy — Smart EV Assistant",
        "version": "1.0.0",
        "description": "Ask me anything about Electric Vehicles!",
        "endpoints": {
            "ask": "POST /ask — single question",
            "chat": "POST /chat — multi-turn conversation",
            "clear": "DELETE /chat/{session_id} — clear history"
        }
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Ask a single question — no conversation history.
    Best for: quick one-off questions
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = ask_greenenergy(
        question=request.question,
        memory=None,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature
    )

    return AskResponse(
        question=request.question,
        answer=answer
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Multi-turn conversation — remembers previous messages.
    Best for: follow-up questions and conversations
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory(max_turns=5)

    memory = sessions[session_id]
    turn   = len(memory.get_history()) // 2 + 1

    answer = ask_greenenergy(
        question=request.question,
        memory=memory,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature
    )

    return ChatResponse(
        session_id=session_id,
        question=request.question,
        answer=answer,
        turn=turn
    )


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    """Clear conversation history for a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Conversation cleared!", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
def list_sessions():
    """List active sessions"""
    return {
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys())
    }
