"""User session history endpoints."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, status

from app.db.supabase import get_supabase_client
from app.routers.analyze import get_current_user_id
from app.models.schemas import SessionResponse, SessionListResponse, ClinicalFlag, MessageResponse

router = APIRouter()


@router.get("/", response_model=SessionListResponse)
async def get_history(
        user_id: UUID = Depends(get_current_user_id),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0)
):
    """Get user's session history."""
    supabase = get_supabase_client()

    # Get sessions
    result = supabase.table("sessions").select(
        "*, clinical_flags(*)"
    ).eq(
        "user_id", str(user_id)
    ).order(
        "created_at", desc=True
    ).range(
        offset, offset + limit - 1
    ).execute()

    # Get total count
    count_result = supabase.table("sessions").select(
        "id", count="exact"
    ).eq("user_id", str(user_id)).execute()

    sessions = []
    for session in result.data:
        flags = [
            ClinicalFlag(
                indicator_type=f["indicator_type"],
                matched_keywords=f["matched_keywords"],
                severity=f["severity"],
                source=f["source"]
            )
            for f in session.get("clinical_flags", [])
        ]

        sessions.append(SessionResponse(
            id=UUID(session["id"]),
            input_text=session["input_text"],
            risk_level=session["risk_level"],
            confidence=session["confidence"],
            probabilities=session.get("probabilities", {}),
            clinical_flags=flags,
            created_at=session["created_at"]
        ))

    return SessionListResponse(
        sessions=sessions,
        total=count_result.count or 0
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
        session_id: UUID,
        user_id: UUID = Depends(get_current_user_id)
):
    """Get a specific session by ID."""
    supabase = get_supabase_client()

    result = supabase.table("sessions").select(
        "*, clinical_flags(*)"
    ).eq(
        "id", str(session_id)
    ).eq(
        "user_id", str(user_id)
    ).single().execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    session = result.data
    flags = [
        ClinicalFlag(
            indicator_type=f["indicator_type"],
            matched_keywords=f["matched_keywords"],
            severity=f["severity"],
            source=f["source"]
        )
        for f in session.get("clinical_flags", [])
    ]

    return SessionResponse(
        id=UUID(session["id"]),
        input_text=session["input_text"],
        risk_level=session["risk_level"],
        confidence=session["confidence"],
        probabilities=session.get("probabilities", {}),
        clinical_flags=flags,
        created_at=session["created_at"]
    )


@router.delete("/{session_id}", response_model=MessageResponse)
async def delete_session(
        session_id: UUID,
        user_id: UUID = Depends(get_current_user_id)
):
    """Delete a session."""
    supabase = get_supabase_client()

    result = supabase.table("sessions").delete().eq(
        "id", str(session_id)
    ).eq(
        "user_id", str(user_id)
    ).execute()

    return MessageResponse(message="Session deleted successfully")


@router.delete("/", response_model=MessageResponse)
async def delete_all_sessions(
        user_id: UUID = Depends(get_current_user_id)
):
    """Delete all sessions for user (GDPR compliance)."""
    supabase = get_supabase_client()

    supabase.table("sessions").delete().eq(
        "user_id", str(user_id)
    ).execute()

    return MessageResponse(message="All sessions deleted successfully")