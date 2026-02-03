"""Analysis endpoints for mental health screening."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Header, status

from app.db.supabase import get_supabase_client
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, ClinicalFlag

router = APIRouter()


def get_current_user_id(authorization: str = Header(...)) -> UUID:
    """Extract user ID from authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )

    token = authorization.replace("Bearer ", "")
    supabase = get_supabase_client()

    try:
        response = supabase.auth.get_user(token)
        if not response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return UUID(response.user.id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


@router.post("/", response_model=AnalyzeResponse)
async def analyze_text(
        data: AnalyzeRequest,
        user_id: UUID = Depends(get_current_user_id)
):
    """
    Analyze text for mental health risk indicators.

    Returns:
    - Risk level (Low/Medium/High)
    - Confidence score
    - Clinical flags (PHQ-9/GAD-7)
    - SHAP explanations
    - Similar past cases
    """
    supabase = get_supabase_client()

    # TODO: Replace with actual ML model prediction
    # For now, return mock data

    # Mock prediction
    risk_level = "Medium"
    confidence = 0.78
    probabilities = {"Low": 0.15, "Medium": 0.78, "High": 0.07}

    # Mock clinical flags
    clinical_flags = [
        ClinicalFlag(
            indicator_type="sleep_disturbance",
            matched_keywords="can't sleep, tired",
            severity="Medium",
            source="PHQ-9"
        )
    ]

    # Save session to database
    session_data = {
        "user_id": str(user_id),
        "input_text": data.text,
        "processed_text": data.text.lower(),  # TODO: Add proper preprocessing
        "risk_level": risk_level,
        "confidence": confidence,
        "probabilities": probabilities,
        "shap_summary": {}
    }

    result = supabase.table("sessions").insert(session_data).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save session"
        )

    session = result.data[0]

    # Save clinical flags
    for flag in clinical_flags:
        supabase.table("clinical_flags").insert({
            "session_id": session["id"],
            "indicator_type": flag.indicator_type,
            "matched_keywords": flag.matched_keywords,
            "severity": flag.severity,
            "source": flag.source
        }).execute()

    return AnalyzeResponse(
        session_id=UUID(session["id"]),
        risk_level=risk_level,
        confidence=confidence,
        probabilities=probabilities,
        clinical_flags=clinical_flags,
        shap_summary={},
        similar_sessions=[],
        created_at=datetime.fromisoformat(session["created_at"].replace("Z", "+00:00"))
    )