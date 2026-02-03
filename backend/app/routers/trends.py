"""Trend analysis endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.db.supabase import get_supabase_client
from app.routers.analyze import get_current_user_id
from app.models.schemas import TrendsResponse, DailyTrend

router = APIRouter()


@router.get("/", response_model=TrendsResponse)
async def get_trends(
        user_id: UUID = Depends(get_current_user_id),
        days: int = Query(30, ge=7, le=90)
):
    """Get user's risk trends over time."""
    supabase = get_supabase_client()

    # Get daily trends
    result = supabase.table("daily_trends").select("*").eq(
        "user_id", str(user_id)
    ).order(
        "date", desc=True
    ).limit(days).execute()

    trends = [
        DailyTrend(
            date=t["date"],
            avg_risk_score=t["avg_risk_score"],
            session_count=t["session_count"],
            dominant_flag=t.get("dominant_flag")
        )
        for t in result.data
    ]

    # TODO: Add LSTM prediction
    prediction = None
    prediction_confidence = None

    return TrendsResponse(
        trends=trends,
        prediction=prediction,
        prediction_confidence=prediction_confidence
    )