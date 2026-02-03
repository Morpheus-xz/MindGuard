"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ============================================
# AUTH SCHEMAS
# ============================================

class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    display_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: UUID
    email: str
    display_name: Optional[str]
    created_at: datetime


class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str


# ============================================
# ANALYSIS SCHEMAS
# ============================================

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)


class ClinicalFlag(BaseModel):
    indicator_type: str
    matched_keywords: str
    severity: str
    source: str


class AnalyzeResponse(BaseModel):
    session_id: UUID
    risk_level: str
    confidence: float
    probabilities: Dict[str, float]
    clinical_flags: List[ClinicalFlag]
    shap_summary: Optional[Dict] = None
    similar_sessions: Optional[List[Dict]] = None
    created_at: datetime


# ============================================
# SESSION SCHEMAS
# ============================================

class SessionResponse(BaseModel):
    id: UUID
    input_text: str
    risk_level: str
    confidence: float
    probabilities: Dict[str, float]
    clinical_flags: List[ClinicalFlag]
    created_at: datetime


class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int


# ============================================
# TRENDS SCHEMAS
# ============================================

class DailyTrend(BaseModel):
    date: str
    avg_risk_score: float
    session_count: int
    dominant_flag: Optional[str]


class TrendsResponse(BaseModel):
    trends: List[DailyTrend]
    prediction: Optional[List[float]] = None
    prediction_confidence: Optional[float] = None


# ============================================
# GENERAL
# ============================================

class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None