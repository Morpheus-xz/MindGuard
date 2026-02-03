"""Authentication endpoints."""

from fastapi import APIRouter, HTTPException, status

from app.db.supabase import get_supabase_client
from app.models.schemas import (
    UserSignup,
    UserLogin,
    AuthResponse,
    UserResponse,
    MessageResponse
)

router = APIRouter()


@router.post("/signup")
async def signup(data: UserSignup):
    """Register a new user."""
    supabase = get_supabase_client()

    try:
        response = supabase.auth.sign_up({
            "email": data.email,
            "password": data.password,
            "options": {
                "data": {
                    "display_name": data.display_name or data.email.split("@")[0]
                }
            }
        })

        if not response.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signup failed"
            )

        # Check if email confirmation is required
        if response.session is None:
            # Email confirmation is enabled
            return {
                "message": "Signup successful! Please check your email to confirm your account.",
                "user_id": str(response.user.id),
                "email": response.user.email,
                "requires_confirmation": True
            }

        # Email confirmation is disabled, return tokens
        return AuthResponse(
            user=UserResponse(
                id=response.user.id,
                email=response.user.email,
                display_name=data.display_name,
                created_at=response.user.created_at
            ),
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=AuthResponse)
async def login(data: UserLogin):
    """Login with email and password."""
    supabase = get_supabase_client()

    try:
        response = supabase.auth.sign_in_with_password({
            "email": data.email,
            "password": data.password
        })

        if not response.user or not response.session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Get profile
        profile_data = None
        try:
            profile = supabase.table("profiles").select("*").eq(
                "id", str(response.user.id)
            ).single().execute()
            profile_data = profile.data
        except:
            pass  # Profile might not exist yet

        display_name = None
        if profile_data:
            display_name = profile_data.get("display_name")

        return AuthResponse(
            user=UserResponse(
                id=response.user.id,
                email=response.user.email,
                display_name=display_name,
                created_at=response.user.created_at
            ),
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )


@router.post("/logout", response_model=MessageResponse)
async def logout():
    """Logout current user."""
    supabase = get_supabase_client()

    try:
        supabase.auth.sign_out()
        return MessageResponse(message="Logged out successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token."""
    supabase = get_supabase_client()

    try:
        response = supabase.auth.refresh_session(refresh_token)

        if not response.user or not response.session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        return AuthResponse(
            user=UserResponse(
                id=response.user.id,
                email=response.user.email,
                display_name=None,
                created_at=response.user.created_at
            ),
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )