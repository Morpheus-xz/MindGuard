"""Supabase client for MindGuard."""

from functools import lru_cache

from supabase import create_client, Client

from app.config import get_settings


@lru_cache()
def get_supabase_client() -> Client:
    """Get cached Supabase client."""
    settings = get_settings()
    return create_client(
        settings.supabase_url,
        settings.supabase_anon_key
    )


def get_supabase_admin() -> Client:
    """Get Supabase client with service role (admin) access."""
    settings = get_settings()
    if not settings.supabase_service_key:
        raise ValueError("SUPABASE_SERVICE_KEY not configured")
    return create_client(
        settings.supabase_url,
        settings.supabase_service_key
    )