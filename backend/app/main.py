"""FastAPI main application for MindGuard."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import auth, analyze, history, trends

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI-powered mental health screening API",
    version="2.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(analyze.router, prefix="/api/analyze", tags=["Analysis"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
app.include_router(trends.router, prefix="/api/trends", tags=["Trends"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": "2.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "supabase": "connected"
    }