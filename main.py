import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.middleware import SlowAPIMiddleware
from dotenv import load_dotenv

from database import init_db
from logger import setup_logging, get_logger
from rate_limiter import init_rate_limiter
from routers.auth_router import router as auth_router
from routers.wardrobe_router import router as wardrobe_router
from routers.outfit_router import router as outfit_router
from routers.ai_router import router as ai_router
from routers.style_router import router as style_router
from routers.user_router import router as user_router
from routers.recommend_router import router as recommend_router
from routers.health_router import router as health_router
from routers.luna_router import router as luna_router  

load_dotenv()
setup_logging()
logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
CLOUDFRONT_DOMAIN = os.getenv("CLOUDFRONT_DOMAIN", "dsbml6kwxecah.cloudfront.net")

# ── CORS Origins ──────────────────────────────────────────────────────────────
allowed_origins: list[str] = [
    f"https://{CLOUDFRONT_DOMAIN}",
    "http://luna-stylist.s3-website.ap-south-1.amazonaws.com",  # Allowed AWS S3 Production bucket host
]

# Development origins (only if DEBUG=true)
if DEBUG:
    allowed_origins += [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

# Extra origins from .env (e.g., EXTRA_ORIGINS=https://a.com,https://b.com)
extra = os.getenv("EXTRA_ORIGINS", "")
if extra:
    allowed_origins += [o.strip() for o in extra.split(",") if o.strip()]

logger.info("CORS origins configured: %d domains", len(allowed_origins))

# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager (replaces deprecated @app.on_event)"""
    # Startup
    init_db()
    logger.info("WYA backend started — DEBUG=%s, origins=%d", DEBUG, len(allowed_origins))
    yield
    # Shutdown
    logger.info("WYA backend shutting down")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="WYA — What's Your Aesthetic",
    description="AI-powered fashion & wardrobe intelligence API",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,      # Hide Swagger in production
    redoc_url="/redoc" if DEBUG else None,    # Hide ReDoc in production
    lifespan=lifespan,                        # Modern FastAPI lifecycle
)

init_rate_limiter(app)
app.add_middleware(SlowAPIMiddleware)

# ── Request Logging Middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
    expose_headers=["Content-Length", "X-Request-ID"],
    max_age=3600,
)

# ── Root Endpoint ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint — no infra details exposed in production"""
    return {
        "app": "WYA — What's Your Aesthetic",
        "version": "1.0.0",
        "health": "/api/health",
    }

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(wardrobe_router)
app.include_router(outfit_router)
app.include_router(ai_router)
app.include_router(style_router)
app.include_router(user_router)
app.include_router(recommend_router)
app.include_router(health_router)
app.include_router(luna_router)
