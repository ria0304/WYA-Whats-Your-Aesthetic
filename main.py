import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from database import init_db
from logger import setup_logging, get_logger
from routers.auth_router import router as auth_router
from routers.wardrobe_router import router as wardrobe_router
from routers.outfit_router import router as outfit_router
from routers.ai_router import router as ai_router
from routers.style_router import router as style_router
from routers.user_router import router as user_router
from routers.recommend_router import router as recommend_router
from routers.health_router import router as health_router  

load_dotenv()
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="WYA — What's Your Aesthetic",
    description="AI-powered fashion & wardrobe intelligence API",
    version="1.0.0",
)

# ── Request logging middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %s  (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

# ── CORS ───────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://wya-whats-your-aesthetic.s3-website.ap-south-1.amazonaws.com",
        "https://dsbml6kwxecah.cloudfront.net",
        "http://3.110.159.133:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("WYA backend started — all routers registered")

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(wardrobe_router)
app.include_router(outfit_router)
app.include_router(ai_router)
app.include_router(style_router)
app.include_router(user_router)
app.include_router(recommend_router)
app.include_router(health_router)  
