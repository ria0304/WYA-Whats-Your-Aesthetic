from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from database import init_db
from routers.auth_router import router as auth_router
from routers.wardrobe_router import router as wardrobe_router
from routers.outfit_router import router as outfit_router
from routers.ai_router import router as ai_router
from routers.style_router import router as style_router
from routers.user_router import router as user_router

load_dotenv()

app = FastAPI(
    title="WYA — What's Your Aesthetic",
    description="AI-powered fashion & wardrobe intelligence API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        # S3 frontend
        "http://wya-whats-your-aesthetic.s3-website.ap-south-1.amazonaws.com",
        # CloudFront frontend (current)
        "https://dsbml6kwxecah.cloudfront.net",
        # EC2 backend direct
        "http://3.110.159.133:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(auth_router)
app.include_router(wardrobe_router)
app.include_router(outfit_router)
app.include_router(ai_router)
app.include_router(style_router)
app.include_router(user_router)
