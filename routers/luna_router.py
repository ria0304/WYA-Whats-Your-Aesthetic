import os
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Any

# Adjust these imports depending on your exact backend directory naming structure
from routers.auth_router import get_current_user  # Assuming you have an auth dependency
from routers.ai_router import curate_outfits_engine, gap_analysis_engine 

router = APIRouter(prefix="/api/luna", tags=["Luna AI Orchestrator"])

# ── Pydantic Request/Response Schemas ──────────────────────────────────────────
class LunaChatRequest(BaseModel):
    message: str
    intent: str

class LunaChatResponse(BaseModel):
    text: str
    outfits: Optional[List[Any]] = None
    gapAnalysis: Optional[dict] = None

# ── Chat Routing Endpoint ──────────────────────────────────────────────────────
@router.post("/chat", response_model=LunaChatResponse)
async def handle_luna_chat(
    payload: LunaChatRequest,
    current_user: Any = Depends(get_current_user)  # Secures the endpoint with your JWT token
):
    """
    Orchestrated natural language processing endpoint for the Luna styling interface.
    Transforms raw algorithmic data structures into consumer-friendly chat payloads.
    """
    intent = payload.intent.lower().strip()
    user_id = getattr(current_user, "id", None) or getattr(current_user, "user_id", None)

    # ── 1. Outfit Curation Pipeline ───────────────────────────────────────────
    if intent == "outfit-help":
        try:
            # Connect direct to your internal backend algorithm processing layer
            # Passing None or empty fallback defaults if user has no explicit anchor items
            raw_curation = await curate_outfits_engine(user_id=user_id)
            
            return LunaChatResponse(
                text=f"I've computed some layout combinations based on your current closet vectors. Let's start with these options:",
                outfits=raw_curation.get("outfits", [])
            )
        except Exception as e:
            # Graceful error handling degradation fallback
            return LunaChatResponse(
                text="I noticed your closet metrics aren't populated yet! Head over to the main app dashboard, snap a quick photo of your favorite top, and I'll assemble some styles.",
                outfits=[]
            )

    # ── 2. Structural Wardrobe Gap Analysis Pipeline ──────────────────────────
    elif intent == "gap-analysis":
        try:
            # Query your internal vector/analytics database modules
            raw_gaps = await gap_analysis_engine(user_id=user_id)
            
            return LunaChatResponse(
                text=raw_gaps.get("summary", "Here is a breakdown of target areas where your wardrobe lacks coverage for your current aesthetic profile:"),
                gapAnalysis=raw_gaps
            )
        except Exception as e:
            return LunaChatResponse(
                text="I couldn't run a complete scan on your clothing metrics right now. Give it another try in a moment!",
                gapAnalysis={"gaps": [], "wardrobeCount": 0}
            )

    # ── 3. Fallback Exception ──────────────────────────────────────────────────
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Intent tracking configuration '{intent}' is not routed natively within the Luna pipeline."
        )
