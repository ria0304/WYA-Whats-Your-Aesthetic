from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, Any
import json
import logging

from database import get_db
from auth_utils import get_current_user, UserProfile
from ai_model import FashionAIModel
from schemas import WeatherRequest, GreenAuditRequest

router = APIRouter(prefix="/api/ai", tags=["ai"])
logger = logging.getLogger("uvicorn.error")


@router.post("/fabric-scan")
async def fabric_scan(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    image = data.get('image')
    if not image:
        raise HTTPException(400, "Image required")
    return await FashionAIModel.autotag_garment(image)


@router.post("/outfit-match")
async def outfit_match(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    image = data.get('image')
    variation = data.get('variation', 0)
    if not image:
        raise HTTPException(400, "Image required")

    conn = get_db()
    items = conn.execute(
        "SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at DESC",
        (user.user_id,)
    ).fetchall()
    conn.close()
    wardrobe_items = [dict(item) for item in items]

    from ai_matcher import fashion_matcher
    inspiration_item = {"category": "Top", "color": "Unknown", "fabric": "Unknown"}
    ranked = fashion_matcher.rank_closet_matches(inspiration_item, wardrobe_items)

    suggestion = await FashionAIModel.get_outfit_suggestion(image, variation, user.user_id)
    suggestion['closet_matches'] = ranked[:8]
    return suggestion


@router.get("/vacation-packer")
async def vacation_packer(
    vacation_type: str = Query("city"),
    duration_days: int = Query(3),
    city: str = Query("Delhi"),
    user: UserProfile = Depends(get_current_user)
):
    return FashionAIModel.curate_trip(city, duration_days, vacation_type)


@router.post("/curate-outfits")
async def curate_outfits(data: Dict[str, Any] = Body(...), user: UserProfile = Depends(get_current_user)):
    items = data.get('items', [])
    if not items:
        raise HTTPException(400, "Wardrobe items required")
    return await FashionAIModel.generate_outfits_from_wardrobe(items)


@router.post("/weather-search")
async def weather_search(data: WeatherRequest, user: UserProfile = Depends(get_current_user)):
    return FashionAIModel.weather_styling(data.city)


@router.post("/green-audit")
async def green_audit(data: GreenAuditRequest, user: UserProfile = Depends(get_current_user)):
    return await FashionAIModel.audit_brand(data.brand)


@router.post("/gap-analysis")
async def gap_analysis(data: Dict[str, Any] = Body(default={}), user: UserProfile = Depends(get_current_user)):
    from services.gap_analyzer import gap_analyzer

    conn = get_db()
    items = conn.execute(
        "SELECT * FROM wardrobe_items WHERE user_id = ?", (user.user_id,)
    ).fetchall()
    wardrobe_items = [dict(item) for item in items]

    dna_row = conn.execute(
        "SELECT styles FROM style_dna WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user.user_id,)
    ).fetchone()
    conn.close()

    style_dna = []
    if dna_row:
        try:
            style_dna = json.loads(dna_row["styles"])
        except Exception:
            style_dna = []

    inspired_category = (data.get("inspired_category") or "").strip()
    result = gap_analyzer.analyze(style_dna, wardrobe_items, inspired_category=inspired_category)

    gender = (user.gender or "Female").strip().lower()
    gender_label = "women's" if gender in ("female", "f", "woman", "women") else "men's"

    gaps = [{
        "category":          g.get("category", "Unknown"),
        "description":       g.get("description", "Missing item"),
        "reason":            g.get("reason", "Fills a wardrobe gap"),
        "priority":          g.get("priority", "medium"),
        "affiliateQuery":    f"{gender_label} {g.get('affiliate_query', '')}",
        "affiliateBrand":    g.get("affiliate_brand", ""),
        "affiliateUrl":      g.get("affiliate_url", ""),
        "dnaAlignmentScore": g.get("dna_alignment_score", 0),
        "gender":            gender_label,
    } for g in result.get("gaps", [])]

    return {
        "gaps":              gaps,
        "primaryAesthetic":  result.get("primary_aesthetic", "casual"),
        "dnaAlignmentScore": result.get("dna_alignment_score", 0),
        "neutralRatio":      result.get("neutral_ratio", 0),
        "patternRatio":      result.get("pattern_ratio", 0),
        "wardrobeCount":     result.get("wardrobe_count", 0),
    }
