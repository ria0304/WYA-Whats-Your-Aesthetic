from fastapi import APIRouter, HTTPException, Depends, Query, Body, Request
from typing import Dict, Any
import json
import logging

from database import get_db
from auth_utils import get_current_user, UserProfile
from ai_model import FashionAIModel
from schemas import WeatherRequest, GreenAuditRequest
from rate_limiter import limiter

router = APIRouter(prefix="/api/ai", tags=["ai"])
logger = logging.getLogger("uvicorn.error")


@router.post("/fabric-scan")
@limiter.limit("10/minute")
async def fabric_scan(request: Request, data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    image = data.get('image')
    if not image:
        raise HTTPException(400, "Image required")
    return await FashionAIModel.autotag_garment(image)


@router.post("/outfit-match")
@limiter.limit("10/minute")
async def outfit_match(request: Request, data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
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

    from ai_matcher import fashion_matcher, _text_to_pseudo_embedding
    import embedding_store

    inspiration_item = {"category": "Top", "color": "Unknown", "fabric": "Unknown"}

    # Try FAISS first (O(log n)), fall back to linear scan O(n) if no index
    if embedding_store.index_exists(user.user_id):
        query_emb = _text_to_pseudo_embedding(inspiration_item)
        similar_ids = embedding_store.search(user.user_id, query_emb, top_k=8)
        if similar_ids:
            id_set = set(similar_ids)
            ranked = [w for w in wardrobe_items if w.get("item_id") in id_set]
            # Preserve FAISS ordering
            id_order = {sid: i for i, sid in enumerate(similar_ids)}
            ranked.sort(key=lambda x: id_order.get(x.get("item_id", ""), 999))
        else:
            ranked = fashion_matcher.rank_closet_matches(inspiration_item, wardrobe_items)
    else:
        ranked = fashion_matcher.rank_closet_matches(inspiration_item, wardrobe_items)

    suggestion = await FashionAIModel.get_outfit_suggestion(image, variation, user.user_id)
    suggestion['closet_matches'] = ranked[:8]
    return suggestion


@router.get("/vacation-packer")
@limiter.limit("10/minute")
async def vacation_packer(
    request: Request,
    vacation_type: str = Query("city"),
    duration_days: int = Query(3),
    city: str = Query("Delhi"),
    user: UserProfile = Depends(get_current_user)
):
    return FashionAIModel.curate_trip(city, duration_days, vacation_type)


@router.post("/curate-outfits")
@limiter.limit("10/minute")
async def curate_outfits(request: Request, data: Dict[str, Any] = Body(...), user: UserProfile = Depends(get_current_user)):
    items = data.get('items', [])
    if not items:
        raise HTTPException(400, "Wardrobe items required")
    return await FashionAIModel.generate_outfits_from_wardrobe(items)


@router.post("/weather-search")
@limiter.limit("20/minute")
async def weather_search(request: Request, data: WeatherRequest, user: UserProfile = Depends(get_current_user)):
    return FashionAIModel.weather_styling(data.city)


@router.post("/green-audit")
@limiter.limit("20/minute")
async def green_audit(request: Request, data: GreenAuditRequest, user: UserProfile = Depends(get_current_user)):
    return await FashionAIModel.audit_brand(data.brand)


@router.post("/gap-analysis")
@limiter.limit("10/minute")
async def gap_analysis(request: Request, data: Dict[str, Any] = Body(default={}), user: UserProfile = Depends(get_current_user)):
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
