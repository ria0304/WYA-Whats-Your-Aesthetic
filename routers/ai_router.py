from fastapi import APIRouter, HTTPException, Depends, Query, Body, Request, Response
from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime

from database import get_db
from auth_utils import get_current_user, UserProfile
from ai_model import FashionAIModel
from schemas import WeatherRequest, GreenAuditRequest
from rate_limiter import limiter
from services.outfit_generator import OutfitGenerator
from services.gap_analyzer import gap_analyzer
from services.style_profile import style_profile

router = APIRouter(prefix="/api/ai", tags=["ai"])
logger = logging.getLogger("uvicorn.error")
outfit_generator = OutfitGenerator()


# ── WEATHER ENDPOINT ──────────────────────────────────────────────────────

@router.get("/weather")
@limiter.limit("20/minute")
async def get_weather(
    request: Request,
    response: Response,
    city: str = Query("Delhi"),
    user: UserProfile = Depends(get_current_user)
):
    """
    Get current weather for a city.
    Uses the weather_service to fetch real-time weather data.
    """
    try:
        from services.weather_service import weather_styling
        result = weather_styling(city)
        return result
    except Exception as e:
        logger.error(f"Weather error: {e}")
        return {
            "condition": "unknown",
            "temperature": 0,
            "humidity": 0,
            "wind_speed": 0,
            "location": city,
            "error": str(e)
        }


# ── FEATURE 1: Personalized Outfit Scoring ──────────────────────────────────

@router.post("/outfit-score")
@limiter.limit("10/minute")
async def outfit_score(
    request: Request,
    response: Response,
    data: Dict[str, Any] = Body(...),
    user: UserProfile = Depends(get_current_user)
):
    """
    Score an outfit based on Style DNA, wear history, and color harmony.
    Returns score (0-100), breakdown, and reasoning.
    """
    outfit = data.get('outfit', {})
    if not outfit:
        raise HTTPException(400, "Outfit data required")

    conn = get_db()

    dna_row = conn.execute(
        "SELECT styles, color_preference FROM style_dna WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user.user_id,)
    ).fetchone()

    style_dna = {}
    if dna_row:
        try:
            style_dna = {
                "styles": json.loads(dna_row["styles"]) if dna_row["styles"] else [],
                "color_preference": dna_row["color_preference"]
            }
        except Exception:
            pass

    wear_history = conn.execute(
        "SELECT * FROM wear_logs WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
        (user.user_id,)
    ).fetchall()
    wear_history = [dict(row) for row in wear_history]

    pref_row = conn.execute(
        "SELECT preferred_categories FROM user_preferences WHERE user_id = ?",
        (user.user_id,)
    ).fetchone()
    
    color_preferences = None
    if pref_row and pref_row["preferred_categories"]:
        try:
            color_preferences = json.loads(pref_row["preferred_categories"])
        except Exception:
            pass

    conn.close()

    scored = outfit_generator.score_outfit(
        outfit=outfit,
        style_dna=style_dna,
        wear_history=wear_history,
        color_preferences=color_preferences
    )

    return scored


# ── FEATURE 2: Context-Aware Recommendations ──────────────────────────────

@router.post("/outfit-match-context")
@limiter.limit("10/minute")
async def outfit_match_context(
    request: Request,
    response: Response,
    data: Dict[str, Any] = Body(...),
    user: UserProfile = Depends(get_current_user)
):
    """
    Generate outfit suggestions with context awareness.
    Context includes: time of day, day of week, weather, temperature, occasion.
    """
    context = data.get('context', {})
    limit = data.get('limit', 5)

    conn = get_db()

    items = conn.execute(
        "SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at DESC",
        (user.user_id,)
    ).fetchall()
    wardrobe_items = [dict(item) for item in items]

    conn.close()

    if not wardrobe_items:
        return {
            "outfits": [],
            "context": context,
            "message": "No wardrobe items found"
        }

    basic_outfits = outfit_generator.generate_outfits_from_wardrobe(wardrobe_items, count=limit * 2)
    scored_outfits = outfit_generator.filter_by_context(basic_outfits, context)

    conn = get_db()
    dna_row = conn.execute(
        "SELECT styles FROM style_dna WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user.user_id,)
    ).fetchone()
    conn.close()

    style_dna = {}
    if dna_row:
        try:
            style_dna = {"styles": json.loads(dna_row["styles"]) if dna_row["styles"] else []}
        except Exception:
            pass

    final_outfits = []
    for scored in scored_outfits[:limit]:
        outfit = scored.get('outfit', {})
        outfit['context_reasoning'] = scored.get('context_reasoning', [])
        outfit['context_score'] = scored.get('context_score', 0)
        final_outfits.append(outfit)

    return {
        "outfits": final_outfits,
        "context": context,
        "total_considered": len(basic_outfits),
        "style_dna": style_dna
    }


# ── EXISTING: Fabric Scan ───────────────────────────────────────────────────

@router.post("/fabric-scan")
@limiter.limit("10/minute")
async def fabric_scan(
    request: Request,
    response: Response,
    data: Dict[str, Any],
    user: UserProfile = Depends(get_current_user)
):
    image = data.get('image')
    if not image:
        raise HTTPException(400, "Image required")
    return await FashionAIModel.autotag_garment(image)


# ── EXISTING: Outfit Match ──────────────────────────────────────────────────

@router.post("/outfit-match")
@limiter.limit("10/minute")
async def outfit_match(
    request: Request,
    response: Response,
    data: Dict[str, Any],
    user: UserProfile = Depends(get_current_user)
):
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

    if embedding_store.index_exists(user.user_id):
        query_emb = _text_to_pseudo_embedding(inspiration_item)
        similar_ids = embedding_store.search(user.user_id, query_emb, top_k=8)
        if similar_ids:
            id_set = set(similar_ids)
            ranked = [w for w in wardrobe_items if w.get("item_id") in id_set]
            id_order = {sid: i for i, sid in enumerate(similar_ids)}
            ranked.sort(key=lambda x: id_order.get(x.get("item_id", ""), 999))
        else:
            ranked = fashion_matcher.rank_closet_matches(inspiration_item, wardrobe_items)
    else:
        ranked = fashion_matcher.rank_closet_matches(inspiration_item, wardrobe_items)

    suggestion = await FashionAIModel.get_outfit_suggestion(image, variation, user.user_id)
    suggestion['closet_matches'] = ranked[:8]
    
    conn = get_db()
    dna_row = conn.execute(
        "SELECT styles FROM style_dna WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user.user_id,)
    ).fetchone()
    conn.close()
    if dna_row:
        try:
            suggestion['style_dna'] = json.loads(dna_row["styles"]) if dna_row["styles"] else []
        except Exception:
            pass
    
    return suggestion


# ── FEATURE 4: Gap Analysis with Shopping Links ────────────────────────────

@router.post("/gap-analysis")
@limiter.limit("10/minute")
async def gap_analysis(
    request: Request,
    response: Response,
    data: Dict[str, Any] = Body(default={}),
    user: UserProfile = Depends(get_current_user)
):
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

    pref_row = conn.execute(
        "SELECT preferred_categories, brands FROM user_preferences WHERE user_id = ?",
        (user.user_id,)
    ).fetchone()
    user_preferences = {}
    if pref_row:
        try:
            user_preferences["preferred_categories"] = json.loads(pref_row["preferred_categories"]) if pref_row["preferred_categories"] else []
        except Exception:
            pass
        try:
            user_preferences["preferred_brands"] = json.loads(pref_row["brands"]) if pref_row["brands"] else []
        except Exception:
            pass

    conn.close()

    style_dna = []
    if dna_row:
        try:
            style_dna = json.loads(dna_row["styles"])
        except Exception:
            style_dna = []

    inspired_category = (data.get("inspired_category") or "").strip()
    include_links = data.get("include_shopping_links", True)

    result = gap_analyzer.analyze(
        style_dna,
        wardrobe_items,
        inspired_category=inspired_category,
        include_shopping_links=include_links,
        user_preferences=user_preferences if user_preferences else None
    )

    gender = (user.gender or "Female").strip().lower()
    gender_label = "women's" if gender in ("female", "f", "woman", "women") else "men's"

    gaps = [{
        "category": g.get("category", "Unknown"),
        "description": g.get("description", "Missing item"),
        "reason": g.get("reason", "Fills a wardrobe gap"),
        "priority": g.get("priority", "medium"),
        "affiliateQuery": f"{gender_label} {g.get('affiliate_query', '')}",
        "affiliateBrand": g.get("affiliate_brand", ""),
        "affiliateUrl": g.get("affiliate_url", ""),
        "dnaAlignmentScore": g.get("dna_alignment_score", 0),
        "gender": gender_label,
        "shopping_suggestions": g.get("shopping_suggestions", {}),
        "price_range": g.get("price_range", {})
    } for g in result.get("gaps", [])]

    return {
        "gaps": gaps,
        "primaryAesthetic": result.get("primary_aesthetic", "casual"),
        "dnaAlignmentScore": result.get("dna_alignment_score", 0),
        "neutralRatio": result.get("neutral_ratio", 0),
        "patternRatio": result.get("pattern_ratio", 0),
        "wardrobeCount": result.get("wardrobe_count", 0),
        "sustainability_score": result.get("sustainability_score", 0),
        "shopping_links_included": include_links
    }


# ── FEATURE 6: Outfit Feedback ─────────────────────────────────────────────

@router.post("/outfit-feedback")
@limiter.limit("20/minute")
async def outfit_feedback(
    request: Request,
    response: Response,
    data: Dict[str, Any] = Body(...),
    user: UserProfile = Depends(get_current_user)
):
    action = data.get('action')
    outfit_id = data.get('outfit_id')
    item_id = data.get('item_id')
    context = data.get('context', {})

    if not action:
        raise HTTPException(400, "Action required")

    if action not in ['like', 'dislike', 'save', 'wear', 'skip']:
        raise HTTPException(400, "Invalid action")

    conn = get_db()
    conn.execute("""
        INSERT INTO outfit_feedback (user_id, outfit_id, item_id, action, context, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user.user_id,
        outfit_id,
        item_id,
        action,
        json.dumps(context),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "message": f"Feedback '{action}' recorded",
        "action": action
    }


# ── EXISTING: Vacation Packer ──────────────────────────────────────────────

@router.get("/vacation-packer")
@limiter.limit("10/minute")
async def vacation_packer(
    request: Request,
    response: Response,
    vacation_type: str = Query("city"),
    duration_days: int = Query(3),
    city: str = Query("Delhi"),
    user: UserProfile = Depends(get_current_user)
):
    return FashionAIModel.curate_trip(city, duration_days, vacation_type)


# ── EXISTING: Curate Outfits ───────────────────────────────────────────────

@router.post("/curate-outfits")
@limiter.limit("10/minute")
async def curate_outfits(
    request: Request,
    response: Response,
    data: Dict[str, Any] = Body(...),
    user: UserProfile = Depends(get_current_user)
):
    items = data.get('items', [])
    if not items:
        raise HTTPException(400, "Wardrobe items required")
    return await FashionAIModel.generate_outfits_from_wardrobe(items)


# ── EXISTING: Weather Search ────────────────────────────────────────────────

@router.post("/weather-search")
@limiter.limit("20/minute")
async def weather_search(
    request: Request,
    response: Response,
    data: WeatherRequest,
    user: UserProfile = Depends(get_current_user)
):
    return FashionAIModel.weather_styling(data.city)


# ── EXISTING: Green Audit ──────────────────────────────────────────────────

@router.post("/green-audit")
@limiter.limit("20/minute")
async def green_audit(
    request: Request,
    response: Response,
    data: GreenAuditRequest,
    user: UserProfile = Depends(get_current_user)
):
    return await FashionAIModel.audit_brand(data.brand)
