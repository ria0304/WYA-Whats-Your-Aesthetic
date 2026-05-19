from fastapi import APIRouter, Depends
from datetime import datetime
import json
import logging

from database import get_db
from auth_utils import get_current_user, UserProfile
from ai_model import FashionAIModel
from schemas import StyleDNACreate

router = APIRouter(prefix="/api", tags=["style"])
logger = logging.getLogger("uvicorn.error")


@router.get("/dashboard/stats")
async def get_stats(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM wardrobe_items WHERE user_id = ?", (user.user_id,)
    ).fetchone()[0]

    dna_row = conn.execute("SELECT * FROM style_dna WHERE user_id = ?", (user.user_id,)).fetchone()
    archetype = "Pending"
    if dna_row:
        try:
            styles = json.loads(dna_row['styles'])
            if styles:
                archetype = styles[0].capitalize()
        except Exception:
            archetype = "Mapped"

    conn.close()
    return {"wardrobe_count": count, "style_archetype": archetype, "style_confidence": 91}


@router.get("/style/evolution")
async def get_evolution(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    items = conn.execute(
        "SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at ASC", (user.user_id,)
    ).fetchall()
    history = conn.execute(
        "SELECT * FROM style_history WHERE user_id = ? ORDER BY created_at DESC", (user.user_id,)
    ).fetchall()
    conn.close()
    return FashionAIModel.get_evolution_data([dict(i) for i in items], [dict(h) for h in history])


@router.get("/style/dna/{user_id}")
async def get_style_dna(user_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM style_dna WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        return {"has_dna": True, **dict(row)}
    return {"has_dna": False}


@router.post("/style/dna")
async def save_style_dna(data: StyleDNACreate, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    now = datetime.utcnow().isoformat()
    styles_json = json.dumps(data.styles)

    conn.execute(
        "INSERT OR REPLACE INTO style_dna (user_id, styles, comfort_level, summary, created_at) VALUES (?,?,?,?,?)",
        (user.user_id, styles_json, data.comfort_level, data.summary, now)
    )
    primary_style = data.styles[0] if data.styles else 'Evolution'
    conn.execute(
        "INSERT INTO style_history (user_id, styles, comfort_level, archetype, summary, created_at) VALUES (?,?,?,?,?,?)",
        (user.user_id, styles_json, data.comfort_level, primary_style, data.summary, now)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@router.get("/style/aura")
async def get_aesthetic_aura(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    items = conn.execute(
        "SELECT * FROM wardrobe_items WHERE user_id = ?", (user.user_id,)
    ).fetchall()
    wardrobe_items = [dict(item) for item in items]

    dna_row = conn.execute("SELECT * FROM style_dna WHERE user_id = ?", (user.user_id,)).fetchone()
    conn.close()

    colors = {}
    for item in wardrobe_items:
        color = item.get('color', 'Unknown')
        colors[color] = colors.get(color, 0) + 1

    top_colors = sorted(colors.items(), key=lambda x: x[1], reverse=True)[:4]
    color_hexes = []
    for color_name, _ in top_colors:
        from services.color_matcher import ColorMatcher
        rgb = ColorMatcher.get_color_properties((128, 128, 128))
        color_hexes.append(f"#{rgb.get('hue', 0):02x}{rgb.get('saturation', 0):02x}{rgb.get('value', 0):02x}")

    categories = {}
    for item in wardrobe_items:
        cat = item.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1

    top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "Outerwear"

    primary_aesthetic = "Classic Chic"
    if dna_row:
        try:
            styles = json.loads(dna_row['styles'])
            if styles:
                aesthetic_map = {
                    'minimalist': 'Minimalist', 'classic': 'Classic Chic',
                    'boho': 'Bohemian', 'streetwear': 'Streetwear', 'avant-garde': 'Avant-Garde'
                }
                primary_aesthetic = aesthetic_map.get(styles[0], 'Classic Chic')
        except Exception:
            pass

    return {
        "primary_aesthetic": primary_aesthetic, "primary_percent": 62,
        "secondary_aesthetic": "Minimalist", "secondary_percent": 28,
        "tertiary_aesthetic": "Streetwear", "tertiary_percent": 10,
        "mood_tag": "Effortlessly Curated", "season_tag": "Perennial Soul",
        "dominant_colors": color_hexes if color_hexes else ["#c4a882", "#2d2d2d", "#f5f0e5", "#6b3f2a"],
        "wardrobe_count": len(wardrobe_items),
        "top_category": top_category
    }
