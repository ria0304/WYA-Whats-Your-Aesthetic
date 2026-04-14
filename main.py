from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Body, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import uuid
import json
import logging
import asyncio
from dotenv import load_dotenv

#CUSTOM IMPORTS
from database import get_db, init_db
from auth_utils import get_current_user, hash_password, verify_password, create_access_token, UserProfile
from ai_model import FashionAIModel

# Load env vars
load_dotenv()
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="WYA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()


# ====================== AUTHENTICATION ======================

@app.post("/api/auth/register")
async def register(data: Dict[str, Any]):
    conn = get_db()
    try:
        user_id = str(uuid.uuid4())
        hashed = hash_password(data['password'])
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO users (user_id, email, full_name, birthday, gender, location, hashed_password, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, data['email'], data['full_name'], data.get('birthday', ''), data.get('gender', 'Female'), data.get('location', 'Global'), hashed, now)
        )
        conn.commit()
        token = create_access_token(user_id)
        user_row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return {"access_token": token, "user": dict(user_row)}
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=400, detail="Registration failed.")
    finally:
        conn.close()

@app.post("/api/auth/login")
async def login(credentials: Dict[str, str]):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (credentials['email'],)).fetchone()
    conn.close()
    if not user or not verify_password(credentials['password'], user['hashed_password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user['user_id'])
    return {"access_token": token, "user": dict(user)}


# ====================== AI ENDPOINTS ======================

@app.post("/api/ai/fabric-scan")
async def fabric_scan(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    image = data.get('image')
    if not image:
        raise HTTPException(400, "Image required")
    return await FashionAIModel.autotag_garment(image)

@app.post("/api/ai/outfit-match")
async def outfit_match(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    image = data.get('image')
    variation = data.get('variation', 0)
    if not image:
        raise HTTPException(400, "Image required")
    
    # Get closet matches with real similarity scores
    conn = get_db()
    items = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at DESC", (user.user_id,)).fetchall()
    conn.close()
    
    wardrobe_items = [dict(item) for item in items]
    
    # Use FashionAIModel to get similarity matches
    from ai_matcher import fashion_matcher
    
    # Create inspiration item from image (simplified - in production would extract features)
    inspiration_item = {"category": "Top", "color": "Unknown", "fabric": "Unknown"}
    
    # Rank closet items by similarity
    ranked = fashion_matcher.rank_closet_matches(inspiration_item, wardrobe_items)
    
    # Get outfit suggestion
    suggestion = await FashionAIModel.get_outfit_suggestion(image, variation, user.user_id)
    suggestion['closet_matches'] = ranked[:8]  # Add top 8 matches
    
    return suggestion

@app.get("/api/ai/vacation-packer")
async def vacation_packer(
    vacation_type: str = Query("city"),
    duration_days: int = Query(3),
    city: str = Query("Delhi"),
    user: UserProfile = Depends(get_current_user)
):
    return FashionAIModel.curate_trip(city, duration_days, vacation_type)

@app.post("/api/ai/curate-outfits")
async def curate_outfits(data: Dict[str, Any] = Body(...), user: UserProfile = Depends(get_current_user)):
    items = data.get('items', [])
    if not items:
        raise HTTPException(400, "Wardrobe items required")
    return await FashionAIModel.generate_outfits_from_wardrobe(items)

@app.post("/api/ai/weather-search")
async def weather_search(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    city = data.get('city', 'Delhi')
    return FashionAIModel.weather_styling(city)

@app.post("/api/ai/green-audit")
async def green_audit(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    brand = data.get('brand')
    if not brand:
        raise HTTPException(400, "Brand required")
    return await FashionAIModel.audit_brand(brand)

@app.post("/api/ai/daily-drop")
async def daily_drop(user: UserProfile = Depends(get_current_user)):
    """Generate today's Daily Drop using color harmony logic"""
    conn = get_db()
    
    # Get user's wardrobe items
    items = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ?", (user.user_id,)).fetchall()
    conn.close()
    
    wardrobe_items = [dict(item) for item in items]
    
    if len(wardrobe_items) < 2:
        return {
            "greeting": "Add more items to your closet!",
            "weather_snippet": "",
            "outfit_name": "Build Your Wardrobe",
            "outfit_vibe": "Start adding pieces",
            "harmony_type": "N/A",
            "pieces": [],
            "style_note": "Add at least 2 items to receive your Daily Drop.",
            "day_score": 0,
            "color_palette": []
        }
    
    # Use FashionAIModel to generate daily drop with color harmony
    from ai_matcher import fashion_matcher
    
    # Get weather for location
    weather_data = {}
    if user.location:
        try:
            weather = FashionAIModel.weather_styling(user.location)
            weather_data = {
                "temp": weather.get("temp", 22),
                "condition": weather.get("condition", "Sunny")
            }
        except:
            weather_data = {"temp": 22, "condition": "Sunny"}
    
    # Create outfit using color harmony
    outfit = fashion_matcher.create_complete_outfit(wardrobe_items, style="casual")
    
    # Get color harmony type
    harmony_type = "Analogous"
    if outfit.get("items") and len(outfit["items"]) >= 2:
        colors = [item.get("color", "") for item in outfit["items"][:2]]
        from ai_matcher import _colors_harmonize
        is_harmonious, h_type = _colors_harmonize(colors[0], colors[1])
        harmony_type = h_type.replace("_", " ").title() if is_harmonious else "Complementary"
    
    # Build pieces list
    pieces = []
    color_palette = []
    for item in outfit.get("items", []):
        color = item.get("color", "Gray")
        pieces.append({
            "name": item.get("name", "Item"),
            "category": item.get("category", "Top"),
            "color": color,
            "image_url": item.get("image_url", "")
        })
        color_palette.append(item.get("hex_color", "#808080"))
    
    weather_snippet = f"{weather_data.get('temp', 22)}°C · {weather_data.get('condition', 'Sunny')}"
    
    return {
        "greeting": "Your Daily Drop is ready ✨",
        "weather_snippet": weather_snippet,
        "outfit_name": outfit.get("name", "Today's Look"),
        "outfit_vibe": outfit.get("vibe", "Casual"),
        "harmony_type": harmony_type,
        "pieces": pieces,
        "style_note": outfit.get("styling_tips", ["Wear with confidence"])[0] if outfit.get("styling_tips") else "Style is personal - wear what makes you feel confident!",
        "day_score": outfit.get("compatibility_score", 85),
        "color_palette": color_palette[:5]
    }

@app.post("/api/ai/gap-analysis")
async def gap_analysis(user: UserProfile = Depends(get_current_user)):
    """Analyze wardrobe gaps based on Style DNA"""
    conn = get_db()
    
    # Get user's wardrobe
    items = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ?", (user.user_id,)).fetchall()
    wardrobe_items = [dict(item) for item in items]
    
    # Get user's Style DNA
    dna_row = conn.execute("SELECT * FROM style_dna WHERE user_id = ?", (user.user_id,)).fetchone()
    conn.close()
    
    style_dna = []
    if dna_row:
        try:
            style_dna = json.loads(dna_row['styles'])
        except:
            style_dna = []
    
    # Analyze gaps
    from ai_matcher import fashion_matcher
    analysis = fashion_matcher.analyze_wardrobe_gaps(wardrobe_items, style_dna)
    
    # Format gaps for frontend
    gaps = []
    for suggestion in analysis.get("gap_analysis", []):
        gaps.append({
            "category": suggestion.get("category", "Unknown"),
            "description": suggestion.get("piece", "Missing item"),
            "reason": suggestion.get("reason", "Fills a wardrobe gap"),
            "affiliateQuery": suggestion.get("affiliate_tag", f"{suggestion.get('piece', '')} sustainable fashion"),
            "priority": "high" if suggestion.get("piece", "").startswith("Essential") else "medium"
        })
    
    return {"gaps": gaps[:5]}


# ====================== WARDROBE ENDPOINTS ======================

@app.get("/api/wardrobe")
async def get_wardrobe(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    items = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at DESC", (user.user_id,)).fetchall()
    conn.close()
    return [dict(row) for row in items]

@app.post("/api/wardrobe")
async def add_wardrobe_item(
    name: str = Form(...),
    category: str = Form(...),
    color: str = Form(""),
    fabric: str = Form(""),
    image_url: Optional[str] = Form(None),
    user: UserProfile = Depends(get_current_user)
):
    item_id = str(uuid.uuid4())
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO wardrobe_items (item_id, user_id, name, category, color, fabric, image_url, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (item_id, user.user_id, name, category, color, fabric, image_url, now)
    )
    conn.commit()
    conn.close()
    return {"success": True}

@app.delete("/api/wardrobe/{item_id}")
async def delete_wardrobe_item(item_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    conn.execute("DELETE FROM wardrobe_items WHERE item_id = ? AND user_id = ?", (item_id, user.user_id))
    conn.commit()
    conn.close()
    return {"success": True}

@app.post("/api/wardrobe/{item_id}/wear")
async def wear_item(item_id: str, data: Dict[str, Any] = None, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    worn_at = data.get('worn_at', datetime.utcnow().isoformat()) if data else datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE wardrobe_items SET wear_count = wear_count + 1, last_worn = ? WHERE item_id = ? AND user_id = ?",
        (worn_at, item_id, user.user_id)
    )
    conn.commit()
    conn.close()
    return {"success": True}

@app.put("/api/wardrobe/{item_id}")
async def update_wardrobe_item(
    item_id: str,
    name: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    fabric: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    user: UserProfile = Depends(get_current_user)
):
    conn = get_db()
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if category is not None:
        fields.append("category = ?")
        values.append(category)
    if color is not None:
        fields.append("color = ?")
        values.append(color)
    if fabric is not None:
        fields.append("fabric = ?")
        values.append(fabric)
    if image_url is not None:
        fields.append("image_url = ?")
        values.append(image_url)
    
    if fields:
        sql = f"UPDATE wardrobe_items SET {', '.join(fields)} WHERE item_id = ? AND user_id = ?"
        values.append(item_id)
        values.append(user.user_id)
        conn.execute(sql, tuple(values))
        conn.commit()
    
    conn.close()
    return {"success": True}

@app.post("/api/wardrobe/{item_id}/remove-bg")
async def remove_background(item_id: str, user: UserProfile = Depends(get_current_user)):
    """Remove background from wardrobe item image"""
    conn = get_db()
    item = conn.execute("SELECT * FROM wardrobe_items WHERE item_id = ? AND user_id = ?", (item_id, user.user_id)).fetchone()
    
    if not item:
        conn.close()
        raise HTTPException(404, "Item not found")
    
    # In production, integrate rembg or similar service
    # For now, return the original URL
    conn.close()
    
    return {"bg_removed_url": item['image_url']}

@app.post("/api/wardrobe/{item_id}/archive")
async def archive_item(item_id: str, data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    """Move item to archive instead of permanent deletion"""
    conn = get_db()
    
    # Get the item
    item = conn.execute("SELECT * FROM wardrobe_items WHERE item_id = ? AND user_id = ?", (item_id, user.user_id)).fetchone()
    
    if not item:
        conn.close()
        raise HTTPException(404, "Item not found")
    
    # Insert into archive
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO wardrobe_archive 
           (item_id, user_id, name, category, color, fabric, brand, image_url, wear_count, created_at, deleted_at, archive_reason, memory_note) 
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (item_id, user.user_id, item['name'], item['category'], item['color'], 
         item['fabric'], item.get('brand', ''), item['image_url'], 
         item.get('wear_count', 0), item['created_at'], now, 
         data.get('reason', 'sold'), data.get('memory_note', ''))
    )
    
    # Delete from main wardrobe
    conn.execute("DELETE FROM wardrobe_items WHERE item_id = ? AND user_id = ?", (item_id, user.user_id))
    
    conn.commit()
    conn.close()
    return {"success": True}


# ====================== OUTFITS ENDPOINTS (Backend Migration) ======================

@app.get("/api/outfits")
async def get_outfits(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    outfits = conn.execute(
        "SELECT * FROM saved_outfits WHERE user_id = ? ORDER BY created_date DESC",
        (user.user_id,)
    ).fetchall()
    conn.close()
    
    result = []
    for outfit in outfits:
        items_json = outfit['items_json']
        try:
            items = json.loads(items_json) if items_json else []
        except:
            items = []
        
        result.append({
            "id": outfit['outfit_id'],
            "name": outfit['name'],
            "vibe": outfit['vibe'],
            "items": items,
            "is_daily": outfit['is_daily'],
            "created_date": outfit['created_date'],
            "worn_count": outfit['worn_count'],
            "last_worn": outfit['last_worn']
        })
    
    return result

@app.post("/api/outfits")
async def save_outfit(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    outfit_id = str(uuid.uuid4())
    conn = get_db()
    now = datetime.utcnow().isoformat()
    
    conn.execute(
        """INSERT INTO saved_outfits 
           (outfit_id, user_id, name, vibe, items_json, is_daily, created_date) 
           VALUES (?,?,?,?,?,?,?)""",
        (outfit_id, user.user_id, data.get('name', 'My Outfit'), data.get('vibe', 'Casual'),
         json.dumps(data.get('items', [])), data.get('is_daily', 0), 
         data.get('created_date', now))
    )
    conn.commit()
    conn.close()
    return {"success": True, "id": outfit_id}

@app.delete("/api/outfits/{outfit_id}")
async def delete_outfit(outfit_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    conn.execute(
        "DELETE FROM saved_outfits WHERE outfit_id = ? AND user_id = ?",
        (outfit_id, user.user_id)
    )
    conn.commit()
    conn.close()
    return {"success": True}

@app.post("/api/outfits/{outfit_id}/worn")
async def log_outfit_wear(outfit_id: str, data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    """Log when an outfit was worn with timestamp"""
    conn = get_db()
    worn_at = data.get('worn_at', datetime.utcnow().isoformat())
    
    # Check if outfit belongs to user
    outfit = conn.execute(
        "SELECT * FROM saved_outfits WHERE outfit_id = ? AND user_id = ?",
        (outfit_id, user.user_id)
    ).fetchone()
    
    if not outfit:
        conn.close()
        raise HTTPException(status_code=404, detail="Outfit not found")
    
    # Insert wear record
    conn.execute(
        "INSERT INTO outfit_wear_history (outfit_id, user_id, worn_at) VALUES (?,?,?)",
        (outfit_id, user.user_id, worn_at)
    )
    
    # Update worn_count and last_worn in saved_outfits
    conn.execute(
        "UPDATE saved_outfits SET worn_count = worn_count + 1, last_worn = ? WHERE outfit_id = ?",
        (worn_at, outfit_id)
    )
    
    conn.commit()
    conn.close()
    return {"success": True, "worn_at": worn_at}

@app.get("/api/outfits/{outfit_id}/wear-history")
async def get_outfit_wear_history(outfit_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    history = conn.execute(
        "SELECT worn_at FROM outfit_wear_history WHERE outfit_id = ? AND user_id = ? ORDER BY worn_at DESC",
        (outfit_id, user.user_id)
    ).fetchall()
    conn.close()
    return [dict(row) for row in history]

@app.get("/api/user/wear-timeline")
async def get_user_wear_timeline(days: int = 90, user: UserProfile = Depends(get_current_user)):
    """Get all outfit wear events for timeline visualization"""
    conn = get_db()
    start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    timeline = conn.execute(
        """SELECT 
            owh.worn_at,
            owh.outfit_id,
            so.name as outfit_name,
            so.vibe,
            so.items_json
        FROM outfit_wear_history owh
        JOIN saved_outfits so ON owh.outfit_id = so.outfit_id
        WHERE owh.user_id = ? AND owh.worn_at >= ?
        ORDER BY owh.worn_at DESC""",
        (user.user_id, start_date)
    ).fetchall()
    
    conn.close()
    return [dict(row) for row in timeline]


# ====================== ARCHIVE ENDPOINTS ======================

@app.get("/api/archive")
async def get_archive(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    archived = conn.execute(
        "SELECT * FROM wardrobe_archive WHERE user_id = ? ORDER BY deleted_at DESC",
        (user.user_id,)
    ).fetchall()
    conn.close()
    
    result = []
    for item in archived:
        result.append({
            "id": item['item_id'],
            "name": item['name'],
            "category": item['category'],
            "color": item['color'],
            "fabric": item['fabric'],
            "image_url": item['image_url'],
            "wear_count": item['wear_count'],
            "archived_date": item['deleted_at'],
            "archive_reason": item['archive_reason'],
            "memory_note": item['memory_note']
        })
    
    return result

@app.delete("/api/archive/{item_id}")
async def permanent_delete_archive(item_id: str, user: UserProfile = Depends(get_current_user)):
    """Permanently delete from archive"""
    conn = get_db()
    conn.execute(
        "DELETE FROM wardrobe_archive WHERE item_id = ? AND user_id = ?",
        (item_id, user.user_id)
    )
    conn.commit()
    conn.close()
    return {"success": True}


# ====================== STYLE DNA ENDPOINTS ======================

@app.get("/api/dashboard/stats")
async def get_stats(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM wardrobe_items WHERE user_id = ?", (user.user_id,)).fetchone()[0]
    
    dna_row = conn.execute("SELECT * FROM style_dna WHERE user_id = ?", (user.user_id,)).fetchone()
    archetype = "Pending"
    if dna_row:
        try:
            styles = json.loads(dna_row['styles'])
            if styles:
                archetype = styles[0].capitalize()
        except:
            archetype = "Mapped"
    
    conn.close()
    return {"wardrobe_count": count, "style_archetype": archetype, "style_confidence": 91}

@app.get("/api/style/evolution")
async def get_evolution(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    items = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at ASC", (user.user_id,)).fetchall()
    history = conn.execute("SELECT * FROM style_history WHERE user_id = ? ORDER BY created_at DESC", (user.user_id,)).fetchall()
    conn.close()
    return FashionAIModel.get_evolution_data([dict(i) for i in items], [dict(h) for h in history])

@app.get("/api/style/dna/{user_id}")
async def get_style_dna(user_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM style_dna WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        return {"has_dna": True, **dict(row)}
    return {"has_dna": False}

@app.post("/api/style/dna")
async def save_style_dna(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    now = datetime.utcnow().isoformat()
    styles_json = json.dumps(data.get('styles', []))
    
    # Update Current Active DNA
    conn.execute(
        "INSERT OR REPLACE INTO style_dna (user_id, styles, comfort_level, summary, created_at) VALUES (?,?,?,?,?)",
        (user.user_id, styles_json, data.get('comfort_level', 50), data.get('summary', ''), now)
    )
    
    # Add to Evolution History
    primary_style = data.get('styles', ['Undefined'])[0] if data.get('styles') else 'Evolution'
    
    conn.execute(
        "INSERT INTO style_history (user_id, styles, comfort_level, archetype, summary, created_at) VALUES (?,?,?,?,?,?)",
        (user.user_id, styles_json, data.get('comfort_level', 50), primary_style, data.get('summary', ''), now)
    )
    
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/style/aura")
async def get_aesthetic_aura(user: UserProfile = Depends(get_current_user)):
    """Generate Aesthetic Aura data for share card"""
    conn = get_db()
    
    # Get wardrobe stats
    items = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ?", (user.user_id,)).fetchall()
    wardrobe_items = [dict(item) for item in items]
    
    # Get Style DNA
    dna_row = conn.execute("SELECT * FROM style_dna WHERE user_id = ?", (user.user_id,)).fetchone()
    conn.close()
    
    # Calculate dominant colors
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
    
    # Categorize wardrobe by aesthetic
    categories = {}
    for item in wardrobe_items:
        cat = item.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "Outerwear"
    
    # Determine primary aesthetic from DNA
    primary_aesthetic = "Classic Chic"
    if dna_row:
        try:
            styles = json.loads(dna_row['styles'])
            if styles:
                aesthetic_map = {
                    'minimalist': 'Minimalist',
                    'classic': 'Classic Chic',
                    'boho': 'Bohemian',
                    'streetwear': 'Streetwear',
                    'avant-garde': 'Avant-Garde'
                }
                primary_aesthetic = aesthetic_map.get(styles[0], 'Classic Chic')
        except:
            pass
    
    return {
        "primary_aesthetic": primary_aesthetic,
        "primary_percent": 62,
        "secondary_aesthetic": "Minimalist",
        "secondary_percent": 28,
        "tertiary_aesthetic": "Streetwear",
        "tertiary_percent": 10,
        "mood_tag": "Effortlessly Curated",
        "season_tag": "Perennial Soul",
        "dominant_colors": color_hexes if color_hexes else ["#c4a882", "#2d2d2d", "#f5f0e5", "#6b3f2a"],
        "wardrobe_count": len(wardrobe_items),
        "top_category": top_category
    }


# ====================== USER PROFILE ENDPOINTS ======================

@app.get("/api/user/profile")
async def get_profile(user: UserProfile = Depends(get_current_user)):
    return user.dict()

@app.put("/api/user/profile")
async def update_profile(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE users SET full_name = ?, location = ?, birthday = ?, gender = ?, email_notifications = ?, updated_at = ? WHERE user_id = ?",
        (data.get('full_name', user.full_name), data.get('location', user.location), data.get('birthday', user.birthday), 
         data.get('gender', user.gender), data.get('email_notifications', user.email_notifications), now, user.user_id)
    )
    conn.commit()
    user_row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user.user_id,)).fetchone()
    conn.close()
    return dict(user_row)

@app.get("/api/user/preferences")
async def get_preferences(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user.user_id,)).fetchone()
    conn.close()
    if row:
        return {"colors": json.loads(row['colors']), "brands": json.loads(row['brands'])}
    return {"colors": [], "brands": []}

@app.put("/api/user/preferences")
async def update_preferences(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    now = datetime.utcnow().isoformat()
    colors = json.dumps(data.get('colors', []))
    brands = json.dumps(data.get('brands', []))
    conn.execute(
        "INSERT OR REPLACE INTO user_preferences (user_id, colors, brands, updated_at) VALUES (?,?,?,?)",
        (user.user_id, colors, brands, now)
    )
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/user/activity")
async def get_activity(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    items = conn.execute("SELECT name as item, 'Added Item' as action, created_at as date FROM wardrobe_items WHERE user_id = ? ORDER BY created_at DESC LIMIT 5", (user.user_id,)).fetchall()
    conn.close()
    return [dict(row) for row in items]


# ====================== NOTIFICATION ENDPOINTS ======================

@app.post("/api/notifications/subscribe")
async def subscribe_notifications(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    """Save push notification subscription"""
    conn = get_db()
    now = datetime.utcnow().isoformat()
    
    conn.execute(
        """INSERT OR REPLACE INTO push_subscriptions 
           (user_id, endpoint, p256dh, auth, created_at, updated_at) 
           VALUES (?,?,?,?,?,?)""",
        (user.user_id, data.get('endpoint', ''), data.get('p256dh', ''), 
         data.get('auth', ''), now, now)
    )
    conn.commit()
    conn.close()
    return {"success": True}
