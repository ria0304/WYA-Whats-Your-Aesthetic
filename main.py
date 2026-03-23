
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import uuid
import json
import logging
from dotenv import load_dotenv

#CUSTOM IMPORTS
from database import get_db, init_db  #from database.py
from auth_utils import get_current_user, hash_password, verify_password, create_access_token, UserProfile #from auth_utils.py
from ai_model import FashionAIModel  #from ai_model.py

# Load env vars but let Uvicorn handle the logging config to avoid Windows multiprocessing errors
load_dotenv() # Loads variable from .env files
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="WYA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runs init_db() on app launch to create DB tables 
@app.on_event("startup")
def on_startup():
    init_db()

# Creates a new user account.
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

#Authenticates an existing user.
@app.post("/api/auth/login")
async def login(credentials: Dict[str, str]):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (credentials['email'],)).fetchone()
    conn.close()
    if not user or not verify_password(credentials['password'], user['hashed_password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user['user_id'])
    return {"access_token": token, "user": dict(user)}

#  Analyzes an image to tag garment details 
@app.post("/api/ai/fabric-scan")
async def fabric_scan(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    image = data.get('image')
    if not image: raise HTTPException(400, "Image required")
    return await FashionAIModel.autotag_garment(image)


# Suggests outfits based on an uploaded image.
@app.post("/api/ai/outfit-match")
async def outfit_match(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    image = data.get('image')
    variation = data.get('variation', 0)
    if not image: raise HTTPException(400, "Image required")
    return await FashionAIModel.get_outfit_suggestion(image, variation)


#Curates outfits for a trip based on location/duration
@app.get("/api/ai/vacation-packer")
async def vacation_packer(
    vacation_type: str = Query("city"),
    duration_days: int = Query(3),
    city: str = Query("Delhi"),
    user: UserProfile = Depends(get_current_user)
):
    return FashionAIModel.curate_trip(city, duration_days, vacation_type)


#Generates outfits from user's wardrobe items.
@app.post("/api/ai/curate-outfits")
async def curate_outfits(
    data: Dict[str, Any] = Body(...), 
    user: UserProfile = Depends(get_current_user)
):
    items = data.get('items', [])
    if not items: raise HTTPException(400, "Wardrobe items required")
    return await FashionAIModel.generate_outfits_from_wardrobe(items)

#Provides styling advice based on weather.
@app.post("/api/ai/weather-search")
async def weather_search(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    city = data.get('city', 'Delhi')
    return FashionAIModel.weather_styling(city)

#Audits a brand for sustainability.
@app.post("/api/ai/green-audit")
async def green_audit(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    brand = data.get('brand')
    if not brand: raise HTTPException(400, "Brand required")
    return await FashionAIModel.audit_brand(brand)


#Retrieves user's wardrobe items.
@app.get("/api/wardrobe")
async def get_wardrobe(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    items = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at DESC", (user.user_id,)).fetchall()
    conn.close()
    return [dict(row) for row in items]


#Adds a new wardrobe item.
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


#Deletes an item.
@app.delete("/api/wardrobe/{item_id}")
async def delete_wardrobe_item(item_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    conn.execute("DELETE FROM wardrobe_items WHERE item_id = ? AND user_id = ?", (item_id, user.user_id))
    conn.commit()
    conn.close()
    return {"success": True}

#Logs wearing an item (increments wear count).
@app.post("/api/wardrobe/{item_id}/wear")
async def wear_item(item_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute("UPDATE wardrobe_items SET wear_count = wear_count + 1, last_worn = ? WHERE item_id = ? AND user_id = ?", (now, item_id, user.user_id))
    conn.commit()
    conn.close()
    return {"success": True}

#Updates an item.
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


#Gets dashboard stats (wardrobe count, style archetype).
@app.get("/api/dashboard/stats")
async def get_stats(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM wardrobe_items WHERE user_id = ?", (user.user_id,)).fetchone()[0]
    
    dna_row = conn.execute("SELECT * FROM style_dna WHERE user_id = ?", (user.user_id,)).fetchone()
    archetype = "Pending"
    if dna_row:
        # Extract first style from list if exists
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
    # Fetch wardrobe for stats
    items = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at ASC", (user.user_id,)).fetchall()
    # Fetch style history for timeline
    history = conn.execute("SELECT * FROM style_history WHERE user_id = ? ORDER BY created_at DESC", (user.user_id,)).fetchall()
    
    conn.close()
    return FashionAIModel.get_evolution_data([dict(i) for i in items], [dict(h) for h in history])


#Provides style evolution data.
@app.get("/api/style/dna/{user_id}")
async def get_style_dna(user_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM style_dna WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        return {"has_dna": True, **dict(row)}
    return {"has_dna": False}


#Checks if user has style DNA.
@app.post("/api/style/dna")
async def save_style_dna(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    now = datetime.utcnow().isoformat()
    styles_json = json.dumps(data.get('styles', []))
    
    # 1. Update Current Active DNA
    conn.execute(
        "INSERT OR REPLACE INTO style_dna (user_id, styles, comfort_level, summary, created_at) VALUES (?,?,?,?,?)",
        (user.user_id, styles_json, data.get('comfort_level', 50), data.get('summary', ''), now)
    )
    
    # 2. Add to Evolution History
    # Extract a primary archetype for the history label if possible
    primary_style = data.get('styles', ['Undefined'])[0] if data.get('styles') else 'Evolution'
    
    conn.execute(
        "INSERT INTO style_history (user_id, styles, comfort_level, archetype, summary, created_at) VALUES (?,?,?,?,?,?)",
        (user.user_id, styles_json, data.get('comfort_level', 50), primary_style, data.get('summary', ''), now)
    )
    
    conn.commit()
    conn.close()
    return {"success": True}


# Saves/updates style DNA.
@app.get("/api/user/profile")
async def get_profile(user: UserProfile = Depends(get_current_user)):
    return user.dict()


#Returns user profile data.
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


#retrieves color/brand preferences.
@app.get("/api/user/preferences")
async def get_preferences(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user.user_id,)).fetchone()
    conn.close()
    if row:
        return {"colors": json.loads(row['colors']), "brands": json.loads(row['brands'])}
    return {"colors": [], "brands": []}


#Gets recent activity (e.g., added items).
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
