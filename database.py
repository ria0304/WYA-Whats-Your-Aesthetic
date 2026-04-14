import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_db():
    conn = sqlite3.connect('wya.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Ensure users table exists (base schema)
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY, 
        email TEXT UNIQUE NOT NULL, 
        full_name TEXT NOT NULL, 
        gender TEXT, 
        location TEXT, 
        hashed_password TEXT NOT NULL, 
        created_at TEXT, 
        updated_at TEXT)''')
    
    # Schema Migration: Check for missing columns in 'users' table
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'birthday' not in columns:
        logger.info("Migrating database: Adding 'birthday' column to users table.")
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN birthday TEXT")
        except Exception as e:
            logger.error(f"Failed to add birthday column: {e}")

    if 'email_notifications' not in columns:
        logger.info("Migrating database: Adding 'email_notifications' column to users table.")
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN email_notifications INTEGER DEFAULT 1")
        except Exception as e:
            logger.error(f"Failed to add email_notifications column: {e}")
    
    # Style DNA table (Current Active DNA)
    cursor.execute('''CREATE TABLE IF NOT EXISTS style_dna (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        user_id TEXT UNIQUE, 
        styles TEXT, 
        comfort_level INTEGER, 
        archetype TEXT, 
        summary TEXT, 
        created_at TEXT, 
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')

    # Style History table (Evolution Timeline)
    cursor.execute('''CREATE TABLE IF NOT EXISTS style_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        styles TEXT,
        comfort_level INTEGER,
        archetype TEXT,
        summary TEXT,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # Wardrobe items table
    cursor.execute('''CREATE TABLE IF NOT EXISTS wardrobe_items (
        item_id TEXT PRIMARY KEY, 
        user_id TEXT, 
        name TEXT NOT NULL, 
        category TEXT, 
        color TEXT, 
        fabric TEXT, 
        brand TEXT, 
        image_url TEXT, 
        last_worn TEXT, 
        wear_count INTEGER DEFAULT 0, 
        created_at TEXT, 
        embedding TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')

    # Activity Log
    cursor.execute('''CREATE TABLE IF NOT EXISTS activity_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        action_type TEXT,
        description TEXT,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')

    # User Preferences
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_preferences (
        user_id TEXT PRIMARY KEY,
        colors TEXT,
        brands TEXT,
        updated_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # Saved Outfits table (for manual curation and daily drops)
    cursor.execute('''CREATE TABLE IF NOT EXISTS saved_outfits (
        outfit_id TEXT PRIMARY KEY,
        user_id TEXT,
        name TEXT,
        vibe TEXT,
        items_json TEXT,
        is_daily INTEGER DEFAULT 0,
        created_date TEXT,
        worn_count INTEGER DEFAULT 0,
        last_worn TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # Outfit Wear History (for tracking when outfits were worn)
    cursor.execute('''CREATE TABLE IF NOT EXISTS outfit_wear_history (
        wear_id INTEGER PRIMARY KEY AUTOINCREMENT,
        outfit_id TEXT,
        user_id TEXT,
        worn_at TEXT,
        FOREIGN KEY (outfit_id) REFERENCES saved_outfits (outfit_id),
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # Wardrobe Archive (for soft-deleted items)
    cursor.execute('''CREATE TABLE IF NOT EXISTS wardrobe_archive (
        item_id TEXT PRIMARY KEY,
        user_id TEXT,
        name TEXT,
        category TEXT,
        color TEXT,
        fabric TEXT,
        brand TEXT,
        image_url TEXT,
        wear_count INTEGER,
        created_at TEXT,
        deleted_at TEXT,
        stats_json TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialization and migration check complete.")
