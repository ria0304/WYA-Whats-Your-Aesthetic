import sqlite3
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def get_db():
    DB_PATH = os.getenv('DB_PATH', '/app/data/wya.db')
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # ── Users Table ────────────────────────────────────────────────────────────
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
    
    # ── Style DNA Table (Current Active DNA) ──────────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS style_dna (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        user_id TEXT UNIQUE, 
        styles TEXT, 
        comfort_level INTEGER, 
        archetype TEXT, 
        summary TEXT, 
        created_at TEXT, 
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')

    # ── Style History Table (Evolution Timeline) ──────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS style_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        styles TEXT,
        comfort_level INTEGER,
        archetype TEXT,
        summary TEXT,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''")
    
    # ── Wardrobe Items Table ──────────────────────────────────────────────────
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
        price REAL DEFAULT 0,
        sustainability_score INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')

    # ── Activity Log ───────────────────────────────────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS activity_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        action_type TEXT,
        description TEXT,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')

    # ── User Preferences ──────────────────────────────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_preferences (
        user_id TEXT PRIMARY KEY,
        colors TEXT,
        brands TEXT,
        preferred_categories TEXT,
        disliked_categories TEXT,
        updated_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # ── Saved Outfits Table ──────────────────────────────────────────────────
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
    
    # ── Outfit Wear History ──────────────────────────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS outfit_wear_history (
        wear_id INTEGER PRIMARY KEY AUTOINCREMENT,
        outfit_id TEXT,
        user_id TEXT,
        worn_at TEXT,
        occasion TEXT,
        weather TEXT,
        FOREIGN KEY (outfit_id) REFERENCES saved_outfits (outfit_id),
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # ── Wardrobe Archive (for soft-deleted items) ────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS wardrobe_archive (
        item_id TEXT PRIMARY KEY,
        user_id TEXT,
        name TEXT,
        category TEXT,
        color TEXT,
        fabric TEXT,
        brand TEXT,
        image_url TEXT,
        bg_removed_url TEXT,
        wear_count INTEGER,
        created_at TEXT,
        deleted_at TEXT,
        archive_reason TEXT,
        memory_note TEXT,
        stats_json TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # ── Push Notifications Subscriptions ─────────────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS push_subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT UNIQUE,
        endpoint TEXT,
        p256dh TEXT,
        auth TEXT,
        created_at TEXT,
        updated_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # ── NEW: Outfit Feedback Table (Feature 6) ──────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS outfit_feedback (
        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        outfit_id TEXT,
        item_id TEXT,
        action TEXT CHECK(action IN ('like', 'dislike', 'save', 'wear', 'skip')),
        context TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id),
        FOREIGN KEY (outfit_id) REFERENCES saved_outfits (outfit_id),
        FOREIGN KEY (item_id) REFERENCES wardrobe_items (item_id))''')
    
    # ── NEW: Search History Table (Feature 3) ──────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS search_history (
        search_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        query TEXT NOT NULL,
        results_count INTEGER,
        intent TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # ── NEW: Wear Logs Table (Feature 6) ────────────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS wear_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        outfit_id TEXT,
        occasion TEXT,
        weather TEXT,
        temperature REAL,
        time_of_day TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id),
        FOREIGN KEY (item_id) REFERENCES wardrobe_items (item_id),
        FOREIGN KEY (outfit_id) REFERENCES saved_outfits (outfit_id))''')
    
    # ── NEW: Conversation Memory Table (Feature 8) ──────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS conversation_memory (
        conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        message TEXT NOT NULL,
        response TEXT,
        intent TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # ── NEW: Style Evolution Snapshots (Feature 3) ──────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS style_evolution (
        evolution_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        styles TEXT,
        color_preference TEXT,
        comfort_level TEXT,
        silhouette TEXT,
        snapshot_date TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # ── NEW: Item Pair Scores (Feature 9) ───────────────────────────────────
    cursor.execute('''CREATE TABLE IF NOT EXISTS item_pairs (
        pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        item_id_1 TEXT NOT NULL,
        item_id_2 TEXT NOT NULL,
        pair_score INTEGER DEFAULT 0,
        last_worn_together TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id),
        FOREIGN KEY (item_id_1) REFERENCES wardrobe_items (item_id),
        FOREIGN KEY (item_id_2) REFERENCES wardrobe_items (item_id))''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialization and migration check complete.")


# ── Embedding Helpers ──────────────────────────────────────────────────────────

def save_embedding(conn, item_id: str, embedding_list) -> None:
    """
    Persist a numpy array (or plain list) as JSON in the embedding column.
    Pass embedding_list as _text_to_pseudo_embedding(item).tolist()
    """
    import json
    import numpy as np

    if isinstance(embedding_list, np.ndarray):
        embedding_list = embedding_list.tolist()
    serialized = json.dumps(embedding_list)
    conn.execute(
        "UPDATE wardrobe_items SET embedding = ? WHERE item_id = ?",
        (serialized, item_id)
    )


def load_embedding(conn, item_id: str):
    """
    Load and deserialize the embedding for an item.
    Returns a numpy float32 array, or None if no embedding is stored.
    """
    import json
    import numpy as np

    row = conn.execute(
        "SELECT embedding FROM wardrobe_items WHERE item_id = ?",
        (item_id,)
    ).fetchone()
    if row and row["embedding"]:
        try:
            return np.array(json.loads(row["embedding"]), dtype=np.float32)
        except Exception:
            return None
    return None


# ── NEW: Feedback Helpers ─────────────────────────────────────────────────────

def save_feedback(conn, user_id: str, action: str, outfit_id: str = None, item_id: str = None, context: str = None):
    """Save user feedback for outfit or item."""
    conn.execute(
        """INSERT INTO outfit_feedback (user_id, outfit_id, item_id, action, context, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, outfit_id, item_id, action, context, datetime.now().isoformat())
    )
    conn.commit()


def get_feedback_history(conn, user_id: str, limit: int = 50):
    """Get user's feedback history."""
    return conn.execute(
        """SELECT * FROM outfit_feedback 
           WHERE user_id = ? 
           ORDER BY created_at DESC 
           LIMIT ?""",
        (user_id, limit)
    ).fetchall()


# ── NEW: Wear Log Helpers ────────────────────────────────────────────────────

def log_wear(conn, user_id: str, item_id: str, outfit_id: str = None, 
             occasion: str = None, weather: str = None, temperature: float = None, time_of_day: str = None):
    """Log when an item or outfit is worn."""
    conn.execute(
        """INSERT INTO wear_logs (user_id, item_id, outfit_id, occasion, weather, temperature, time_of_day, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, item_id, outfit_id, occasion, weather, temperature, time_of_day, datetime.now().isoformat())
    )
    conn.commit()


def get_wear_history(conn, user_id: str, days: int = 30):
    """Get wear history for a user."""
    return conn.execute(
        """SELECT * FROM wear_logs 
           WHERE user_id = ? AND created_at >= datetime('now', ?)
           ORDER BY created_at DESC""",
        (user_id, f'-{days} days')
    ).fetchall()


# ── NEW: Search History Helpers ─────────────────────────────────────────────

def log_search(conn, user_id: str, query: str, results_count: int, intent: str = None):
    """Log user search queries."""
    conn.execute(
        """INSERT INTO search_history (user_id, query, results_count, intent, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, query, results_count, intent, datetime.now().isoformat())
    )
    conn.commit()


def get_recent_searches(conn, user_id: str, limit: int = 10):
    """Get user's recent searches."""
    return conn.execute(
        """SELECT * FROM search_history 
           WHERE user_id = ? 
           ORDER BY created_at DESC 
           LIMIT ?""",
        (user_id, limit)
    ).fetchall()


# ── NEW: Conversation Memory Helpers ────────────────────────────────────────

def save_conversation(conn, user_id: str, message: str, response: str, intent: str = None):
    """Save conversation history."""
    conn.execute(
        """INSERT INTO conversation_memory (user_id, message, response, intent, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, message, response, intent, datetime.now().isoformat())
    )
    conn.commit()


def get_recent_conversations(conn, user_id: str, limit: int = 10):
    """Get user's recent conversations."""
    return conn.execute(
        """SELECT * FROM conversation_memory 
           WHERE user_id = ? 
           ORDER BY created_at DESC 
           LIMIT ?""",
        (user_id, limit)
    ).fetchall()


# ── NEW: Style Evolution Helpers ────────────────────────────────────────────

def save_evolution_snapshot(conn, user_id: str, styles: str, color_preference: str = None, 
                            comfort_level: str = None, silhouette: str = None):
    """Save a style evolution snapshot."""
    conn.execute(
        """INSERT INTO style_evolution (user_id, styles, color_preference, comfort_level, silhouette, snapshot_date)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, styles, color_preference, comfort_level, silhouette, datetime.now().isoformat())
    )
    conn.commit()


def get_evolution_history(conn, user_id: str):
    """Get user's style evolution history."""
    return conn.execute(
        """SELECT * FROM style_evolution 
           WHERE user_id = ? 
           ORDER BY snapshot_date ASC""",
        (user_id,)
    ).fetchall()


# ── NEW: Item Pair Helpers ──────────────────────────────────────────────────

def update_pair_score(conn, user_id: str, item_id_1: str, item_id_2: str):
    """Update the pair score when items are worn together."""
    existing = conn.execute(
        """SELECT pair_id, pair_score FROM item_pairs 
           WHERE user_id = ? AND item_id_1 = ? AND item_id_2 = ?""",
        (user_id, item_id_1, item_id_2)
    ).fetchone()
    
    if existing:
        conn.execute(
            """UPDATE item_pairs 
               SET pair_score = pair_score + 1, last_worn_together = ?
               WHERE pair_id = ?""",
            (datetime.now().isoformat(), existing['pair_id'])
        )
    else:
        conn.execute(
            """INSERT INTO item_pairs (user_id, item_id_1, item_id_2, pair_score, last_worn_together)
               VALUES (?, ?, ?, 1, ?)""",
            (user_id, item_id_1, item_id_2, datetime.now().isoformat())
        )
    conn.commit()


def get_top_pairs(conn, user_id: str, limit: int = 10):
    """Get most frequently worn item pairs."""
    return conn.execute(
        """SELECT * FROM item_pairs 
           WHERE user_id = ? 
           ORDER BY pair_score DESC 
           LIMIT ?""",
        (user_id, limit)
    ).fetchall()
