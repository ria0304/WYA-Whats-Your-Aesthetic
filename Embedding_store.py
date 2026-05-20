# embedding_store.py — FAISS index manager for WYA semantic recommendation engine
# Manages one FAISS flat L2 index per user, persisted to /app/data/faiss/
# Falls back gracefully if faiss-cpu is not installed.

import json
import logging
import os
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FAISS_DIR = os.getenv("FAISS_DIR", "/app/data/faiss")

# ---------------------------------------------------------------------------
# Optional FAISS import — fall back silently so the app boots without it
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
    logger.info("FAISS loaded successfully — semantic search enabled")
except ImportError:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed — semantic search will fall back to linear scan")


def _index_path(user_id: str) -> str:
    return os.path.join(FAISS_DIR, f"{user_id}.index")


def _ids_path(user_id: str) -> str:
    return os.path.join(FAISS_DIR, f"{user_id}.ids.json")


def _ensure_dir() -> None:
    os.makedirs(FAISS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(user_id: str, items: List[Dict[str, Any]]) -> bool:
    """
    Build (or rebuild) a FAISS flat L2 index from a list of wardrobe item dicts.
    Stores the index and a parallel item_id list to disk.
    Returns True on success, False on failure / FAISS unavailable.
    """
    if not _FAISS_AVAILABLE or not items:
        return False

    from ai_matcher import _text_to_pseudo_embedding  # local import to avoid circular deps

    try:
        _ensure_dir()
        embeddings = []
        item_ids = []

        for item in items:
            emb = _text_to_pseudo_embedding(item)
            embeddings.append(emb)
            item_ids.append(item.get("item_id") or item.get("id", ""))

        vectors = np.array(embeddings, dtype=np.float32)
        dim = vectors.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        faiss.write_index(index, _index_path(user_id))
        with open(_ids_path(user_id), "w") as f:
            json.dump(item_ids, f)

        logger.info("FAISS index built — user=%s items=%d dim=%d", user_id[:8], len(items), dim)
        return True

    except Exception as exc:
        logger.error("FAISS build_index failed — user=%s error=%s", user_id[:8], exc)
        return False


def search(user_id: str, query_embedding: np.ndarray, top_k: int = 10) -> List[str]:
    """
    Search the FAISS index for the top-K most similar item_ids.
    Returns a list of item_id strings (may be shorter than top_k if wardrobe is small).
    Returns [] if index doesn't exist or FAISS is unavailable.
    """
    if not _FAISS_AVAILABLE:
        return []

    idx_path = _index_path(user_id)
    ids_path = _ids_path(user_id)

    if not os.path.exists(idx_path) or not os.path.exists(ids_path):
        return []

    try:
        index = faiss.read_index(idx_path)
        with open(ids_path) as f:
            item_ids = json.load(f)

        k = min(top_k, index.ntotal)
        if k == 0:
            return []

        vec = np.array([query_embedding], dtype=np.float32)
        _distances, indices = index.search(vec, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(item_ids):
                results.append(item_ids[idx])

        logger.debug("FAISS search — user=%s top_k=%d results=%d", user_id[:8], top_k, len(results))
        return results

    except Exception as exc:
        logger.error("FAISS search failed — user=%s error=%s", user_id[:8], exc)
        return []


def add_item(user_id: str, item: Dict[str, Any]) -> bool:
    """
    Add a single item to an existing FAISS index.
    If no index exists yet, builds one from scratch with just this item.
    Returns True on success.
    """
    if not _FAISS_AVAILABLE:
        return False

    from ai_matcher import _text_to_pseudo_embedding

    idx_path = _index_path(user_id)
    ids_path = _ids_path(user_id)

    try:
        _ensure_dir()
        emb = _text_to_pseudo_embedding(item)
        vec = np.array([emb], dtype=np.float32)
        item_id = item.get("item_id") or item.get("id", "")

        if os.path.exists(idx_path) and os.path.exists(ids_path):
            index = faiss.read_index(idx_path)
            with open(ids_path) as f:
                item_ids = json.load(f)

            index.add(vec)
            item_ids.append(item_id)

            faiss.write_index(index, idx_path)
            with open(ids_path, "w") as f:
                json.dump(item_ids, f)
        else:
            # No index yet — bootstrap with this single item
            dim = vec.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(vec)
            faiss.write_index(index, idx_path)
            with open(ids_path, "w") as f:
                json.dump([item_id], f)

        logger.info("FAISS item added — user=%s item=%s", user_id[:8], str(item_id)[:8])
        return True

    except Exception as exc:
        logger.error("FAISS add_item failed — user=%s error=%s", user_id[:8], exc)
        return False


def delete_index(user_id: str) -> bool:
    """
    Delete persisted FAISS index for a user (e.g. after a wardrobe delete).
    Next search will return [] and trigger a rebuild via the rebuild endpoint.
    """
    removed = False
    for path in (_index_path(user_id), _ids_path(user_id)):
        if os.path.exists(path):
            try:
                os.remove(path)
                removed = True
            except Exception as exc:
                logger.error("FAISS delete_index failed — user=%s path=%s error=%s", user_id[:8], path, exc)
    if removed:
        logger.info("FAISS index deleted — user=%s", user_id[:8])
    return removed


def index_exists(user_id: str) -> bool:
    """Return True if a valid index file exists for this user."""
    return os.path.exists(_index_path(user_id)) and os.path.exists(_ids_path(user_id))
