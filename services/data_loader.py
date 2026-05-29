import os
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def load_json_data(file_path: str, default: Any = None) -> Any:
    """Load data from a JSON file with graceful fallback."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning("JSON file not found: %s", file_path)
    except Exception as exc:
        logger.error("Error loading %s: %s", file_path, exc)
    return default if default is not None else {}

FASHION_DATA      = load_json_data(os.path.join(DATA_DIR, "fashion_data.json"), {})
BRAND_SCORES      = load_json_data(os.path.join(DATA_DIR, "brand_score.json"), {})
REGIONAL_ITEMS    = load_json_data(os.path.join(DATA_DIR, "regional_items.json"), {"regions": {}})
WEATHER_CODES     = load_json_data(os.path.join(DATA_DIR, "weather_codes.json"), {})
CATEGORY_MAP      = load_json_data(os.path.join(DATA_DIR, "category_map.json"), {})
COLOR_DICTIONARY  = load_json_data(os.path.join(DATA_DIR, "color_dictionary.json"), {})
COLOR_HARMONY     = load_json_data(os.path.join(DATA_DIR, "color_harmony.json"), {})
COUNTRY_TO_REGION = load_json_data(os.path.join(DATA_DIR, "country_to_region.json"), {})
GLOBAL_CHAINS     = load_json_data(os.path.join(DATA_DIR, "global_chains.json"), [])
LOCAL_INDICATORS  = load_json_data(os.path.join(DATA_DIR, "local_indicators.json"), [])
