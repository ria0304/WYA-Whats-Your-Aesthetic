# services/data_loader.py


import os
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

GEOAPIFY_API_KEY = "3e48b0904db44b08993cd8c2aa999b4f"


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


FASHION_DATA     = load_json_data("fashion_data.json", {})
BRAND_SCORES     = load_json_data("brand_score.json", {})
REGIONAL_ITEMS   = load_json_data("regional_items.json", {"regions": {}})
WEATHER_CODES    = load_json_data("weather_codes.json", {})
CATEGORY_MAP     = load_json_data("category_map.json", {})
COLOR_DICTIONARY = load_json_data("color_dictionary.json", {})
COLOR_HARMONY    = load_json_data("color_harmony.json", {})
COUNTRY_TO_REGION = load_json_data("country_to_region.json", {})
GLOBAL_CHAINS    = load_json_data("global_chains.json", [])
LOCAL_INDICATORS = load_json_data("local_indicators.json", [])
