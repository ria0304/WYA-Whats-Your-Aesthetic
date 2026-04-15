# services/__init__.py
# Expose the most-used names at package level.

from .brand_auditor       import audit_brand
from .color_matcher       import ColorMatcher
from .computer_vision     import LocalComputerVision, load_fashionclip, load_sam
from .data_loader         import (
    BRAND_SCORES, CATEGORY_MAP, COLOR_DICTIONARY, COLOR_HARMONY,
    COUNTRY_TO_REGION, FASHION_DATA, GEOAPIFY_API_KEY, GLOBAL_CHAINS,
    LOCAL_INDICATORS, REGIONAL_ITEMS, WEATHER_CODES, load_json_data,
)
from .fabric_classifier   import FabricClassifier
from .notification_service import NotificationService
from .outfit_generator    import OutfitGenerator
from .style_profile       import StyleProfile
from .trip_curator        import curate_trip
from .weather_service     import get_weather_data, weather_styling

__all__ = [
    "audit_brand", "ColorMatcher", "LocalComputerVision", "load_sam", "load_fashionclip",
    "FabricClassifier", "NotificationService", "OutfitGenerator",
    "StyleProfile", "curate_trip", "weather_styling", "get_weather_data",
    "load_json_data",
    "BRAND_SCORES", "CATEGORY_MAP", "COLOR_DICTIONARY", "COLOR_HARMONY",
    "COUNTRY_TO_REGION", "FASHION_DATA", "GEOAPIFY_API_KEY", "GLOBAL_CHAINS",
    "LOCAL_INDICATORS", "REGIONAL_ITEMS", "WEATHER_CODES",
]
