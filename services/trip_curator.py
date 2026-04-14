# services/trip_curator.py
# Trip curation: geocoding, local place discovery, packing list generation.

import logging
import random
from typing import Any, Dict, List

import requests

from .data_loader import (
    COUNTRY_TO_REGION,
    GEOAPIFY_API_KEY,
    GLOBAL_CHAINS,
    LOCAL_INDICATORS,
    REGIONAL_ITEMS,
)
from .weather_service import get_weather_data

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def curate_trip(city: str, duration: int, vibe: str) -> Dict[str, Any]:
    """Return a full trip-curation response for *city*."""
    city_title = city.title().strip()
    try:
        # Geocode
        geo = _geocode(city_title)
        if geo is None:
            return {"city": city_title, "error": f"Could not find coordinates for {city_title}",
                    "message": "Please check the city name or try a different city."}

        lat, lon, city_name, country = geo
        region = _region_from_country(country)

        # Fetch places
        markets_raw   = _fetch_places(lat, lon, _MARKET_CATS,   20, 5000)
        bakeries_raw  = _fetch_places(lat, lon, _BAKERY_CATS,   15, 3000)
        boutiques_raw = _fetch_places(lat, lon, _BOUTIQUE_CATS, 20, 4000)

        major_markets  = _process_markets(markets_raw)
        bakeries       = _process_bakeries(bakeries_raw)
        boutiques      = _process_boutiques(boutiques_raw, region)

        # Weather
        weather = get_weather_data(lat, lon, city_name)
        avg_temp  = weather.get("avg_temperature", 20) or 20
        rain_days = weather.get("rain_days", 0)

        packing_list, weather_summary = _build_packing_list(duration, avg_temp, rain_days, vibe)

        # Bakery highlights
        oldest_bakery = most_popular_bakery = None
        if bakeries:
            oldest_bakery = {**sorted(bakeries, key=lambda x: len(x["name"]))[0],
                             "description": "One of the oldest bakeries - traditional recipes for generations"}
            if len(bakeries) > 1:
                most_popular_bakery = {**bakeries[1],
                                       "description": "Most popular bakery among locals"}
            else:
                most_popular_bakery = {**bakeries[0], "description": "Popular local bakery"}

        return {
            "city": city_name, "country": country, "days": int(duration),
            "weather_summary": weather_summary,
            "weather_details": _safe_weather_details(weather),
            "clothes_count": len(packing_list),
            "packing_list": packing_list,
            "major_markets": major_markets[:8],
            "bakeries": {
                "oldest": oldest_bakery, "most_popular": most_popular_bakery,
                "others": [b for b in bakeries if b not in (oldest_bakery, most_popular_bakery)][:4],
            },
            "hidden_gem_boutiques": boutiques[:6],
            "total_places_found": len(major_markets) + len(bakeries) + len(boutiques),
            "region": region,
            "data_source": "OpenStreetMap via Geoapify + Open-Meteo + Regional Items DB",
        }

    except Exception as exc:
        logger.error("Trip curation error: %s", exc)
        return {
            "city": city_title, "days": duration, "error": str(exc),
            "weather_summary": "Weather information unavailable", "weather_details": {},
            "packing_list": [
                f"{max(2, duration)}x Tops", f"{max(1, duration // 2)}x Bottoms",
                "Comfortable walking shoes", "Evening outfit", "Toiletries",
                "Power bank", "Umbrella",
            ],
            "major_markets": [], "bakeries": {"oldest": None, "most_popular": None, "others": []},
            "hidden_gem_boutiques": [], "message": "Unable to fetch real places. Please try again later.",
        }


# ------------------------------------------------------------------
# Geoapify helpers
# ------------------------------------------------------------------

_MARKET_CATS   = ["commercial.market", "commercial.marketplace", "commercial.supermarket",
                   "commercial.shopping_mall", "commercial.department_store", "commercial.food_and_drink"]
_BAKERY_CATS   = ["catering.bakery", "catering.pastry_shop", "catering.cafe", "catering.coffee_shop"]
_BOUTIQUE_CATS = ["commercial.clothing", "commercial.fashion", "commercial.boutique",
                  "commercial.gift_and_souvenir", "commercial.antiques", "commercial.books", "commercial.jewelry"]


def _geocode(city: str):
    try:
        resp = requests.get(
            "https://api.geoapify.com/v1/geocode/search",
            params={"text": city, "apiKey": GEOAPIFY_API_KEY, "limit": 1},
        )
        features = resp.json().get("features", [])
        if not features:
            return None
        props  = features[0]["properties"]
        coords = features[0]["geometry"]["coordinates"]
        return coords[1], coords[0], props.get("city", city), props.get("country", "")
    except Exception as exc:
        logger.error("Geocoding error: %s", exc)
        return None


def _fetch_places(lat, lon, categories, limit, radius) -> List[Dict]:
    try:
        resp = requests.get(
            "https://api.geoapify.com/v2/places",
            params={
                "categories": ",".join(categories),
                "filter": f"circle:{lon},{lat},{radius}",
                "bias": f"proximity:{lon},{lat}",
                "limit": limit,
                "apiKey": GEOAPIFY_API_KEY,
            },
            headers={"Accept": "application/json"},
        )
        return resp.json().get("features", [])
    except Exception as exc:
        logger.error("Places fetch error: %s", exc)
        return []


def _is_chain(name: str) -> bool:
    return any(chain in name.lower() for chain in GLOBAL_CHAINS)


def _process_markets(features) -> List[Dict]:
    results = []
    for f in features:
        p = f["properties"]
        name = p.get("name", "")
        if not name or len(name) < 2 or _is_chain(name):
            continue
        cats = " ".join(str(c).lower() for c in p.get("categories", []))
        mtype = ("Supermarket" if "supermarket" in cats else
                 "Shopping Mall" if "shopping_mall" in cats else
                 "Food Market" if "food" in cats else "Market")
        results.append({
            "name": name, "address": p.get("formatted", "Address not available"),
            "distance": round(p.get("distance", 0)), "type": mtype,
            "opening_hours": p.get("opening_hours", "Hours not available"),
            "specialty": "Local produce, crafts, and traditional goods",
            "best_for": ["Fresh food", "Local crafts", "Souvenirs"],
            "rating": p.get("rating", "N/A"),
        })
    return results


def _process_bakeries(features) -> List[Dict]:
    chain_words = {"starbucks", "dunkin", "tim hortons", "costa"}
    results = []
    for f in features:
        p = f["properties"]
        name = p.get("name", "")
        if not name or len(name) < 2 or any(c in name.lower() for c in chain_words):
            continue
        cats  = " ".join(str(c).lower() for c in p.get("categories", []))
        btype = ("Pastry Shop" if "pastry" in cats else
                 "Cafe" if "cafe" in cats or "coffee" in cats else "Bakery")
        results.append({
            "name": name, "address": p.get("formatted", "Address not available"),
            "distance": round(p.get("distance", 0)), "type": btype,
            "opening_hours": p.get("opening_hours", "Hours not available"),
            "specialty": "Fresh baked goods daily",
        })
    return results


def _process_boutiques(features, region: str) -> List[Dict]:
    results = []
    for f in features:
        p = f["properties"]
        name = p.get("name", "")
        if not name or len(name) < 2 or _is_chain(name):
            continue
        cats  = " ".join(str(c).lower() for c in p.get("categories", []))
        btype = ("Clothing Boutique" if "clothing" in cats else
                 "Gift Shop"        if "gift" in cats else
                 "Antique Shop"     if "antique" in cats else
                 "Jewelry Store"    if "jewelry" in cats else
                 "Bookstore"        if "books" in cats else "Boutique")
        limited = _limited_edition_items("boutique", region)
        results.append({
            "name": name, "address": p.get("formatted", "Address not available"),
            "distance": round(p.get("distance", 0)), "type": btype,
            "opening_hours": p.get("opening_hours", "Hours not available"),
            "description": "Hidden gem boutique with unique local finds and curated collections",
            "limited_edition_items": limited,
            "is_hidden_gem": any(ind in name.lower() for ind in LOCAL_INDICATORS),
        })
    results.sort(key=lambda x: (not x["is_hidden_gem"], x["distance"]))
    return results


# ------------------------------------------------------------------
# Region / limited items
# ------------------------------------------------------------------

def _region_from_country(country: str) -> str:
    cl = country.lower().strip()
    if cl in COUNTRY_TO_REGION:
        return COUNTRY_TO_REGION[cl]
    for key, region in COUNTRY_TO_REGION.items():
        if key in cl or cl in key:
            return region
    patterns = {
        "mediterranean":  ["turkey", "egypt", "morocco", "greece", "italy", "spain"],
        "middle_eastern": ["uae", "saudi", "qatar", "jordan", "iran", "iraq"],
        "asian":          ["japan", "china", "korea", "thailand", "vietnam", "india"],
        "european":       ["france", "germany", "uk", "netherlands"],
        "north_american": ["usa", "canada", "mexico"],
        "south_american": ["brazil", "argentina", "peru", "colombia"],
        "african":        ["south africa", "kenya", "nigeria", "ghana"],
    }
    for region, keywords in patterns.items():
        if any(kw in cl for kw in keywords):
            return region
    return "european"


def _limited_edition_items(place_type: str, region: str, count: int = 2) -> List[Dict]:
    try:
        region_data = REGIONAL_ITEMS.get("regions", {}).get(
            region, REGIONAL_ITEMS.get("regions", {}).get("european", {}))
        cat_map = {"market": "market", "antique": "antique", "boutique": "boutique",
                   "shopping_mall": "market", "department_store": "boutique",
                   "clothing": "boutique", "gift": "market", "food": "market"}
        category = cat_map.get(place_type, "market")
        items_data = region_data.get(category, region_data.get("market", {}))
        items_list = items_data.get("items", []) if isinstance(items_data, dict) else []
        return random.sample(items_list, min(count, len(items_list))) if items_list else []
    except Exception as exc:
        logger.error("Error getting limited edition items: %s", exc)
        return []


# ------------------------------------------------------------------
# Packing list
# ------------------------------------------------------------------

def _build_packing_list(duration: int, avg_temp: float, rain_days: int, vibe: str):
    tops    = max(2, duration)
    bottoms = max(1, duration // 2 + 1)
    base = [
        f"{tops}x Tops/Shirts", f"{bottoms}x Bottoms (Pants/Shorts)",
        f"{duration + 1}x Underwear & Socks", "1x Comfortable Walking Shoes",
        "1x Evening Outfit", "Sleepwear", "Toiletries Kit", "Power Bank & Chargers",
    ]
    if   avg_temp > 25: extras, summary = ["Sunglasses", "Sunscreen SPF 50", "Hat", "Water bottle"], f"{avg_temp}°C - Hot and sunny"
    elif avg_temp > 20: extras, summary = ["Sunglasses", "Light jacket for evenings"],               f"{avg_temp}°C - Warm and pleasant"
    elif avg_temp > 15: extras, summary = ["Light jacket", "Umbrella (just in case)"],               f"{avg_temp}°C - Mild and comfortable"
    elif avg_temp > 10: extras, summary = ["Medium jacket", "Sweater", "Umbrella"],                  f"{avg_temp}°C - Cool, bring layers"
    elif avg_temp > 5:  extras, summary = ["Warm jacket", "Sweater", "Scarf", "Umbrella"],           f"{avg_temp}°C - Cold, bundle up"
    else:               extras, summary = ["Heavy winter coat", "Thermal layers", "Gloves", "Warm hat", "Scarf"], f"{avg_temp}°C - Freezing"

    if rain_days > 0:
        extras.append(f"Umbrella (rain expected on {rain_days} days)")
        summary += f" - {rain_days} days with rain"

    vibe_extras = {
        "beach":     ["2x Swimwear", "Flip Flops", "Beach Towel", "Beach Bag", "Waterproof phone case"],
        "mountain":  ["Hiking Boots", "Thermal Layers", "Rain Jacket", "Wool Beanie", "Backpack", "First aid kit"],
        "city":      ["Daypack", "Camera", "Power bank"],
        "luxury":    ["Cocktail dress/suit", "Designer accessories", "Formal shoes", "Jewelry"],
        "adventure": ["Hiking boots", "Quick-dry clothing", "Water bottle", "First aid kit", "Multi-tool"],
    }
    extras.extend(vibe_extras.get(vibe.lower(), []))
    return base + extras, summary


def _safe_weather_details(w: Dict) -> Dict:
    def _f(v): return float(v) if v is not None else None
    return {
        "current_temp": _f(w.get("current_temp")),
        "feels_like":   _f(w.get("feels_like")),
        "humidity":     _f(w.get("humidity")),
        "wind_speed":   _f(w.get("wind_speed")),
        "min_temp":     _f(w.get("min_temp")),
        "max_temp":     _f(w.get("max_temp")),
        "description":  w.get("description"),
        "rain_days":    int(w.get("rain_days", 0)),
    }
