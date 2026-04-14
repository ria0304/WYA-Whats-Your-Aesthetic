# services/weather_service.py
# Real-weather retrieval (Open-Meteo) + outfit/advice generation.

import logging
from typing import Any, Dict

import requests

from .data_loader import GEOAPIFY_API_KEY, WEATHER_CODES

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def weather_styling(city: str) -> Dict[str, Any]:
    """Return real-time weather data and outfit recommendations for *city*."""
    city_title = city.title().strip()
    try:
        lat, lon, city_name, country = _geocode(city_title)
        if lat is None:
            return _fallback(city_title, f"Could not find coordinates for {city_title}")

        weather = _fetch_weather(lat, lon, forecast_days=1)
        if not weather:
            return _fallback(city_name)

        temp        = round(weather["current"].get("temperature_2m", 22))
        feels_like  = round(weather["current"].get("apparent_temperature", temp))
        humidity    = int(weather["current"].get("relative_humidity_2m", 50))
        wind_speed  = round(weather["current"].get("wind_speed_10m", 5))
        code        = weather["current"].get("weather_code", 0)
        condition   = WEATHER_CODES.get(str(code), f"Unknown ({code})")

        daily = weather.get("daily", {})
        min_temp = round(daily.get("temperature_2m_min", [temp])[0]) if daily.get("temperature_2m_min") else temp
        max_temp = round(daily.get("temperature_2m_max", [temp])[0]) if daily.get("temperature_2m_max") else temp

        outfit  = _suggest_outfit(temp, feels_like, condition, wind_speed)
        advice  = _generate_advice(temp, feels_like, condition, wind_speed, humidity, min_temp, max_temp)

        return {
            "city": city_name, "country": country,
            "temp": temp, "feels_like": feels_like,
            "min_temp": min_temp, "max_temp": max_temp,
            "condition": condition, "humidity": humidity, "wind_speed": wind_speed,
            "outfit": outfit, "advice": advice,
            "data_source": "Open-Meteo Weather API (FREE)",
        }
    except Exception as exc:
        logger.error("Weather styling error: %s", exc)
        return _fallback(city_title, str(exc))


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _geocode(city: str):
    """Return (lat, lon, city_name, country) or (None, ...) on failure."""
    try:
        resp = requests.get(
            "https://api.geoapify.com/v1/geocode/search",
            params={"text": city, "apiKey": GEOAPIFY_API_KEY, "limit": 1},
        )
        features = resp.json().get("features", [])
        if not features:
            return None, None, city, ""
        props = features[0]["properties"]
        coords = features[0]["geometry"]["coordinates"]
        return coords[1], coords[0], props.get("city", city), props.get("country", "")
    except Exception as exc:
        logger.error("Geocoding error: %s", exc)
        return None, None, city, ""


def _fetch_weather(lat: float, lon: float, forecast_days: int = 7) -> Dict[str, Any]:
    """Call Open-Meteo and return raw JSON, or {} on failure."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": ["temperature_2m", "relative_humidity_2m",
                            "apparent_temperature", "weather_code", "wind_speed_10m"],
                "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                "timezone": "auto",
                "forecast_days": forecast_days,
            },
        )
        return resp.json()
    except Exception as exc:
        logger.error("Weather API error: %s", exc)
        return {}


def get_weather_data(lat: float, lon: float, city_name: str) -> Dict[str, Any]:
    """Summarised weather dict used by the trip curator."""
    data = _fetch_weather(lat, lon, forecast_days=7)
    if not data:
        return {}
    current = data.get("current", {})
    daily   = data.get("daily", {})
    code    = current.get("weather_code", 0)

    rain_days = sum(1 for p in daily.get("precipitation_sum", []) if p and p > 0)
    avg_temp  = None
    if daily.get("temperature_2m_max") and daily.get("temperature_2m_min"):
        hi, lo = daily["temperature_2m_max"], daily["temperature_2m_min"]
        avg_temp = round(sum((hi[i] + lo[i]) / 2 for i in range(len(hi))) / len(hi), 1)

    return {
        "current_temp":    round(current.get("temperature_2m", 0), 1),
        "feels_like":      round(current.get("apparent_temperature", 0), 1),
        "humidity":        current.get("relative_humidity_2m", 0),
        "wind_speed":      round(current.get("wind_speed_10m", 0), 1),
        "description":     WEATHER_CODES.get(str(code), f"Unknown Code {code}"),
        "min_temp":        round(min(daily.get("temperature_2m_min", [0])), 1) if daily.get("temperature_2m_min") else None,
        "max_temp":        round(max(daily.get("temperature_2m_max", [0])), 1) if daily.get("temperature_2m_max") else None,
        "avg_temperature": avg_temp,
        "rain_days":       rain_days,
    }


def _suggest_outfit(temp: float, feels_like: float, condition: str, wind_speed: float) -> Dict[str, str]:
    ct = feels_like
    is_rainy = any(x in condition.lower() for x in ("rain", "drizzle", "shower", "thunderstorm"))
    is_snowy = any(x in condition.lower() for x in ("snow", "sleet"))

    if   ct > 35: outfit = {"top": "Cotton Vest / Tank Top",   "bottom": "Linen Shorts",           "outerwear": "None",                      "footwear": "Flip Flops / Sandals"}
    elif ct > 32: outfit = {"top": "T-Shirt (Light Cotton)",   "bottom": "Shorts / Linen Pants",   "outerwear": "None",                      "footwear": "Sandals / Breathable Sneakers"}
    elif ct > 28: outfit = {"top": "T-Shirt",                  "bottom": "Shorts / Light Pants",   "outerwear": "None",                      "footwear": "Sneakers / Loafers"}
    elif ct > 25: outfit = {"top": "T-Shirt / Polo",           "bottom": "Jeans / Chinos",         "outerwear": "None",                      "footwear": "Sneakers"}
    elif ct > 22: outfit = {"top": "T-Shirt / Blouse",         "bottom": "Jeans",                  "outerwear": "None",                      "footwear": "Sneakers"}
    elif ct > 20: outfit = {"top": "Long Sleeve Shirt",        "bottom": "Jeans",                  "outerwear": "Light Cardigan (optional)", "footwear": "Sneakers"}
    elif ct > 18: outfit = {"top": "Long Sleeve Shirt",        "bottom": "Jeans / Trousers",       "outerwear": "Light Jacket",              "footwear": "Sneakers / Loafers"}
    elif ct > 15: outfit = {"top": "Sweater / Hoodie",         "bottom": "Jeans",                  "outerwear": "Jacket",                    "footwear": "Boots / Sneakers"}
    elif ct > 12: outfit = {"top": "Sweater",                  "bottom": "Jeans",                  "outerwear": "Heavy Jacket / Coat",       "footwear": "Boots"}
    elif ct > 8:  outfit = {"top": "Thermal + Sweater",        "bottom": "Jeans (thermals opt.)",  "outerwear": "Winter Coat",               "footwear": "Insulated Boots"}
    elif ct > 5:  outfit = {"top": "Thermal + Sweater",        "bottom": "Jeans with Thermals",    "outerwear": "Heavy Winter Coat",         "footwear": "Winter Boots"}
    else:         outfit = {"top": "Thermal + Wool Sweater",   "bottom": "Insulated Pants",        "outerwear": "Heavy Winter Coat",         "footwear": "Snow Boots"}

    if is_rainy:
        outfit["outerwear"] = "Light Rain Jacket" if ct > 20 else "Waterproof Coat"
        outfit["footwear"]  = "Waterproof Sneakers" if ct > 20 else "Waterproof Boots"
        outfit["accessories"] = "Umbrella"
    if is_snowy:
        outfit["outerwear"]   = "Insulated Snow Jacket"
        outfit["footwear"]    = "Snow Boots"
        outfit["accessories"] = "Scarf, Gloves, Beanie"
    if wind_speed > 30:
        if ct > 15 and "Jacket" not in outfit["outerwear"] and "Coat" not in outfit["outerwear"]:
            outfit["outerwear"] = "Windbreaker"
        outfit["accessories"] = outfit.get("accessories", "") + " (secure your hat)"

    return outfit


def _generate_advice(temp, feels_like, condition, wind_speed, humidity, min_temp, max_temp) -> str:
    ct = feels_like
    if   ct > 35: adv = "Extreme heat! Opt for loose, breathable fabrics in light colors. "
    elif ct > 30: adv = "Hot and sunny. Choose lightweight cotton or linen. Sunglasses and a hat are essential. "
    elif ct > 25: adv = "Warm and pleasant. Perfect for casual summer outfits. Light colors will keep you cool. "
    elif ct > 20: adv = "Mild and comfortable. A t-shirt and jeans combo works perfectly. "
    elif ct > 15: adv = "Slightly cool. A light jacket or sweater is recommended. "
    elif ct > 10: adv = "Cool weather. Layer up with a jacket or hoodie. "
    elif ct > 5:  adv = "Cold outside. Wear a warm coat, scarf, and gloves. "
    else:         adv = "Freezing temperatures! Bundle up with thermal layers, heavy coat, and winter accessories. "

    if max_temp - min_temp > 10:
        adv += f"Temperature will vary from {min_temp}°C to {max_temp}°C today - dress in layers. "
    cond_lower = condition.lower()
    if "rain" in cond_lower or "drizzle" in cond_lower: adv += "Don't forget an umbrella and waterproof footwear. "
    elif "snow" in cond_lower:                          adv += "Snow expected! Wear waterproof boots and warm layers. "
    elif "thunderstorm" in cond_lower:                  adv += "Thunderstorms likely. Stay dry. "
    elif "fog" in cond_lower:                           adv += "Foggy - wear bright or reflective clothing. "
    if   wind_speed > 40: adv += "Very strong winds - secure your hat. "
    elif wind_speed > 25: adv += "Windy - a windbreaker might be useful. "
    elif wind_speed > 15: adv += "Breezy - light jacket recommended. "
    if humidity > 80: adv += "High humidity - choose moisture-wicking fabrics. "
    elif humidity < 30: adv += "Low humidity - moisturize and stay hydrated. "
    if "Clear" in condition and ct > 25: adv += "Don't forget sunscreen - UV rays are strong today. "
    return adv.strip()


def _fallback(city: str, error: str = "") -> Dict[str, Any]:
    base = {
        "city": city, "temp": 22, "feels_like": 22, "min_temp": 18, "max_temp": 25,
        "condition": "Unknown", "humidity": 50, "wind_speed": 5,
        "outfit": {"top": "T-Shirt", "bottom": "Jeans", "outerwear": "None", "footwear": "Sneakers"},
        "advice": "Weather data unavailable. Showing default suggestions.",
        "data_source": "Fallback Data",
    }
    if error:
        base["error"] = error
    return base
