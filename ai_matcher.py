# ai_model.py - Thin orchestrator for all AI features
# Delegates to specialized services for similarity matching, outfit generation, gap analysis, etc.

import json
import logging
import random
import base64
import os
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

from services.brand_auditor import audit_brand
from services.color_matcher import ColorMatcher
from services.computer_vision import LocalComputerVision
from services.data_loader import COLOR_HARMONY, FASHION_DATA
from services.fabric_classifier import FabricClassifier
from services.style_profile import StyleProfile
from services.trip_curator import curate_trip
from services.weather_service import weather_styling
from services.outfit_generator import OutfitGenerator
from services.notification_service import NotificationService

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

logger = logging.getLogger(__name__)

try:
    from ai_matcher import fashion_matcher
except ImportError:
    fashion_matcher = None
    logger.warning("ai_matcher not found, using fallback matching")


# ====================== FASHION AI MODEL ======================

class FashionAIModel:
    """
    Thin orchestrator. Each method delegates to the relevant service.
    Supports: similarity matching, color harmony, gap analysis, outfit generation,
    background removal, aesthetic aura, and push notifications.
    """

    vision = LocalComputerVision()
    classifier = FabricClassifier()
    outfit_generator = OutfitGenerator()

    _user_profiles: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Style-DNA / profile management
    # ------------------------------------------------------------------

    @staticmethod
    async def set_user_style_profile(user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        if "answers" in profile_data:
            profile = StyleProfile.extract_from_questionnaire(profile_data["answers"])
        else:
            profile = {**StyleProfile.get_default_profile(), **profile_data}
        FashionAIModel._user_profiles[user_id] = profile
        return {
            "success": True,
            "profile": profile,
            "message": f"Style DNA mapped to {profile['style_archetype']}",
        }

    @staticmethod
    def get_user_style_profile(user_id: str) -> Dict[str, Any]:
        return FashionAIModel._user_profiles.get(user_id, StyleProfile.get_default_profile())

    @staticmethod
    def get_style_evolution(user_id: str) -> Dict[str, Any]:
        profile = FashionAIModel.get_user_style_profile(user_id)
        today = datetime.now().strftime("%b %d").upper()
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%b %d").upper()
        two_wks_ago = (datetime.now() - timedelta(days=14)).strftime("%b %d").upper()

        timeline = [
            {"date": today, "archetype": profile.get("style_archetype", "Classic"),
             "style": (profile.get("style_vibes") or ["Classic"])[0], "mood": "Current",
             "progress": profile.get("comfort_level", 50)},
            {"date": week_ago, "archetype": "Classic", "style": "Classic & Sophisticated",
             "mood": "Exploring", "progress": 45},
            {"date": two_wks_ago, "archetype": "Casual", "style": "Casual Comfort",
             "mood": "Exploring", "progress": 40},
        ]
        cp_name = profile.get("color_preference_name", "Classic Monochrome")
        cp_colors = profile.get("color_preference_colors", ["Black", "White", "Gray"])

        return {
            "current_style": {
                "archetype": profile.get("style_archetype", "Classic"),
                "aesthetic": profile.get("everyday_look", "Classic & Sophisticated"),
                "color_preference": cp_name,
                "colors": cp_colors,
                "comfort_level": profile.get("comfort_level", 50),
                "silhouette": profile.get("silhouette_preference", "Draped & Flowing"),
            },
            "timeline": timeline,
            "insights": {
                "dominant_style": profile.get("style_archetype", "Classic"),
                "style_change": "+10%",
                "color_preferences": profile.get("preferred_colors", ["Black", "White"])[:5],
                "style_confidence": profile.get("comfort_level", 50),
                "recommendations": [
                    f"Try adding more {cp_colors[0] if cp_colors else 'neutral'} pieces",
                    "Experiment with layering this season",
                    "Your style is evolving toward more structured silhouettes",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Garment analysis (autotag)
    # ------------------------------------------------------------------

    @staticmethod
    async def autotag_garment(image_data: str) -> Dict[str, Any]:
        """Use HuggingFace API for garment classification + local color extraction."""
        try:
            if not image_data or not isinstance(image_data, str):
                raise ValueError("Invalid image_data: must be non-empty string")

            # Extract base64 image bytes
            if "," in image_data:
                b64 = image_data.split(",", 1)[1]
            else:
                b64 = image_data
            image_bytes = base64.b64decode(b64)

            # ── Category classification via HuggingFace ────────────────────────
            category_labels = [
                "t-shirt", "dress", "jeans", "jacket", "skirt", "blouse",
                "sweater", "shorts", "coat", "shoes", "bag", "accessories",
                "trousers", "jumpsuit", "top"
            ]
            hf_response = requests.post(
                "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32",
                headers=HF_HEADERS,
                json={
                    "inputs": {"image": b64, "candidate_labels": category_labels}
                },
                timeout=30
            )

            category = "Top"
            confidence = 0.7
            if hf_response.status_code == 200:
                results = hf_response.json()
                if isinstance(results, list) and results:
                    top = results[0]
                    label = top.get("label", "top").lower()
                    confidence = top.get("score", 0.7)
                    category_map = {
                        "t-shirt": "T-Shirt", "dress": "Dress", "jeans": "Jeans",
                        "jacket": "Jacket", "skirt": "Skirt", "blouse": "Top",
                        "sweater": "Sweater", "shorts": "Shorts", "coat": "Outerwear",
                        "shoes": "Shoes", "bag": "Bag", "accessories": "Accessories",
                        "trousers": "Trousers", "jumpsuit": "Jumpsuit", "top": "Top"
                    }
                    category = category_map.get(label, "Top")

            # ── Color extraction via local CV ──────────────────────────────────
            img = FashionAIModel.vision.decode_image(image_data)
            color_name = "Gray"
            hex_color = "#808080"
            rgb = [128, 128, 128]
            secondary_color = ""

            if img is not None and img.size > 0:
                try:
                    mask = FashionAIModel.vision.get_improved_mask(img)
                    hex_color, color_name, rgb_tuple = FashionAIModel.vision.get_dominant_color(img, mask)
                    rgb = [int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2])]
                    secondary_color = getattr(FashionAIModel.vision, "_last_secondary_color", None) or ""
                except Exception:
                    pass

            # ── Fabric classification ──────────────────────────────────────────
            fabric_labels = ["cotton", "denim", "silk", "linen", "wool", "leather", "polyester", "chiffon", "satin"]
            fabric = "Cotton"
            fabric_response = requests.post(
                "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32",
                headers=HF_HEADERS,
                json={"inputs": {"image": b64, "candidate_labels": fabric_labels}},
                timeout=30
            )
            if fabric_response.status_code == 200:
                fab_results = fabric_response.json()
                if isinstance(fab_results, list) and fab_results:
                    fabric = fab_results[0].get("label", "cotton").capitalize()

            # ── Build name ────────────────────────────────────────────────────
            _GENERIC_FABRICS = {"Cotton", "Polyester", "Synthetic"}
            name_parts = []
            if secondary_color and secondary_color != color_name:
                name_parts.append(f"{color_name} & {secondary_color}")
            else:
                name_parts.append(color_name)
            if fabric not in _GENERIC_FABRICS:
                name_parts.append(fabric)
            name_parts.append(category)
            name = " ".join(name_parts)

            return {
                "success": True,
                "name": name,
                "category": str(category),
                "fabric": str(fabric),
                "color": str(color_name),
                "secondary_color": str(secondary_color),
                "hex_color": str(hex_color),
                "rgb": rgb,
                "pattern": "solid",
                "has_pattern": False,
                "details": f"AI Scan: {fabric} {category} | Color: {color_name} ({hex_color})",
                "confidence": float(confidence),
                "texture_variance": 0.0,
                "brightness": 0.0,
                "mask_coverage": 0.0,
            }

        except Exception as exc:
            logger.error("Autotag error: %s", exc)
            return {
                "success": False, "error": str(exc),
                "name": "Cotton Item", "category": "Top", "fabric": "Cotton",
                "color": "Gray", "hex_color": "#808080", "rgb": [128, 128, 128], "confidence": 0.0,
            }

    # ------------------------------------------------------------------
    # Outfit suggestions with similarity matching (Feature 1)
    # ------------------------------------------------------------------

    @staticmethod
    async def get_outfit_suggestion(
        image_data: str, variation: int = 0, user_id: str = None, season: str = "summer"
    ) -> Dict[str, Any]:
        """Get outfit suggestions with real similarity matching."""
        try:
            tag = await FashionAIModel.autotag_garment(image_data)
            if not tag.get("success"):
                raise ValueError("Failed to identify garment")

            category = tag["category"]
            color = tag["color"]
            rgb = tag["rgb"]
            detected = tag["name"]

            profile = (FashionAIModel.get_user_style_profile(user_id)
                       if user_id else StyleProfile.get_default_profile())
            archetype = profile.get("style_archetype", "Casual")
            vibes = profile.get("style_vibes", [])
            silhouette = profile.get("silhouette_preference", "Draped & Flowing")

            vibe_map = {"Minimalist": "minimalist", "Avant-Garde": "avant-garde",
                        "Classic": "classic", "Boho": "boho", "Streetwear": "streetwear", "Casual": "casual"}
            vibe_key = vibe_map.get(archetype.title(), "casual")

            random.seed(hash(image_data[:100]) + variation * 1000 + int(datetime.utcnow().timestamp() % 100))

            matching = ColorMatcher.get_matching_colors((rgb[0], rgb[1], rgb[2]), variation, count=6)
            best = matching[1 if variation % 3 == 0 and len(matching) > 2 else 0] if matching else {
                "color": "White", "hex": "#ffffff", "rgb": [255, 255, 255],
                "match_type": "fallback", "confidence": 0.5, "reason": "Classic white",
            }
            best_color = best["color"]

            shoe_sug = await FashionAIModel._style_suggestion("shoes", vibe_key, season, variation, best_color)
            jewelry_sug = await FashionAIModel._style_suggestion("jewelry/accessory", vibe_key, season, variation + 1, best_color)
            bag_sug = await FashionAIModel._style_suggestion("bag", vibe_key, season, variation + 2, best_color)

            match_piece = (f"{best_color} Top" if category in ("Pants", "Shorts", "Skirt") else
                           f"{best_color} Bottom" if category == "Top" else
                           "" if category in ("Dress", "Jumpsuit") else
                           f"{best_color} Matching Piece")

            styling_tip = await FashionAIModel._style_tip(vibe_key, best_color, season, variation)
            dna_msg = f"Based on your {', '.join(vibes) if vibes else archetype} Style DNA"

            return {
                "style_dna": dna_msg,
                "season": f" {season.capitalize()}",
                "vibe": archetype.title(),
                "identified_item": detected,
                "match_piece": match_piece,
                "jewelry": jewelry_sug,
                "shoes": shoe_sug,
                "bag": bag_sug,
                "best_match": best,
                "matching_colors": matching,
                "styling_tips": styling_tip,
                "silhouette": silhouette,
            }

        except Exception as exc:
            logger.error("Suggestion failed: %s", exc)
            return {
                "style_dna": "Based on your Casual Style DNA", "season": " Summer",
                "vibe": "Casual", "identified_item": "Cotton Item", "match_piece": "",
                "jewelry": "Simple accessory", "shoes": "Classic sneakers", "bag": "Versatile tote",
                "best_match": {"color": "White", "hex": "#ffffff", "rgb": [255, 255, 255],
                               "match_type": "fallback", "confidence": 0.5, "reason": "Classic white"},
                "matching_colors": [], "styling_tips": "Keep it simple and comfortable",
                "silhouette": "Relaxed",
            }

    @staticmethod
    async def _style_suggestion(item_type, vibe_key, season, variation, best_color) -> str:
        try:
            items = (FASHION_DATA.get(vibe_key, FASHION_DATA.get("casual", {}))
                     .get(season, FASHION_DATA.get("casual", {}).get("summer", {}))
                     .get(item_type, []))
            if not items:
                items = (FASHION_DATA.get("casual", {}).get(season,
                          FASHION_DATA.get("casual", {}).get("summer", {}))
                         .get(item_type, ["Classic option"]))
            suggestion = items[(variation + hash(item_type)) % len(items)]
            if (variation + hash(suggestion)) % 2 == 0:
                suggestion = f"{suggestion} in {best_color}"
            return suggestion
        except Exception:
            return {"shoes": "Classic sneakers", "jewelry/accessory": "Simple accessory",
                    "bag": "Versatile tote"}.get(item_type, "Stylish option")

    _STYLE_TIPS: Dict[str, Dict[str, List[str]]] = {
        "minimalist": {
            "summer": ["Keep accessories minimal - let the clean lines speak.",
                       "Choose breathable linens and cottons for hot days.",
                       "Less is more - let your accent piece be the statement."],
            "winter": ["Focus on quality wool and cashmere for warmth without bulk.",
                       "Layer thoughtfully - a fine turtleneck under a structured coat.",
                       "Invest in well-tailored basics that last."],
        },
        "boho": {
            "summer": ["Layer lightweight textures like crochet, lace, and gauze.",
                       "Mix earthy tones with pops of turquoise or coral.",
                       "Don't be afraid to mix patterns and textures."],
            "winter": ["Layer chunky knits over flowing skirts with tights.",
                       "Mix warm textures like suede, wool, and velvet."],
        },
        "streetwear": {
            "summer": ["Play with proportions - oversized tees with bike shorts.",
                       "Fresh white sneakers complete any summer streetwear look.",
                       "Add technical fabrics for an urban edge."],
            "winter": ["Oversized puffers are both warm and stylish.",
                       "Layer hoodies under denim jackets for warmth."],
        },
        "classic": {
            "summer": ["Invest in quality linen and cotton basics.",
                       "A well-tailored white shirt elevates any summer outfit.",
                       "Stick to a neutral palette with one accent color."],
            "winter": ["A cashmere sweater in a neutral tone is worth the investment.",
                       "Well-tailored wool trousers create clean lines."],
        },
        "avant-garde": {
            "summer": ["Let your outfit be a conversation starter.",
                       "Mix unexpected lightweight textures like mesh and organza.",
                       "Asymmetric cuts create visual intrigue."],
            "winter": ["Sculptural coats become the focal point.",
                       "Mix structured and flowing elements."],
        },
        "casual": {
            "summer": ["Comfort is key - choose soft, breathable fabrics.",
                       "Well-fitted basics create an effortlessly cool look.",
                       "A great pair of white sneakers grounds any casual outfit."],
            "winter": ["Comfort is key - choose soft sweaters and warm layers.",
                       "Boots and beanies complete the cozy look."],
        },
    }

    @staticmethod
    async def _style_tip(vibe_key: str, accent_color: str, season: str, variation: int = 0) -> str:
        tips = FashionAIModel._STYLE_TIPS.get(vibe_key, FashionAIModel._STYLE_TIPS["casual"])
        season_tips = tips.get(season, tips.get("summer", ["Style is personal - wear what makes you feel confident!"]))
        tip = season_tips[variation % len(season_tips)]
        return tip.replace("{accent_color}", accent_color)

    # ------------------------------------------------------------------
    # Wardrobe-level methods (Gap Analysis, Outfit Generation)
    # ------------------------------------------------------------------

    @staticmethod
    async def generate_outfits_from_wardrobe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate outfits using color harmony (Feature 2)."""
        if not fashion_matcher:
            return []
        return FashionAIModel.outfit_generator.generate_outfits_from_wardrobe(items)

    @staticmethod
    async def get_gap_analysis(user_id: str, wardrobe_items: List[Dict[str, Any]], db_conn) -> Dict[str, Any]:
        """Analyze wardrobe gaps based on Style DNA (Feature 3)."""
        return FashionAIModel.outfit_generator.analyze_wardrobe_gaps(user_id, wardrobe_items, db_conn)

    @staticmethod
    async def get_aesthetic_aura(user_id: str, wardrobe_items: List[Dict[str, Any]], db_conn) -> Dict[str, Any]:
        """Generate aesthetic aura data for share card (Feature 8)."""
        return FashionAIModel.outfit_generator.generate_aesthetic_aura(user_id, wardrobe_items, db_conn)

    @staticmethod
    def get_evolution_data(items: List[Dict[str, Any]], history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
        """Get style evolution timeline data."""
        if not fashion_matcher:
            return {
                "timeline": [],
                "insights": {"dominant_style": "Casual", "style_change": "0%",
                             "color_preferences": [], "style_confidence": 50,
                             "wardrobe_size": len(items), "recommendations": []},
            }
        analysis = fashion_matcher.analyze_wardrobe_gaps(items)

        def _fmt(iso_str):
            try:
                return datetime.fromisoformat(iso_str).strftime("%b %d")
            except Exception:
                return "Unknown"

        timeline = []
        for i, entry in enumerate(history):
            try:
                style_list = json.loads(entry.get("styles", "[]"))
            except Exception:
                style_list = []
            primary = style_list[0] if style_list else entry.get("archetype", "Mapped")
            mood = ("Clean & Sharp" if "Minimalist" in primary else
                    "Bold & Expressive" if "Streetwear" in primary else
                    "Nostalgic" if "Vintage" in primary else "Exploring")
            timeline.append({
                "period": _fmt(entry.get("created_at", "")),
                "stage": primary,
                "style": " & ".join(style_list[:2]),
                "color": "Personalized",
                "mood": mood,
                "progress": entry.get("comfort_level", 50),
                "items": len(items),
                "key_item": "DNA Profile",
                "is_current": i == 0,
            })

        return {
            "timeline": timeline,
            "insights": {
                "dominant_style": analysis.get("dominant_style", "Casual"),
                "style_change": f"{analysis.get('wardrobe_health_score', 50)}%",
                "color_preferences": analysis.get("color_preferences", [])[:5],
                "style_confidence": analysis.get("wardrobe_health_score", 50),
                "wardrobe_size": analysis.get("total_items", len(items)),
                "recommendations": analysis.get("recommendations", []),
            },
        }

    # ------------------------------------------------------------------
    # Background Removal (Feature 7)
    # ------------------------------------------------------------------

    @staticmethod
    async def remove_background(image_data: str) -> Dict[str, Any]:
        """Remove background using HuggingFace API."""
        try:
            if "," in image_data:
                b64 = image_data.split(",", 1)[1]
            else:
                b64 = image_data
            image_bytes = base64.b64decode(b64)

            response = requests.post(
                "https://api-inference.huggingface.co/models/briaai/RMBG-1.4",
                headers=HF_HEADERS,
                data=image_bytes,
                timeout=60
            )

            if response.status_code == 200:
                result_b64 = base64.b64encode(response.content).decode("utf-8")
                return {
                    "success": True,
                    "bg_removed_image": result_b64,
                    "message": "Background removed successfully"
                }
            else:
                raise ValueError(f"HuggingFace API error: {response.status_code}")

        except Exception as exc:
            logger.error(f"Background removal failed: {exc}")
            return {
                "success": False,
                "error": str(exc),
                "bg_removed_image": None
            }

    # ------------------------------------------------------------------
    # Push Notifications (Feature 10)
    # ------------------------------------------------------------------

    @staticmethod
    async def save_push_subscription(user_id: str, subscription_data: Dict[str, Any], db_conn) -> bool:
        """Save push notification subscription."""
        return await NotificationService.save_subscription(user_id, subscription_data, db_conn)

    # ------------------------------------------------------------------
    # Legacy / compat helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_color_suggestions(base_color: str, category: str = None) -> Dict[str, Any]:
        suggestions: Dict[str, Any] = {
            "complementary": [], "analogous": [], "monochromatic": [],
            "seasonal": {"summer": [], "winter": [], "spring": [], "fall": []},
        }
        for key in ("complementary", "analogous"):
            if base_color in COLOR_HARMONY.get(key, {}):
                suggestions[key] = COLOR_HARMONY[key][base_color]
        for family, colors in COLOR_HARMONY.get("monochromatic", {}).items():
            if base_color in colors or base_color == family:
                suggestions["monochromatic"] = [c for c in colors if c != base_color]
                break
        if not suggestions["monochromatic"]:
            suggestions["monochromatic"] = ["Black", "White", "Gray"]
        if not suggestions["complementary"]:
            suggestions["complementary"] = COLOR_HARMONY.get("neutrals", [])
        for season, colors in COLOR_HARMONY.get("seasonal", {}).items():
            suggestions["seasonal"][season] = [c for c in colors if c != base_color][:3]
        for key in ("complementary", "analogous", "monochromatic"):
            suggestions[key] = list(dict.fromkeys(suggestions[key]))[:5]
        return suggestions

    @staticmethod
    def _fallback_color_match(base_color: str) -> str:
        return {
            "Black": "White", "White": "Black", "Navy": "Beige", "Blue": "White",
            "Denim": "White", "Gray": "Black", "Red": "Black", "Green": "Beige",
            "Yellow": "Navy", "Pink": "Gray", "Purple": "Black", "Orange": "Navy",
            "Brown": "Cream", "Beige": "Navy",
        }.get(base_color, "White")

    # Delegate trip / weather / brand to service modules
    @staticmethod
    def curate_trip(city: str, duration: int, vibe: str) -> Dict[str, Any]:
        from services.trip_curator import curate_trip as _ct
        return _ct(city, duration, vibe)

    @staticmethod
    def weather_styling(city: str) -> Dict[str, Any]:
        from services.weather_service import weather_styling as _ws
        return _ws(city)

    @staticmethod
    async def audit_brand(brand: str) -> Dict[str, Any]:
        from services.brand_auditor import audit_brand as _ab
        return await _ab(brand)


# ====================== MODULE-LEVEL HELPERS ======================

def best_match_color_from_context(color_context: str) -> str:
    known = ["Black", "White", "Navy", "Beige", "Denim", "Gray", "Olive", "Camel",
             "Red", "Silver", "Burgundy", "Emerald", "Mustard", "Coral", "Lavender",
             "Rust", "Sage", "Blush"]
    for c in known:
        if c.lower() in color_context.lower():
            return c
    return "complementary"


def item_to_dict(item_str: str) -> Dict[str, Any]:
    parts = item_str.split()
    color = parts[0] if parts else "Unknown"
    category = next(
        (cat for cat in ("Top", "Pants", "Shorts", "Skirt", "Dress", "Jumpsuit")
         if cat in item_str), "Unknown"
    )
    if category == "Unknown" and "Jeans" in item_str:
        category = "Pants"
    return {"color": color, "category": category, "fabric": "Unknown"}


def suggestion_to_dict(suggestion: str, color: str, item_type: str) -> Dict[str, Any]:
    cat = {"shoes": "Shoes", "jewelry/accessory": "Accessory", "bag": "Bag",
           "top": "Top", "bottom": "Pants", "outerwear": "Outerwear",
           "jacket/blazer": "Jacket"}.get(item_type, "Unknown")
    return {"color": color, "category": cat, "fabric": "Unknown"}
