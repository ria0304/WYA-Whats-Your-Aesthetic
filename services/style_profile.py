# services/style_profile.py
# Converts questionnaire answers into a structured style-DNA profile.

from typing import Any, Dict, List


class StyleProfile:
    """Manages user style preferences and profiles."""

    _LOOK_MAPPING = {
        "Minimalist & Clean": {
            "archetype": "Minimalist",
            "keywords": ["clean-lined", "understated", "monochromatic", "streamlined"],
            "colors": ["Monochrome"],
            "silhouette": "Tailored & Structured",
            "color_preference_name": "Modern Minimalist",
            "color_preference_colors": ["Black", "White", "Gray", "Charcoal"],
        },
        "Bold & Experimental": {
            "archetype": "Avant-Garde",
            "keywords": ["bold", "experimental", "statement", "unique", "artistic"],
            "colors": ["Bright", "Contrast"],
            "silhouette": "Architectural",
            "color_preference_name": "Avant-Garde Edge",
            "color_preference_colors": ["Red", "Royal Blue", "Emerald", "Yellow", "Fuchsia"],
        },
        "Classic & Sophisticated": {
            "archetype": "Classic",
            "keywords": ["timeless", "elegant", "polished", "refined", "sophisticated"],
            "colors": ["Monochrome", "Neutrals"],
            "silhouette": "Tailored & Structured",
            "color_preference_name": "Timeless Elegance",
            "color_preference_colors": ["Navy", "Beige", "Cream", "Burgundy", "Brown"],
        },
        "Bohemian & Relaxed": {
            "archetype": "Boho",
            "keywords": ["flowy", "relaxed", "earthy", "layered", "free-spirited"],
            "colors": ["Earthy", "Warm"],
            "silhouette": "Draped & Flowing",
            "color_preference_name": "Warm Sunrise",
            "color_preference_colors": ["Orange", "Pink", "Coral", "Peach", "Terracotta"],
        },
        "Streetwear & Edgy": {
            "archetype": "Streetwear",
            "keywords": ["edgy", "urban", "oversized", "graphic", "casual-cool"],
            "colors": ["Monochrome", "Bold"],
            "silhouette": "Oversized & Relaxed",
            "color_preference_name": "Urban Edge",
            "color_preference_colors": ["Black", "Charcoal", "Red", "Olive", "Mustard"],
        },
    }

    @staticmethod
    def get_default_profile() -> Dict[str, Any]:
        return {
            "everyday_look": "Casual Comfort",
            "style_archetype": "Casual",
            "style_vibes": ["Casual"],
            "preferred_colors": ["Black", "White", "Gray"],
            "color_categories": ["Monochrome"],
            "color_preference_name": "Classic Monochrome",
            "color_preference_colors": ["Black", "White", "Gray"],
            "silhouette_preference": "Draped & Flowing",
            "avoid_colors": [],
            "preferred_fabrics": ["Cotton"],
            "comfort_level": 5,
            "style_keywords": ["comfortable", "versatile"],
            "body_type": "Unknown",
            "occasion_preferences": {
                "casual": "Relaxed and comfortable",
                "work": "Smart casual",
                "formal": "Classic and elegant",
                "party": "Bold and stylish",
            },
        }

    @staticmethod
    def extract_from_questionnaire(answers: Dict[str, Any]) -> Dict[str, Any]:
        """Convert questionnaire answers to a style profile dict."""
        profile = StyleProfile.get_default_profile()
        lm = StyleProfile._LOOK_MAPPING

        everyday_looks: List[str] = answers.get("everyday_look", [])
        if isinstance(everyday_looks, str):
            everyday_looks = [everyday_looks]

        profile["style_vibes"] = everyday_looks
        all_keywords: List[str] = []

        for look in everyday_looks:
            if look in lm:
                all_keywords.extend(lm[look]["keywords"])

        if everyday_looks:
            primary = everyday_looks[0]
            mapped = lm.get(primary, {})
            profile["style_archetype"]          = mapped.get("archetype", "Casual")
            profile["everyday_look"]             = primary
            profile["color_preference_name"]     = mapped.get("color_preference_name", "Custom Palette")
            profile["color_preference_colors"]   = mapped.get("color_preference_colors", ["Black", "White", "Gray"])

        profile["style_keywords"] = list(dict.fromkeys(all_keywords))

        # Colour preference
        cp = answers.get("color_preference", "")
        profile["color_preference"] = cp
        if "Monochrome" in cp:
            profile.update(color_categories=["Monochrome"],
                           preferred_colors=["Black", "White", "Gray", "Charcoal", "Silver"],
                           color_preference_name="Classic Monochrome",
                           color_preference_colors=["Black", "White", "Gray", "Charcoal"])
        elif "Pastels" in cp:
            profile.update(color_categories=["Pastels"],
                           preferred_colors=["Blush", "Lavender", "Mint", "Baby Blue", "Cream", "Powder Pink"],
                           color_preference_name="Soft Pastels",
                           color_preference_colors=["Blush", "Lavender", "Mint", "Baby Blue"])
        elif "Earthy" in cp or "Warm" in cp:
            profile.update(color_categories=["Earthy"],
                           preferred_colors=["Olive", "Brown", "Tan", "Rust", "Sage", "Camel", "Terracotta"],
                           color_preference_name="Earthy Tones",
                           color_preference_colors=["Olive", "Brown", "Tan", "Rust", "Terracotta"])
        elif "Bright" in cp:
            profile.update(color_categories=["Bright"],
                           preferred_colors=["Red", "Royal Blue", "Emerald", "Yellow", "Coral", "Fuchsia"],
                           color_preference_name="Vibrant Brights",
                           color_preference_colors=["Red", "Royal Blue", "Emerald", "Yellow", "Coral"])

        # Silhouette
        silhouette = answers.get("silhouette", "Draped & Flowing")
        profile["silhouette_preference"] = silhouette
        extra = {
            "Tailored & Structured": ["tailored", "structured", "sharp"],
            "Draped & Flowing":      ["flowy", "draped", "soft"],
            "Oversized & Relaxed":   ["oversized", "relaxed", "comfortable"],
        }.get(silhouette, [])
        profile["style_keywords"] = list(dict.fromkeys(profile["style_keywords"] + extra))

        return profile
