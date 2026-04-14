# services/color_matcher.py
# Suggests harmonious matching colors for a given garment RGB value.

import colorsys
import random
from typing import Any, Dict, List, Tuple

from .data_loader import COLOR_DICTIONARY


class ColorMatcher:
    """
    Intelligent color matching for fashion.
    Suggests matching colors based on exact shade, saturation, and brightness.
    """

    COLOR_FAMILIES = {
        "red":    ["Red", "Burgundy", "Maroon", "Wine", "Brick Red", "Crimson"],
        "pink":   ["Pink", "Blush", "Rose", "Fuchsia", "Magenta", "Coral", "Peach"],
        "orange": ["Orange", "Coral", "Peach", "Rust", "Terracotta", "Apricot"],
        "yellow": ["Yellow", "Mustard", "Gold", "Amber", "Honey"],
        "green":  ["Green", "Emerald", "Mint", "Sage", "Olive", "Forest Green", "Lime", "Hunter Green"],
        "blue":   ["Blue", "Navy", "Royal Blue", "Sky Blue", "Baby Blue", "Teal", "Turquoise",
                   "Denim", "Cobalt", "Midnight Blue"],
        "purple": ["Purple", "Lavender", "Lilac", "Mauve", "Plum", "Eggplant", "Violet"],
        "brown":  ["Brown", "Tan", "Camel", "Beige", "Cream", "Taupe", "Chocolate", "Coffee", "Caramel"],
        "gray":   ["Gray", "Charcoal", "Silver", "Slate", "Ash"],
        "black":  ["Black", "Jet Black", "Onyx"],
        "white":  ["White", "Off-White", "Ivory", "Cream", "Eggshell"],
    }

    # ------------------------------------------------------------------
    # Color-space helpers
    # ------------------------------------------------------------------

    @staticmethod
    def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        r, g, b = (x / 255.0 for x in rgb)
        return colorsys.rgb_to_hsv(r, g, b)

    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    # ------------------------------------------------------------------
    # Color analysis
    # ------------------------------------------------------------------

    @staticmethod
    def get_color_properties(rgb: Tuple[int, int, int]) -> Dict[str, Any]:
        h, s, v = ColorMatcher.rgb_to_hsv(rgb)
        hue_deg = h * 360

        saturation_level = (
            "very_low" if s < 0.15 else
            "low"      if s < 0.30 else
            "medium"   if s < 0.50 else
            "high"     if s < 0.70 else
            "very_high"
        )
        brightness_level = (
            "very_dark"   if v < 0.20 else
            "dark"        if v < 0.35 else
            "medium_dark" if v < 0.50 else
            "medium"      if v < 0.65 else
            "light"       if v < 0.80 else
            "very_light"
        )

        return {
            "rgb": rgb,
            "hue": hue_deg,
            "saturation": s,
            "value": v,
            "color_family":      ColorMatcher._get_color_family(hue_deg, s, v),
            "color_name":        ColorMatcher._rgb_to_color_name(rgb),
            "saturation_level":  saturation_level,
            "brightness_level":  brightness_level,
        }

    @staticmethod
    def _get_color_family(hue_deg: float, s: float, v: float) -> str:
        if v < 0.15:              return "black"
        if v > 0.9 and s < 0.1:  return "white"
        if s < 0.1:               return "gray"
        if hue_deg < 15 or hue_deg >= 345: return "red"
        if hue_deg < 35:  return "orange"
        if hue_deg < 50:  return "yellow"
        if hue_deg < 80:  return "yellow_green"
        if hue_deg < 150: return "green"
        if hue_deg < 190: return "teal"
        if hue_deg < 260: return "blue"
        if hue_deg < 330: return "purple"
        return "pink"

    @staticmethod
    def _rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
        r, g, b = rgb
        best, min_dist = "Gray", float("inf")
        for name, val in COLOR_DICTIONARY.items():
            dist = (r - val[0]) ** 2 + (g - val[1]) ** 2 + (b - val[2]) ** 2
            if dist < min_dist:
                min_dist, best = dist, name
        return best

    # ------------------------------------------------------------------
    # Suggestion engine
    # ------------------------------------------------------------------

    @staticmethod
    def get_matching_colors(
        garment_rgb: Tuple[int, int, int],
        variation: int = 0,
        count: int = 6,
    ) -> List[Dict[str, Any]]:
        """Return up to *count* harmonious color suggestions with variation."""
        props = ColorMatcher.get_color_properties(garment_rgb)
        random.seed(variation + int(props["hue"] * 100))
        suggestions: List[Dict[str, Any]] = []

        def _swatch(rgb, match_type, reasons):
            name = ColorMatcher._rgb_to_color_name(rgb)
            return {
                "color": name,
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                "rgb": [int(x) for x in rgb],
                "match_type": match_type,
                "confidence": round(random.uniform(0.82, 0.98), 2),
                "reason": random.choice(reasons),
            }

        # Complementary
        comp_hue = ((props["hue"] + 180 + variation * 3) % 360) / 360
        comp_rgb = ColorMatcher.hsv_to_rgb(
            comp_hue,
            min(1.0, props["saturation"] * random.uniform(1.1, 1.3)),
            min(1.0, props["value"] * random.uniform(1.05, 1.15)),
        )
        suggestions.append(_swatch(comp_rgb, "complementary", [
            "Creates a striking contrast that makes both colors pop",
            "Opposite colors that complement each other perfectly",
            "Bold contrast for a statement look",
        ]))

        # Split-complementary (2 swatches)
        for offset in random.sample([150, 210], 2):
            split_hue = ((props["hue"] + offset + variation * 5) % 360) / 360
            suggestions.append(_swatch(
                ColorMatcher.hsv_to_rgb(split_hue,
                    props["saturation"] * random.uniform(0.85, 0.95),
                    props["value"]      * random.uniform(0.85, 0.95)),
                "split_complementary",
                ["Softer contrast than complementary, very harmonious",
                 "Elegant and subtle - a sophisticated choice"],
            ))

        # Analogous (2 swatches)
        for offset in random.sample([30, -30], 2):
            analog_hue = ((props["hue"] + offset + variation * 2) % 360) / 360
            suggestions.append(_swatch(
                ColorMatcher.hsv_to_rgb(analog_hue,
                    props["saturation"] * random.uniform(0.75, 0.85),
                    min(1.0, props["value"] * random.uniform(0.95, 1.05))),
                "analogous",
                ["Harmonious and easy on the eyes", "Serene and cohesive color palette"],
            ))

        # Monochromatic
        bl = props["brightness_level"]
        if bl in ("dark", "very_dark", "medium_dark"):
            new_v = min(1.0, props["value"] + random.uniform(0.35, 0.45))
        elif bl in ("light", "very_light"):
            new_v = max(0.2, props["value"] - random.uniform(0.35, 0.45))
        else:
            new_v = (min(1.0, props["value"] + random.uniform(0.25, 0.35))
                     if variation % 2 == 0
                     else max(0.2, props["value"] - random.uniform(0.25, 0.35)))
        suggestions.append(_swatch(
            ColorMatcher.hsv_to_rgb(props["hue"] / 360,
                props["saturation"] * random.uniform(0.65, 0.85), new_v),
            "monochromatic",
            ["Elegant tonal dressing - sophisticated and chic",
             "Subtle variation on your base color"],
        ))

        # Neutrals
        neutrals = [
            {"name": "White",  "rgb": (255, 255, 255), "reasons": ["Crisp and clean - lets your garment shine"]},
            {"name": "Black",  "rgb": (0, 0, 0),       "reasons": ["Timeless and elegant - creates definition"]},
            {"name": "Cream",  "rgb": (255, 253, 208),  "reasons": ["Soft and warm - effortless sophistication"]},
            {"name": "Beige",  "rgb": (245, 245, 220),  "reasons": ["Versatile neutral that complements any color"]},
            {"name": "Gray",   "rgb": (128, 128, 128),  "reasons": ["Modern and understated - perfect balance"]},
            {"name": "Navy",   "rgb": (0, 0, 128),      "reasons": ["Classic alternative to black - rich and deep"]},
        ]
        random.shuffle(neutrals)
        added = 0
        for n in neutrals:
            if added >= random.randint(2, 3):
                break
            nh, ns, nv = ColorMatcher.rgb_to_hsv(n["rgb"])
            if ColorMatcher._get_color_family(nh * 360, ns, nv) != props["color_family"]:
                suggestions.append({
                    "color": n["name"],
                    "hex": "#{:02x}{:02x}{:02x}".format(*n["rgb"]),
                    "rgb": list(n["rgb"]),
                    "match_type": "neutral",
                    "confidence": round(random.uniform(0.75, 0.85), 2),
                    "reason": random.choice(n["reasons"]),
                })
                added += 1

        # Deduplicate
        seen: set = set()
        unique = [s for s in suggestions
                  if s["color"] not in seen
                  and not seen.add(s["color"])  
                  and s["color"] != props["color_name"]]

        unique.sort(key=lambda x: x["confidence"], reverse=True)
        top = unique[: count + 2]
        random.shuffle(top)
        return top[:count]

    @staticmethod
    def get_best_match(garment_rgb: Tuple[int, int, int], variation: int = 0) -> Dict[str, Any]:
        matches = ColorMatcher.get_matching_colors(garment_rgb, variation, count=3)
        if matches:
            return random.choice(matches)
        return {
            "color": "White", "hex": "#ffffff", "rgb": [255, 255, 255],
            "match_type": "fallback", "confidence": 0.5,
            "reason": "Classic white - always a safe choice",
        }
