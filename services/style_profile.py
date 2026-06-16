# services/style_profile.py
# Converts questionnaire answers into a structured style-DNA profile.
# Also tracks style evolution over time.

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
            profile["style_archetype"] = mapped.get("archetype", "Casual")
            profile["everyday_look"] = primary
            profile["color_preference_name"] = mapped.get("color_preference_name", "Custom Palette")
            profile["color_preference_colors"] = mapped.get("color_preference_colors", ["Black", "White", "Gray"])

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
            "Draped & Flowing": ["flowy", "draped", "soft"],
            "Oversized & Relaxed": ["oversized", "relaxed", "comfortable"],
        }.get(silhouette, [])
        profile["style_keywords"] = list(dict.fromkeys(profile["style_keywords"] + extra))

        return profile

   

    @staticmethod
    def track_evolution(
        current_profile: Dict[str, Any],
        previous_profile: Dict[str, Any],
        db_conn = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Compare current and previous style profiles to track evolution.
        Stores snapshot in database if db_conn and user_id provided.
        """
        if not previous_profile:
            return {
                "has_changed": False,
                "message": "No previous profile found — this is your first profile."
            }

        changes = []
        evolution_summary = []

        # 1. Check archetype change
        current_archetype = current_profile.get("style_archetype", "Unknown")
        previous_archetype = previous_profile.get("style_archetype", "Unknown")
        if current_archetype != previous_archetype:
            changes.append({
                "field": "archetype",
                "from": previous_archetype,
                "to": current_archetype,
                "description": f"Style shifted from {previous_archetype} to {current_archetype}"
            })
            evolution_summary.append(f"Changed from {previous_archetype} to {current_archetype}")

        # 2. Check color preference change
        current_colors = set(current_profile.get("color_preference_colors", []))
        previous_colors = set(previous_profile.get("color_preference_colors", []))
        if current_colors != previous_colors:
            added = current_colors - previous_colors
            removed = previous_colors - current_colors
            changes.append({
                "field": "colors",
                "from": list(previous_colors),
                "to": list(current_colors),
                "added": list(added),
                "removed": list(removed),
                "description": f"Color palette evolved"
            })
            if added:
                evolution_summary.append(f"Added colors: {', '.join(added)}")
            if removed:
                evolution_summary.append(f"Dropped colors: {', '.join(removed)}")

        # 3. Check silhouette change
        current_silhouette = current_profile.get("silhouette_preference", "Unknown")
        previous_silhouette = previous_profile.get("silhouette_preference", "Unknown")
        if current_silhouette != previous_silhouette:
            changes.append({
                "field": "silhouette",
                "from": previous_silhouette,
                "to": current_silhouette,
                "description": f"Silhouette changed from {previous_silhouette} to {current_silhouette}"
            })
            evolution_summary.append(f"Changed silhouette from {previous_silhouette} to {current_silhouette}")

        # 4. Check comfort level change
        current_comfort = current_profile.get("comfort_level", 5)
        previous_comfort = previous_profile.get("comfort_level", 5)
        if current_comfort != previous_comfort:
            changes.append({
                "field": "comfort_level",
                "from": previous_comfort,
                "to": current_comfort,
                "description": f"Comfort level changed from {previous_comfort} to {current_comfort}"
            })
            evolution_summary.append(f"Comfort level changed from {previous_comfort} to {current_comfort}")

        # 5. Check style vibes change
        current_vibes = set(current_profile.get("style_vibes", []))
        previous_vibes = set(previous_profile.get("style_vibes", []))
        if current_vibes != previous_vibes:
            added_vibes = current_vibes - previous_vibes
            removed_vibes = previous_vibes - current_vibes
            changes.append({
                "field": "style_vibes",
                "from": list(previous_vibes),
                "to": list(current_vibes),
                "added": list(added_vibes),
                "removed": list(removed_vibes),
                "description": "Style vibes evolved"
            })
            if added_vibes:
                evolution_summary.append(f"Added vibes: {', '.join(added_vibes)}")
            if removed_vibes:
                evolution_summary.append(f"Dropped vibes: {', '.join(removed_vibes)}")

        # 6. Save snapshot to database
        if db_conn and user_id:
            try:
                db_conn.execute("""
                    INSERT INTO style_evolution 
                    (user_id, styles, color_preference, comfort_level, silhouette, snapshot_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    json.dumps(current_profile.get("style_vibes", [])),
                    json.dumps(current_profile.get("color_preference_colors", [])),
                    str(current_profile.get("comfort_level", 5)),
                    current_profile.get("silhouette_preference", "Unknown"),
                    datetime.now().isoformat()
                ))
                db_conn.commit()
            except Exception as e:
                logger.warning(f"Failed to save evolution snapshot: {e}")

        return {
            "has_changed": len(changes) > 0,
            "changes": changes,
            "summary": " -> ".join(evolution_summary) if evolution_summary else "No significant changes detected",
            "evolution_score": len(changes) * 10,
            "timestamp": datetime.now().isoformat(),
            "current_archetype": current_archetype,
            "previous_archetype": previous_archetype,
        }


    @staticmethod
    def get_profile_analytics(
        profile: Dict[str, Any],
        wardrobe_items: List[Dict[str, Any]],
        wear_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate analytics based on style profile + wardrobe data.
        """
        total_items = len(wardrobe_items)
        archetype = profile.get("style_archetype", "Unknown")

        # Count items matching style archetype
        matching_items = 0
        for item in wardrobe_items:
            item_category = item.get("category", "")
            if item_category.lower() in [v.lower() for v in profile.get("style_vibes", [])]:
                matching_items += 1

        # Calculate style alignment score
        alignment_score = round((matching_items / max(total_items, 1)) * 100, 1)

        # Color alignment
        preferred_colors = set(profile.get("color_preference_colors", []))
        color_matches = 0
        for item in wardrobe_items:
            item_color = item.get("color", "")
            if item_color in preferred_colors:
                color_matches += 1
        color_alignment = round((color_matches / max(total_items, 1)) * 100, 1)

        # Silhouette alignment
        preferred_silhouette = profile.get("silhouette_preference", "")
        silhouette_matches = 0
        for item in wardrobe_items:
            item_silhouette = item.get("silhouette", "")
            if preferred_silhouette.lower() in item_silhouette.lower():
                silhouette_matches += 1
        silhouette_alignment = round((silhouette_matches / max(total_items, 1)) * 100, 1)

        return {
            "archetype": archetype,
            "total_items": total_items,
            "matching_items": matching_items,
            "alignment_score": alignment_score,
            "color_alignment": color_alignment,
            "silhouette_alignment": silhouette_alignment,
            "preferred_colors": list(preferred_colors),
            "style_vibes": profile.get("style_vibes", []),
            "comfort_level": profile.get("comfort_level", 5),
            "recommendations": StyleProfile._generate_profile_recommendations(
                profile, alignment_score, color_alignment, silhouette_alignment
            )
        }


    @staticmethod
    def _generate_profile_recommendations(
        profile: Dict[str, Any],
        alignment_score: float,
        color_alignment: float,
        silhouette_alignment: float
    ) -> List[str]:
        """Generate personalized recommendations based on profile."""
        recommendations = []

        if alignment_score < 50:
            recommendations.append(
                f"Your wardrobe doesn't strongly reflect your {profile.get('style_archetype', '')} style. "
                f"Consider adding more items that match your aesthetic."
            )
        elif alignment_score < 75:
            recommendations.append(
                f"Good start on your {profile.get('style_archetype', '')} style! "
                f"Adding a few more key pieces will complete the look."
            )
        else:
            recommendations.append(
                f"Your wardrobe strongly reflects your {profile.get('style_archetype', '')} style. "
                f"Well curated!"
            )

        if color_alignment < 50:
            recommendations.append(
                f"Your preferred colors are {', '.join(profile.get('color_preference_colors', [])[:3])}. "
                f"Try adding more items in these colors."
            )

        if silhouette_alignment < 50:
            recommendations.append(
                f"Your preferred silhouette is {profile.get('silhouette_preference', '')}. "
                f"Look for items with this silhouette to enhance your style."
            )

        return recommendations

 

    @staticmethod
    def analyze_evolution_over_time(
        snapshots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze style evolution across multiple snapshots.
        """
        if not snapshots or len(snapshots) < 2:
            return {
                "has_evolution": False,
                "message": "Not enough data to analyze evolution",
                "trajectory": [],
                "current": snapshots[0] if snapshots else None
            }

        trajectory = []
        current_archetype = snapshots[-1].get("styles", ["Unknown"])[0] if snapshots[-1].get("styles") else "Unknown"
        first_archetype = snapshots[0].get("styles", ["Unknown"])[0] if snapshots[0].get("styles") else "Unknown"

        for i, snapshot in enumerate(snapshots):
            styles = snapshot.get("styles", [])
            trajectory.append({
                "timestamp": snapshot.get("snapshot_date", ""),
                "archetype": styles[0] if styles else "Unknown",
                "colors": snapshot.get("color_preference", []),
                "comfort_level": snapshot.get("comfort_level", 5),
            })

        evolution_direction = "stable"
        if current_archetype != first_archetype:
            evolution_direction = "changed"
            # Determine direction
            archetype_order = ["Minimalist", "Classic", "Boho", "Streetwear", "Avant-Garde"]
            if current_archetype in archetype_order and first_archetype in archetype_order:
                if archetype_order.index(current_archetype) > archetype_order.index(first_archetype):
                    evolution_direction = "becoming more experimental"
                else:
                    evolution_direction = "becoming more classic/minimal"

        return {
            "has_evolution": True,
            "trajectory": trajectory,
            "current_archetype": current_archetype,
            "first_archetype": first_archetype,
            "evolution_direction": evolution_direction,
            "total_snapshots": len(snapshots),
            "timespan": {
                "start": snapshots[0].get("snapshot_date", ""),
                "end": snapshots[-1].get("snapshot_date", "")
            }
        }


# Module-level singleton
style_profile = StyleProfile()
