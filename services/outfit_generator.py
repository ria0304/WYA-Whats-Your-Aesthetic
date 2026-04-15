# services/outfit_generator.py
# Outfit generation, daily drop, gap analysis, and aesthetic aura.
# Delegates heavy lifting to ai_matcher.AdvancedFashionMatcher.

import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    from ai_matcher import fashion_matcher, _text_to_pseudo_embedding, cosine_similarity
    MATCHER_AVAILABLE = True
except ImportError:
    MATCHER_AVAILABLE = False
    fashion_matcher = None
    logger.warning("ai_matcher not available — outfit_generator running in fallback mode")

from .data_loader import COLOR_HARMONY
from .style_profile import StyleProfile


class OutfitGenerator:
    """
    Generates outfits, daily drops, aesthetic auras, and gap analysis results.
    All methods are pure functions that receive wardrobe data — no DB state.
    """

    # ------------------------------------------------------------------
    # Outfit generation (Feature 2 – color harmony seed logic)
    # ------------------------------------------------------------------

    def generate_outfits_from_wardrobe(
        self, items: List[Dict[str, Any]], count: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate `count` outfit suggestions from the wardrobe using color harmony."""
        if not MATCHER_AVAILABLE or not items:
            return []
        outfits = []
        styles = ['casual', 'formal', 'boho', 'streetwear', 'classic']
        for i in range(min(count, len(styles))):
            outfit = fashion_matcher.create_complete_outfit(items, style=styles[i])
            if outfit:
                outfits.append(outfit)
        return outfits

    # ------------------------------------------------------------------
    # Daily Drop (Feature 2)
    # ------------------------------------------------------------------

    def generate_daily_drop(
        self,
        user_id: str,
        wardrobe_items: List[Dict[str, Any]],
        location: str = None,
    ) -> Dict[str, Any]:
        """Generate today's outfit drop using color harmony seed logic."""
        if not wardrobe_items:
            return {'error': 'No wardrobe items found', 'outfit': None}

        # Deterministic seed per user per day so the drop is stable
        day_seed = int(datetime.utcnow().strftime('%Y%m%d')) + hash(user_id or 'anon')
        random.seed(day_seed)

        season = self._current_season()
        style = random.choice(['casual', 'classic', 'streetwear', 'boho', 'formal'])

        if MATCHER_AVAILABLE:
            outfit = fashion_matcher.create_complete_outfit(wardrobe_items, style=style, occasion='daytime')
        else:
            outfit = self._fallback_outfit(wardrobe_items)

        return {
            'outfit': outfit,
            'season': season,
            'style': style,
            'generated_at': datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Gap Analysis (Feature 3) — delegates to AdvancedFashionMatcher
    # ------------------------------------------------------------------

    def analyze_wardrobe_gaps(
        self,
        user_id: str,
        wardrobe_items: List[Dict[str, Any]],
        db_conn,
    ) -> Dict[str, Any]:
        """
        Compare user's Style DNA against inventory using cosine embedding similarity.
        Pulls DNA style labels from the DB, then asks fashion_matcher to find gaps.
        """
        style_dna_labels: List[str] = []

        try:
            row = db_conn.execute(
                "SELECT styles FROM style_dna WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (user_id,)
            ).fetchone()
            if row and row['styles']:
                style_dna_labels = json.loads(row['styles'])
        except Exception as exc:
            logger.warning("Could not fetch Style DNA for gap analysis: %s", exc)

        if MATCHER_AVAILABLE:
            return fashion_matcher.analyze_wardrobe_gaps(wardrobe_items, style_dna_labels)
        return self._fallback_gap_analysis(wardrobe_items)

    # ------------------------------------------------------------------
    # Aesthetic Aura (Feature 11)
    # ------------------------------------------------------------------

    def generate_aesthetic_aura(
        self,
        user_id: str,
        wardrobe_items: List[Dict[str, Any]],
        db_conn,
    ) -> Dict[str, Any]:
        """
        Derive aesthetic percentage breakdown from Style DNA + wardrobe composition.
        Returns data consumed by AestheticAura.tsx.
        """
        style_dna: Dict[str, Any] = {}
        try:
            row = db_conn.execute(
                """SELECT styles, archetype, comfort_level, created_at
                   FROM style_dna WHERE user_id = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id,)
            ).fetchone()
            if row:
                style_dna = {
                    'styles': json.loads(row['styles'] or '[]'),
                    'archetype': row['archetype'],
                    'comfort_level': row['comfort_level'],
                }
        except Exception as exc:
            logger.warning("Could not fetch Style DNA for aura: %s", exc)

        styles = style_dna.get('styles', ['Classic Chic'])
        archetype = style_dna.get('archetype', styles[0] if styles else 'Classic Chic')

        # Map archetype → aesthetic label used in frontend AESTHETIC_CONFIGS
        archetype_map = {
            'Minimalist': 'Minimalist', 'Classic': 'Classic Chic', 'Boho': 'Bohemian',
            'Streetwear': 'Streetwear', 'Avant-Garde': 'Avant-Garde', 'Casual': 'Classic Chic',
        }
        primary = archetype_map.get(archetype, archetype)
        secondary = archetype_map.get(styles[1], styles[1]) if len(styles) > 1 else 'Minimalist'
        tertiary = archetype_map.get(styles[2], styles[2]) if len(styles) > 2 else 'Eclectic'

        # Wardrobe-based adjustments: dominant color → palette
        color_counts: Dict[str, int] = defaultdict(int)
        category_counts: Dict[str, int] = defaultdict(int)
        for item in wardrobe_items:
            color_counts[item.get('color', 'Unknown')] += 1
            category_counts[item.get('category', 'Unknown')] += 1

        top_category = max(category_counts, key=category_counts.get) if category_counts else 'Top'
        dominant_colors = self._colors_to_hex(
            sorted(color_counts, key=color_counts.get, reverse=True)[:5]
        )

        season = self._current_season()
        mood_map = {
            'Minimalist': 'Effortlessly Curated', 'Classic Chic': 'Timelessly Refined',
            'Bohemian': 'Free-Spirited Soul', 'Streetwear': 'Culture Architect',
            'Avant-Garde': 'Fearlessly Original', 'Y2K Futurist': 'Born Iconic',
            'Grunge-Core': 'Raw & Unapologetic',
        }

        return {
            'primary_aesthetic': primary,
            'primary_percent': 60,
            'secondary_aesthetic': secondary,
            'secondary_percent': 28,
            'tertiary_aesthetic': tertiary,
            'tertiary_percent': 12,
            'mood_tag': mood_map.get(primary, 'Effortlessly Curated'),
            'season_tag': f'{season.capitalize()} Soul',
            'dominant_colors': dominant_colors,
            'wardrobe_count': len(wardrobe_items),
            'top_category': top_category,
            'has_dna': bool(style_dna),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_season(self) -> str:
        month = datetime.utcnow().month
        if month in (3, 4, 5):
            return 'spring'
        if month in (6, 7, 8):
            return 'summer'
        if month in (9, 10, 11):
            return 'fall'
        return 'winter'

    _COLOR_HEX: Dict[str, str] = {
        'Black': '#1a1a1a', 'White': '#f5f5f5', 'Gray': '#9e9e9e', 'Charcoal': '#3d3d3d',
        'Navy': '#1b2a4a', 'Blue': '#2979ff', 'Denim': '#4a6fa5', 'Teal': '#00796b',
        'Green': '#388e3c', 'Sage': '#7c9a73', 'Olive': '#6d7c47', 'Mint': '#a8d5b5',
        'Red': '#c62828', 'Burgundy': '#6a1b1b', 'Rust': '#b84c20', 'Coral': '#e8735a',
        'Orange': '#ef6c00', 'Yellow': '#f9a825', 'Mustard': '#c8930a', 'Gold': '#c9a227',
        'Brown': '#6d4c41', 'Camel': '#c19a6b', 'Beige': '#d4b896', 'Tan': '#c4a06e',
        'Cream': '#f5efe0', 'Pink': '#e91e8c', 'Blush': '#f4a7b9', 'Rose': '#c9748a',
        'Mauve': '#9e6b7e', 'Lavender': '#9575cd', 'Purple': '#6a1b9a', 'Plum': '#4a0e5e',
        'Silver': '#bdbdbd', 'Terracotta': '#b5541c',
    }

    def _colors_to_hex(self, color_names: List[str]) -> List[str]:
        return [self._COLOR_HEX.get(c, '#c4a882') for c in color_names]

    def _fallback_outfit(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            'name': 'Daily Look',
            'vibe': 'Casual',
            'item_ids': [it.get('id') for it in items[:3]],
            'items': items[:3],
            'compatibility_score': 75.0,
            'styling_tips': ['Great combination for the day!'],
        }

    def _fallback_gap_analysis(self, wardrobe: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, int] = defaultdict(int)
        for item in wardrobe:
            counts[item.get('category', 'Unknown')] += 1
        return {
            'total_items': len(wardrobe),
            'category_distribution': dict(counts),
            'missing_essentials': [],
            'gap_analysis': [],
            'recommendations': [],
            'wardrobe_health_score': 50,
            'dominant_style': 'Versatile',
            'color_preferences': [],
        }
