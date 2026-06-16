# services/outfit_generator.py
# Outfit generation, gap analysis, and aesthetic aura.
# Delegates heavy lifting to ai_matcher.AdvancedFashionMatcher.

import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    Generates outfits, aesthetic auras, and gap analysis results.
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
    # FEATURE 1: Personalized Outfit Scoring
    # ------------------------------------------------------------------

    def score_outfit(
        self,
        outfit: Dict[str, Any],
        style_dna: Dict[str, Any],
        wear_history: List[Dict[str, Any]],
        color_preferences: List[str] = None
    ) -> Dict[str, Any]:
        """
        Score an outfit based on:
        - Style DNA match (30%)
        - Color harmony (25%)
        - Wear history (25%)
        - Color preferences (20%)
        
        Returns: score (0-100), breakdown, reasoning
        """
        score = 0
        breakdown = {}
        reasoning = []

        # 1. Style DNA match (30%)
        dna_score = 0
        if style_dna and style_dna.get('styles'):
            dna_styles = style_dna.get('styles', [])
            outfit_vibe = outfit.get('vibe', 'casual')
            
            # Check if outfit vibe matches any style DNA
            for style in dna_styles:
                if style.lower() in outfit_vibe.lower() or outfit_vibe.lower() in style.lower():
                    dna_score = 30
                    reasoning.append(f"Matches your {style} style")
                    break
            else:
                dna_score = 15
                reasoning.append("Partially matches your style DNA")
        else:
            dna_score = 20
            reasoning.append("Style DNA not available — using default scoring")
        
        score += dna_score
        breakdown['style_dna'] = dna_score

        # 2. Color harmony (25%)
        color_score = 0
        outfit_colors = [item.get('color', '') for item in outfit.get('items', []) if item.get('color')]
        
        if len(outfit_colors) >= 2:
            # Check if colors harmonize using COLOR_HARMONY
            harmonious = False
            for i, color1 in enumerate(outfit_colors):
                for color2 in outfit_colors[i+1:]:
                    if color1 in COLOR_HARMONY and color2 in COLOR_HARMONY[color1]:
                        harmonious = True
                        break
                    elif color2 in COLOR_HARMONY and color1 in COLOR_HARMONY[color2]:
                        harmonious = True
                        break
            
            if harmonious:
                color_score = 25
                reasoning.append("Colors are harmonious")
            else:
                color_score = 10
                reasoning.append("Colors could be better matched")
        else:
            color_score = 20
            reasoning.append("Not enough colors to evaluate harmony")
        
        score += color_score
        breakdown['color_harmony'] = color_score

        # 3. Wear history (25%) — prefer items worn less
        wear_score = 0
        if wear_history:
            item_ids = [item.get('item_id') for item in outfit.get('items', [])]
            wear_counts = {}
            for log in wear_history:
                item_id = log.get('item_id')
                if item_id:
                    wear_counts[item_id] = wear_counts.get(item_id, 0) + 1
            
            # Calculate average wear count for outfit items
            avg_wear = sum(wear_counts.get(item_id, 0) for item_id in item_ids) / max(len(item_ids), 1)
            
            # Lower wear count = higher score (promote under-worn items)
            if avg_wear <= 1:
                wear_score = 25
                reasoning.append("All items are fresh and under-worn")
            elif avg_wear <= 3:
                wear_score = 18
                reasoning.append("Some items have been worn before")
            elif avg_wear <= 5:
                wear_score = 10
                reasoning.append("These items are frequently worn")
            else:
                wear_score = 5
                reasoning.append("These items are heavily worn — try something new")
        else:
            wear_score = 20
            reasoning.append("No wear history available")
        
        score += wear_score
        breakdown['wear_history'] = wear_score

        # 4. Color preferences (20%)
        pref_score = 0
        if color_preferences:
            matched_colors = [c for c in outfit_colors if c in color_preferences]
            if matched_colors:
                pref_score = 20
                reasoning.append(f"Includes your preferred colors: {', '.join(matched_colors)}")
            else:
                pref_score = 5
                reasoning.append("Doesn't include your preferred colors")
        else:
            pref_score = 15
            reasoning.append("No color preferences set")
        
        score += pref_score
        breakdown['color_preferences'] = pref_score

        return {
            'score': round(score, 1),
            'breakdown': breakdown,
            'reasoning': reasoning,
            'max_score': 100,
            'rating': self._get_rating(score)
        }

    def _get_rating(self, score: float) -> str:
        if score >= 85:
            return 'excellent'
        elif score >= 70:
            return 'good'
        elif score >= 50:
            return 'average'
        else:
            return 'needs_improvement'

    # ------------------------------------------------------------------
    # FEATURE 2: Context-Aware Recommendations
    # ------------------------------------------------------------------

    def filter_by_context(
        self,
        outfits: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter and re-rank outfits based on context:
        - Time of day
        - Day of week
        - Weather
        - Temperature
        - Occasion
        """
        if not outfits:
            return []
        
        scored_outfits = []
        for outfit in outfits:
            score = 0
            reasoning = []
            
            # 1. Time of day (30%)
            time_of_day = context.get('time_of_day', 'afternoon')
            outfit_vibe = outfit.get('vibe', 'casual')
            
            if time_of_day == 'morning':
                if outfit_vibe in ['casual', 'comfortable']:
                    score += 30
                    reasoning.append("Good for morning")
            elif time_of_day == 'evening':
                if outfit_vibe in ['formal', 'party', 'smart-casual']:
                    score += 30
                    reasoning.append("Good for evening")
            else:  # afternoon
                if outfit_vibe in ['casual', 'formal', 'smart-casual']:
                    score += 25
                    reasoning.append("Good for afternoon")
            
            # 2. Day of week (20%)
            day = context.get('day_of_week', 'weekday')
            if day == 'weekend':
                if outfit_vibe in ['casual', 'party', 'boho']:
                    score += 20
                    reasoning.append("Great for weekend")
            else:  # weekday
                if outfit_vibe in ['formal', 'smart-casual', 'classic']:
                    score += 20
                    reasoning.append("Good for weekday")
            
            # 3. Weather (30%)
            weather = context.get('weather', 'unknown')
            outfit_colors = [item.get('color', '') for item in outfit.get('items', [])]
            outfit_fabrics = [item.get('fabric', '') for item in outfit.get('items', [])]
            
            if weather == 'cold' or weather == 'winter':
                if 'Jacket' in [item.get('category') for item in outfit.get('items', [])]:
                    score += 30
                    reasoning.append("Has jacket for cold weather")
                elif any(f in ['wool', 'cotton', 'denim'] for f in outfit_fabrics):
                    score += 20
                    reasoning.append("Has warm fabrics")
            elif weather == 'hot' or weather == 'summer':
                if any(f in ['linen', 'cotton', 'silk'] for f in outfit_fabrics):
                    score += 30
                    reasoning.append("Has breathable fabrics")
                elif not any(f in ['wool', 'polyester'] for f in outfit_fabrics):
                    score += 20
                    reasoning.append("Lightweight materials")
            elif weather == 'rainy':
                if 'Jacket' in [item.get('category') for item in outfit.get('items', [])]:
                    score += 30
                    reasoning.append("Has jacket for rain")
                elif not any(f in ['silk', 'linen'] for f in outfit_fabrics):
                    score += 15
                    reasoning.append("Weather-resistant fabrics")
            else:  # mild
                score += 25
                reasoning.append("Suitable for mild weather")
            
            # 4. Occasion (20%)
            occasion = context.get('occasion', 'everyday')
            if occasion == 'formal':
                if outfit_vibe in ['formal', 'classic']:
                    score += 20
                    reasoning.append("Matches formal occasion")
            elif occasion == 'party':
                if outfit_vibe in ['party', 'bold', 'vibrant']:
                    score += 20
                    reasoning.append("Matches party occasion")
            elif occasion == 'casual':
                if outfit_vibe in ['casual', 'comfortable']:
                    score += 20
                    reasoning.append("Matches casual occasion")
            elif occasion == 'sport':
                if outfit_vibe in ['athleisure', 'sporty']:
                    score += 20
                    reasoning.append("Matches sporty occasion")
            else:
                score += 15
                reasoning.append("Suitable for everyday")
            
            scored_outfits.append({
                'outfit': outfit,
                'context_score': score,
                'context_reasoning': reasoning
            })
        
        # Sort by context score (highest first)
        scored_outfits.sort(key=lambda x: x['context_score'], reverse=True)
        return scored_outfits

    # ------------------------------------------------------------------
    # FEATURE 4: Gap Analysis with Actionable Suggestions
    # ------------------------------------------------------------------

    def analyze_wardrobe_gaps(
        self,
        user_id: str,
        wardrobe_items: List[Dict[str, Any]],
        db_conn,
        include_shopping_links: bool = True
    ) -> Dict[str, Any]:
        """
        Compare user's Style DNA against inventory using cosine embedding similarity.
        Pulls DNA style labels from the DB, then asks fashion_matcher to find gaps.
        Includes shopping links for actionable suggestions.
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
            result = fashion_matcher.analyze_wardrobe_gaps(wardrobe_items, style_dna_labels)
        else:
            result = self._fallback_gap_analysis(wardrobe_items)
        
        # Add shopping links (Feature 4)
        if include_shopping_links and result.get('gaps'):
            result = self._add_shopping_links(result)
        
        return result

    def _add_shopping_links(self, gaps_result: Dict[str, Any]) -> Dict[str, Any]:
        """Add actionable shopping suggestions to gap analysis."""
        for gap in gaps_result.get('gaps', []):
            category = gap.get('category', 'clothing')
            # Create search links for major platforms
            gap['shopping_suggestions'] = {
                'amazon': f"https://www.amazon.in/s?k={category.replace(' ', '+')}",
                'myntra': f"https://www.myntra.com/{category.replace(' ', '-')}",
                'search_query': f"best {category} for minimalist wardrobe"
            }
            gap['affiliate_link'] = f"https://www.amazon.in/s?k={category.replace(' ', '+')}&ref=luna_recommend"
        return gaps_result

    # ------------------------------------------------------------------
    # FEATURE 5: Wardrobe Analytics
    # ------------------------------------------------------------------

    def get_wardrobe_analytics(
        self,
        wardrobe_items: List[Dict[str, Any]],
        wear_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate wardrobe analytics:
        - Most worn items
        - Least worn items
        - Cost per wear (if price data available)
        - Category distribution
        - Color distribution
        """
        if not wardrobe_items:
            return {
                'total_items': 0,
                'most_worn': [],
                'least_worn': [],
                'cost_per_wear': [],
                'category_distribution': {},
                'color_distribution': {},
                'sustainability_score': 0
            }
        
        # Category distribution
        category_counts = defaultdict(int)
        color_counts = defaultdict(int)
        wear_counts = []
        total_value = 0
        
        for item in wardrobe_items:
            category_counts[item.get('category', 'Unknown')] += 1
            color_counts[item.get('color', 'Unknown')] += 1
            wear_count = item.get('wear_count', 0)
            wear_counts.append({
                'item_id': item.get('item_id'),
                'name': item.get('name', 'Unknown'),
                'wear_count': wear_count,
                'price': item.get('price', 0)
            })
            total_value += item.get('price', 0)
        
        # Sort by wear count
        wear_counts.sort(key=lambda x: x['wear_count'], reverse=True)
        
        # Calculate cost per wear
        cost_per_wear = []
        for item in wear_counts:
            if item['price'] > 0 and item['wear_count'] > 0:
                cost_per_wear.append({
                    'name': item['name'],
                    'cost_per_wear': round(item['price'] / item['wear_count'], 2),
                    'wear_count': item['wear_count'],
                    'price': item['price']
                })
        
        # Sustainability score (simplified)
        sustainable_fabrics = ['cotton', 'linen', 'hemp', 'bamboo', 'wool', 'silk']
        sustainable_items = 0
        for item in wardrobe_items:
            fabric = item.get('fabric', '').lower()
            if any(sf in fabric for sf in sustainable_fabrics):
                sustainable_items += 1
        sustainability_score = round((sustainable_items / len(wardrobe_items)) * 100, 1) if wardrobe_items else 0
        
        return {
            'total_items': len(wardrobe_items),
            'total_value': round(total_value, 2),
            'most_worn': wear_counts[:3],
            'least_worn': [w for w in wear_counts if w['wear_count'] == 0][:3],
            'cost_per_wear': sorted(cost_per_wear, key=lambda x: x['cost_per_wear'])[:5],
            'category_distribution': dict(category_counts),
            'color_distribution': dict(color_counts),
            'sustainability_score': sustainability_score,
            'average_wear_count': round(sum(w['wear_count'] for w in wear_counts) / len(wear_counts), 1) if wear_counts else 0
        }

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
            'gaps': []  # Added for consistency with new features
        }

    # ------------------------------------------------------------------
    # FEATURE 6: Outfit Memory & Feedback Loop
    # ------------------------------------------------------------------

    def apply_feedback_to_scoring(self, outfits: List[Dict[str, Any]], feedback_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adjust outfit scores based on past feedback.
        """
        if not feedback_history:
            return outfits
        
        # Extract liked/disliked categories and vibes
        liked_categories = set()
        liked_vibes = set()
        disliked_categories = set()
        disliked_vibes = set()
        
        for feedback in feedback_history:
            action = feedback.get('action')
            context = feedback.get('context', {})
            
            if action == 'like':
                if context.get('category'):
                    liked_categories.add(context['category'])
                if context.get('vibe'):
                    liked_vibes.add(context['vibe'])
            elif action == 'dislike':
                if context.get('category'):
                    disliked_categories.add(context['category'])
                if context.get('vibe'):
                    disliked_vibes.add(context['vibe'])
        
        # Re-rank outfits
        for outfit in outfits:
            boost = 0
            penalty = 0
            
            outfit_vibe = outfit.get('vibe', '')
            outfit_items = outfit.get('items', [])
            
            # Boost from liked categories
            for item in outfit_items:
                category = item.get('category', '')
                if category in liked_categories:
                    boost += 5
                if category in disliked_categories:
                    penalty += 5
            
            # Boost from liked vibes
            if outfit_vibe in liked_vibes:
                boost += 10
            if outfit_vibe in disliked_vibes:
                penalty += 10
            
            # Apply adjustments
            if 'score' in outfit:
                outfit['score'] = min(100, outfit['score'] + boost - penalty)
            if 'reasoning' in outfit:
                if boost > 0:
                    outfit['reasoning'].append(f"Based on your past likes (+{boost})")
                if penalty > 0:
                    outfit['reasoning'].append(f"Based on your past dislikes (-{penalty})")
        
        return outfits
