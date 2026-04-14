# ai_matcher.py - ADVANCED FASHION MATCHING ENGINE
# Uses real cosine similarity for matching (FashionCLIP-style logic via numpy)
# Supports: True Similarity Matching (F1), Smarter Curate & Daily Drop (F2), Gap Analysis (F3)

import logging
import random
import json
import os
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helpers — replaces random.uniform(85, 95) with real cosine sim
# ---------------------------------------------------------------------------

def _text_to_pseudo_embedding(item: Dict[str, Any]) -> np.ndarray:
    """Convert item attributes to embedding vector for similarity comparison."""
    COLOR_HUES = {
        'Red': 0, 'Orange': 30, 'Yellow': 60, 'Green': 120, 'Teal': 180,
        'Blue': 240, 'Purple': 270, 'Pink': 300, 'Brown': 20, 'Beige': 45,
        'White': 0, 'Black': 0, 'Gray': 0, 'Navy': 240, 'Denim': 210,
        'Cream': 60, 'Camel': 30, 'Olive': 80, 'Burgundy': 350, 'Rust': 15,
        'Mint': 150, 'Lavender': 270, 'Sage': 100, 'Taupe': 30, 'Gold': 45,
        'Silver': 0, 'Charcoal': 200, 'Rose': 330, 'Coral': 15, 'Mustard': 55,
        'Blush': 340, 'Mauve': 310, 'Plum': 290, 'Emerald': 130, 'Terracotta': 20,
    }
    CATEGORY_VEC = {
        'Top': [1,0,0,0,0,0,0,0], 'T-Shirt': [1,0,0,0,0,0,0,0],
        'Blouse': [1,0,0,0,0,0,0,0], 'Shirt': [1,0,0,0,0,0,0,0],
        'Sweater': [0.8,0,0,0,0,0,0.2,0], 'Tank': [1,0,0,0,0,0,0,0],
        'Crop Top': [1,0,0,0,0,0,0,0],
        'Bottom': [0,1,0,0,0,0,0,0], 'Trousers': [0,1,0,0,0,0,0,0],
        'Jeans': [0,1,0,0,0,0,0,0], 'Skirt': [0,1,0,0,0,0,0,0],
        'Shorts': [0,1,0,0,0,0,0,0], 'Pants': [0,1,0,0,0,0,0,0],
        'Jacket': [0,0,1,0,0,0,0,0], 'Blazer': [0,0,1,0,0,0,0,0],
        'Coat': [0,0,1,0,0,0,0,0], 'Cardigan': [0,0,0.7,0,0,0,0.3,0],
        'Outerwear': [0,0,1,0,0,0,0,0],
        'Dress': [0,0,0,1,0,0,0,0], 'Jumpsuit': [0,0,0,1,0,0,0,0],
        'Romper': [0,0,0,1,0,0,0,0],
        'Shoes': [0,0,0,0,1,0,0,0], 'Boots': [0,0,0,0,1,0,0,0],
        'Sandals': [0,0,0,0,1,0,0,0], 'Sneakers': [0,0,0,0,1,0,0,0],
        'Heels': [0,0,0,0,1,0,0,0],
        'Bag': [0,0,0,0,0,1,0,0], 'Accessory': [0,0,0,0,0,1,0,0],
        'Jewelry': [0,0,0,0,0,1,0,0], 'Hat': [0,0,0,0,0,1,0,0],
        'Scarf': [0,0,0,0,0,1,0,0], 'Belt': [0,0,0,0,0,0.8,0,0.2],
        'Necklace': [0,0,0,0,0,1,0,0], 'Ring': [0,0,0,0,0,1,0,0],
        'Earrings': [0,0,0,0,0,1,0,0], 'Watch': [0,0,0,0,0,0.8,0,0.2],
        'Suit': [0,0,0.5,0,0,0,0,0.5], 'Dress Pants': [0,1,0,0,0,0,0,0],
    }
    FABRIC_VEC = {
        'Linen': [1,0,0,0], 'Cotton': [0.8,0.2,0,0], 'Silk': [0,1,0,0],
        'Chiffon': [0,1,0,0], 'Jersey': [0.5,0.5,0,0], 'Wool': [0,0,1,0],
        'Cashmere': [0,0,1,0], 'Tweed': [0,0,0.8,0.2], 'Fleece': [0,0,0.7,0.3],
        'Velvet': [0,0.3,0.7,0], 'Denim': [0.5,0,0,0.5], 'Polyester': [0.3,0.3,0.3,0.1],
        'Rayon': [0.5,0.5,0,0], 'Spandex': [0,0,0,1], 'Leather': [0,0,0,1],
        'Metal': [0,0,0,1], 'Gold': [0,0,0,1], 'Silver': [0,0,0,1],
        'Unknown': [0.25,0.25,0.25,0.25],
    }
    cat = item.get('category', 'Top')
    color = item.get('color', 'Black')
    fabric = item.get('fabric', 'Unknown')
    name = item.get('name', '').lower()

    cat_vec = CATEGORY_VEC.get(cat, [0.125]*8)
    hue = COLOR_HUES.get(color, 0)
    hue_rad = np.radians(hue)
    is_neutral = 1.0 if color in ['Black','White','Gray','Beige','Cream','Taupe','Navy','Denim'] else 0.0
    brightness = 1.0 if color in ['White','Cream','Yellow','Mint','Lavender'] else (0.0 if color in ['Black','Navy','Burgundy'] else 0.5)
    color_vec = [
        float(np.sin(hue_rad)), float(np.cos(hue_rad)),
        float(np.sin(hue_rad * 0.5)), float(np.cos(hue_rad * 0.5)),
        is_neutral, 1.0 - is_neutral, brightness, 1.0 - brightness
    ]
    fab_vec = FABRIC_VEC.get(fabric, FABRIC_VEC['Unknown'])
    style_keywords = {
        0: ['casual','tee','jeans','sneaker','cotton','basic'],
        1: ['formal','suit','blazer','silk','tailored'],
        2: ['boho','flowy','embroid','fringe','maxi','linen'],
        3: ['street','hoodie','oversized','graphic','cargo'],
    }
    style_vec = [0.25, 0.25, 0.25, 0.25]
    for i, kws in style_keywords.items():
        if any(kw in name for kw in kws):
            style_vec = [0.0]*4
            style_vec[i] = 1.0
            break

    vec = np.array(cat_vec + color_vec + fab_vec + style_vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors (0-1 scale)."""
    dot = float(np.dot(v1, v2))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (n1 * n2)))


def compute_similarity_score(item1: Dict[str, Any], item2: Dict[str, Any]) -> float:
    """
    Compute real similarity score between two items (0-100 scale).
    Replaces random.uniform(85, 95) with actual cosine similarity.
    """
    e1 = _text_to_pseudo_embedding(item1)
    e2 = _text_to_pseudo_embedding(item2)
    raw = cosine_similarity(e1, e2)

    TOPS = {'Top','T-Shirt','Blouse','Shirt','Sweater','Tank','Crop Top'}
    BOTTOMS = {'Bottom','Trousers','Jeans','Skirt','Shorts','Pants'}
    OUTERWEAR = {'Jacket','Blazer','Coat','Cardigan','Outerwear'}
    DRESSES = {'Dress','Jumpsuit','Romper'}
    SHOES = {'Shoes','Boots','Sandals','Sneakers','Heels'}
    ACCESSORIES = {'Bag','Accessory','Jewelry','Hat','Scarf','Belt','Necklace','Ring','Earrings','Watch'}

    cat1 = item1.get('category', '')
    cat2 = item2.get('category', '')
    bonus = 0.0
    if (cat1 in TOPS and cat2 in BOTTOMS) or (cat2 in TOPS and cat1 in BOTTOMS):
        bonus = 0.35
    elif (cat1 in DRESSES and cat2 in OUTERWEAR) or (cat2 in DRESSES and cat1 in OUTERWEAR):
        bonus = 0.28
    elif (cat1 in SHOES and cat2 not in SHOES) or (cat2 in SHOES and cat1 not in SHOES):
        bonus = 0.20
    elif (cat1 in ACCESSORIES and cat2 not in ACCESSORIES) or (cat2 in ACCESSORIES and cat1 not in ACCESSORIES):
        bonus = 0.15

    combined = raw * 0.55 + bonus * 0.45
    return round(50 + combined * 49, 1)


def _load_color_harmony() -> Dict[str, Any]:
    """Load color harmony rules from JSON file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'color_harmony.json')
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

COLOR_HARMONY = _load_color_harmony()


def _colors_harmonize(c1: str, c2: str) -> Tuple[bool, str]:
    """Check if two colors harmonize and return harmony type."""
    for harmony_type in ['complementary', 'analogous', 'monochromatic']:
        table = COLOR_HARMONY.get(harmony_type, {})
        if c1 in table and c2 in table.get(c1, []):
            return True, harmony_type
        if c2 in table and c1 in table.get(c2, []):
            return True, harmony_type
    neutrals = set(COLOR_HARMONY.get('neutrals', []))
    if c1 in neutrals or c2 in neutrals:
        return True, 'neutral'
    return False, 'none'


class AdvancedFashionMatcher:
    """
    Advanced matching engine for complete wardrobe coordination.
    Supports: True Similarity Matching (F1), Smarter Curate (F2), Gap Analysis (F3)
    """

    def __init__(self):
        self.color_harmony_data = COLOR_HARMONY
        self.category_groups = {
            'tops': ['Top','T-Shirt','Blouse','Shirt','Sweater','Tank','Crop Top'],
            'bottoms': ['Bottom','Trousers','Jeans','Skirt','Shorts','Pants'],
            'outerwear': ['Jacket','Blazer','Coat','Cardigan','Outerwear'],
            'dresses': ['Dress','Jumpsuit','Romper'],
            'shoes': ['Shoes','Boots','Sandals','Sneakers','Heels'],
            'accessories': ['Accessory','Jewelry','Bag','Hat','Scarf','Belt','Necklace','Ring','Earrings','Watch'],
            'formal': ['Suit','Blazer','Dress Pants','Cocktail Dress'],
        }
        self.fabric_seasonality = {
            'summer': ['Linen','Cotton','Silk','Chiffon','Jersey'],
            'winter': ['Wool','Cashmere','Tweed','Fleece','Velvet'],
            'all_season': ['Denim','Polyester','Rayon','Spandex','Leather','Metal','Gold','Silver'],
        }

    def match_items(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match two fashion items using real cosine similarity (not random).
        Returns compatibility score and detailed breakdown.
        """
        color1 = item1.get('color', 'Unknown')
        color2 = item2.get('color', 'Unknown')
        cat1 = item1.get('category', 'Unknown')
        cat2 = item2.get('category', 'Unknown')
        fabric1 = item1.get('fabric', 'Unknown')
        fabric2 = item2.get('fabric', 'Unknown')
        explanations = []

        # REAL similarity score (replaces random.uniform)
        overall_score = compute_similarity_score(item1, item2)

        # Color compatibility score
        color_score = 0
        harmonize, harmony_type = _colors_harmonize(color1, color2)
        if color1 == color2:
            color_score = 35
            explanations.append(f"Monochromatic: Both items are {color1}")
        elif harmonize:
            label = harmony_type.replace('_', ' ').title()
            color_score = 30
            explanations.append(f"{label} colors: {color1} and {color2}")
        elif color2 in self.color_harmony_data.get('neutrals', []):
            color_score = 25
            explanations.append(f"Neutral pairing: {color1} works with neutral {color2}")
        elif color1 in self.color_harmony_data.get('neutrals', []):
            color_score = 25
            explanations.append(f"Neutral pairing: {color2} works with neutral {color1}")
        else:
            color_score = 10

        # Category compatibility score
        category_score = 0
        tops_s = set(self.category_groups['tops'])
        bots_s = set(self.category_groups['bottoms'])
        dr_s = set(self.category_groups['dresses'])
        out_s = set(self.category_groups['outerwear'])
        sh_s = set(self.category_groups['shoes'])
        ac_s = set(self.category_groups['accessories'])

        if (cat1 in tops_s and cat2 in bots_s) or (cat2 in tops_s and cat1 in bots_s):
            category_score = 28
            explanations.append("Perfect combination: Top with Bottom")
        elif (cat1 in dr_s and cat2 in out_s) or (cat2 in dr_s and cat1 in out_s):
            category_score = 22
            explanations.append("Layered look: Dress with Outerwear")
        elif (cat1 in ac_s and cat2 not in ac_s) or (cat2 in ac_s and cat1 not in ac_s):
            category_score = 16
            explanations.append("Accessory complements clothing item")
        elif (cat1 in sh_s and cat2 not in sh_s) or (cat2 in sh_s and cat1 not in sh_s):
            category_score = 20
            explanations.append("Shoes complete the outfit")

        # Fabric compatibility score
        fabric_score = 5
        for season, fabrics in self.fabric_seasonality.items():
            if fabric1 in fabrics and fabric2 in fabrics:
                fabric_score = 18
                explanations.append(f"Seasonally appropriate: Both are {season} fabrics")
                break

        # Style compatibility score
        style_score = 0
        name1 = item1.get('name', '').lower()
        name2 = item2.get('name', '').lower()
        for style, kws in {'casual':['casual','tee','jeans'],'formal':['suit','blazer','silk'],
                           'bohemian':['boho','maxi','linen'],'streetwear':['hoodie','oversized','graphic']}.items():
            if any(k in name1 for k in kws) and any(k in name2 for k in kws):
                style_score = 8
                explanations.append(f"Style match: Both items have {style} elements")
                break

        # Determine compatibility level
        if overall_score >= 80:
            compatibility, recommendation = "Excellent", "Perfect match! Wear together confidently."
        elif overall_score >= 65:
            compatibility, recommendation = "Good", "Works well together. Consider adding accessories."
        elif overall_score >= 50:
            compatibility, recommendation = "Fair", "Could work with the right styling."
        else:
            compatibility, recommendation = "Poor", "Consider different combinations for better harmony."

        return {
            'compatibility_score': round(overall_score),
            'compatibility_level': compatibility,
            'category_match': f"{cat1} + {cat2}",
            'color_match': f"{color1} + {color2}",
            'fabric_match': f"{fabric1} + {fabric2}",
            'explanations': explanations,
            'recommendation': recommendation,
            'breakdown': {'color': color_score, 'category': category_score, 'fabric': fabric_score, 'style': style_score}
        }

    def rank_closet_matches(self, inspiration_item: Dict[str, Any], wardrobe: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank all wardrobe items by actual cosine similarity to inspiration_item.
        Replaces random.uniform(85, 95) with real similarity scores.
        """
        scored = []
        for item in wardrobe:
            score = compute_similarity_score(inspiration_item, item)
            scored.append({**item, 'match_score': round(score, 1)})
        scored.sort(key=lambda x: x['match_score'], reverse=True)
        return scored

    def create_complete_outfit(self, items: List[Dict[str, Any]], style: str = "casual", occasion: str = "daytime") -> Dict[str, Any]:
        """
        Build outfit using color harmony seed logic instead of random picks (Feature 2).
        Picks a seed item and builds outfit using complementary/analogous color logic.
        """
        if len(items) < 2:
            return {}
        categorized = defaultdict(list)
        for item in items:
            cat = item.get('category', 'Unknown')
            for gname, gcats in self.category_groups.items():
                if cat in gcats:
                    categorized[gname].append(item)
                    break
            else:
                categorized['other'].append(item)

        # Pick seed item (most worn or random from tops/bottoms)
        seed_pool = categorized.get('tops', []) + categorized.get('bottoms', []) or items
        seed = max(seed_pool, key=lambda x: x.get('wear_count', 0))

        def best(pool):
            if not pool:
                return None
            return max(pool, key=lambda x: compute_similarity_score(seed, x))

        selected = [seed]
        partner = best(categorized.get('bottoms' if seed.get('category') in self.category_groups['tops'] else 'tops', []))
        if not partner and seed.get('category') in self.category_groups['dresses']:
            partner = best(categorized.get('outerwear', []))
        if partner:
            selected.append(partner)
        
        shoes = best(categorized.get('shoes', []))
        if shoes:
            selected.append(shoes)
        if random.random() > 0.3:
            acc = best(categorized.get('accessories', []))
            if acc:
                selected.append(acc)

        # Determine harmony type for badge
        seed_color = seed.get('color', 'Black')
        partner_color = (partner or seed).get('color', seed_color)
        harmonize, h_type = _colors_harmonize(seed_color, partner_color)
        vibe = h_type.replace('_', ' ').title() if harmonize and h_type != 'none' else style.capitalize()
        
        outfit_name = random.choice([
            f"{vibe} {seed_color} Edit",
            f"{seed_color} {style.capitalize()} Story",
            f"The {seed_color} Moment"
        ])

        # Calculate average compatibility score
        scores = [compute_similarity_score(selected[i], selected[j]) 
                  for i in range(len(selected)) for j in range(i+1, len(selected))]
        avg = sum(scores)/len(scores) if scores else 75.0

        return {
            'name': outfit_name,
            'vibe': vibe,
            'item_ids': [it['id'] for it in selected],
            'items': selected,
            'compatibility_score': round(avg, 1),
            'styling_tips': self._generate_styling_tips(selected, style, occasion),
        }

    def _generate_styling_tips(self, items: List[Dict], style: str, occasion: str) -> List[str]:
        """Generate styling tips based on outfit composition."""
        tips = []
        colors = list({it.get('color') for it in items if it.get('color')})
        if len(colors) >= 2:
            ok, h_type = _colors_harmonize(colors[0], colors[1])
            if ok:
                tips.append(f"{h_type.replace('_',' ').title()} color pairing — very harmonious.")
        if style == 'casual':
            tips.append("Tuck in the front for a relaxed look.")
        elif style == 'formal':
            tips.append("Keep accessories minimal for a polished finish.")
        return tips or ["Mix of textures adds visual interest."]

    def analyze_wardrobe_gaps(self, wardrobe: List[Dict[str, Any]], style_dna: List[str] = None) -> Dict[str, Any]:
        """
        Gap analysis: compare Style DNA against actual inventory (Feature 3).
        Identifies imbalances and suggests missing pieces with affiliate links.
        """
        category_counts: Dict[str, int] = defaultdict(int)
        color_counts: Dict[str, int] = defaultdict(int)
        style_counts: Dict[str, int] = defaultdict(int)
        
        for item in wardrobe:
            category_counts[item.get('category', 'Unknown')] += 1
            color_counts[item.get('color', 'Unknown')] += 1
            n = item.get('name', '').lower()
            if any(k in n for k in ['jeans', 'tee', 'sneaker', 'casual']):
                style_counts['Casual'] += 1
            elif any(k in n for k in ['suit', 'blazer', 'formal', 'silk']):
                style_counts['Formal'] += 1
            elif any(k in n for k in ['boho', 'maxi', 'flowy', 'fringe']):
                style_counts['Bohemian'] += 1
            else:
                style_counts['Versatile'] += 1

        missing_essentials = [c for c in ['Top', 'Bottom', 'Shoes', 'Jacket'] if category_counts.get(c, 0) == 0]
        dna_suggestions = []

        # DNA-based gap suggestions
        if style_dna:
            for dna_style in style_dna:
                dl = dna_style.lower()
                if 'minimalist' in dl or 'clean' in dl:
                    patterned = sum(1 for it in wardrobe if any(k in it.get('name', '').lower() 
                                    for k in ['pattern', 'floral', 'stripe', 'print', 'graphic']))
                    neutral_bots = sum(1 for it in wardrobe if it.get('category') in ('Bottom', 'Trousers', 'Jeans', 'Pants') 
                                       and it.get('color', '') in self.color_harmony_data.get('neutrals', []))
                    if patterned > 5 and neutral_bots < 3:
                        dna_suggestions.append({
                            'piece': 'Neutral Wide-Leg Trousers',
                            'reason': f'Your DNA is {dna_style} but you have {patterned} patterned tops — neutral bottoms will balance the look.',
                            'category': 'Bottom',
                            'suggested_colors': ['Black', 'Cream', 'Beige'],
                            'affiliate_tag': 'minimalist-bottoms'
                        })
                if 'bohemian' in dl or 'boho' in dl:
                    if sum(1 for it in wardrobe if any(k in it.get('name', '').lower() 
                            for k in ['maxi', 'flowy', 'boho', 'linen'])) < 2:
                        dna_suggestions.append({
                            'piece': 'Linen Maxi Dress or Skirt',
                            'reason': f'Your DNA is {dna_style} but your closet lacks flowing silhouettes.',
                            'category': 'Dress',
                            'suggested_colors': ['Cream', 'Sage', 'Terracotta'],
                            'affiliate_tag': 'boho-maxi'
                        })
                if 'streetwear' in dl or 'street' in dl:
                    if sum(1 for it in wardrobe if any(k in it.get('name', '').lower() 
                            for k in ['oversized', 'cargo', 'hoodie', 'graphic'])) < 2:
                        dna_suggestions.append({
                            'piece': 'Oversized Graphic Tee or Cargo Pants',
                            'reason': f'Your DNA is {dna_style} but lacks core streetwear silhouettes.',
                            'category': 'Top',
                            'suggested_colors': ['Black', 'White', 'Gray'],
                            'affiliate_tag': 'streetwear-essentials'
                        })
                if any(k in dl for k in ['old money', 'classic', 'quiet luxury', 'preppy']):
                    if sum(1 for it in wardrobe if it.get('category') in ('Blazer', 'Coat', 'Jacket', 'Outerwear') 
                           and it.get('fabric', '') in ('Wool', 'Cashmere', 'Tweed')) < 1:
                        dna_suggestions.append({
                            'piece': 'Structured Wool Blazer',
                            'reason': f'Your DNA is {dna_style} — a structured blazer is the cornerstone piece.',
                            'category': 'Jacket',
                            'suggested_colors': ['Navy', 'Camel', 'Charcoal'],
                            'affiliate_tag': 'classic-outerwear'
                        })

        # Essential missing items
        for missing in missing_essentials:
            dna_suggestions.append({
                'piece': f'Essential {missing}',
                'reason': f'You have no {missing.lower()} — a versatile one unlocks many outfit combinations.',
                'category': missing,
                'suggested_colors': ['Black', 'White', 'Navy'],
                'affiliate_tag': f'essential-{missing.lower()}'
            })

        # Color variety suggestion
        if color_counts:
            total = sum(color_counts.values())
            dom_c, dom_n = max(color_counts.items(), key=lambda x: x[1])
            if total > 5 and dom_n / total > 0.6:
                complements = self.color_harmony_data.get('complementary', {}).get(dom_c, [])
                if complements:
                    dna_suggestions.append({
                        'piece': f'{complements[0]} Statement Piece',
                        'reason': f'{dom_n}/{total} items are {dom_c} — adding {complements[0]} unlocks new combinations.',
                        'category': 'Top',
                        'suggested_colors': complements[:3],
                        'affiliate_tag': f'color-variety-{dom_c.lower()}'
                    })

        dom_style = max(style_counts, key=style_counts.get) if style_counts else 'Versatile'
        health_score = min(100, len(wardrobe) * 4 + len(color_counts) * 4 + (20 if not missing_essentials else 0))
        
        return {
            'total_items': len(wardrobe),
            'category_distribution': dict(category_counts),
            'missing_essentials': missing_essentials,
            'gap_analysis': dna_suggestions,
            'recommendations': [s['piece'] + ' — ' + s['reason'] for s in dna_suggestions[:3]],
            'wardrobe_health_score': health_score,
            'dominant_style': dom_style,
            'color_preferences': list(color_counts.keys()),
        }


# Singleton instance
fashion_matcher = AdvancedFashionMatcher()
