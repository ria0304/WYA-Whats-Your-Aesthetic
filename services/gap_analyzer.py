# services/gap_analyzer.py
# "Shop Your Closet" — Gap Analysis Engine
#
# Architecture:
#   1. Encode Style DNA labels → pseudo-embeddings (FashionCLIP-style vectors)
#   2. Encode every wardrobe item → pseudo-embedding
#   3. For each DNA-required archetype dimension, compute cosine distance to
#      the user's inventory centroid → surfaces imbalances.
#   4. Rule-layer on top: detect literal category/color holes.
#   5. Map gaps to Green Score approved brands with affiliate-ready search queries.

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style DNA → required wardrobe "blueprint"
# ---------------------------------------------------------------------------

# What a wardrobe ideally looks like for each aesthetic.
# keys: category, neutral_ratio (0-1), pattern_penalty (applied if user has
#       many patterned items), must_have_colors, avoid_colors
AESTHETIC_BLUEPRINTS: Dict[str, Dict[str, Any]] = {
    "minimalist": {
        "required_categories": ["Top", "Pants", "Outerwear", "Shoes"],
        "neutral_ratio_target": 0.80,        # ≥80 % of wardrobe should be neutrals
        "max_pattern_ratio": 0.10,           # ≤10 % patterned items
        "must_have_colors": ["Black", "White", "Gray", "Beige", "Cream", "Navy"],
        "preferred_fabrics": ["Cotton", "Linen", "Wool", "Cashmere"],
        "missing_piece_templates": [
            {"category": "Top",      "color": "White",  "label": "White linen blouse"},
            {"category": "Pants",    "color": "Black",  "label": "Tailored black trousers"},
            {"category": "Outerwear","color": "Camel",  "label": "Camel wool coat"},
            {"category": "Shoes",    "color": "White",  "label": "Clean white leather sneakers"},
            {"category": "Accessory","color": "Black",  "label": "Minimalist black leather belt"},
        ],
    },
    "classic": {
        "required_categories": ["Top", "Pants", "Blazer", "Shoes", "Bag"],
        "neutral_ratio_target": 0.65,
        "max_pattern_ratio": 0.20,
        "must_have_colors": ["Navy", "White", "Black", "Beige"],
        "preferred_fabrics": ["Cotton", "Wool", "Silk", "Cashmere"],
        "missing_piece_templates": [
            {"category": "Top",    "color": "White",  "label": "Classic white button-down shirt"},
            {"category": "Blazer", "color": "Navy",   "label": "Navy wool blazer"},
            {"category": "Pants",  "color": "Beige",  "label": "Tailored chino trousers"},
            {"category": "Shoes",  "color": "Brown",  "label": "Leather loafers"},
            {"category": "Bag",    "color": "Tan",    "label": "Structured leather tote"},
        ],
    },
    "boho": {
        "required_categories": ["Dress", "Top", "Outerwear", "Shoes"],
        "neutral_ratio_target": 0.35,
        "max_pattern_ratio": 0.60,
        "must_have_colors": ["Rust", "Olive", "Mustard", "Cream", "Terracotta"],
        "preferred_fabrics": ["Linen", "Cotton", "Chiffon", "Silk"],
        "missing_piece_templates": [
            {"category": "Dress",    "color": "Rust",     "label": "Flowy midi wrap dress"},
            {"category": "Top",      "color": "Cream",    "label": "Embroidered peasant blouse"},
            {"category": "Outerwear","color": "Camel",    "label": "Fringed suede vest"},
            {"category": "Shoes",    "color": "Tan",      "label": "Leather ankle boots"},
            {"category": "Accessory","color": "Gold",     "label": "Layered gold necklaces"},
        ],
    },
    "streetwear": {
        "required_categories": ["Top", "Pants", "Outerwear", "Shoes", "Accessory"],
        "neutral_ratio_target": 0.45,
        "max_pattern_ratio": 0.40,
        "must_have_colors": ["Black", "White", "Gray"],
        "preferred_fabrics": ["Cotton", "Denim", "Polyester"],
        "missing_piece_templates": [
            {"category": "Top",      "color": "White",  "label": "Oversized graphic tee"},
            {"category": "Pants",    "color": "Black",  "label": "Cargo trousers"},
            {"category": "Outerwear","color": "Black",  "label": "Bomber jacket"},
            {"category": "Shoes",    "color": "White",  "label": "High-top sneakers"},
            {"category": "Accessory","color": "Black",  "label": "Baseball cap"},
        ],
    },
    "avant-garde": {
        "required_categories": ["Top", "Pants", "Outerwear", "Shoes"],
        "neutral_ratio_target": 0.40,
        "max_pattern_ratio": 0.50,
        "must_have_colors": ["Black", "White"],
        "preferred_fabrics": ["Silk", "Leather", "Velvet", "Chiffon"],
        "missing_piece_templates": [
            {"category": "Top",      "color": "Black",  "label": "Asymmetric draped blouse"},
            {"category": "Pants",    "color": "Black",  "label": "Wide-leg structured trousers"},
            {"category": "Outerwear","color": "Black",  "label": "Sculptural cocoon coat"},
            {"category": "Shoes",    "color": "Black",  "label": "Architectural platform boots"},
            {"category": "Accessory","color": "Silver", "label": "Statement geometric earrings"},
        ],
    },
    "casual": {
        "required_categories": ["Top", "Pants", "Shoes"],
        "neutral_ratio_target": 0.50,
        "max_pattern_ratio": 0.30,
        "must_have_colors": ["White", "Gray", "Navy", "Denim"],
        "preferred_fabrics": ["Cotton", "Jersey", "Denim"],
        "missing_piece_templates": [
            {"category": "Top",   "color": "White",  "label": "Soft cotton crew-neck tee"},
            {"category": "Pants", "color": "Denim",  "label": "Classic straight-leg jeans"},
            {"category": "Shoes", "color": "White",  "label": "White canvas sneakers"},
            {"category": "Bag",   "color": "Beige",  "label": "Canvas tote bag"},
        ],
    },
}

# ---------------------------------------------------------------------------
# Green Score approved brands (from brand_score.json logic) per aesthetic
# ---------------------------------------------------------------------------

AESTHETIC_BRANDS: Dict[str, List[Dict[str, str]]] = {
    "minimalist": [
        {"name": "Everlane", "affiliate_base": "everlane.com", "search_suffix": "everlane minimalist"},
        {"name": "COS", "affiliate_base": "cosstores.com", "search_suffix": "COS clean minimal"},
        {"name": "Toteme", "affiliate_base": "toteme-studio.com", "search_suffix": "toteme wardrobe essential"},
    ],
    "classic": [
        {"name": "Reformation", "affiliate_base": "thereformation.com", "search_suffix": "reformation classic"},
        {"name": "Everlane", "affiliate_base": "everlane.com", "search_suffix": "everlane classic"},
        {"name": "M.M. LaFleur", "affiliate_base": "mmlafleur.com", "search_suffix": "mmlafleur workwear"},
    ],
    "boho": [
        {"name": "Free People", "affiliate_base": "freepeople.com", "search_suffix": "free people boho"},
        {"name": "Anthropologie", "affiliate_base": "anthropologie.com", "search_suffix": "anthropologie boho"},
        {"name": "Christy Dawn", "affiliate_base": "christydawn.com", "search_suffix": "christy dawn sustainable dress"},
    ],
    "streetwear": [
        {"name": "Patagonia", "affiliate_base": "patagonia.com", "search_suffix": "patagonia streetwear"},
        {"name": "Adidas Originals", "affiliate_base": "adidas.com", "search_suffix": "adidas originals sustainable"},
        {"name": "Pangaia", "affiliate_base": "pangaia.com", "search_suffix": "pangaia streetwear"},
    ],
    "avant-garde": [
        {"name": "Veja", "affiliate_base": "veja-store.com", "search_suffix": "veja avant-garde"},
        {"name": "Stella McCartney", "affiliate_base": "stellamccartney.com", "search_suffix": "stella mccartney sustainable"},
        {"name": "Nanushka", "affiliate_base": "nanushka.com", "search_suffix": "nanushka vegan leather"},
    ],
    "casual": [
        {"name": "Patagonia", "affiliate_base": "patagonia.com", "search_suffix": "patagonia everyday"},
        {"name": "Everlane", "affiliate_base": "everlane.com", "search_suffix": "everlane casual"},
        {"name": "Pact", "affiliate_base": "wearpact.com", "search_suffix": "pact organic casual"},
    ],
}

# Neutral colors set
NEUTRAL_COLORS = {
    "Black", "White", "Gray", "Charcoal", "Beige", "Cream", "Camel",
    "Navy", "Ivory", "Taupe", "Sand", "Khaki",
}


# ---------------------------------------------------------------------------
# Embedding helpers (FashionCLIP-style pseudo-embeddings)
# ---------------------------------------------------------------------------

def _dna_to_embedding(aesthetic: str) -> np.ndarray:
    """
    Encode a style aesthetic label into a 24-dim pseudo-embedding.
    Mirrors the category+color+fabric axes used in ai_matcher._text_to_pseudo_embedding
    so we can compute cosine similarity between DNA and inventory centroid.
    """
    AESTHETIC_VECTORS = {
        # [top, bottom, outerwear, dress, shoes, accessory, casual, formal] (8 category dims)
        # + [warm_hue, cool_hue, neutral, saturated] (4 color dims)
        # + [light, structured, casual_fab, performance] (4 fabric dims)
        # + [min, cls, boho, street, avant, eco] (6 style dims)
        "minimalist":  [0.8, 0.8, 0.6, 0.3, 0.5, 0.4, 0.3, 0.7, 0.1, 0.3, 0.9, 0.1, 0.5, 0.8, 0.3, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.7],
        "classic":     [0.7, 0.8, 0.8, 0.4, 0.6, 0.5, 0.2, 0.9, 0.2, 0.4, 0.7, 0.3, 0.4, 0.9, 0.4, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.6],
        "boho":        [0.6, 0.5, 0.4, 0.9, 0.5, 0.8, 0.4, 0.3, 0.8, 0.3, 0.3, 0.9, 0.8, 0.2, 0.7, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8],
        "streetwear":  [0.9, 0.7, 0.7, 0.2, 0.9, 0.7, 0.8, 0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.3, 0.8, 0.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
        "avant-garde": [0.7, 0.7, 0.8, 0.6, 0.8, 0.8, 0.3, 0.5, 0.2, 0.6, 0.4, 0.9, 0.3, 0.7, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4],
        "casual":      [0.9, 0.9, 0.4, 0.3, 0.7, 0.3, 0.9, 0.1, 0.3, 0.2, 0.6, 0.4, 0.3, 0.1, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
    }
    key = aesthetic.lower().replace("-", "").replace(" ", "")
    # fuzzy key match
    for k in AESTHETIC_VECTORS:
        if k in key or key in k:
            v = np.array(AESTHETIC_VECTORS[k], dtype=np.float32)
            n = np.linalg.norm(v)
            return v / n if n > 0 else v
    # fallback: casual
    v = np.array(AESTHETIC_VECTORS["casual"], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _item_to_embedding(item: Dict[str, Any]) -> np.ndarray:
    """Encode a wardrobe item to the same 24-dim space as DNA embeddings."""
    CATEGORY_GROUPS = {
        "top": {"Top", "T-Shirt", "Blouse", "Shirt", "Tank", "Crop Top"},
        "bottom": {"Bottom", "Pants", "Trousers", "Jeans", "Shorts", "Skirt"},
        "outerwear": {"Jacket", "Blazer", "Coat", "Cardigan", "Outerwear"},
        "dress": {"Dress", "Jumpsuit", "Romper"},
        "shoes": {"Shoes", "Boots", "Sandals", "Sneakers", "Heels"},
        "accessory": {"Bag", "Accessory", "Jewelry", "Hat", "Scarf", "Belt",
                      "Necklace", "Ring", "Earrings", "Watch"},
    }
    cat = item.get("category", "Top")
    color = item.get("color", "Black")
    fabric = item.get("fabric", "Unknown")
    name = item.get("name", "").lower()

    # category axes [top, bottom, outerwear, dress, shoes, accessory, casual, formal]
    cat_vec = [0.0] * 8
    for i, (grp, members) in enumerate(CATEGORY_GROUPS.items()):
        if cat in members:
            cat_vec[i] = 1.0
            break
    is_formal = any(w in name for w in ["blazer", "suit", "dress pants", "silk", "formal"])
    is_casual = any(w in name for w in ["jeans", "tee", "hoodie", "sweat", "denim"])
    cat_vec[6] = 1.0 if is_casual else 0.3
    cat_vec[7] = 1.0 if is_formal else 0.3

    # color axes [warm_hue, cool_hue, neutral, saturated]
    WARM = {"Red", "Orange", "Yellow", "Rust", "Camel", "Coral", "Mustard", "Terracotta",
            "Gold", "Rose", "Brown", "Burgundy"}
    COOL = {"Blue", "Navy", "Teal", "Mint", "Lavender", "Purple", "Green", "Sage",
            "Denim", "Emerald", "Plum"}
    neutral = 1.0 if color in NEUTRAL_COLORS else 0.0
    warm = 1.0 if color in WARM else 0.0
    cool = 1.0 if color in COOL else 0.0
    saturated = 0.0 if color in NEUTRAL_COLORS else 1.0
    color_vec = [warm, cool, neutral, saturated]

    # fabric axes [light, structured, casual_fab, performance]
    LIGHT = {"Linen", "Chiffon", "Silk", "Cotton", "Rayon"}
    STRUCTURED = {"Wool", "Cashmere", "Tweed", "Blazer-weight"}
    CASUAL_FAB = {"Denim", "Cotton", "Jersey", "Fleece"}
    PERF = {"Spandex", "Polyester", "Nylon", "Technical"}
    fab_vec = [
        1.0 if fabric in LIGHT else 0.2,
        1.0 if fabric in STRUCTURED else 0.2,
        1.0 if fabric in CASUAL_FAB else 0.2,
        1.0 if fabric in PERF else 0.1,
    ]

    # style axes [min, cls, boho, street, avant, eco]
    style_vec = [0.0] * 6
    style_keywords = {
        0: ["minimal", "clean", "simple", "basic", "linen", "white"],
        1: ["classic", "tailored", "blazer", "button", "oxford"],
        2: ["boho", "flowy", "embroid", "fringe", "wrap", "maxi"],
        3: ["oversized", "graphic", "cargo", "hoodie", "street"],
        4: ["asymmetric", "sculptural", "avant", "architectural"],
        5: ["organic", "sustainable", "recycled", "eco"],
    }
    for i, kws in style_keywords.items():
        if any(kw in name for kw in kws):
            style_vec[i] = 1.0

    vec = np.array(cat_vec + color_vec + fab_vec + style_vec, dtype=np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (n1 * n2)))


# ---------------------------------------------------------------------------
# Main gap analyzer
# ---------------------------------------------------------------------------

class GapAnalyzer:
    """
    Compares Style DNA embedding against wardrobe inventory centroid,
    then surfaces missing pieces with Green Score affiliate links.
    """

    def analyze(
        self,
        style_dna: List[str],
        wardrobe_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Resolve primary aesthetic from DNA labels
        2. Compute DNA embedding and inventory centroid embedding
        3. Measure alignment (cosine similarity)
        4. Detect category holes and color imbalances
        5. Return gap suggestions with affiliate links
        """
        primary_aesthetic = self._resolve_aesthetic(style_dna)
        blueprint = AESTHETIC_BLUEPRINTS.get(primary_aesthetic, AESTHETIC_BLUEPRINTS["casual"])
        brands = AESTHETIC_BRANDS.get(primary_aesthetic, AESTHETIC_BRANDS["casual"])

        # --- Embedding alignment -------------------------------------------------
        dna_emb = _dna_to_embedding(primary_aesthetic)

        inventory_alignment = 0.0
        if wardrobe_items:
            item_embs = [_item_to_embedding(it) for it in wardrobe_items]
            centroid = np.mean(item_embs, axis=0)
            n = np.linalg.norm(centroid)
            centroid = centroid / n if n > 0 else centroid
            inventory_alignment = _cosine_similarity(dna_emb, centroid)

        # --- Inventory stats -----------------------------------------------------
        category_counts: Dict[str, int] = defaultdict(int)
        color_counts: Dict[str, int] = defaultdict(int)
        neutral_count = 0
        patterned_count = 0  # proxy: items named with "stripe", "floral", "print", etc.

        for item in wardrobe_items:
            cat = item.get("category", "Unknown")
            color = item.get("color", "Unknown")
            name = item.get("name", "").lower()

            category_counts[cat] += 1
            color_counts[color] += 1

            if color in NEUTRAL_COLORS:
                neutral_count += 1
            if any(p in name for p in ["stripe", "floral", "print", "pattern", "check", "plaid", "polka"]):
                patterned_count += 1

        total = max(len(wardrobe_items), 1)
        actual_neutral_ratio = neutral_count / total
        actual_pattern_ratio = patterned_count / total

        # --- Gap detection -------------------------------------------------------
        gaps: List[Dict[str, Any]] = []

        # 1. Category holes
        for required_cat in blueprint["required_categories"]:
            count = sum(
                v for k, v in category_counts.items()
                if k.lower() in required_cat.lower() or required_cat.lower() in k.lower()
            )
            if count == 0:
                template = next(
                    (t for t in blueprint["missing_piece_templates"] if t["category"] == required_cat),
                    blueprint["missing_piece_templates"][0],
                )
                brand = brands[len(gaps) % len(brands)]
                gaps.append({
                    "gap_type": "missing_category",
                    "category": required_cat,
                    "description": template["label"],
                    "reason": f"Your {primary_aesthetic.capitalize()} DNA needs a {required_cat.lower()} anchor piece — you have none.",
                    "priority": "high",
                    "affiliate_query": f"{template['label']} {brand['search_suffix']}",
                    "affiliate_brand": brand["name"],
                    "affiliate_url": f"https://www.{brand['affiliate_base']}/search?q={template['label'].replace(' ', '+')}",
                    "dna_alignment_score": round(inventory_alignment * 100, 1),
                })

        # 2. Neutral ratio imbalance
        target_neutral = blueprint["neutral_ratio_target"]
        if actual_neutral_ratio < target_neutral - 0.15:
            gap_delta = target_neutral - actual_neutral_ratio
            needed_must_haves = [
                c for c in blueprint["must_have_colors"] if color_counts.get(c, 0) == 0
            ]
            if needed_must_haves:
                missing_color = needed_must_haves[0]
                template = next(
                    (t for t in blueprint["missing_piece_templates"] if t["color"] == missing_color),
                    blueprint["missing_piece_templates"][0],
                )
                brand = brands[len(gaps) % len(brands)]
                gaps.append({
                    "gap_type": "color_imbalance",
                    "category": template["category"],
                    "description": f"{missing_color} {template['category'].lower()} — DNA anchor color",
                    "reason": (
                        f"Your wardrobe is {round(actual_neutral_ratio*100)}% neutral "
                        f"but your {primary_aesthetic.capitalize()} DNA targets {round(target_neutral*100)}%. "
                        f"A {missing_color.lower()} piece closes {round(gap_delta*100)}% of the gap."
                    ),
                    "priority": "high" if gap_delta > 0.25 else "medium",
                    "affiliate_query": f"{missing_color} {template['category']} {brand['search_suffix']}",
                    "affiliate_brand": brand["name"],
                    "affiliate_url": f"https://www.{brand['affiliate_base']}/search?q={missing_color}+{template['category'].replace(' ', '+')}",
                    "dna_alignment_score": round(inventory_alignment * 100, 1),
                })

        # 3. Pattern overload (e.g. Minimalist with too many prints)
        max_pattern = blueprint["max_pattern_ratio"]
        if actual_pattern_ratio > max_pattern + 0.10:
            excess = round((actual_pattern_ratio - max_pattern) * total)
            brand = brands[0]
            gaps.append({
                "gap_type": "pattern_overload",
                "category": "Top",
                "description": "Solid-color foundational top",
                "reason": (
                    f"You have ~{round(actual_pattern_ratio*100)}% patterned items — "
                    f"your {primary_aesthetic.capitalize()} DNA prefers ≤{round(max_pattern*100)}%. "
                    f"Adding {excess} solid basics will unlock more outfit combinations."
                ),
                "priority": "medium",
                "affiliate_query": f"solid neutral top {brand['search_suffix']}",
                "affiliate_brand": brand["name"],
                "affiliate_url": f"https://www.{brand['affiliate_base']}/search?q=solid+top",
                "dna_alignment_score": round(inventory_alignment * 100, 1),
            })

        # 4. If inventory is well-aligned but thin, suggest "unlock" pieces
        if len(gaps) == 0 and len(wardrobe_items) < 10:
            template = blueprint["missing_piece_templates"][0]
            brand = brands[0]
            gaps.append({
                "gap_type": "wardrobe_thin",
                "category": template["category"],
                "description": template["label"],
                "reason": (
                    f"Your style DNA alignment is strong ({round(inventory_alignment*100)}%), "
                    "but a larger wardrobe unlocks more outfit combinations."
                ),
                "priority": "low",
                "affiliate_query": f"{template['label']} {brand['search_suffix']}",
                "affiliate_brand": brand["name"],
                "affiliate_url": f"https://www.{brand['affiliate_base']}/search?q={template['label'].replace(' ', '+')}",
                "dna_alignment_score": round(inventory_alignment * 100, 1),
            })

        return {
            "primary_aesthetic": primary_aesthetic,
            "dna_alignment_score": round(inventory_alignment * 100, 1),
            "neutral_ratio": round(actual_neutral_ratio * 100, 1),
            "pattern_ratio": round(actual_pattern_ratio * 100, 1),
            "gaps": gaps[:6],
            "wardrobe_count": len(wardrobe_items),
            "analysis_method": "fashionclip_pseudo_embedding_cosine",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_aesthetic(self, dna_labels: List[str]) -> str:
        LABEL_MAP = {
            "minimalist": "minimalist", "minimal": "minimalist",
            "classic": "classic", "timeless": "classic", "chic": "classic",
            "boho": "boho", "bohemian": "boho", "cottagecore": "boho",
            "streetwear": "streetwear", "street": "streetwear", "urban": "streetwear",
            "avant-garde": "avant-garde", "avant garde": "avant-garde", "editorial": "avant-garde",
            "casual": "casual", "everyday": "casual",
        }
        for label in dna_labels:
            key = label.lower().strip()
            for pattern, aesthetic in LABEL_MAP.items():
                if pattern in key:
                    return aesthetic
        return "casual"


# Module-level singleton
gap_analyzer = GapAnalyzer()
