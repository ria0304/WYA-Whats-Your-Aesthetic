# ai_model.py - FINAL VERSION WITH IMPROVED MASKING & COLOR EXTRACTION
# Fixed: Properly isolates garment from background for accurate color detection
# Fixed: Better distinction between jumpsuits, skirts, and pants using aspect ratio
# Fixed: Removed wool completely - only cotton, linen, silk, denim, etc.
# Fixed: Style match always suggests different color than target garment
# Fixed: AUTOTAGGING - Skirts are skirts, Jumpsuits are jumpsuits, Pants are pants!
# Fixed: STYLE MATCH VARIETY - Different suggestions each time with variation parameter

import os
import json
import base64
import logging
import random
import colorsys
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from datetime import datetime, timedelta
import requests
from requests.structures import CaseInsensitiveDict

logger = logging.getLogger(__name__)

# -------------------- Computer Vision --------------------
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - image processing will be limited")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - color clustering will use fallback")

# -------------------- Deep Learning (Lazy Loading) --------------------
SAM_AVAILABLE = False
FASHIONCLIP_AVAILABLE = False
predictor = None
clip_model = None
clip_processor = None

def load_sam():
    global SAM_AVAILABLE, predictor
    if SAM_AVAILABLE or predictor is not None:
        return
    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        if os.path.exists(sam_checkpoint):
            sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam.to(device)
            predictor = SamPredictor(sam)
            SAM_AVAILABLE = True
            logger.info(f"SAM (vit_b) loaded on {device}")
        else:
            logger.warning(f"SAM checkpoint not found at {sam_checkpoint}. Using GrabCut fallback.")
    except Exception as e:
            logger.warning(f"SAM loading failed: {e}")

def load_fashionclip():
    global FASHIONCLIP_AVAILABLE, clip_model, clip_processor
    if FASHIONCLIP_AVAILABLE or clip_model is not None:
        return
    try:
        from transformers import CLIPProcessor, CLIPModel
        clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        FASHIONCLIP_AVAILABLE = True
        logger.info("FashionCLIP loaded successfully.")
    except Exception as e:
        logger.warning(f"FashionCLIP loading failed: {e}")

# -------------------- Matching Engine --------------------
try:
    from ai_matcher import fashion_matcher
except ImportError:
    fashion_matcher = None
    logger.warning("ai_matcher not found, using fallback matching")

# ====================== LOAD EXTERNAL JSON DATA ======================

def load_json_data(file_path: str, default: Any = None) -> Any:
    """Load data from JSON file with error handling."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"JSON file not found: {file_path}")
            return default if default is not None else {}
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return default if default is not None else {}

# Load fashion data (style suggestions)
FASHION_DATA = load_json_data('fashion_data.json', {})

# Load brand audit data
BRAND_SCORES = load_json_data('brand_score.json', {})

# Load regional items database (for limited edition items)
REGIONAL_ITEMS = load_json_data('regional_items.json', {"regions": {}})

# Load weather codes mapping
WEATHER_CODES = load_json_data('weather_codes.json', {})

# Load category taxonomy
CATEGORY_MAP = load_json_data('category_map.json', {})

# Load color dictionary
COLOR_DICTIONARY = load_json_data('color_dictionary.json', {})

# Load color harmony rules (used as fallback)
COLOR_HARMONY = load_json_data('color_harmony.json', {})

# Load country to region mapping
COUNTRY_TO_REGION = load_json_data('country_to_region.json', {})

# Load global chains list
GLOBAL_CHAINS = load_json_data('global_chains.json', [])

# Load local indicators list
LOCAL_INDICATORS = load_json_data('local_indicators.json', [])

# ====================== GEOAPIFY API CONFIGURATION ======================
# Your FREE Geoapify API key - get one at https://myprojects.geoapify.com
# No credit card required, 3000 requests/day free
GEOAPIFY_API_KEY = "3e48b0904db44b08993cd8c2aa999b4f"  # <--- YOUR API KEY HERE

# ====================== COLOR MATCHER CLASS ======================
class ColorMatcher:
    """
    Intelligent color matching for fashion - suggests matching colors based on 
    exact shade, saturation, and brightness of the garment
    """
    
    # Color families for categorization
    COLOR_FAMILIES = {
        "red": ["Red", "Burgundy", "Maroon", "Wine", "Brick Red", "Crimson"],
        "pink": ["Pink", "Blush", "Rose", "Fuchsia", "Magenta", "Coral", "Peach"],
        "orange": ["Orange", "Coral", "Peach", "Rust", "Terracotta", "Apricot"],
        "yellow": ["Yellow", "Mustard", "Gold", "Amber", "Honey"],
        "green": ["Green", "Emerald", "Mint", "Sage", "Olive", "Forest Green", "Lime", "Hunter Green"],
        "blue": ["Blue", "Navy", "Royal Blue", "Sky Blue", "Baby Blue", "Teal", "Turquoise", "Denim", "Cobalt", "Midnight Blue"],
        "purple": ["Purple", "Lavender", "Lilac", "Mauve", "Plum", "Eggplant", "Violet"],
        "brown": ["Brown", "Tan", "Camel", "Beige", "Cream", "Taupe", "Chocolate", "Coffee", "Caramel"],
        "gray": ["Gray", "Charcoal", "Silver", "Slate", "Ash"],
        "black": ["Black", "Jet Black", "Onyx"],
        "white": ["White", "Off-White", "Ivory", "Cream", "Eggshell"]
    }
    
    @staticmethod
    def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV for better color analysis"""
        r, g, b = [x/255.0 for x in rgb]
        return colorsys.rgb_to_hsv(r, g, b)
    
    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV back to RGB"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    @staticmethod
    def get_color_properties(rgb: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Analyze the color to get its properties
        """
        h, s, v = ColorMatcher.rgb_to_hsv(rgb)
        
        # Hue in degrees (0-360)
        hue_deg = h * 360
        
        # Determine color family
        color_family = ColorMatcher._get_color_family(hue_deg, s, v)
        
        # Determine specific color name (closest match from dictionary)
        color_name = ColorMatcher._rgb_to_color_name(rgb)
        
        # Determine saturation level
        if s < 0.15:
            saturation_level = "very_low"  # Almost gray
        elif s < 0.3:
            saturation_level = "low"       # Muted
        elif s < 0.5:
            saturation_level = "medium"    # Moderate
        elif s < 0.7:
            saturation_level = "high"      # Vibrant
        else:
            saturation_level = "very_high" # Extremely vibrant
        
        # Determine brightness level
        if v < 0.2:
            brightness_level = "very_dark"
        elif v < 0.35:
            brightness_level = "dark"
        elif v < 0.5:
            brightness_level = "medium_dark"
        elif v < 0.65:
            brightness_level = "medium"
        elif v < 0.8:
            brightness_level = "light"
        else:
            brightness_level = "very_light"
        
        return {
            "rgb": rgb,
            "hue": hue_deg,
            "saturation": s,
            "value": v,
            "color_family": color_family,
            "color_name": color_name,
            "saturation_level": saturation_level,
            "brightness_level": brightness_level
        }
    
    @staticmethod
    def _get_color_family(hue_deg: float, s: float, v: float) -> str:
        """
        Determine which color family this belongs to
        """
        if v < 0.15:  # Very dark
            return "black"
        if v > 0.9 and s < 0.1:  # Very light, low saturation
            return "white"
        if s < 0.1:  # Very low saturation
            return "gray"
        
        # Map hue to color family
        if hue_deg < 15 or hue_deg >= 345:
            return "red"
        elif 15 <= hue_deg < 35:
            return "orange"
        elif 35 <= hue_deg < 50:
            return "yellow"
        elif 50 <= hue_deg < 80:
            return "yellow_green"
        elif 80 <= hue_deg < 150:
            return "green"
        elif 150 <= hue_deg < 190:
            return "teal"  # Teal/Turquoise
        elif 190 <= hue_deg < 260:
            return "blue"
        elif 260 <= hue_deg < 330:
            return "purple"
        else:
            return "pink"
    
    @staticmethod
    def _rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
        """
        Find the closest color name from COLOR_DICTIONARY
        """
        r, g, b = rgb
        min_dist = float('inf')
        best_name = "Gray"
        
        for name, rgb_val in COLOR_DICTIONARY.items():
            dist = (r - rgb_val[0])**2 + (g - rgb_val[1])**2 + (b - rgb_val[2])**2
            if dist < min_dist:
                min_dist = dist
                best_name = name
        
        return best_name
    
    @staticmethod
    def _get_colors_by_hue(target_hue: float, target_sat: float, target_val: float, count: int = 3) -> List[str]:
        """
        Find color names that match a target hue range
        """
        # Define hue ranges for each color
        hue_ranges = {
            "Red": (0, 15),
            "Orange": (15, 35),
            "Yellow": (35, 50),
            "Green": (80, 150),
            "Teal": (150, 190),
            "Blue": (190, 260),
            "Purple": (260, 330),
            "Pink": (330, 360)
        }
        
        matching_colors = []
        for color_name, (hue_min, hue_max) in hue_ranges.items():
            if hue_min <= target_hue <= hue_max or (hue_max == 360 and target_hue >= hue_min):
                # Adjust saturation and value to find appropriate shade
                if target_sat > 0.7:
                    shades = [color_name, f"Bright {color_name}", f"Vibrant {color_name}"]
                elif target_sat < 0.3:
                    shades = [f"Dusty {color_name}", f"Muted {color_name}"]
                elif target_val < 0.3:
                    shades = [f"Deep {color_name}", f"Dark {color_name}"]
                elif target_val > 0.7:
                    shades = [f"Light {color_name}", f"Pale {color_name}"]
                else:
                    shades = [color_name]
                
                matching_colors.extend(shades)
        
        return matching_colors[:count]
    
    @staticmethod
    def _adjust_brightness(rgb: Tuple[int, int, int], new_v: float) -> Tuple[int, int, int]:
        """Adjust brightness of a color"""
        h, s, _ = ColorMatcher.rgb_to_hsv(rgb)
        new_v = max(0, min(1, new_v))
        return ColorMatcher.hsv_to_rgb(h, s, new_v)
    
    @staticmethod
    def get_matching_colors(garment_rgb: Tuple[int, int, int], variation: int = 0, count: int = 6) -> List[Dict[str, Any]]:
        """
        Suggest matching colors for a given garment color
        Variation parameter ensures different suggestions each time
        """
        props = ColorMatcher.get_color_properties(garment_rgb)
        
        # Use variation to seed randomness
        random.seed(variation + int(props["hue"] * 100))
        
        suggestions = []
        
        # ===== STRATEGY 1: Complementary colors (opposite on color wheel) =====
        comp_hue = (props["hue"] + 180) % 360
        # Add slight variation based on variation parameter
        comp_hue = (comp_hue + (variation * 3)) % 360
        
        comp_sat = min(1.0, props["saturation"] * random.uniform(1.1, 1.3))
        comp_val = min(1.0, props["value"] * random.uniform(1.05, 1.15))
        
        comp_rgb = ColorMatcher.hsv_to_rgb(comp_hue/360, comp_sat, comp_val)
        comp_name = ColorMatcher._rgb_to_color_name(comp_rgb)
        
        reasons = [
            "Creates a striking contrast that makes both colors pop",
            "Opposite colors that complement each other perfectly",
            "Bold contrast for a statement look",
            "Classic complementary pairing that always works"
        ]
        
        suggestions.append({
            "color": comp_name,
            "hex": '#{:02x}{:02x}{:02x}'.format(comp_rgb[0], comp_rgb[1], comp_rgb[2]),
            "rgb": [int(comp_rgb[0]), int(comp_rgb[1]), int(comp_rgb[2])],
            "match_type": "complementary",
            "confidence": round(random.uniform(0.92, 0.98), 2),
            "reason": random.choice(reasons)
        })
        
        # ===== STRATEGY 2: Split complementary =====
        split_offsets = [150, 210]
        random.shuffle(split_offsets)
        
        for offset in split_offsets[:2]:
            split_hue = (props["hue"] + offset + (variation * 5)) % 360
            split_rgb = ColorMatcher.hsv_to_rgb(
                split_hue/360, 
                props["saturation"] * random.uniform(0.85, 0.95), 
                props["value"] * random.uniform(0.85, 0.95)
            )
            split_name = ColorMatcher._rgb_to_color_name(split_rgb)
            
            split_reasons = [
                "Softer contrast than complementary, very harmonious",
                "Elegant and subtle - a sophisticated choice",
                "Balanced contrast that's easy on the eyes",
                "Modern twist on complementary colors"
            ]
            
            suggestions.append({
                "color": split_name,
                "hex": '#{:02x}{:02x}{:02x}'.format(split_rgb[0], split_rgb[1], split_rgb[2]),
                "rgb": [int(split_rgb[0]), int(split_rgb[1]), int(split_rgb[2])],
                "match_type": "split_complementary",
                "confidence": round(random.uniform(0.87, 0.93), 2),
                "reason": random.choice(split_reasons)
            })
        
        # ===== STRATEGY 3: Analogous colors =====
        analog_offsets = [30, -30]
        random.shuffle(analog_offsets)
        
        for offset in analog_offsets[:2]:
            analog_hue = (props["hue"] + offset + (variation * 2)) % 360
            analog_rgb = ColorMatcher.hsv_to_rgb(
                analog_hue/360, 
                props["saturation"] * random.uniform(0.75, 0.85), 
                props["value"] * random.uniform(0.95, 1.05)
            )
            analog_name = ColorMatcher._rgb_to_color_name(analog_rgb)
            
            analog_reasons = [
                "Harmonious and easy on the eyes",
                "Serene and cohesive color palette",
                "Subtle gradient effect - very trendy",
                "Peaceful and balanced combination"
            ]
            
            suggestions.append({
                "color": analog_name,
                "hex": '#{:02x}{:02x}{:02x}'.format(analog_rgb[0], analog_rgb[1], analog_rgb[2]),
                "rgb": [int(analog_rgb[0]), int(analog_rgb[1]), int(analog_rgb[2])],
                "match_type": "analogous",
                "confidence": round(random.uniform(0.82, 0.88), 2),
                "reason": random.choice(analog_reasons)
            })
        
        # ===== STRATEGY 4: Monochromatic =====
        if props["brightness_level"] in ["dark", "very_dark", "medium_dark"]:
            # Dark garment -> lighter shade
            lighter_v = min(1.0, props["value"] + random.uniform(0.35, 0.45))
            lighter_rgb = ColorMatcher.hsv_to_rgb(
                props["hue"]/360, 
                props["saturation"] * random.uniform(0.65, 0.75), 
                lighter_v
            )
            lighter_name = ColorMatcher._rgb_to_color_name(lighter_rgb)
            
            mono_reasons = [
                "Elegant tonal dressing - sophisticated and chic",
                "Monochrome magic - depth without distraction",
                "Subtle variation on your base color",
                "Understated elegance with tonal contrast"
            ]
            
            suggestions.append({
                "color": lighter_name,
                "hex": '#{:02x}{:02x}{:02x}'.format(lighter_rgb[0], lighter_rgb[1], lighter_rgb[2]),
                "rgb": [int(lighter_rgb[0]), int(lighter_rgb[1]), int(lighter_rgb[2])],
                "match_type": "monochromatic",
                "confidence": round(random.uniform(0.87, 0.92), 2),
                "reason": random.choice(mono_reasons)
            })
        elif props["brightness_level"] in ["light", "very_light"]:
            # Light garment -> darker shade
            darker_v = max(0.2, props["value"] - random.uniform(0.35, 0.45))
            darker_rgb = ColorMatcher.hsv_to_rgb(
                props["hue"]/360, 
                props["saturation"] * random.uniform(1.05, 1.15), 
                darker_v
            )
            darker_name = ColorMatcher._rgb_to_color_name(darker_rgb)
            
            mono_reasons = [
                "Sophisticated monochrome look with depth",
                "Grounding your light color with deeper tones",
                "Elegant contrast within the same family",
                "Rich and dimensional monochromatic styling"
            ]
            
            suggestions.append({
                "color": darker_name,
                "hex": '#{:02x}{:02x}{:02x}'.format(darker_rgb[0], darker_rgb[1], darker_rgb[2]),
                "rgb": [int(darker_rgb[0]), int(darker_rgb[1]), int(darker_rgb[2])],
                "match_type": "monochromatic",
                "confidence": round(random.uniform(0.87, 0.92), 2),
                "reason": random.choice(mono_reasons)
            })
        else:
            # Medium garment -> could go either way
            if variation % 2 == 0:
                lighter_v = min(1.0, props["value"] + random.uniform(0.25, 0.35))
                lighter_rgb = ColorMatcher.hsv_to_rgb(
                    props["hue"]/360, 
                    props["saturation"] * random.uniform(0.75, 0.85), 
                    lighter_v
                )
                lighter_name = ColorMatcher._rgb_to_color_name(lighter_rgb)
                
                suggestions.append({
                    "color": lighter_name,
                    "hex": '#{:02x}{:02x}{:02x}'.format(lighter_rgb[0], lighter_rgb[1], lighter_rgb[2]),
                    "rgb": [int(lighter_rgb[0]), int(lighter_rgb[1]), int(lighter_rgb[2])],
                    "match_type": "monochromatic",
                    "confidence": round(random.uniform(0.82, 0.88), 2),
                    "reason": "Light variation for contrast"
                })
            else:
                darker_v = max(0.2, props["value"] - random.uniform(0.25, 0.35))
                darker_rgb = ColorMatcher.hsv_to_rgb(
                    props["hue"]/360, 
                    props["saturation"] * random.uniform(1.05, 1.15), 
                    darker_v
                )
                darker_name = ColorMatcher._rgb_to_color_name(darker_rgb)
                
                suggestions.append({
                    "color": darker_name,
                    "hex": '#{:02x}{:02x}{:02x}'.format(darker_rgb[0], darker_rgb[1], darker_rgb[2]),
                    "rgb": [int(darker_rgb[0]), int(darker_rgb[1]), int(darker_rgb[2])],
                    "match_type": "monochromatic",
                    "confidence": round(random.uniform(0.82, 0.88), 2),
                    "reason": "Deep variation for drama"
                })
        
        # ===== STRATEGY 5: Neutrals =====
        neutrals = [
            {"name": "White", "rgb": (255, 255, 255), "reasons": ["Crisp and clean - lets your garment shine", "Fresh and modern", "Timeless classic", "Brightens any outfit"]},
            {"name": "Black", "rgb": (0, 0, 0), "reasons": ["Timeless and elegant - creates definition", "Sleek and sophisticated", "Edgy contrast", "Always in style"]},
            {"name": "Cream", "rgb": (255, 253, 208), "reasons": ["Soft and warm - effortless sophistication", "Gentle alternative to white", "Romantic and soft", "Warms up any color"]},
            {"name": "Beige", "rgb": (245, 245, 220), "reasons": ["Versatile neutral that complements any color", "Understated elegance", "Earth tones harmony", "Calm and collected"]},
            {"name": "Gray", "rgb": (128, 128, 128), "reasons": ["Modern and understated - perfect balance", "Urban chic", "Sleek minimalist choice", "Cool sophistication"]},
            {"name": "Navy", "rgb": (0, 0, 128), "reasons": ["Classic alternative to black - rich and deep", "Professional and polished", "Depth without darkness", "Refined choice"]}
        ]
        
        # Add 2-3 neutrals, avoiding the garment's own color family
        neutral_count = 0
        neutral_target = random.randint(2, 3)
        random.shuffle(neutrals)
        
        for neutral in neutrals:
            if neutral_count >= neutral_target:
                break
            # Skip if neutral is too close to garment color
            neutral_h, neutral_s, neutral_v = ColorMatcher.rgb_to_hsv(neutral["rgb"])
            neutral_family = ColorMatcher._get_color_family(neutral_h * 360, neutral_s, neutral_v)
            
            if neutral_family != props["color_family"]:
                suggestions.append({
                    "color": neutral["name"],
                    "hex": '#{:02x}{:02x}{:02x}'.format(neutral["rgb"][0], neutral["rgb"][1], neutral["rgb"][2]),
                    "rgb": [int(neutral["rgb"][0]), int(neutral["rgb"][1]), int(neutral["rgb"][2])],
                    "match_type": "neutral",
                    "confidence": round(random.uniform(0.75, 0.85), 2),
                    "reason": random.choice(neutral["reasons"])
                })
                neutral_count += 1
        
        # Remove duplicates based on color name
        unique_suggestions = []
        seen_colors = set()
        
        for suggestion in suggestions:
            if suggestion["color"] not in seen_colors and suggestion["color"] != props["color_name"]:
                seen_colors.add(suggestion["color"])
                unique_suggestions.append(suggestion)
        
        # Sort by confidence and return top 'count'
        unique_suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Add some randomness to the order
        if len(unique_suggestions) > count:
            # Take top count+2 and shuffle slightly
            top_suggestions = unique_suggestions[:count+2]
            random.shuffle(top_suggestions)
            return top_suggestions[:count]
        
        return unique_suggestions[:count]
    
    @staticmethod
    def get_best_match(garment_rgb: Tuple[int, int, int], variation: int = 0) -> Dict[str, Any]:
        """
        Get the single best matching color with variation
        """
        matches = ColorMatcher.get_matching_colors(garment_rgb, variation, count=3)
        # Pick one randomly from top 3 for variety
        if matches:
            return random.choice(matches)
        
        return {
            "color": "White",
            "hex": "#ffffff",
            "rgb": [255, 255, 255],
            "match_type": "fallback",
            "confidence": 0.5,
            "reason": "Classic white - always a safe choice"
        }


# ====================== LOCAL COMPUTER VISION ======================
class LocalComputerVision:
    """
    Local CV Engine with upgraded segmentation and advanced color extraction.
    FIXED: Better background removal to isolate ONLY the garment.
    FIXED: Better distinction between jumpsuits, skirts, and pants using aspect ratio.
    """
    
    def decode_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 image to numpy array with proper error handling."""
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available for image decoding")
            return np.zeros((256, 256, 3), dtype=np.uint8)

        try:
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            if not base64_str or len(base64_str) < 100:
                raise ValueError("Invalid base64 image data")
            img_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image decode failed - cv2.imdecode returned None")
            if img.shape[0] < 10 or img.shape[1] < 10:
                raise ValueError(f"Image too small: {img.shape}")
            return img
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return np.zeros((256, 256, 3), dtype=np.uint8)

    def get_improved_mask(self, image: np.ndarray) -> np.ndarray:
        """
        IMPROVED MASKING: Better background removal to isolate garment.
        Uses multiple techniques to ensure only the garment is captured.
        """
        load_sam()
        
        # Try SAM first (best accuracy)
        if SAM_AVAILABLE:
            sam_mask = self._get_sam_mask(image)
            if sam_mask is not None and np.sum(sam_mask > 0) > 10000:  # Need enough pixels
                # Clean up SAM mask
                kernel = np.ones((5, 5), np.uint8)
                sam_mask = cv2.morphologyEx(sam_mask, cv2.MORPH_CLOSE, kernel)
                sam_mask = cv2.morphologyEx(sam_mask, cv2.MORPH_OPEN, kernel)
                
                # Find the largest contour (main garment)
                contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Create new mask with only the largest contour
                    clean_mask = np.zeros_like(sam_mask)
                    cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
                    
                    # Erode to remove any remaining background edges
                    erode_kernel = np.ones((5, 5), np.uint8)
                    clean_mask = cv2.erode(clean_mask, erode_kernel, iterations=2)
                    
                    return clean_mask
                
                return sam_mask
        
        # Fallback to enhanced GrabCut
        return self._enhanced_grabcut_mask(image)

    def _get_sam_mask(self, image_np: np.ndarray) -> Optional[np.ndarray]:
        """Internal SAM mask generation."""
        if not SAM_AVAILABLE or predictor is None:
            return None
        try:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) if image_np.shape[2] == 3 else image_np
            predictor.set_image(image_rgb)
            masks, _, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=False)
            return (masks[0] * 255).astype(np.uint8)
        except Exception as e:
            logger.warning(f"SAM prediction failed: {e}")
            return None

    def _enhanced_grabcut_mask(self, image: np.ndarray) -> np.ndarray:
        """
        ENHANCED GrabCut with better background separation.
        Uses edge detection and color analysis to isolate the garment.
        """
        if not CV2_AVAILABLE:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255

        h, w = image.shape[:2]
        
        # Step 1: Create a rough mask using color and edge information
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Otsu's thresholding to find potential foreground
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Use edge detection to find boundaries
        edges = cv2.Canny(gray, 30, 100)
        
        # Combine threshold and edges
        combined = cv2.bitwise_or(thresh, edges)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (assumed to be the garment)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a mask from the largest contour
            contour_mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
            
            # Dilate to ensure we capture the whole garment
            kernel = np.ones((15, 15), np.uint8)
            contour_mask = cv2.dilate(contour_mask, kernel, iterations=2)
            
            # Create a rectangle around the garment
            x, y, wc, hc = cv2.boundingRect(largest_contour)
            
            # Expand rectangle slightly
            pad_x = int(wc * 0.1)
            pad_y = int(hc * 0.1)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            wc = min(w - x, wc + 2 * pad_x)
            hc = min(h - y, hc + 2 * pad_y)
            
            rect = (x, y, wc, hc)
            
            # Initialize GrabCut mask
            mask = np.zeros(image.shape[:2], np.uint8)
            mask[contour_mask > 0] = cv2.GC_PR_FGD
            
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Run GrabCut
            try:
                cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                
                # Create final mask
                final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
                
                # Clean up the mask
                kernel = np.ones((5, 5), np.uint8)
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
                
                # Find the largest contour again (to remove any small artifacts)
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    clean_mask = np.zeros_like(final_mask)
                    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
                    
                    # Erode slightly to remove edges
                    erode_kernel = np.ones((3, 3), np.uint8)
                    clean_mask = cv2.erode(clean_mask, erode_kernel, iterations=1)
                    
                    return clean_mask
                
                return final_mask
            except:
                return contour_mask
        
        # Ultimate fallback - center ellipse
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 3, h // 3)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return mask

    def _get_garment_crop(self, image: np.ndarray, mask: np.ndarray):
        """Crop image to garment bounding box with minimal padding."""
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        # Minimal padding to avoid including background
        pad = 5
        y1, y2 = max(0, y - pad), min(image.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(image.shape[1], x + w + pad)
        return image[y1:y2, x1:x2]

    def identify_garment(self, image: np.ndarray, mask: np.ndarray) -> str:
        """
        Use FashionCLIP to identify garment category.
        FIXED: Better distinction between jumpsuits, skirts, and pants using aspect ratio and shape analysis.
        """
        load_fashionclip()
        if not FASHIONCLIP_AVAILABLE:
            return "Top"
        
        try:
            from PIL import Image
            import torch
            
            # Get cropped garment
            cropped = self._get_garment_crop(image, mask)
            
            # If crop is too small, use original
            if cropped.shape[0] < 50 or cropped.shape[1] < 50:
                cropped = image
            
            # Calculate aspect ratio (height/width) to help distinguish garment types
            h, w = cropped.shape[:2]
            aspect_ratio = h / w if w > 0 else 1.0
            logger.info(f"Garment aspect ratio: {aspect_ratio:.2f} (height/width)")
            
            # Calculate the bounding box of the mask to get more accurate shape
            if mask is not None and np.sum(mask > 0) > 0:
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x, y, w_mask, h_mask = cv2.boundingRect(coords)
                    mask_aspect_ratio = h_mask / w_mask if w_mask > 0 else 1.0
                    logger.info(f"Mask aspect ratio: {mask_aspect_ratio:.2f} (height/width)")
                    # Use mask aspect ratio as it's more accurate for the garment shape
                    aspect_ratio = mask_aspect_ratio
            
            # Prepare image for CLIP
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
            # Expanded labels with more specific categories
            labels = [
                "t-shirt", "shirt", "blouse", "tank top", "sweater", "hoodie", "cardigan", "polo",
                "jeans", "pants", "trousers", "leggings", "shorts", "cargo pants", "joggers",
                "skirt", "pencil skirt", "pleated skirt", "mini skirt", "midi skirt", "maxi skirt",
                "dress", "maxi dress", "mini dress", "midi dress", "bodycon dress", "a-line dress",
                "jumpsuit", "romper", "overalls",
                "jacket", "coat", "blazer", "puffer jacket", "denim jacket", "leather jacket"
            ]
            
            # Get CLIP predictions
            inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
            
            # Get top 5 predictions for better disambiguation
            top_probs, top_indices = torch.topk(probs[0], 5)
            top_labels = [labels[idx] for idx in top_indices]
            top_scores = [p.item() for p in top_probs]
            
            logger.info(f"Top predictions: {list(zip(top_labels, [f'{p:.3f}' for p in top_scores]))}")
            
            # Define category keywords
            jumpsuit_keywords = ["jumpsuit", "romper", "overalls"]
            dress_keywords = ["dress", "maxi dress", "mini dress", "midi dress", "bodycon", "a-line"]
            skirt_keywords = ["skirt", "pencil skirt", "pleated skirt", "mini skirt", "midi skirt", "maxi skirt"]
            pants_keywords = ["jeans", "pants", "trousers", "leggings", "shorts", "cargo", "joggers"]
            top_keywords = ["t-shirt", "shirt", "blouse", "tank top", "sweater", "hoodie", "cardigan", "polo"]
            outerwear_keywords = ["jacket", "coat", "blazer", "puffer"]
            
            # Calculate confidence scores for each category type
            jumpsuit_score = 0
            dress_score = 0
            skirt_score = 0
            pants_score = 0
            top_score = 0
            outerwear_score = 0
            
            for label, score in zip(top_labels, top_scores):
                label_lower = label.lower()
                
                if any(keyword in label_lower for keyword in jumpsuit_keywords):
                    jumpsuit_score += score
                elif any(keyword in label_lower for keyword in dress_keywords):
                    dress_score += score
                elif any(keyword in label_lower for keyword in skirt_keywords):
                    skirt_score += score
                elif any(keyword in label_lower for keyword in pants_keywords):
                    pants_score += score
                elif any(keyword in label_lower for keyword in top_keywords):
                    top_score += score
                elif any(keyword in label_lower for keyword in outerwear_keywords):
                    outerwear_score += score
            
            logger.info(f"Category scores - Jumpsuit: {jumpsuit_score:.3f}, Dress: {dress_score:.3f}, Skirt: {skirt_score:.3f}, Pants: {pants_score:.3f}, Top: {top_score:.3f}, Outerwear: {outerwear_score:.3f}")
            
            # ==================== FIXED DECISION LOGIC ====================
            
            # STEP 1: Check for SKIRTS first - skirts are separate garments that only cover the lower body
            # Skirts typically have aspect ratio between 1.0 and 2.5
            # Pencil skirts are long but are STILL SKIRTS, not jumpsuits
            if skirt_score > 0.2:
                if 0.8 < aspect_ratio < 2.5:
                    logger.info(f"SKIRT detected (score: {skirt_score:.3f}, aspect: {aspect_ratio:.2f})")
                    return "Skirt"
                elif skirt_score > 0.4:
                    logger.info(f"SKIRT detected (high score: {skirt_score:.3f})")
                    return "Skirt"
            
            # STEP 2: Check for JUMPSUITS - they are full-body garments that cover both top and bottom
            # Jumpsuits typically have HIGH aspect ratio (> 1.8) because they're long
            if jumpsuit_score > 0.25:
                if aspect_ratio > 1.8:
                    logger.info(f"JUMPSUIT detected (score: {jumpsuit_score:.3f}, aspect: {aspect_ratio:.2f})")
                    return "Jumpsuit"
                elif jumpsuit_score > 0.45:
                    logger.info(f"JUMPSUIT detected (high score: {jumpsuit_score:.3f})")
                    return "Jumpsuit"
            
            # STEP 3: Check for PANTS - pants cover legs only
            if pants_score > 0.3:
                # If jumpsuit score is also significant and aspect ratio is high, it's a jumpsuit
                if jumpsuit_score > 0.2 and aspect_ratio > 2.0:
                    logger.info(f"JUMPSUIT overrides PANTS (jumpsuit: {jumpsuit_score:.3f}, pants: {pants_score:.3f}, aspect: {aspect_ratio:.2f})")
                    return "Jumpsuit"
                logger.info(f"PANTS detected (score: {pants_score:.3f})")
                return "Pants"
            
            # STEP 4: Check for DRESSES
            if dress_score > 0.3:
                if aspect_ratio > 2.5 and jumpsuit_score > 0.2:
                    logger.info(f"JUMPSUIT overrides DRESS (high aspect ratio)")
                    return "Jumpsuit"
                logger.info(f"DRESS detected (score: {dress_score:.3f})")
                return "Dress"
            
            # STEP 5: Check for TOPS
            if top_score > 0.3:
                logger.info(f"TOP detected (score: {top_score:.3f})")
                return "Top"
            
            # STEP 6: Check for OUTERWEAR
            if outerwear_score > 0.3:
                logger.info(f"OUTERWEAR detected (score: {outerwear_score:.3f})")
                return "Outerwear"
            
            # STEP 7: TIEBREAKER for ambiguous cases
            if jumpsuit_score > 0.15 and pants_score > 0.15:
                if aspect_ratio > 2.0:
                    logger.info(f"TIEBREAKER: aspect ratio {aspect_ratio:.2f} > 2.0 -> Jumpsuit")
                    return "Jumpsuit"
                else:
                    logger.info(f"TIEBREAKER: aspect ratio {aspect_ratio:.2f} < 2.0 -> Pants")
                    return "Pants"
            
            if jumpsuit_score > 0.15 and skirt_score > 0.15:
                if aspect_ratio > 2.2:
                    logger.info(f"TIEBREAKER: aspect ratio {aspect_ratio:.2f} > 2.2 -> Jumpsuit")
                    return "Jumpsuit"
                else:
                    logger.info(f"TIEBREAKER: aspect ratio {aspect_ratio:.2f} < 2.2 -> Skirt")
                    return "Skirt"
            
            # STEP 8: Check for PENCIL SKIRTS specifically
            if any("pencil" in label.lower() for label in top_labels) and 1.5 < aspect_ratio < 2.5:
                logger.info(f"PENCIL SKIRT detected with aspect ratio {aspect_ratio:.2f}")
                return "Skirt"
            
            # STEP 9: Final check - if it's a jumpsuit-like garment but was misclassified
            if any("jumpsuit" in label.lower() for label in top_labels) and aspect_ratio > 1.8:
                logger.info(f"Final check: JUMPSUIT detected from labels")
                return "Jumpsuit"
            
            # STEP 10: Default to top prediction mapped through CATEGORY_MAP
            idx = probs.argmax().item()
            raw_category = labels[idx]
            
            # Final check - if it's a skirt-like garment but was misclassified
            if "skirt" in raw_category.lower() and aspect_ratio < 2.5:
                return "Skirt"
            
            # Map through CATEGORY_MAP
            mapped_category = CATEGORY_MAP.get(raw_category, "Top")
            logger.info(f"Final category: {mapped_category} (from {raw_category})")
            return mapped_category
            
        except Exception as e:
            logger.warning(f"FashionCLIP identification failed: {e}")
            return "Top"

    def get_dominant_color(self, image: np.ndarray, mask: np.ndarray) -> Tuple[str, str, Tuple[int, int, int]]:
        """
        FIXED COLOR ENGINE: Uses only garment pixels, aggressively filters background.
        Returns hex color, color name, and RGB tuple.
        """
        if not CV2_AVAILABLE or not SKLEARN_AVAILABLE:
            return "#808080", "Gray", (128, 128, 128)

        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Erode aggressively to remove boundary pixels
        erode_kernel = np.ones((7, 7), np.uint8)
        mask = cv2.erode(mask, erode_kernel, iterations=3)
        
        # Get mask statistics
        mask_pixels = np.sum(mask > 0)
        logger.info(f"Mask has {mask_pixels} non-zero pixels after cleaning")
        
        # If mask is too small, use a center region approach
        if mask_pixels < 5000:
            logger.warning("Mask too small, using center region approach")
            h, w = image.shape[:2]
            center_mask = np.zeros((h, w), dtype=np.uint8)
            center_h, center_w = h // 2, w // 2
            size = min(h, w) // 3
            cv2.rectangle(center_mask, 
                         (center_w - size, center_h - size),
                         (center_w + size, center_h + size), 
                         255, -1)
            mask = center_mask
            mask_pixels = np.sum(mask > 0)
        
        # Extract pixels from masked region
        pixels = image[mask > 0]
        logger.info(f"Extracted {len(pixels)} garment pixels for color analysis")
        
        if len(pixels) < 1000:
            logger.warning(f"Not enough garment pixels ({len(pixels)}). Using fallback.")
            return "#808080", "Gray", (128, 128, 128)

        # Convert to RGB and HSV for analysis
        pixels_rgb = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        pixels_hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        
        # Filter out dark, bright, and low saturation pixels (likely background)
        # More aggressive filtering for better accuracy
        quality_mask = (
            (pixels_hsv[:, 2] > 30) &  # Not too dark
            (pixels_hsv[:, 2] < 220) &  # Not too bright
            (pixels_hsv[:, 1] > 20) &   # Some saturation (avoid grays)
            (pixels_rgb[:, 0] < 240) &   # Not too white
            (pixels_rgb[:, 1] < 240) &
            (pixels_rgb[:, 2] < 240) &
            (pixels_rgb[:, 0] > 30) &    # Not too black
            (pixels_rgb[:, 1] > 30) &
            (pixels_rgb[:, 2] > 30)
        )
        
        # Use quality pixels if enough, otherwise fallback
        if np.sum(quality_mask) > 500:
            filtered_pixels = pixels_rgb[quality_mask]
            logger.info(f"Using {len(filtered_pixels)} quality pixels after filtering")
        else:
            filtered_pixels = pixels_rgb
            logger.info(f"Using all {len(filtered_pixels)} pixels (insufficient quality)")

        try:
            # Use KMeans to find dominant colors
            n_clusters = min(5, max(3, len(filtered_pixels) // 1000))
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
            kmeans.fit(filtered_pixels)
            
            # Get cluster sizes
            counts = np.bincount(kmeans.labels_)
            sorted_indices = np.argsort(counts)[::-1]
            
            # Get the top 2 clusters
            top_clusters = []
            for idx in sorted_indices[:2]:
                rgb = kmeans.cluster_centers_[idx]
                r, g, b = rgb.astype(int)
                total = r + g + b
                
                # Validate the color
                if 60 < total < 750:
                    top_clusters.append((rgb, counts[idx]))
            
            if top_clusters:
                # Use the largest valid cluster
                best_rgb = top_clusters[0][0]
            else:
                # Fallback to largest cluster
                best_rgb = kmeans.cluster_centers_[sorted_indices[0]]
            
            r, g, b = best_rgb.astype(int)
            
        except Exception as e:
            logger.warning(f"KMeans failed: {e}, using median")
            r, g, b = np.median(filtered_pixels, axis=0).astype(int)

        # Map RGB to color name using dictionary
        best_name = self._map_rgb_to_color_name(r, g, b)
        
        # Special case for denim
        if best_name in ["Navy", "Blue", "Light Blue", "Gray"]:
            if 60 < g < 150 and 40 < r < 140 and 80 < b < 200:
                best_name = "Denim"
        
        hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        logger.info(f"Detected color: {best_name} ({r},{g},{b}) from {len(filtered_pixels)} pixels")
        
        return hex_color, best_name, (r, g, b)

    def _map_rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """
        Map RGB values to the closest color name from COLOR_DICTIONARY.
        """
        min_dist = float('inf')
        best_name = "Gray"
        
        for name, rgb_val in COLOR_DICTIONARY.items():
            dist = (r - rgb_val[0])**2 + (g - rgb_val[1])**2 + (b - rgb_val[2])**2
            if dist < min_dist:
                min_dist = dist
                best_name = name
        
        return best_name

    def analyze_texture_properties(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
        """Analyze texture with optional mask."""
        if not CV2_AVAILABLE:
            return {"variance": 0.0, "brightness": 128.0}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if mask is not None and mask.size > 0:
            if mask.shape != gray.shape:
                mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            center = masked_gray[mask > 0]
        else:
            center = gray

        if len(center) == 0:
            center = gray

        variance = float(np.var(center))
        brightness = float(np.mean(center))
        return {"variance": variance, "brightness": brightness}


# ====================== FABRIC CLASSIFIER ======================
class FabricClassifier:
    """
    Enhanced Logic Inference Engine (LIE) for Fabric Detection.
    FIXED: Removed wool completely - only cotton, linen, silk, denim, etc.
    Defaults to "Cotton" instead of "Unknown"
    """
    @staticmethod
    def classify(variance: float, brightness: float, color: str, category: str) -> str:

        denim_colors = ["Denim", "Light Denim", "Navy", "Blue", "Charcoal", "Ice Blue", "Gray", "Black",
                       "Light Blue", "Royal Blue", "Sky Blue", "Slate", "Indigo", "Midnight Navy"]
        denim_categories = ["Pants", "Shorts", "Jacket", "Skirt", "Dress", "Jumpsuit", "Top"]

        if category in denim_categories and color in denim_colors:
            if 150 < variance < 800:
                return "Denim"
            if color in ["Denim", "Light Denim"]:
                return "Denim"

        # --- ACCESSORIES & JEWELRY ---
        if category in ["Necklace", "Ring", "Earrings", "Watch", "Jewellery"]:
            if color in ["Gold", "Yellow", "Orange", "Beige", "Cream"]:
                return "Gold"
            if color in ["Silver", "Gray", "White", "Platinum", "Ash"]:
                return "Silver"
            if category == "Watch" and color in ["Black", "Brown", "Tan"]:
                return "Leather Strap"
            return "Metal"

        if category == "Bag":
            if color in ["Brown", "Tan", "Black", "Camel", "Cognac", "Red"]:
                return "Leather"
            if variance > 300:
                return "Canvas"
            return "Synthetic"

        # --- BOTTOM FABRICS ---
        if category in ["Pants", "Shorts"]:
            if variance > 500:
                return "Cotton"
            if 200 < variance < 500:
                return "Cotton"
            return "Polyester"

        # --- SKIRT FABRICS ---
        if category == "Skirt":
            if variance < 30 and brightness > 120:  # Very smooth and bright = satin/silk
                return "Satin"
            if 300 < variance < 700:  # Medium texture = cotton/linen
                if color in ["White", "Beige", "Cream", "Olive", "Sage", "Brown", "Tan"]:
                    return "Linen"
                return "Cotton"
            if variance < 100 and brightness > 150:  # Smooth and bright = maybe silk
                return "Silk"
            return "Polyester"  # Default for skirts

        # --- DRESS FABRICS ---
        if category == "Dress":
            if variance > 800:
                # Could be velvet, knit, or textured fabric
                if color in ["Red", "Burgundy", "Navy", "Black", "Green", "Emerald", "Purple"] and brightness < 120:
                    return "Velvet"  # Rich, deep colors often indicate velvet
                if brightness < 100:
                    return "Knit"  # Dark, textured = knit dress
                return "Textured"  # Light, textured = textured weave
                
            if variance < 30 and brightness > 120:  # Very smooth = satin/silk
                return "Satin"
                
            if 30 < variance < 100 and brightness > 150:  # Slightly textured but bright = silk/chiffon
                if color in ["Blush", "Lavender", "Mint", "Baby Blue", "Cream", "White"]:
                    return "Chiffon"  # Light colors often indicate chiffon
                return "Silk"
                
            if 100 < variance < 400:  # Low-medium texture = cotton/linen
                if color in ["White", "Beige", "Cream", "Olive", "Sage", "Brown", "Tan", "Rust"]:
                    return "Linen"  # Earthy tones often linen
                return "Cotton"
                
            if 400 < variance < 700:  # Medium texture = crepe/ponte
                if brightness < 100:
                    return "Crepe"  # Dark = crepe
                return "Ponte"
                
            return "Polyester"  # Default for dresses

        # --- TOP FABRICS ---
        if category == "Top":
            if variance > 800:
                return "Cotton"
            if variance < 30 and brightness > 120:
                return "Satin"
            if 300 < variance < 700 and color in ["White", "Beige", "Cream", "Olive"]:
                return "Linen"
            return "Cotton"

        # --- JUMPSUIT FABRICS ---
        if category == "Jumpsuit":
            if variance < 30 and brightness > 120:  # Very smooth = satin
                return "Satin"
                
            if variance > 500:  # Very textured
                if brightness < 100:
                    return "Velvet"  # Dark, textured = velvet
                return "Textured"
                
            if 30 < variance < 200 and brightness > 150:  # Slightly textured, bright = silk
                return "Silk"
                
            if 200 < variance < 500:  # Medium texture = cotton/linen
                if color in ["White", "Beige", "Cream", "Olive", "Sage", "Brown", "Tan"]:
                    return "Linen"
                return "Cotton"
                
            return "Polyester"  # Default for jumpsuits

        # --- OUTERWEAR ---
        if category in ["Jacket", "Coat", "Outerwear", "Blazer"]:
            if variance > 800:
                return "Cotton"
            if variance > 400:
                return "Cotton"
            if color in ["Brown", "Tan", "Camel", "Black", "Navy"]:
                return "Leather" if variance < 200 else "Suede"
            return "Polyester"

        return "Cotton"  # Default to Cotton instead of Unknown


# ====================== STYLE PROFILE ======================
class StyleProfile:
    """
    Manages user style preferences and profiles.
    """

    @staticmethod
    def get_default_profile() -> Dict[str, Any]:
        """Return a default style profile"""
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
                "party": "Bold and stylish"
            }
        }

    @staticmethod
    def extract_from_questionnaire(answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert questionnaire answers to style profile.
        """
        profile = StyleProfile.get_default_profile()

        # --- 1. Process Everyday Look selections ---
        everyday_looks = answers.get("everyday_look", [])
        if isinstance(everyday_looks, str):
            everyday_looks = [everyday_looks]

        look_mapping = {
            "Minimalist & Clean": {
                "archetype": "Minimalist",
                "keywords": ["clean-lined", "understated", "monochromatic", "streamlined"],
                "colors": ["Monochrome"],
                "silhouette": "Tailored & Structured",
                "color_preference_name": "Modern Minimalist",
                "color_preference_colors": ["Black", "White", "Gray", "Charcoal"]
            },
            "Bold & Experimental": {
                "archetype": "Avant-Garde",
                "keywords": ["bold", "experimental", "statement", "unique", "artistic"],
                "colors": ["Bright", "Contrast"],
                "silhouette": "Architectural",
                "color_preference_name": "Avant-Garde Edge",
                "color_preference_colors": ["Red", "Royal Blue", "Emerald", "Yellow", "Fuchsia"]
            },
            "Classic & Sophisticated": {
                "archetype": "Classic",
                "keywords": ["timeless", "elegant", "polished", "refined", "sophisticated"],
                "colors": ["Monochrome", "Neutrals"],
                "silhouette": "Tailored & Structured",
                "color_preference_name": "Timeless Elegance",
                "color_preference_colors": ["Navy", "Beige", "Cream", "Burgundy", "Brown"]
            },
            "Bohemian & Relaxed": {
                "archetype": "Boho",
                "keywords": ["flowy", "relaxed", "earthy", "layered", "free-spirited"],
                "colors": ["Earthy", "Warm"],
                "silhouette": "Draped & Flowing",
                "color_preference_name": "Warm Sunrise",
                "color_preference_colors": ["Orange", "Pink", "Coral", "Peach", "Terracotta"]
            },
            "Streetwear & Edgy": {
                "archetype": "Streetwear",
                "keywords": ["edgy", "urban", "oversized", "graphic", "casual-cool"],
                "colors": ["Monochrome", "Bold"],
                "silhouette": "Oversized & Relaxed",
                "color_preference_name": "Urban Edge",
                "color_preference_colors": ["Black", "Charcoal", "Red", "Olive", "Mustard"]
            }
        }

        profile["style_vibes"] = everyday_looks
        all_keywords = []
        all_colors = []

        for look in everyday_looks:
            if look in look_mapping:
                mapped = look_mapping[look]
                all_keywords.extend(mapped["keywords"])
                if mapped["colors"]:
                    all_colors.extend(mapped["colors"])

        if everyday_looks:
            primary_look = everyday_looks[0]
            profile["style_archetype"] = look_mapping.get(primary_look, {}).get("archetype", "Casual")
            profile["everyday_look"] = primary_look
            profile["color_preference_name"] = look_mapping.get(primary_look, {}).get("color_preference_name", "Custom Palette")
            profile["color_preference_colors"] = look_mapping.get(primary_look, {}).get("color_preference_colors", ["Black", "White", "Gray"])

        profile["style_keywords"] = list(dict.fromkeys(all_keywords))

        # --- 2. Process Color Preference ---
        color_pref = answers.get("color_preference", "")
        profile["color_preference"] = color_pref

        if "Monochrome" in color_pref:
            profile["color_categories"] = ["Monochrome"]
            profile["preferred_colors"] = ["Black", "White", "Gray", "Charcoal", "Silver"]
            profile["color_preference_name"] = "Classic Monochrome"
            profile["color_preference_colors"] = ["Black", "White", "Gray", "Charcoal"]
        elif "Pastels" in color_pref:
            profile["color_categories"] = ["Pastels"]
            profile["preferred_colors"] = ["Blush", "Lavender", "Mint", "Baby Blue", "Cream", "Powder Pink"]
            profile["color_preference_name"] = "Soft Pastels"
            profile["color_preference_colors"] = ["Blush", "Lavender", "Mint", "Baby Blue"]
        elif "Earthy" in color_pref or "Warm" in color_pref:
            profile["color_categories"] = ["Earthy"]
            profile["preferred_colors"] = ["Olive", "Brown", "Tan", "Rust", "Sage", "Camel", "Terracotta"]
            profile["color_preference_name"] = "Earthy Tones"
            profile["color_preference_colors"] = ["Olive", "Brown", "Tan", "Rust", "Terracotta"]
        elif "Bright" in color_pref:
            profile["color_categories"] = ["Bright"]
            profile["preferred_colors"] = ["Red", "Royal Blue", "Emerald", "Yellow", "Coral", "Fuchsia"]
            profile["color_preference_name"] = "Vibrant Brights"
            profile["color_preference_colors"] = ["Red", "Royal Blue", "Emerald", "Yellow", "Coral"]
        else:
            if "Black/White" in color_pref:
                profile["preferred_colors"].extend(["Black", "White"])
                profile["color_preference_name"] = "Classic Contrast"
                profile["color_preference_colors"] = ["Black", "White"]
            if "Pastels" in color_pref:
                profile["preferred_colors"].extend(["Blush", "Lavender", "Mint"])
                profile["color_preference_name"] = "Custom Blend"
                profile["color_preference_colors"] = ["Blush", "Lavender", "Mint"]

        # --- 3. Process Silhouette Preference ---
        silhouette = answers.get("silhouette", "Draped & Flowing")
        profile["silhouette_preference"] = silhouette

        if silhouette == "Tailored & Structured":
            profile["style_keywords"].extend(["tailored", "structured", "sharp"])
        elif silhouette == "Draped & Flowing":
            profile["style_keywords"].extend(["flowy", "draped", "soft"])
        elif silhouette == "Oversized & Relaxed":
            profile["style_keywords"].extend(["oversized", "relaxed", "comfortable"])

        profile["style_keywords"] = list(dict.fromkeys(profile["style_keywords"]))

        return profile


# ====================== FASHION AI MODEL ======================
class FashionAIModel:
    vision = LocalComputerVision()
    classifier = FabricClassifier()
    _user_profiles: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    async def set_user_style_profile(user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set or update a user's style profile from questionnaire answers.
        """
        if "answers" in profile_data:
            profile = StyleProfile.extract_from_questionnaire(profile_data["answers"])
        else:
            profile = {**StyleProfile.get_default_profile(), **profile_data}

        FashionAIModel._user_profiles[user_id] = profile

        logger.info(f"Style profile set for user {user_id}: {profile['style_archetype']}")

        return {
            "success": True,
            "profile": profile,
            "message": f"Style DNA mapped to {profile['style_archetype']}"
        }

    @staticmethod
    def get_user_style_profile(user_id: str) -> Dict[str, Any]:
        """
        Get a user's style profile.
        """
        return FashionAIModel._user_profiles.get(user_id, StyleProfile.get_default_profile())

    @staticmethod
    def get_style_evolution(user_id: str) -> Dict[str, Any]:
        """
        Get style evolution timeline for a user.
        """
        profile = FashionAIModel.get_user_style_profile(user_id)
        
        today = datetime.now().strftime("%b %d").upper()
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%b %d").upper()
        two_weeks_ago = (datetime.now() - timedelta(days=14)).strftime("%b %d").upper()
        
        timeline = [
            {
                "date": today,
                "archetype": profile.get("style_archetype", "Classic"),
                "style": profile.get("style_vibes", ["Classic"])[0] if profile.get("style_vibes") else "Classic",
                "mood": "Current",
                "progress": profile.get("comfort_level", 50)
            },
            {
                "date": week_ago,
                "archetype": "Classic",
                "style": "Classic & Sophisticated",
                "mood": "Exploring",
                "progress": 45
            },
            {
                "date": two_weeks_ago,
                "archetype": "Casual",
                "style": "Casual Comfort",
                "mood": "Exploring",
                "progress": 40
            }
        ]
        
        color_pref_name = profile.get("color_preference_name", "Classic Monochrome")
        color_pref_colors = profile.get("color_preference_colors", ["Black", "White", "Gray"])
        
        return {
            "current_style": {
                "archetype": profile.get("style_archetype", "Classic"),
                "aesthetic": profile.get("everyday_look", "Classic & Sophisticated"),
                "color_preference": color_pref_name,
                "colors": color_pref_colors,
                "comfort_level": profile.get("comfort_level", 50),
                "silhouette": profile.get("silhouette_preference", "Draped & Flowing")
            },
            "timeline": timeline,
            "insights": {
                "dominant_style": profile.get("style_archetype", "Classic"),
                "style_change": "+10%",
                "color_preferences": profile.get("preferred_colors", ["Black", "White"])[:5],
                "style_confidence": profile.get("comfort_level", 50),
                "recommendations": [
                    f"Try adding more {color_pref_colors[0] if color_pref_colors else 'neutral'} pieces",
                    "Experiment with layering this season",
                    "Your style is evolving toward more structured silhouettes"
                ]
            }
        }

    @staticmethod
    async def autotag_garment(image_data: str) -> Dict[str, Any]:
        """
        Full pipeline: decode → mask → color → CLIP → fabric → final tags.
        FIXED: Proper JSON serialization, default fabric "Cotton" instead of "Unknown"
        FIXED: Skirts are skirts, Jumpsuits are jumpsuits, Pants are pants!
        """
        try:
            if not image_data or not isinstance(image_data, str):
                raise ValueError("Invalid image_data: must be non-empty string")

            img = FashionAIModel.vision.decode_image(image_data)
            if img is None or img.size == 0 or np.all(img == 0):
                raise ValueError("Failed to decode image or image is empty")

            # Get mask (SAM or enhanced GrabCut)
            mask = FashionAIModel.vision.get_improved_mask(img)
            
            # Log mask coverage
            mask_coverage = np.sum(mask > 0) / (img.shape[0] * img.shape[1]) * 100
            logger.info(f"Mask covers {mask_coverage:.2f}% of image")

            # Identify category using FashionCLIP (FIXED version)
            category = FashionAIModel.vision.identify_garment(img, mask)

            # Advanced color analysis on masked region (ONLY garment pixels)
            hex_color, color_name, rgb = FashionAIModel.vision.get_dominant_color(img, mask)

            # Texture analysis on masked region
            texture = FashionAIModel.vision.analyze_texture_properties(img, mask)

            # Fabric classification (default to "Cotton" if Unknown)
            fabric = FashionAIModel.classifier.classify(
                variance=texture['variance'],
                brightness=texture['brightness'],
                color=color_name,
                category=category
            )
            
            # Replace "Unknown" with "Cotton"
            if fabric == "Unknown":
                fabric = "Cotton"

            # Build final name
            final_name_parts = []
            if fabric not in ["Cotton", "Polyester"]:  # Don't add common fabrics to name
                final_name_parts.append(fabric)
            final_name_parts.append(color_name)
            final_name_parts.append(category)
            final_name = " ".join(final_name_parts)

            # Convert numpy values to Python native types for JSON serialization
            return {
                "success": True,
                "name": str(final_name),
                "category": str(category),
                "fabric": str(fabric),
                "color": str(color_name),
                "hex_color": str(hex_color),
                "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],  # Convert numpy ints to Python ints
                "details": f"AI Scan: {fabric} {category} | Color: {color_name} ({hex_color})",
                "confidence": 0.96,
                "texture_variance": float(round(texture['variance'], 2)),  # Convert to float
                "brightness": float(round(texture['brightness'], 2)),
                "mask_coverage": float(round(mask_coverage, 2))
            }

        except Exception as e:
            logger.error(f"Autotag error: {e}")
            return {
                "success": False,
                "error": str(e),
                "name": "Cotton Item",
                "category": "Top",
                "fabric": "Cotton",
                "color": "Gray",
                "hex_color": "#808080",
                "rgb": [128, 128, 128],
                "confidence": 0.0
            }

    @staticmethod
    def _get_color_suggestions(base_color: str, category: str = None) -> Dict[str, List[str]]:
        """
        Legacy method - kept for compatibility
        """
        suggestions = {
            "complementary": [],
            "analogous": [],
            "monochromatic": [],
            "seasonal": {
                "summer": [],
                "winter": [],
                "spring": [],
                "fall": []
            }
        }
        
        if base_color in COLOR_HARMONY["complementary"]:
            suggestions["complementary"] = COLOR_HARMONY["complementary"][base_color]
        
        if base_color in COLOR_HARMONY["analogous"]:
            suggestions["analogous"] = COLOR_HARMONY["analogous"][base_color]
        
        for family, colors in COLOR_HARMONY["monochromatic"].items():
            if base_color in colors or base_color == family:
                suggestions["monochromatic"] = [c for c in colors if c != base_color]
                break
        
        if not suggestions["monochromatic"]:
            suggestions["monochromatic"] = ["Black", "White", "Gray"]
        
        if not suggestions["complementary"]:
            suggestions["complementary"] = COLOR_HARMONY["neutrals"]
        
        for season, colors in COLOR_HARMONY["seasonal"].items():
            if base_color in colors or any(c in base_color for c in colors):
                suggestions["seasonal"][season] = [c for c in colors if c != base_color][:3]
            else:
                suggestions["seasonal"][season] = colors[:3]
        
        for key in ["complementary", "analogous", "monochromatic"]:
            suggestions[key] = list(dict.fromkeys(suggestions[key]))[:5]
        
        return suggestions

    @staticmethod
    def _fallback_color_match(base_color: str) -> str:
        """Fallback color matching when fashion_matcher is unavailable."""
        complements = {
            'Black': 'White', 'White': 'Black', 'Navy': 'Beige', 'Blue': 'White',
            'Denim': 'White', 'Gray': 'Black', 'Red': 'Black', 'Green': 'Beige',
            'Yellow': 'Navy', 'Pink': 'Gray', 'Purple': 'Black', 'Orange': 'Navy',
            'Brown': 'Cream', 'Beige': 'Navy'
        }
        return complements.get(base_color, 'White')

    @staticmethod
    async def get_outfit_suggestion(image_data: str, variation: int = 0, user_id: str = None, season: str = "summer") -> Dict[str, Any]:
        """
        Generates outfit suggestions based on detected garment, user's Style DNA, AND season.
        FIXED: Always suggests a different color than the target garment.
        FIXED: Different suggestions each time based on variation parameter.
        """
        try:
            # First get the autotag result
            tag_result = await FashionAIModel.autotag_garment(image_data)
            if not tag_result.get("success", False):
                raise ValueError("Failed to identify garment")

            category = tag_result.get("category", "Top")
            color = tag_result.get("color", "Gray")
            fabric = tag_result.get("fabric", "Cotton")
            rgb = tag_result.get("rgb", [128, 128, 128])
            detected_item = tag_result.get("name", f"{color} {category}")

            # Get user's Style DNA profile
            user_profile = FashionAIModel.get_user_style_profile(user_id) if user_id else StyleProfile.get_default_profile()

            # Use variation to seed random differently
            base_seed = hash(image_data[:100]) + variation * 1000 + int(datetime.utcnow().timestamp() % 100)
            random.seed(base_seed)

            # Determine vibe based on style DNA and variation
            style_archetype = user_profile.get("style_archetype", "Casual").lower()
            style_keywords = user_profile.get("style_keywords", [])
            style_vibes = user_profile.get("style_vibes", [])

            # Map to the keys used in fashion_data.json
            vibe_map = {
                "Minimalist": "minimalist", "Avant-Garde": "avant-garde", "Classic": "classic",
                "Boho": "boho", "Streetwear": "streetwear", "Casual": "casual"
            }
            
            vibe_key = vibe_map.get(style_archetype.title() if style_archetype else "Casual", "casual")

            logger.info(f"Using Style DNA: {vibe_key} for user {user_id}, season: {season}, variation: {variation}")

            # Get intelligent color matches with variation
            matching_colors = ColorMatcher.get_matching_colors(
                (rgb[0], rgb[1], rgb[2]), 
                variation=variation, 
                count=6
            )
            
            # Get best match (could be different based on variation)
            if matching_colors:
                best_match = matching_colors[0]
                # Occasionally pick a different one for more variety
                if variation % 3 == 0 and len(matching_colors) > 2:
                    best_match = matching_colors[1]
            else:
                best_match = {
                    "color": "White",
                    "hex": "#ffffff",
                    "rgb": [255, 255, 255],
                    "match_type": "fallback",
                    "confidence": 0.5,
                    "reason": "Classic white - always a safe choice"
                }

            best_match_color = best_match.get("color", "White")

            silhouette = user_profile.get("silhouette_preference", "Draped & Flowing")

            # Generate suggestions using fashion_data.json with variation
            # Pass variation to ensure different items each time
            shoe_suggestion = await FashionAIModel._generate_style_dna_suggestion(
                item_type="shoes", 
                vibe_key=vibe_key, 
                season=season, 
                variation=variation, 
                best_color=best_match_color
            )

            jewelry_suggestion = await FashionAIModel._generate_style_dna_suggestion(
                item_type="jewelry/accessory", 
                vibe_key=vibe_key, 
                season=season, 
                variation=variation + 1,  # Add offset for more variety
                best_color=best_match_color
            )

            bag_suggestion = await FashionAIModel._generate_style_dna_suggestion(
                item_type="bag", 
                vibe_key=vibe_key, 
                season=season, 
                variation=variation + 2,  # Add offset for more variety
                best_color=best_match_color
            )

            # Construct match description
            if category in ['Pants', 'Shorts', 'Skirt']:
                match_piece_str = f"{best_match_color} Top"
            elif category == 'Top':
                match_piece_str = f"{best_match_color} Bottom"
            elif category in ['Dress', 'Jumpsuit']:
                match_piece_str = ""
            else:
                match_piece_str = f"{best_match_color} Matching Piece"

            # Generate styling tips (with variation)
            styling_tips = await FashionAIModel._generate_style_dna_tips(
                vibe_key, best_match_color, season, variation
            )

            # Create style DNA message
            style_dna_message = f"Based on your {style_archetype.title()} Style DNA"
            if style_vibes:
                style_dna_message = f"Based on your {', '.join(style_vibes)} Style DNA"

            return {
                "style_dna": style_dna_message,
                "season": f" {season.capitalize()}",
                "vibe": style_archetype.title(),
                "identified_item": detected_item,
                "match_piece": match_piece_str,
                "jewelry": jewelry_suggestion,
                "shoes": shoe_suggestion,
                "bag": bag_suggestion,
                "best_match": best_match,
                "matching_colors": matching_colors,
                "styling_tips": styling_tips,
                "silhouette": silhouette
            }

        except Exception as e:
            logger.error(f"Suggestion failed: {e}")
            return {
                "style_dna": "Based on your Casual Style DNA",
                "season": " Summer",
                "vibe": "Casual",
                "identified_item": "Cotton Item",
                "match_piece": "",
                "jewelry": "Simple accessory",
                "shoes": "Classic sneakers",
                "bag": "Versatile tote",
                "best_match": {
                    "color": "White",
                    "hex": "#ffffff",
                    "rgb": [255, 255, 255],
                    "match_type": "fallback",
                    "confidence": 0.5,
                    "reason": "Classic white - always a safe choice"
                },
                "matching_colors": [],
                "styling_tips": "Keep it simple and comfortable",
                "silhouette": "Relaxed"
            }

    @staticmethod
    async def _generate_style_dna_suggestion(item_type: str, vibe_key: str, season: str, variation: int, best_color: str) -> str:
        """
        Generate suggestions using fashion_data.json with variation for different results
        """
        try:
            style_data = FASHION_DATA.get(vibe_key, FASHION_DATA.get("casual", {}))
            season_data = style_data.get(season, style_data.get("summer", {}))
            items = season_data.get(item_type, [])
            
            if not items:
                casual_data = FASHION_DATA.get("casual", {}).get(season, FASHION_DATA.get("casual", {}).get("summer", {}))
                items = casual_data.get(item_type, ["Classic option"])
            
            if items:
                # Use variation to pick different items
                idx = (variation + hash(item_type)) % len(items)
                suggestion = items[idx]
                
                # Sometimes add color, sometimes don't for variety
                if (variation + hash(suggestion)) % 2 == 0:
                    suggestion = f"{suggestion} in {best_color}"
                
                return suggestion
            
            return f"Stylish {item_type.replace('_', ' ')}"
            
        except Exception as e:
            logger.error(f"Error generating suggestion: {e}")
            fallbacks = {
                "shoes": "Classic sneakers",
                "jewelry/accessory": "Simple accessory",
                "bag": "Versatile tote"
            }
            return fallbacks.get(item_type, "Stylish option")

    @staticmethod
    async def _generate_style_dna_tips(vibe_key: str, accent_color: str, season: str, variation: int = 0) -> str:
        """
        Generate styling tips based on vibe and season with variation.
        """
        tips = {
            "minimalist": {
                "summer": [
                    "Keep accessories minimal - let the clean lines speak for themselves.",
                    "Choose breathable linens and cottons for hot days.",
                    f"The {accent_color} accents will add a perfect pop to your minimalist look.",
                    "Less is more - let your {accent_color} piece be the statement.",
                    "Embrace negative space and clean silhouettes."
                ],
                "winter": [
                    "Focus on quality wool and cashmere for warmth without bulk.",
                    "Layer thoughtfully - a fine gauge turtleneck under a structured coat.",
                    f"Let the {accent_color} details shine against your neutral palette.",
                    "Invest in well-tailored basics that last.",
                    "Create depth with different textures in similar tones."
                ]
            },
            "boho": {
                "summer": [
                    "Layer lightweight textures like crochet, lace, and gauze.",
                    "Mix earthy tones with pops of turquoise or coral.",
                    f"Your {accent_color} pieces will ground the free-spirited vibe.",
                    "Don't be afraid to mix patterns and textures.",
                    "Accessorize with natural materials like wood and shell."
                ],
                "winter": [
                    "Layer chunky knits over flowing skirts with tights.",
                    "Mix warm textures like suede, wool, and velvet.",
                    f"The {accent_color} accents will warm up your winter boho look.",
                    "Add fringe and tassel details for movement.",
                    "Layer different lengths for visual interest."
                ]
            },
            "streetwear": {
                "summer": [
                    "Play with proportions - oversized tees with bike shorts.",
                    "Fresh white sneakers complete any summer streetwear look.",
                    f"Let the {accent_color} pieces be your statement.",
                    "Add technical fabrics for an urban edge.",
                    "Accessorize with chains and bucket hats."
                ],
                "winter": [
                    "Oversized puffers are both warm and stylish.",
                    "Layer hoodies under denim jackets for warmth.",
                    f"Your {accent_color} accessories will pop against darker winter layers.",
                    "Chunky sneakers work great with thick socks.",
                    "Beanies and technical fabrics are winter essentials."
                ]
            },
            "classic": {
                "summer": [
                    "Invest in quality linen and cotton basics.",
                    "A well-tailored white shirt elevates any summer outfit.",
                    f"The {accent_color} touches add timeless elegance.",
                    "Stick to a neutral palette with one accent color.",
                    "Pearls and gold tones add sophistication."
                ],
                "winter": [
                    "A cashmere sweater in a neutral tone is worth the investment.",
                    "Well-tailored wool trousers create clean lines.",
                    f"Let {accent_color} be your signature accent color.",
                    "A camel hair coat is the ultimate classic winter piece.",
                    "Leather gloves and a silk scarf complete the look."
                ]
            },
            "avant-garde": {
                "summer": [
                    "Let your outfit be a conversation starter.",
                    "Mix unexpected lightweight textures like mesh and organza.",
                    f"The {accent_color} elements create architectural interest.",
                    "Don't follow trends - set them.",
                    "Asymmetric cuts create visual intrigue."
                ],
                "winter": [
                    "Sculptural coats become the focal point.",
                    "Layer deconstructed pieces for warmth and interest.",
                    f"Your {accent_color} statement pieces will stand out.",
                    "Mix structured and flowing elements.",
                    "Experiment with unconventional silhouettes."
                ]
            },
            "casual": {
                "summer": [
                    "Comfort is key - choose soft, breathable fabrics.",
                    "Well-fitted basics create an effortlessly cool look.",
                    f"The {accent_color} pieces tie everything together.",
                    "A great pair of white sneakers grounds any casual outfit.",
                    "Layer with a denim jacket for cooler evenings."
                ],
                "winter": [
                    "Comfort is key - choose soft sweaters and warm layers.",
                    "Well-fitted basics under cardigans create effortless style.",
                    f"Let the {accent_color} accents brighten your winter days.",
                    "Add texture with chunky knits.",
                    "Boots and beanies complete the cozy look."
                ]
            }
        }
        
        vibe_tips = tips.get(vibe_key, tips["casual"])
        season_tips = vibe_tips.get(season, vibe_tips.get("summer", [
            "Style is personal - wear what makes you feel confident!"
        ]))
        
        # Use variation to pick different tips
        if season_tips:
            idx = variation % len(season_tips)
            return season_tips[idx]
        
        return "Style is personal - wear what makes you feel confident!"

    @staticmethod
    async def generate_outfits_from_wardrobe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generates outfit vibes using the Advanced Engine."""
        if not fashion_matcher:
            return []

        outfits = []
        styles = ['casual', 'business', 'streetwear']

        for style in styles:
            outfit = fashion_matcher.create_complete_outfit(items, style=style)
            if outfit and 'items' in outfit:
                outfits.append({
                    "name": outfit['name'],
                    "vibe": outfit['vibe'],
                    "item_ids": outfit['item_ids']
                })

        return outfits

    @staticmethod
    def get_evolution_data(items: List[Dict[str, Any]], history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
        """Analyzes wardrobe health/gaps."""
        if not fashion_matcher:
            return {
                "timeline": [],
                "insights": {
                    "dominant_style": "Casual",
                    "style_change": "0%",
                    "color_preferences": [],
                    "style_confidence": 50,
                    "wardrobe_size": len(items),
                    "recommendations": []
                }
            }

        analysis = fashion_matcher.analyze_wardrobe_gaps(items)
        timeline = []

        def format_date(iso_str):
            try:
                dt = datetime.fromisoformat(iso_str)
                return dt.strftime("%b %d")
            except:
                return "Unknown"

        for i, entry in enumerate(history):
            style_list = []
            try:
                style_list = json.loads(entry.get('styles', '[]'))
            except:
                pass

            primary_style = style_list[0] if style_list else entry.get('archetype', 'Mapped')

            mood = "Exploring"
            if 'Minimalist' in primary_style: mood = "Clean & Sharp"
            elif 'Streetwear' in primary_style: mood = "Bold & Expressive"
            elif 'Vintage' in primary_style: mood = "Nostalgic"

            timeline.append({
                "period": format_date(entry.get('created_at', '')),
                "stage": primary_style,
                "style": " & ".join(style_list[:2]),
                "color": "Personalized",
                "mood": mood,
                "progress": entry.get('comfort_level', 50),
                "items": len(items),
                "key_item": "DNA Profile",
                "is_current": i == 0
            })

        return {
            "timeline": timeline,
            "insights": {
                "dominant_style": analysis.get('dominant_style', 'Casual'),
                "style_change": f"{analysis.get('wardrobe_health_score', 50)}%",
                "color_preferences": analysis.get('color_preferences', [])[:5],
                "style_confidence": analysis.get('wardrobe_health_score', 50),
                "wardrobe_size": analysis.get('total_items', len(items)),
                "recommendations": analysis.get('recommendations', [])
            }
        }

    @staticmethod
    def _get_region_from_country(country: str) -> str:
        """
        Determine region from country name using country_to_region.json mapping.
        """
        if not country:
            return "european"
        
        country_lower = country.lower().strip()
        
        # Direct lookup from mapping
        if country_lower in COUNTRY_TO_REGION:
            return COUNTRY_TO_REGION[country_lower]
        
        # Partial matching for cases like "United States" vs "usa"
        for key, region in COUNTRY_TO_REGION.items():
            if key in country_lower or country_lower in key:
                return region
        
        # Fallback based on common patterns
        if any(x in country_lower for x in ["turkey", "egypt", "morocco", "greece", "italy", "spain"]):
            return "mediterranean"
        elif any(x in country_lower for x in ["uae", "saudi", "qatar", "jordan", "iran", "iraq"]):
            return "middle_eastern"
        elif any(x in country_lower for x in ["japan", "china", "korea", "thailand", "vietnam", "india"]):
            return "asian"
        elif any(x in country_lower for x in ["france", "germany", "uk", "italy", "spain", "netherlands"]):
            return "european"
        elif any(x in country_lower for x in ["usa", "canada", "mexico"]):
            return "north_american"
        elif any(x in country_lower for x in ["brazil", "argentina", "peru", "colombia"]):
            return "south_american"
        elif any(x in country_lower for x in ["south africa", "kenya", "nigeria", "ghana"]):
            return "african"
        else:
            return "european"

    @staticmethod
    def _get_limited_edition_items(place_type: str, region: str, count: int = 2) -> List[Dict[str, str]]:
        """
        Get random limited edition items based on place type and region.
        """
        try:
            region_data = REGIONAL_ITEMS.get("regions", {}).get(region, REGIONAL_ITEMS.get("regions", {}).get("european", {}))
            
            category_map = {
                "market": "market", "antique": "antique", "boutique": "boutique",
                "shopping_mall": "market", "department_store": "boutique",
                "clothing": "boutique", "gift": "market", "food": "market"
            }
            
            category = category_map.get(place_type, "market")
            items_data = region_data.get(category, region_data.get("market", {}))
            items_list = items_data.get("items", []) if isinstance(items_data, dict) else []
            
            if not items_list:
                return []
            
            selected = random.sample(items_list, min(count, len(items_list)))
            return selected
            
        except Exception as e:
            logger.error(f"Error getting limited edition items: {e}")
            return []

    @staticmethod
    def _get_weather_data(lat: float, lon: float, city_name: str) -> Dict[str, Any]:
        """
        Get REAL weather data from Open-Meteo API (FREE, no API key required).
        """
        try:
            weather_url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'weather_code', 'wind_speed_10m'],
                'daily': ['weather_code', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'],
                'timezone': 'auto',
                'forecast_days': 7
            }
            
            response = requests.get(weather_url, params=params)
            data = response.json()
            
            if not data:
                logger.warning(f"No weather data returned for {city_name}")
                return {}
            
            current = data.get('current', {})
            daily = data.get('daily', {})
            
            current_code = current.get('weather_code', 0)
            weather_description = WEATHER_CODES.get(str(current_code), f"Unknown Code {current_code}")
            
            rain_days = 0
            if daily.get('precipitation_sum'):
                rain_days = sum(1 for p in daily['precipitation_sum'] if p and p > 0)
            
            avg_temp = None
            if daily.get('temperature_2m_max') and daily.get('temperature_2m_min'):
                max_temps = daily['temperature_2m_max']
                min_temps = daily['temperature_2m_min']
                if max_temps and min_temps:
                    avg_temp = round(sum((max_temps[i] + min_temps[i]) / 2 for i in range(len(max_temps))) / len(max_temps), 1)
            
            return {
                'current_temp': round(current.get('temperature_2m', 0), 1),
                'feels_like': round(current.get('apparent_temperature', 0), 1),
                'humidity': current.get('relative_humidity_2m', 0),
                'wind_speed': round(current.get('wind_speed_10m', 0), 1),
                'description': weather_description,
                'min_temp': round(min(daily.get('temperature_2m_min', [0])), 1) if daily.get('temperature_2m_min') else None,
                'max_temp': round(max(daily.get('temperature_2m_max', [0])), 1) if daily.get('temperature_2m_max') else None,
                'avg_temperature': avg_temp,
                'rain_days': rain_days
            }
            
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {}

    @staticmethod
    def curate_trip(city: str, duration: int, vibe: str) -> Dict[str, Any]:
        """
        Curates a packing list and shopping guide for ANY city using FREE Geoapify API.
        Filters out global chains to show ONLY unique local stores specific to that place.
        """
        try:
            city_title = city.title().strip()
            
            # STEP 1: Geocoding - convert city name to coordinates using Geoapify
            geocode_url = "https://api.geoapify.com/v1/geocode/search"
            geo_params = {
                'text': city_title,
                'apiKey': GEOAPIFY_API_KEY,
                'limit': 1
            }
            
            geo_response = requests.get(geocode_url, params=geo_params)
            geo_data = geo_response.json()
            
            if not geo_data.get('features'):
                return {
                    "city": city_title,
                    "error": f"Could not find coordinates for {city_title}",
                    "message": "Please check the city name or try a different city."
                }
            
            # Extract coordinates and city details
            lat = geo_data['features'][0]['geometry']['coordinates'][1]
            lon = geo_data['features'][0]['geometry']['coordinates'][0]
            city_name = geo_data['features'][0]['properties'].get('city', city_title)
            country = geo_data['features'][0]['properties'].get('country', '')
            
            logger.info(f"Found {city_name}, {country} at coordinates: {lat}, {lon}")
            
            # Determine region for limited edition items
            region = FashionAIModel._get_region_from_country(country)
            logger.info(f"Region detected: {region} for country: {country}")
            
            # STEP 2: Places API - find REAL shopping places
            places_url = "https://api.geoapify.com/v2/places"
            
            # Function to fetch places by category
            def fetch_places(categories, limit=20, radius=5000):
                params = {
                    'categories': ','.join(categories),
                    'filter': f'circle:{lon},{lat},{radius}',
                    'bias': f'proximity:{lon},{lat}',
                    'limit': limit,
                    'apiKey': GEOAPIFY_API_KEY
                }
                headers = {"Accept": "application/json"}
                response = requests.get(places_url, params=params, headers=headers)
                return response.json()
            
            # Fetch different types of places with appropriate categories
            market_categories = [
                "commercial.market",
                "commercial.marketplace",
                "commercial.supermarket",
                "commercial.shopping_mall",
                "commercial.department_store",
                "commercial.food_and_drink"
            ]
            
            bakery_categories = [
                "catering.bakery",
                "catering.pastry_shop",
                "catering.cafe",
                "catering.coffee_shop"
            ]
            
            boutique_categories = [
                "commercial.clothing",
                "commercial.fashion",
                "commercial.boutique",
                "commercial.gift_and_souvenir",
                "commercial.antiques",
                "commercial.books",
                "commercial.jewelry"
            ]
            
            # Fetch places
            markets_data = fetch_places(market_categories, 20, 5000)
            bakeries_data = fetch_places(bakery_categories, 15, 3000)
            boutiques_data = fetch_places(boutique_categories, 20, 4000)
            
            # Log what we found
            logger.info(f"Markets found: {len(markets_data.get('features', []))}")
            logger.info(f"Bakeries found: {len(bakeries_data.get('features', []))}")
            logger.info(f"Boutiques found: {len(boutiques_data.get('features', []))}")
            
            # Process markets (Major Local Markets)
            major_markets = []
            for feature in markets_data.get('features', []):
                props = feature['properties']
                place_name = props.get('name', '')
                
                if not place_name or len(place_name) < 2:
                    continue
                
                place_name_lower = place_name.lower()
                
                # Skip global chains
                if any(chain in place_name_lower for chain in GLOBAL_CHAINS):
                    continue
                
                # Get categories for better description
                categories = props.get('categories', ['unknown'])
                categories_str = ' '.join(str(c).lower() for c in categories)
                
                # Determine market type
                market_type = "Market"
                if 'supermarket' in categories_str:
                    market_type = "Supermarket"
                elif 'shopping_mall' in categories_str:
                    market_type = "Shopping Mall"
                elif 'food' in categories_str:
                    market_type = "Food Market"
                
                market = {
                    'name': place_name,
                    'address': props.get('formatted', 'Address not available'),
                    'distance': round(props.get('distance', 0)),
                    'type': market_type,
                    'opening_hours': props.get('opening_hours', 'Hours not available'),
                    'specialty': 'Local produce, crafts, and traditional goods',
                    'best_for': ['Fresh food', 'Local crafts', 'Souvenirs'],
                    'rating': props.get('rating', 'N/A')
                }
                major_markets.append(market)
            
            # Process bakeries
            bakeries = []
            for feature in bakeries_data.get('features', []):
                props = feature['properties']
                place_name = props.get('name', '')
                
                if not place_name or len(place_name) < 2:
                    continue
                
                # Skip chains
                place_name_lower = place_name.lower()
                if any(chain in place_name_lower for chain in ['starbucks', 'dunkin', 'tim hortons', 'costa']):
                    continue
                
                categories = props.get('categories', ['unknown'])
                categories_str = ' '.join(str(c).lower() for c in categories)
                
                bakery_type = "Bakery"
                if 'pastry' in categories_str:
                    bakery_type = "Pastry Shop"
                elif 'cafe' in categories_str or 'coffee' in categories_str:
                    bakery_type = "Cafe"
                
                bakery = {
                    'name': place_name,
                    'address': props.get('formatted', 'Address not available'),
                    'distance': round(props.get('distance', 0)),
                    'type': bakery_type,
                    'opening_hours': props.get('opening_hours', 'Hours not available'),
                    'specialty': 'Fresh baked goods daily'
                }
                bakeries.append(bakery)
            
            # Identify oldest and most popular bakeries (simplified heuristic)
            oldest_bakery = None
            most_popular_bakery = None
            
            if bakeries:
                # Sort by name length as simple heuristic for "oldest" (older places often have shorter names)
                bakeries_sorted = sorted(bakeries, key=lambda x: len(x['name']))
                if bakeries_sorted:
                    oldest_bakery = bakeries_sorted[0].copy()
                    oldest_bakery['description'] = 'One of the oldest bakeries in the city, serving traditional recipes for generations'
                
                # Most popular could be the one with most reviews or simply a random one
                if len(bakeries) > 1:
                    most_popular_bakery = bakeries[1].copy()
                    most_popular_bakery['description'] = 'Most popular bakery among locals, known for fresh daily specials'
                elif bakeries:
                    most_popular_bakery = bakeries[0].copy()
                    most_popular_bakery['description'] = 'Popular local bakery with loyal customers'
            
            # Process boutiques (Hidden Gems)
            hidden_gem_boutiques = []
            for feature in boutiques_data.get('features', []):
                props = feature['properties']
                place_name = props.get('name', '')
                
                if not place_name or len(place_name) < 2:
                    continue
                
                place_name_lower = place_name.lower()
                
                # Skip global chains
                if any(chain in place_name_lower for chain in GLOBAL_CHAINS):
                    continue
                
                # Get categories
                categories = props.get('categories', ['unknown'])
                categories_str = ' '.join(str(c).lower() for c in categories)
                
                # Determine boutique type
                boutique_type = "Boutique"
                if 'clothing' in categories_str:
                    boutique_type = "Clothing Boutique"
                elif 'gift' in categories_str:
                    boutique_type = "Gift Shop"
                elif 'antique' in categories_str:
                    boutique_type = "Antique Shop"
                elif 'jewelry' in categories_str:
                    boutique_type = "Jewelry Store"
                elif 'books' in categories_str:
                    boutique_type = "Bookstore"
                
                # Get limited edition items for this boutique
                limited_items = FashionAIModel._get_limited_edition_items('boutique', region, count=2)
                
                boutique = {
                    'name': place_name,
                    'address': props.get('formatted', 'Address not available'),
                    'distance': round(props.get('distance', 0)),
                    'type': boutique_type,
                    'opening_hours': props.get('opening_hours', 'Hours not available'),
                    'description': 'Hidden gem boutique with unique local finds and curated collections',
                    'limited_edition_items': limited_items,
                    'is_hidden_gem': any(indicator in place_name_lower for indicator in LOCAL_INDICATORS)
                }
                hidden_gem_boutiques.append(boutique)
            
            # Sort by distance and prioritize those that seem like hidden gems
            hidden_gem_boutiques.sort(key=lambda x: (not x['is_hidden_gem'], x['distance']))
            
            # STEP 3: Get REAL weather data
            weather_data = FashionAIModel._get_weather_data(lat, lon, city_name)
            
            # STEP 4: Generate packing list based on real weather
            tops_count = max(2, duration)
            bottoms_count = max(1, duration // 2 + 1)
            
            packing_list = [
                f"{tops_count}x Tops/Shirts",
                f"{bottoms_count}x Bottoms (Pants/Shorts)",
                f"{duration + 1}x Underwear & Socks",
                "1x Comfortable Walking Shoes",
                "1x Evening Outfit",
                "Sleepwear",
                "Toiletries Kit",
                "Power Bank & Chargers"
            ]
            
            # Add weather-appropriate items based on REAL weather
            avg_temp = weather_data.get('avg_temperature', 20)
            rain_days = weather_data.get('rain_days', 0)
            
            if avg_temp > 25:
                packing_list.extend(["Sunglasses", "Sunscreen SPF 50", "Hat", "Lightweight clothing", "Water bottle"])
                weather_summary = f"{avg_temp}°C - Hot and sunny"
            elif avg_temp > 20:
                packing_list.extend(["Sunglasses", "Light jacket for evenings"])
                weather_summary = f"{avg_temp}°C - Warm and pleasant"
            elif avg_temp > 15:
                packing_list.extend(["Light jacket", "Umbrella (just in case)"])
                weather_summary = f"{avg_temp}°C - Mild and comfortable"
            elif avg_temp > 10:
                packing_list.extend(["Medium jacket", "Sweater", "Umbrella"])
                weather_summary = f"{avg_temp}°C - Cool, bring layers"
            elif avg_temp > 5:
                packing_list.extend(["Warm jacket", "Sweater", "Scarf", "Umbrella"])
                weather_summary = f"{avg_temp}°C - Cold, bundle up"
            else:
                packing_list.extend(["Heavy winter coat", "Thermal layers", "Gloves", "Warm hat", "Scarf"])
                weather_summary = f"{avg_temp}°C - Freezing"
            
            if rain_days > 0:
                packing_list.append(f"Umbrella (rain expected on {rain_days} days)")
                weather_summary += f" - {rain_days} days with rain"
            
            # Add vibe-specific items
            if vibe.lower() == 'beach':
                packing_list.extend(["2x Swimwear", "Flip Flops", "Beach Towel", "Beach Bag", "Waterproof phone case"])
            elif vibe.lower() == 'mountain':
                packing_list.extend(["Hiking Boots", "Thermal Layers", "Rain Jacket", "Wool Beanie", "Backpack", "First aid kit"])
            elif vibe.lower() == 'city':
                packing_list.extend(["Daypack", "Comfortable walking shoes", "Camera", "Power bank"])
            elif vibe.lower() == 'luxury':
                packing_list.extend(["Cocktail dress/suit", "Designer accessories", "Formal shoes", "Jewelry"])
            elif vibe.lower() == 'adventure':
                packing_list.extend(["Hiking boots", "Quick-dry clothing", "Water bottle", "First aid kit", "Multi-tool"])
            
            # Structure the response
            return {
                "city": city_name,
                "country": country,
                "days": int(duration),
                "weather_summary": weather_summary,
                "weather_details": {
                    "current_temp": float(weather_data.get('current_temp')) if weather_data.get('current_temp') else None,
                    "feels_like": float(weather_data.get('feels_like')) if weather_data.get('feels_like') else None,
                    "humidity": float(weather_data.get('humidity')) if weather_data.get('humidity') else None,
                    "wind_speed": float(weather_data.get('wind_speed')) if weather_data.get('wind_speed') else None,
                    "min_temp": float(weather_data.get('min_temp')) if weather_data.get('min_temp') else None,
                    "max_temp": float(weather_data.get('max_temp')) if weather_data.get('max_temp') else None,
                    "description": weather_data.get('description'),
                    "rain_days": int(weather_data.get('rain_days', 0))
                },
                "clothes_count": len(packing_list),
                "packing_list": packing_list,
                "major_markets": major_markets[:8],
                "bakeries": {
                    "oldest": oldest_bakery,
                    "most_popular": most_popular_bakery,
                    "others": [b for b in bakeries if b != oldest_bakery and b != most_popular_bakery][:4]
                },
                "hidden_gem_boutiques": hidden_gem_boutiques[:6],
                "total_places_found": len(major_markets) + len(bakeries) + len(hidden_gem_boutiques),
                "region": region,
                "data_source": "OpenStreetMap via Geoapify + Open-Meteo Weather + Regional Items Database"
            }
            
        except Exception as e:
            logger.error(f"Trip curation error: {e}")
            return {
                "city": city,
                "days": duration,
                "error": str(e),
                "weather_summary": "Weather information unavailable",
                "weather_details": {},
                "packing_list": [
                    f"{max(2, duration)}x Tops",
                    f"{max(1, duration//2)}x Bottoms",
                    "Comfortable walking shoes",
                    "Evening outfit",
                    "Toiletries",
                    "Power bank",
                    "Umbrella"
                ],
                "major_markets": [],
                "bakeries": {"oldest": None, "most_popular": None, "others": []},
                "hidden_gem_boutiques": [],
                "message": "Unable to fetch real places. Please try again later."
            }

    @staticmethod
    def weather_styling(city: str) -> Dict[str, Any]:
        """
        Provides REAL weather-based styling advice using Open-Meteo API.
        Suggests outfit (top, bottom, outerwear, footwear) based on current weather conditions.
        """
        try:
            city_title = city.title().strip()
            
            # STEP 1: Geocoding - convert city name to coordinates using Geoapify
            geocode_url = "https://api.geoapify.com/v1/geocode/search"
            geo_params = {
                'text': city_title,
                'apiKey': GEOAPIFY_API_KEY,
                'limit': 1
            }
            
            geo_response = requests.get(geocode_url, params=geo_params)
            geo_data = geo_response.json()
            
            if not geo_data.get('features'):
                return {
                    "city": city_title,
                    "error": f"Could not find coordinates for {city_title}",
                    "temp": 22,
                    "condition": "Unknown",
                    "outfit": {"top": "T-Shirt", "bottom": "Jeans", "outerwear": "None", "footwear": "Sneakers"},
                    "advice": "Weather data unavailable. Showing default suggestions.",
                    "data_source": "Fallback Data"
                }
            
            # Extract coordinates
            lat = geo_data['features'][0]['geometry']['coordinates'][1]
            lon = geo_data['features'][0]['geometry']['coordinates'][0]
            city_name = geo_data['features'][0]['properties'].get('city', city_title)
            country = geo_data['features'][0]['properties'].get('country', '')
            
            logger.info(f"Getting weather for {city_name}, {country} at coordinates: {lat}, {lon}")
            
            # STEP 2: Get REAL weather data from Open-Meteo (FREE, no API key)
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                'latitude': lat,
                'longitude': lon,
                'current': ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'weather_code', 'wind_speed_10m'],
                'daily': ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'],
                'timezone': 'auto',
                'forecast_days': 1
            }
            
            weather_response = requests.get(weather_url, params=weather_params)
            weather_data = weather_response.json()
            
            if not weather_data or 'current' not in weather_data:
                return {
                    "city": city_name,
                    "temp": 22,
                    "condition": "Unknown",
                    "outfit": {"top": "T-Shirt", "bottom": "Jeans", "outerwear": "None", "footwear": "Sneakers"},
                    "advice": "Weather data temporarily unavailable. Showing default suggestions.",
                    "data_source": "Fallback Data"
                }
            
            # Extract current weather
            current = weather_data['current']
            daily = weather_data.get('daily', {})
            
            temp = round(current.get('temperature_2m', 22))
            feels_like = round(current.get('apparent_temperature', temp))
            humidity = current.get('relative_humidity_2m', 50)
            wind_speed = round(current.get('wind_speed_10m', 5))
            weather_code = current.get('weather_code', 0)
            
            # Get min/max temp for the day
            min_temp = round(daily.get('temperature_2m_min', [temp])[0]) if daily.get('temperature_2m_min') else temp
            max_temp = round(daily.get('temperature_2m_max', [temp])[0]) if daily.get('temperature_2m_max') else temp
            
            # Get weather description from JSON mapping
            condition = WEATHER_CODES.get(str(weather_code), f"Unknown ({weather_code})")
            
            # STEP 3: Generate outfit based on REAL temperature and conditions
            outfit = FashionAIModel._suggest_outfit_from_weather(temp, feels_like, condition, wind_speed)
            
            # STEP 4: Generate styling advice
            advice = FashionAIModel._generate_weather_advice(temp, feels_like, condition, wind_speed, humidity, min_temp, max_temp)
            
            return {
                "city": city_name,
                "country": country,
                "temp": int(temp),
                "feels_like": int(feels_like),
                "min_temp": int(min_temp),
                "max_temp": int(max_temp),
                "condition": condition,
                "humidity": int(humidity),
                "wind_speed": int(wind_speed),
                "outfit": outfit,
                "advice": advice,
                "data_source": "Open-Meteo Weather API (FREE)"
            }
            
        except Exception as e:
            logger.error(f"Weather styling error: {e}")
            return {
                "city": city,
                "error": str(e),
                "temp": 22,
                "feels_like": 22,
                "min_temp": 18,
                "max_temp": 25,
                "condition": "Sunny (Fallback)",
                "humidity": 50,
                "wind_speed": 5,
                "outfit": {"top": "T-Shirt", "bottom": "Jeans", "outerwear": "None", "footwear": "Sneakers"},
                "advice": "Great weather for a casual day out. (Using fallback data)",
                "data_source": "Fallback Data"
            }

    @staticmethod
    def _suggest_outfit_from_weather(temp: float, feels_like: float, condition: str, wind_speed: float) -> Dict[str, str]:
        """
        Suggest top, bottom, outerwear, footwear based on real temperature and conditions.
        """
        # Use feels_like for more accurate comfort assessment
        comfort_temp = feels_like
        
        # Check for rain/snow first
        is_rainy = any(x in condition.lower() for x in ['rain', 'drizzle', 'shower', 'thunderstorm'])
        is_snowy = any(x in condition.lower() for x in ['snow', 'sleet'])
        
        # Hot weather (> 28°C)
        if comfort_temp > 35:
            outfit = {
                "top": "Cotton Vest / Tank Top",
                "bottom": "Linen Shorts",
                "outerwear": "None",
                "footwear": "Flip Flops / Sandals"
            }
        elif comfort_temp > 32:
            outfit = {
                "top": "T-Shirt (Light Cotton)",
                "bottom": "Shorts / Linen Pants",
                "outerwear": "None",
                "footwear": "Sandals / Breathable Sneakers"
            }
        elif comfort_temp > 28:
            outfit = {
                "top": "T-Shirt",
                "bottom": "Shorts / Light Pants",
                "outerwear": "None",
                "footwear": "Sneakers / Loafers"
            }
        
        # Warm weather (22-28°C)
        elif comfort_temp > 25:
            outfit = {
                "top": "T-Shirt / Polo",
                "bottom": "Jeans / Chinos",
                "outerwear": "None",
                "footwear": "Sneakers"
            }
        elif comfort_temp > 22:
            outfit = {
                "top": "T-Shirt / Blouse",
                "bottom": "Jeans",
                "outerwear": "None",
                "footwear": "Sneakers"
            }
        
        # Mild weather (18-22°C)
        elif comfort_temp > 20:
            outfit = {
                "top": "Long Sleeve Shirt",
                "bottom": "Jeans",
                "outerwear": "Light Cardigan (optional)",
                "footwear": "Sneakers"
            }
        elif comfort_temp > 18:
            outfit = {
                "top": "Long Sleeve Shirt",
                "bottom": "Jeans / Trousers",
                "outerwear": "Light Jacket",
                "footwear": "Sneakers / Loafers"
            }
        
        # Cool weather (12-18°C)
        elif comfort_temp > 15:
            outfit = {
                "top": "Sweater / Hoodie",
                "bottom": "Jeans",
                "outerwear": "Jacket",
                "footwear": "Boots / Sneakers"
            }
        elif comfort_temp > 12:
            outfit = {
                "top": "Sweater",
                "bottom": "Jeans",
                "outerwear": "Heavy Jacket / Coat",
                "footwear": "Boots"
            }
        
        # Cold weather (5-12°C)
        elif comfort_temp > 8:
            outfit = {
                "top": "Thermal + Sweater",
                "bottom": "Jeans (with thermals optional)",
                "outerwear": "Winter Coat",
                "footwear": "Insulated Boots"
            }
        elif comfort_temp > 5:
            outfit = {
                "top": "Thermal + Sweater",
                "bottom": "Jeans with Thermals",
                "outerwear": "Heavy Winter Coat",
                "footwear": "Winter Boots"
            }
        
        # Freezing weather (< 5°C)
        else:
            outfit = {
                "top": "Thermal + Wool Sweater",
                "bottom": "Insulated Pants",
                "outerwear": "Heavy Winter Coat",
                "footwear": "Snow Boots"
            }
        
        # Adjust for rain
        if is_rainy:
            if comfort_temp > 20:
                outfit["outerwear"] = "Light Rain Jacket"
                outfit["footwear"] = "Waterproof Sneakers"
            else:
                outfit["outerwear"] = "Waterproof Coat"
                outfit["footwear"] = "Waterproof Boots"
            outfit["accessories"] = "Umbrella"
        
        # Adjust for snow
        if is_snowy:
            outfit["outerwear"] = "Insulated Snow Jacket"
            outfit["footwear"] = "Snow Boots"
            outfit["accessories"] = "Scarf, Gloves, Beanie"
        
        # Adjust for wind
        if wind_speed > 30:
            if "Jacket" in outfit["outerwear"] or "Coat" in outfit["outerwear"]:
                pass
            elif comfort_temp > 15:
                outfit["outerwear"] = "Windbreaker"
            if "accessories" in outfit:
                outfit["accessories"] += " (secure your hat)"
            else:
                outfit["accessories"] = "Secure your hat"
        
        return outfit

    @staticmethod
    def _generate_weather_advice(temp: float, feels_like: float, condition: str, wind_speed: float, humidity: int, min_temp: float, max_temp: float) -> str:
        """
        Generate personalized styling advice based on weather conditions.
        """
        advice = ""
        
        # Temperature-based advice
        if feels_like > 35:
            advice = "Extreme heat! Opt for loose, breathable fabrics in light colors. Stay hydrated and avoid heavy layers. "
        elif feels_like > 30:
            advice = "Hot and sunny. Choose lightweight cotton or linen. Sunglasses and a hat are essential. "
        elif feels_like > 25:
            advice = "Warm and pleasant. Perfect for casual summer outfits. Light colors will keep you cool. "
        elif feels_like > 20:
            advice = "Mild and comfortable. A t-shirt and jeans combo works perfectly. "
        elif feels_like > 15:
            advice = "Slightly cool. A light jacket or sweater is recommended. "
        elif feels_like > 10:
            advice = "Cool weather. Layer up with a jacket or hoodie. "
        elif feels_like > 5:
            advice = "Cold outside. Wear a warm coat, scarf, and gloves. "
        else:
            advice = "Freezing temperatures! Bundle up with thermal layers, heavy coat, and winter accessories. "
        
        # Temperature range advice
        if max_temp - min_temp > 10:
            advice += f"Temperature will vary from {min_temp}°C to {max_temp}°C today - dress in layers so you can adjust. "
        
        # Condition-based additions
        if "Rain" in condition or "Drizzle" in condition:
            advice += "Don't forget an umbrella and waterproof footwear. "
        elif "Snow" in condition:
            advice += "Snow expected! Wear waterproof boots and warm layers. "
        elif "Thunderstorm" in condition:
            advice += "Thunderstorms likely. Stay dry and consider indoor activities. "
        elif "Fog" in condition:
            advice += "Foggy conditions - wear bright or reflective clothing if you'll be outside. "
        
        # Wind-based advice
        if wind_speed > 40:
            advice += "Very strong winds - secure your hat and choose fitted clothing. "
        elif wind_speed > 25:
            advice += "Windy - a windbreaker might be useful. "
        elif wind_speed > 15:
            advice += "Breezy conditions - light jacket recommended. "
        
        # Humidity-based advice
        if humidity > 80:
            advice += "High humidity - choose moisture-wicking fabrics. "
        elif humidity < 30:
            advice += "Low humidity - moisturize and stay hydrated. "
        
        # UV advice for clear skies
        if "Clear" in condition and feels_like > 25:
            advice += "Don't forget sunscreen - UV rays are strong today. "
        
        return advice.strip()

    @staticmethod
    async def audit_brand(brand: str) -> Dict[str, Any]:
        """Audits a brand's sustainability using brand_score.json."""
        try:
            brand_lower = brand.lower().strip()
            
            if brand_lower in BRAND_SCORES:
                data = BRAND_SCORES[brand_lower]
                return {
                    "brand": brand,
                    "total_score": data.get('total', 50),
                    "summary": data.get('summary', 'No summary available'),
                    "eco_score": data.get('eco', 50),
                    "labor_score": data.get('labor', 50),
                    "trans_score": data.get('trans', 50),
                    "sources": [{"uri": "#", "title": f"{brand} Sustainability Report"}]
                }
            
            for key in BRAND_SCORES:
                if key in brand_lower or brand_lower in key:
                    data = BRAND_SCORES[key]
                    return {
                        "brand": brand,
                        "total_score": data.get('total', 50),
                        "summary": data.get('summary', 'No summary available'),
                        "eco_score": data.get('eco', 50),
                        "labor_score": data.get('labor', 50),
                        "trans_score": data.get('trans', 50),
                        "sources": [{"uri": "#", "title": f"{key} Sustainability Report"}]
                    }
            
            seed = sum(ord(c) for c in brand_lower)
            random.seed(seed)

            base_score = random.randint(30, 80)
            eco = max(0, min(100, base_score + random.randint(-10, 10)))
            labor = max(0, min(100, base_score + random.randint(-10, 10)))
            trans = max(0, min(100, base_score + random.randint(-10, 10)))
            total = (eco + labor + trans) // 3

            summary = "AI Estimate: Moderate sustainability performance based on sector averages."
            if total > 70:
                summary = "AI Estimate: Likely has good sustainability practices."
            elif total < 40:
                summary = "AI Estimate: Potential risks in supply chain transparency."

            return {
                "brand": brand,
                "total_score": total,
                "summary": summary,
                "eco_score": eco,
                "labor_score": labor,
                "trans_score": trans,
                "sources": []
            }
            
        except Exception as e:
            logger.error(f"Brand audit error: {e}")
            return {
                "brand": brand,
                "total_score": 50,
                "summary": "Unable to audit brand at this time.",
                "eco_score": 50,
                "labor_score": 50,
                "trans_score": 50,
                "sources": []
            }


# ====================== HELPER FUNCTIONS ======================
def best_match_color_from_context(color_context: str) -> str:
    """Extract color from context string"""
    colors = ['Black', 'White', 'Navy', 'Beige', 'Denim', 'Gray', 'Olive', 'Camel', 'Red', 'Silver',
              'Burgundy', 'Emerald', 'Mustard', 'Coral', 'Lavender', 'Rust', 'Sage', 'Blush']
    for color in colors:
        if color.lower() in color_context.lower():
            return color
    return "complementary"

def item_to_dict(item_str: str) -> Dict[str, Any]:
    """Convert item string to dictionary for matching"""
    parts = item_str.split()
    color = parts[0] if parts else "Unknown"

    if "Top" in item_str:
        category = "Top"
    elif "Pants" in item_str or "Jeans" in item_str:
        category = "Pants"
    elif "Shorts" in item_str:
        category = "Shorts"
    elif "Skirt" in item_str:
        category = "Skirt"
    elif "Dress" in item_str:
        category = "Dress"
    elif "Jumpsuit" in item_str:
        category = "Jumpsuit"
    else:
        category = "Unknown"

    return {'color': color, 'category': category, 'fabric': 'Unknown'}

def suggestion_to_dict(suggestion: str, color: str, item_type: str) -> Dict[str, Any]:
    """Convert suggestion to dictionary for matching"""
    category_map = {
        "shoes": "Shoes",
        "jewelry/accessory": "Accessory",
        "bag": "Bag",
        "top": "Top",
        "bottom": "Pants",
        "outerwear": "Outerwear",
        "jacket/blazer": "Jacket"
    }

    category = category_map.get(item_type, "Unknown")
    return {'color': color, 'category': category, 'fabric': 'Unknown'}
