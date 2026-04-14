# services/computer_vision.py
# Lazy-loaded deep learning models + local CV pipeline for garment analysis.
# Includes FashionCLIP embeddings, background removal, and improved color extraction.

import base64
import logging
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import io

from .data_loader import CATEGORY_MAP, COLOR_DICTIONARY

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Optional heavy dependencies
# ------------------------------------------------------------------
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - deep learning features disabled")

# ------------------------------------------------------------------
# Background removal (rembg)
# ------------------------------------------------------------------
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg not available - background removal will use fallback")

# ------------------------------------------------------------------
# Lazy model loaders
# ------------------------------------------------------------------
SAM_AVAILABLE = False
FASHIONCLIP_AVAILABLE = False
predictor = None
clip_model = None
clip_processor = None


def load_sam() -> None:
    """Load Segment Anything Model for improved masking."""
    global SAM_AVAILABLE, predictor
    if SAM_AVAILABLE or predictor is not None:
        return
    try:
        import os
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        checkpoint = "sam_vit_b_01ec64.pth"
        if os.path.exists(checkpoint):
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam.to(device)
            predictor = SamPredictor(sam)
            SAM_AVAILABLE = True
            logger.info("SAM (vit_b) loaded on %s", device)
        else:
            logger.warning("SAM checkpoint not found at %s. Using GrabCut fallback.", checkpoint)
    except Exception as exc:
        logger.warning("SAM loading failed: %s", exc)


def load_fashionclip() -> None:
    """Load FashionCLIP model for embeddings and similarity matching."""
    global FASHIONCLIP_AVAILABLE, clip_model, clip_processor
    if FASHIONCLIP_AVAILABLE or clip_model is not None:
        return
    try:
        from transformers import CLIPModel, CLIPProcessor

        clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        FASHIONCLIP_AVAILABLE = True
        logger.info("FashionCLIP loaded successfully.")
    except Exception as exc:
        logger.warning("FashionCLIP loading failed: %s", exc)


# ------------------------------------------------------------------
# LocalComputerVision
# ------------------------------------------------------------------
class LocalComputerVision:
    """Local CV engine: segmentation, dominant-color extraction, texture analysis, embeddings, background removal."""

    # ---- image decoding ----

    def decode_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available for image decoding")
            return np.zeros((256, 256, 3), dtype=np.uint8)
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            if not base64_str or len(base64_str) < 100:
                raise ValueError("Invalid base64 image data")
            img_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            if img.shape[0] < 10 or img.shape[1] < 10:
                raise ValueError(f"Image too small: {img.shape}")
            return img
        except Exception as exc:
            logger.error("Image decode error: %s", exc)
            return np.zeros((256, 256, 3), dtype=np.uint8)

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 string."""
        if not CV2_AVAILABLE:
            return ""
        try:
            _, buffer = cv2.imencode('.png', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as exc:
            logger.error("Image encode error: %s", exc)
            return ""

    # ---- background removal ----

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """Remove background from image using rembg."""
        if not CV2_AVAILABLE:
            return image
        
        if REMBG_AVAILABLE:
            try:
                # Convert BGR to RGB for rembg
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Remove background
                output = remove(rgb_image)
                # Convert back to BGR
                if len(output.shape) == 3:
                    result = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                else:
                    result = image
                return result
            except Exception as exc:
                logger.warning("rembg failed: %s, using fallback", exc)
        
        # Fallback: use grabcut-based masking
        mask = self.get_improved_mask(image)
        # Apply mask to create transparent background (approximated with white)
        result = image.copy()
        result[mask == 0] = [255, 255, 255]
        return result

    # ---- FashionCLIP embeddings ----

    def get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """Generate FashionCLIP embedding for similarity comparison."""
        load_fashionclip()
        if not FASHIONCLIP_AVAILABLE or not TORCH_AVAILABLE:
            # Fallback to pseudo-embedding based on color/texture
            return self._get_pseudo_embedding(image)
        
        try:
            from PIL import Image
            
            # Convert OpenCV BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_image)
            
            # Process through FashionCLIP
            inputs = clip_processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                embedding = clip_model.get_image_features(**inputs)
            
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy()[0]
            
        except Exception as exc:
            logger.warning("FashionCLIP embedding failed: %s", exc)
            return self._get_pseudo_embedding(image)
    
    def _get_pseudo_embedding(self, image: np.ndarray) -> np.ndarray:
        """Fallback: generate pseudo-embedding from color and texture."""
        if not CV2_AVAILABLE:
            return np.zeros(512, dtype=np.float32)
        
        # Resize to consistent size
        h, w = image.shape[:2]
        resized = cv2.resize(image, (224, 224))
        
        # Extract color histogram
        hist = cv2.calcHist([resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Extract texture (variance)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        texture = np.var(gray).reshape(1)
        
        # Combine
        embedding = np.concatenate([hist, texture])
        
        # Pad to 512 dimensions
        if len(embedding) < 512:
            embedding = np.pad(embedding, (0, 512 - len(embedding)))
        else:
            embedding = embedding[:512]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)

    # ---- masking ----

    def get_improved_mask(self, image: np.ndarray) -> np.ndarray:
        """Get improved mask for garment isolation."""
        load_sam()
        if SAM_AVAILABLE:
            sam_mask = self._get_sam_mask(image)
            if sam_mask is not None and np.sum(sam_mask > 0) > 10_000:
                k = np.ones((5, 5), np.uint8)
                sam_mask = cv2.morphologyEx(sam_mask, cv2.MORPH_CLOSE, k)
                sam_mask = cv2.morphologyEx(sam_mask, cv2.MORPH_OPEN, k)
                contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    clean = np.zeros_like(sam_mask)
                    cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], -1, 255, -1)
                    clean = cv2.erode(clean, np.ones((5, 5), np.uint8), iterations=2)
                    return clean
                return sam_mask
        return self._enhanced_grabcut_mask(image)

    def _get_sam_mask(self, image_np: np.ndarray) -> Optional[np.ndarray]:
        """Internal SAM mask generation."""
        if not SAM_AVAILABLE or predictor is None:
            return None
        try:
            rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) if image_np.shape[2] == 3 else image_np
            predictor.set_image(rgb)
            masks, _, _ = predictor.predict(
                point_coords=None, point_labels=None, multimask_output=False
            )
            return (masks[0] * 255).astype(np.uint8)
        except Exception as exc:
            logger.warning("SAM prediction failed: %s", exc)
            return None

    def _enhanced_grabcut_mask(self, image: np.ndarray) -> np.ndarray:
        """Enhanced GrabCut mask with better background separation."""
        if not CV2_AVAILABLE:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, 30, 100)
        combined = cv2.bitwise_or(thresh, edges)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour_mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(contour_mask, [largest], -1, 255, -1)
            contour_mask = cv2.dilate(contour_mask, np.ones((15, 15), np.uint8), iterations=2)
            x, y, wc, hc = cv2.boundingRect(largest)
            pad_x, pad_y = int(wc * 0.1), int(hc * 0.1)
            rect = (max(0, x - pad_x), max(0, y - pad_y),
                    min(w - x, wc + 2 * pad_x), min(h - y, hc + 2 * pad_y))
            mask = np.zeros(image.shape[:2], np.uint8)
            mask[contour_mask > 0] = cv2.GC_PR_FGD
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
                final = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
                k = np.ones((5, 5), np.uint8)
                final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, k)
                final = cv2.morphologyEx(final, cv2.MORPH_OPEN, k)
                contours2, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours2:
                    clean = np.zeros_like(final)
                    cv2.drawContours(clean, [max(contours2, key=cv2.contourArea)], -1, 255, -1)
                    return cv2.erode(clean, np.ones((3, 3), np.uint8), iterations=1)
                return final
            except Exception:
                return contour_mask
        # Ultimate fallback - centre ellipse
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 255, -1)
        return mask

    def _get_garment_crop(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop image to garment bounding box."""
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        pad = 5
        return image[max(0, y - pad): min(image.shape[0], y + h + pad),
                     max(0, x - pad): min(image.shape[1], x + w + pad)]

    # ---- garment identification ----

    def identify_garment(self, image: np.ndarray, mask: np.ndarray) -> str:
        """Identify garment category using FashionCLIP."""
        load_fashionclip()
        if not FASHIONCLIP_AVAILABLE:
            return "Top"
        try:
            from PIL import Image
            import torch

            cropped = self._get_garment_crop(image, mask)
            if cropped.shape[0] < 50 or cropped.shape[1] < 50:
                cropped = image

            h, w = cropped.shape[:2]
            aspect_ratio = h / w if w > 0 else 1.0
            if mask is not None and np.sum(mask > 0) > 0:
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    _, _, wm, hm = cv2.boundingRect(coords)
                    aspect_ratio = hm / wm if wm > 0 else aspect_ratio

            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            labels = [
                "t-shirt", "shirt", "blouse", "tank top", "sweater", "hoodie", "cardigan", "polo",
                "jeans", "pants", "trousers", "leggings", "shorts", "cargo pants", "joggers",
                "skirt", "pencil skirt", "pleated skirt", "mini skirt", "midi skirt", "maxi skirt",
                "dress", "maxi dress", "mini dress", "midi dress", "bodycon dress", "a-line dress",
                "jumpsuit", "romper", "overalls",
                "jacket", "coat", "blazer", "puffer jacket", "denim jacket", "leather jacket",
            ]
            inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
            with torch.no_grad():
                probs = clip_model(**inputs).logits_per_image.softmax(dim=1)

            top_probs, top_indices = torch.topk(probs[0], 5)
            top_labels = [labels[i] for i in top_indices]
            top_scores = [p.item() for p in top_probs]

            scores: Dict[str, float] = {
                "jumpsuit": 0, "dress": 0, "skirt": 0,
                "pants": 0, "top": 0, "outerwear": 0,
            }
            buckets = {
                "jumpsuit": ["jumpsuit", "romper", "overalls"],
                "dress": ["dress", "maxi dress", "mini dress", "midi dress", "bodycon", "a-line"],
                "skirt": ["skirt", "pencil skirt", "pleated skirt", "mini skirt", "midi skirt", "maxi skirt"],
                "pants": ["jeans", "pants", "trousers", "leggings", "shorts", "cargo", "joggers"],
                "top": ["t-shirt", "shirt", "blouse", "tank top", "sweater", "hoodie", "cardigan", "polo"],
                "outerwear": ["jacket", "coat", "blazer", "puffer"],
            }
            for label, score in zip(top_labels, top_scores):
                for bucket, keywords in buckets.items():
                    if any(kw in label.lower() for kw in keywords):
                        scores[bucket] += score
                        break

            # Decision tree
            if scores["skirt"] > 0.2 and (0.8 < aspect_ratio < 2.5 or scores["skirt"] > 0.4):
                return "Skirt"
            if scores["jumpsuit"] > 0.25 and (aspect_ratio > 1.8 or scores["jumpsuit"] > 0.45):
                return "Jumpsuit"
            if scores["pants"] > 0.3:
                if scores["jumpsuit"] > 0.2 and aspect_ratio > 2.0:
                    return "Jumpsuit"
                return "Pants"
            if scores["dress"] > 0.3:
                if aspect_ratio > 2.5 and scores["jumpsuit"] > 0.2:
                    return "Jumpsuit"
                return "Dress"
            if scores["top"] > 0.3:
                return "Top"
            if scores["outerwear"] > 0.3:
                return "Outerwear"

            # Tiebreakers
            if scores["jumpsuit"] > 0.15 and scores["pants"] > 0.15:
                return "Jumpsuit" if aspect_ratio > 2.0 else "Pants"
            if scores["jumpsuit"] > 0.15 and scores["skirt"] > 0.15:
                return "Jumpsuit" if aspect_ratio > 2.2 else "Skirt"

            raw = labels[probs.argmax().item()]
            if "skirt" in raw.lower() and aspect_ratio < 2.5:
                return "Skirt"
            return CATEGORY_MAP.get(raw, "Top")

        except Exception as exc:
            logger.warning("FashionCLIP identification failed: %s", exc)
            return "Top"

    # ---- colour extraction ----

    def get_dominant_color(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[str, str, Tuple[int, int, int]]:
        """Return (hex, color_name, rgb_tuple) for the dominant garment colour."""
        if not CV2_AVAILABLE or not SKLEARN_AVAILABLE:
            return "#808080", "Gray", (128, 128, 128)

        mask = mask.astype(np.uint8)
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=3)

        if np.sum(mask > 0) < 5_000:
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2
            s = min(h, w) // 3
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (cx - s, cy - s), (cx + s, cy + s), 255, -1)

        pixels = image[mask > 0]
        if len(pixels) < 1_000:
            return "#808080", "Gray", (128, 128, 128)

        px_rgb = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        px_hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

        quality = (
            (px_hsv[:, 2] > 30) & (px_hsv[:, 2] < 220) & (px_hsv[:, 1] > 20) &
            (px_rgb[:, 0] < 240) & (px_rgb[:, 1] < 240) & (px_rgb[:, 2] < 240) &
            (px_rgb[:, 0] > 30) & (px_rgb[:, 1] > 30) & (px_rgb[:, 2] > 30)
        )
        filtered = px_rgb[quality] if np.sum(quality) > 500 else px_rgb

        try:
            n = min(5, max(3, len(filtered) // 1_000))
            km = KMeans(n_clusters=n, n_init=5, random_state=42)
            km.fit(filtered)
            counts = np.bincount(km.labels_)
            for idx in np.argsort(counts)[::-1][:2]:
                rgb = km.cluster_centers_[idx].astype(int)
                if 60 < int(rgb.sum()) < 750:
                    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
                    break
            else:
                rgb = km.cluster_centers_[np.argsort(counts)[-1]].astype(int)
                r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        except Exception:
            r, g, b = (int(x) for x in np.median(filtered, axis=0))

        name = self._map_rgb_to_color_name(r, g, b)
        if name in ("Navy", "Blue", "Light Blue", "Gray") and 60 < g < 150 and 40 < r < 140 and 80 < b < 200:
            name = "Denim"

        # Also return hex for color palette
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)

        return hex_color, name, (r, g, b)

    def _map_rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """Map RGB values to closest color name from dictionary."""
        best, min_dist = "Gray", float("inf")
        for name, val in COLOR_DICTIONARY.items():
            dist = (r - val[0]) ** 2 + (g - val[1]) ** 2 + (b - val[2]) ** 2
            if dist < min_dist:
                min_dist, best = dist, name
        return best

    # ---- texture analysis ----

    def analyze_texture_properties(
        self, image: np.ndarray, mask: np.ndarray = None
    ) -> Dict[str, float]:
        """Analyze texture variance and brightness of garment."""
        if not CV2_AVAILABLE:
            return {"variance": 0.0, "brightness": 128.0}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if mask is not None and mask.size > 0:
            if mask.shape != gray.shape:
                mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            center = cv2.bitwise_and(gray, gray, mask=mask)[mask > 0]
        else:
            center = gray.ravel()
        if len(center) == 0:
            center = gray.ravel()
        return {"variance": float(np.var(center)), "brightness": float(np.mean(center))}

    # ---- pattern detection for gap analysis ----

    def detect_pattern(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """Detect if garment has patterns (floral, striped, etc.)"""
        if not CV2_AVAILABLE:
            return {"has_pattern": False, "pattern_type": "solid", "confidence": 0.5}
        
        # Use masked region if provided
        if mask is not None and np.sum(mask > 0) > 0:
            masked = cv2.bitwise_and(image, image, mask=mask)
            # Crop to bounding box
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                cropped = masked[y:y+h, x:x+w]
            else:
                cropped = image
        else:
            cropped = image
        
        if cropped.size == 0:
            return {"has_pattern": False, "pattern_type": "solid", "confidence": 0.5}
        
        # Convert to grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge density (patterns have more edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate color variance
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        hue_variance = np.var(hsv[:, :, 0])
        
        has_pattern = edge_density > 0.05 or hue_variance > 500
        
        # Determine pattern type (simplified)
        pattern_type = "solid"
        if has_pattern:
            # Check for stripes vs floral vs geometric
            # Analyze edge orientation
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            angle = np.arctan2(sobely, sobelx)
            
            # If edges are strongly oriented in one direction, likely stripes
            angle_hist, _ = np.histogram(angle, bins=36)
            max_orientation = np.max(angle_hist) / np.sum(angle_hist)
            
            if max_orientation > 0.3:
                pattern_type = "striped"
            elif edge_density > 0.15:
                pattern_type = "floral"
            else:
                pattern_type = "geometric"
        
        return {
            "has_pattern": bool(has_pattern),
            "pattern_type": pattern_type,
            "confidence": min(0.95, edge_density * 5 + 0.3)
        }
