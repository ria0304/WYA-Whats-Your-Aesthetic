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
except (ImportError, OSError):
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - image processing will be limited")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except (ImportError, OSError):
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - color clustering will use fallback")

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - deep learning features disabled")

# ------------------------------------------------------------------
# Background removal (rembg)
# ------------------------------------------------------------------
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except (ImportError, OSError):
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
    except (ImportError, OSError, Exception) as exc:
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
    except (ImportError, OSError, Exception) as exc:
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
        """
        Remove background using rembg (returns RGBA numpy array with true transparency).
        Falls back to GrabCut masking with white background if rembg is unavailable.
        """
        if not CV2_AVAILABLE:
            return image

        if REMBG_AVAILABLE:
            try:
                from PIL import Image as PILImage
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_in = PILImage.fromarray(rgb)
                pil_out = remove(pil_in)   # RGBA PIL Image
                return np.array(pil_out)   # shape (H, W, 4)
            except Exception as exc:
                logger.warning("rembg failed: %s — using GrabCut fallback", exc)

        # GrabCut fallback — white background, no alpha
        mask = self.get_improved_mask(image)
        result = image.copy()
        result[mask == 0] = [255, 255, 255]
        return result

    def encode_image_to_base64_png(self, image: np.ndarray) -> str:
        """Encode RGB/BGR/RGBA numpy array to base64 PNG, preserving alpha if present."""
        try:
            from PIL import Image as PILImage
            if image.ndim == 3 and image.shape[2] == 4:
                pil = PILImage.fromarray(image, "RGBA")
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else image
                pil = PILImage.fromarray(rgb, "RGB")
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.error("PNG encode error: %s", exc)
            return ""

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

    # ── Dedicated shoe sub-classifier ───────────────────────────────────────

    def _classify_shoe_subtype(self, pil_img, image_np: np.ndarray) -> str:
        """
        Two-pass shoe identification:
          Pass 1 — broad CLIP categories to confirm it's a shoe.
          Pass 2 — focused CLIP call with only shoe labels.
          Pass 3 — CV shape heuristics as tiebreaker (heel height, sole thickness, openness).
        """
        import torch

        # ── Pass 2: focused shoe-only CLIP ──────────────────────────────────
        shoe_labels = [
            "sneakers",          # general athletic / casual
            "running shoes",     # sporty, mesh upper
            "canvas sneakers",   # low-top canvas like Converse/Vans
            "high-top sneakers", # high ankle athletic
            "ankle boots",       # short shaft, any heel
            "knee-high boots",   # tall shaft
            "combat boots",      # chunky, lace-up, military
            "chelsea boots",     # elastic side panel, no laces
            "cowboy boots",      # pointed toe, stacked heel, western
            "stiletto heels",    # very thin high heel
            "block heels",       # chunky square heel
            "kitten heels",      # low thin heel
            "platform heels",    # thick platform + heel
            "wedge heels",       # solid wedge sole
            "strappy heels",     # open, strappy construction
            "flat sandals",      # no heel, open toe
            "sports sandals",    # velcro / buckle straps
            "gladiator sandals", # multiple straps up the leg
            "loafers",           # slip-on, closed toe, low heel
            "oxford shoes",      # lace-up, closed toe, formal
            "brogues",           # oxford with decorative perforations
            "ballet flats",      # very flat, rounded toe, no fastening
            "pointed flats",     # flat with pointed toe
            "mules",             # backless, closed toe
            "slide sandals",     # backless, open toe
            "flip flops",        # thong sandal
            "mary janes",        # rounded toe, single strap across instep
            "platform shoes",    # thick sole all around, no distinct heel
            "espadrilles",       # rope/jute sole
            "monk strap shoes",  # buckle strap, no laces, formal
        ]

        inputs = clip_processor(text=shoe_labels, images=pil_img, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]

        top_probs, top_idx = torch.topk(probs, 5)
        top_labels = [shoe_labels[i] for i in top_idx]
        top_scores = [p.item() for p in top_probs]

        # Aggregate into sub-type buckets
        sub_scores = {
            "Sneakers":       0.0,
            "Boots":          0.0,
            "Heels":          0.0,
            "Sandals":        0.0,
            "Loafers":        0.0,
            "Oxfords":        0.0,
            "Flats":          0.0,
            "Slides":         0.0,
            "Mary Janes":     0.0,
            "Platform Shoes": 0.0,
        }
        sub_buckets = {
            "Sneakers":       ["sneakers", "running shoes", "canvas sneakers", "high-top sneakers"],
            "Boots":          ["ankle boots", "knee-high boots", "combat boots", "chelsea boots", "cowboy boots"],
            "Heels":          ["stiletto heels", "block heels", "kitten heels", "platform heels", "wedge heels", "strappy heels"],
            "Sandals":        ["flat sandals", "sports sandals", "gladiator sandals"],
            "Loafers":        ["loafers", "monk strap shoes", "espadrilles"],
            "Oxfords":        ["oxford shoes", "brogues"],
            "Flats":          ["ballet flats", "pointed flats"],
            "Slides":         ["mules", "slide sandals", "flip flops"],
            "Mary Janes":     ["mary janes"],
            "Platform Shoes": ["platform shoes"],
        }
        for lbl, score in zip(top_labels, top_scores):
            for sub, keywords in sub_buckets.items():
                if any(kw in lbl for kw in keywords):
                    sub_scores[sub] += score
                    break

        # ── Pass 3: CV shape heuristics ─────────────────────────────────────
        # Only used as a tiebreaker when top two CLIP subs are close (diff < 0.08)
        cv_hint = self._shoe_shape_heuristic(image_np)

        sorted_subs = sorted(sub_scores.items(), key=lambda x: x[1], reverse=True)
        best_sub, best_score   = sorted_subs[0]
        second_sub, second_score = sorted_subs[1]

        # If clear winner — use it
        if best_score - second_score > 0.08:
            return best_sub

        # Tiebreaker — cv_hint decides
        if cv_hint and cv_hint in sub_scores:
            return cv_hint
        return best_sub

    def _shoe_shape_heuristic(self, image_np: np.ndarray) -> str:
        """
        Fast CV heuristics on the shoe silhouette:
        - Shaft height ratio  → boots vs low shoes
        - Sole/base thickness → platform / wedge
        - Open pixel ratio    → sandals / slides
        - Aspect ratio        → tall boots vs ankle boots
        Returns a sub-type string or "" if inconclusive.
        """
        if not CV2_AVAILABLE:
            return ""
        try:
            h, w = image_np.shape[:2]
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

            # Threshold to isolate shoe silhouette against background
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return ""
            cnt = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw == 0 or ch == 0:
                return ""

            aspect = ch / cw          # tall = boots; wide-ish = flats/sneakers
            fill_ratio = cv2.contourArea(cnt) / (cw * ch)   # low fill = open / strappy

            # Sole band: bottom 20% of bounding box — thick sole → platform
            sole_region = thresh[y + int(ch * 0.8): y + ch, x: x + cw]
            sole_fill = np.sum(sole_region > 0) / max(sole_region.size, 1)

            # Heel column: rightmost 15% of bounding box vertical strip
            heel_col = thresh[y: y + ch, x + int(cw * 0.85): x + cw]
            heel_density = np.sum(heel_col > 0) / max(heel_col.size, 1)

            if aspect > 2.5:
                return "Boots"         # very tall
            if aspect > 1.5:
                return "Boots"         # ankle / mid shaft
            if sole_fill > 0.75 and aspect < 0.8:
                return "Platform Shoes"
            if fill_ratio < 0.45:
                return "Sandals"       # very open / strappy
            if heel_density < 0.25 and aspect < 1.0:
                return "Flats"         # almost no heel column
            if heel_density > 0.65 and aspect < 1.4:
                return "Heels"         # tall narrow heel column
            if aspect < 1.1 and fill_ratio > 0.65:
                return "Sneakers"      # chunky, filled, low
            return ""
        except Exception:
            return ""

    # ── HuggingFace API fallback ─────────────────────────────────────────────

    def _identify_garment_hf_api(self, image: np.ndarray) -> str:
        """Fallback: use SageMaker Serverless FashionCLIP for garment classification."""
        import os, base64, io, json
        import boto3
        from PIL import Image as PILImage

        endpoint_name = os.getenv("SAGEMAKER_ENDPOINT", "wya-fashionclip-serverless")
        region = os.getenv("AWS_REGION", "ap-south-1")

        try:
            pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            labels = [
                "t-shirt", "dress", "jeans", "trousers", "jacket", "coat",
                "skirt", "shoes", "sneakers", "heels", "boots", "bag",
                "shorts", "hoodie", "blazer", "jumpsuit", "blouse",
                "watch", "necklace", "ring", "earrings"
            ]

            runtime = boto3.client("sagemaker-runtime", region_name=region)
            payload = {"inputs": img_b64, "parameters": {"candidate_labels": labels}}
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            results = json.loads(response["Body"].read())

            if isinstance(results, list) and results:
                top = results[0].get("label", "Top").lower()
                mapping = {
                    "t-shirt": "Top", "blouse": "Top", "hoodie": "Top",
                    "dress": "Dress", "jumpsuit": "Jumpsuit",
                    "jeans": "Jeans", "trousers": "Trousers", "shorts": "Shorts",
                    "skirt": "Skirt",
                    "jacket": "Jacket", "coat": "Jacket", "blazer": "Jacket",
                    "shoes": "Shoes", "sneakers": "Shoes", "heels": "Shoes", "boots": "Shoes",
                    "bag": "Bag",
                    "watch": "Watch", "necklace": "Necklace", "ring": "Ring", "earrings": "Earrings"
                }
                return mapping.get(top, "Top")
        except Exception as e:
            import traceback
            logger.error(
                f"SageMaker garment call FAILED — endpoint={endpoint_name} region={region}\n"
                f"Error: {e}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
        return "Top"

    # ── Main garment identifier ──────────────────────────────────────────────

    def identify_garment(self, image: np.ndarray, mask: np.ndarray) -> str:
        """Identify garment category using FashionCLIP (two-stage for shoes)."""
        load_fashionclip()
        if not FASHIONCLIP_AVAILABLE:
            return self._identify_garment_hf_api(image)
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

            # ── Stage 1: broad category detection ───────────────────────────
            broad_labels = [
                # Clothing
                "t-shirt", "shirt", "blouse", "tank top", "crop top", "sweater", "hoodie",
                "cardigan", "polo", "turtleneck",
                "jeans", "pants", "trousers", "leggings", "shorts", "cargo pants", "joggers",
                "skirt", "mini skirt", "midi skirt", "maxi skirt",
                "dress", "maxi dress", "mini dress", "midi dress", "bodycon dress",
                "jumpsuit", "romper", "overalls",
                "jacket", "coat", "blazer", "puffer jacket", "leather jacket",
                # Non-clothing (single representative label per category)
                "shoes",       # shoes catch-all — Stage 2 will refine
                "handbag", "tote bag", "backpack", "crossbody bag", "clutch",
                "belt", "hat", "scarf", "sunglasses",
                "necklace", "earrings", "ring", "watch",
            ]
            inputs = clip_processor(text=broad_labels, images=pil_img, return_tensors="pt", padding=True)
            with torch.no_grad():
                probs = clip_model(**inputs).logits_per_image.softmax(dim=1)

            top_probs, top_indices = torch.topk(probs[0], 5)
            top_labels = [broad_labels[i] for i in top_indices]
            top_scores = [p.item() for p in top_probs]

            scores: Dict[str, float] = {
                "jumpsuit": 0, "dress": 0, "skirt": 0, "pants": 0,
                "top": 0, "outerwear": 0, "shoes": 0,
                "bag": 0, "accessory": 0, "jewellery": 0,
            }
            buckets = {
                "jumpsuit":  ["jumpsuit", "romper", "overalls"],
                "dress":     ["dress", "maxi dress", "mini dress", "midi dress", "bodycon", "a-line"],
                "skirt":     ["skirt", "mini skirt", "midi skirt", "maxi skirt"],
                "pants":     ["jeans", "pants", "trousers", "leggings", "shorts", "cargo", "joggers"],
                "top":       ["t-shirt", "shirt", "blouse", "tank top", "crop top", "sweater", "hoodie", "cardigan", "polo", "turtleneck"],
                "outerwear": ["jacket", "coat", "blazer", "puffer", "leather jacket"],
                "shoes":     ["shoes"],
                "bag":       ["handbag", "tote bag", "backpack", "crossbody bag", "clutch"],
                "accessory": ["belt", "hat", "scarf", "sunglasses"],
                "jewellery": ["necklace", "earrings", "ring", "watch"],
            }
            for label, score in zip(top_labels, top_scores):
                for bucket, keywords in buckets.items():
                    if any(kw in label.lower() for kw in keywords):
                        scores[bucket] += score
                        break

            # Reset side-effect slots
            self._last_secondary_color: Optional[str] = None
            self._last_shoe_subtype: str = "Shoes"

            # ── Accessory / jewellery / bag — high specificity ───────────────
            if scores["jewellery"] > 0.2:
                raw = broad_labels[probs.argmax().item()]
                if "watch"    in raw: return "Watch"
                if "necklace" in raw: return "Necklace"
                if "ring"     in raw: return "Ring"
                if "earring"  in raw: return "Earrings"
                return "Necklace"
            if scores["bag"] > 0.2:
                return "Bag"
            if scores["accessory"] > 0.2:
                return "Accessories"

            # ── Shoes — Stage 2 focused sub-classifier ───────────────────────
            if scores["shoes"] > 0.18:
                subtype = self._classify_shoe_subtype(pil_img, cropped)
                self._last_shoe_subtype = subtype
                return "Shoes"

            # ── Clothing decision tree ───────────────────────────────────────
            if scores["skirt"] > 0.2 and (0.8 < aspect_ratio < 2.5 or scores["skirt"] > 0.4):
                return "Skirt"
            if scores["jumpsuit"] > 0.25 and (aspect_ratio > 1.8 or scores["jumpsuit"] > 0.45):
                return "Jumpsuit"
            if scores["pants"] > 0.3:
                if scores["jumpsuit"] > 0.2 and aspect_ratio > 2.0:
                    return "Jumpsuit"
                raw = broad_labels[probs.argmax().item()]
                if "short"  in raw: return "Shorts"
                if "jean"   in raw: return "Jeans"
                if "legging" in raw: return "Trousers"
                return "Trousers"
            if scores["dress"] > 0.3:
                if aspect_ratio > 2.5 and scores["jumpsuit"] > 0.2:
                    return "Jumpsuit"
                return "Dress"
            if scores["top"] > 0.3:
                raw = broad_labels[probs.argmax().item()]
                if "sweater"  in raw or "cardigan" in raw: return "Sweater"
                if "t-shirt"  in raw:                      return "T-Shirt"
                return "Top"
            if scores["outerwear"] > 0.3:
                raw = broad_labels[probs.argmax().item()]
                return "Jacket" if "blazer" in raw or "jacket" in raw else "Outerwear"

            # Tiebreakers
            if scores["jumpsuit"] > 0.15 and scores["pants"] > 0.15:
                return "Jumpsuit" if aspect_ratio > 2.0 else "Trousers"
            if scores["jumpsuit"] > 0.15 and scores["skirt"] > 0.15:
                return "Jumpsuit" if aspect_ratio > 2.2 else "Skirt"

            raw = broad_labels[probs.argmax().item()]
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

        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)

        # ── Secondary color (2nd largest cluster, if distinct enough) ──
        self._last_secondary_color: Optional[str] = None
        self._last_shoe_subtype: str = "Shoes"
        if SKLEARN_AVAILABLE:
            try:
                counts_sorted = np.argsort(np.bincount(km.labels_))[::-1]
                if len(counts_sorted) > 1:
                    sec_rgb = km.cluster_centers_[counts_sorted[1]].astype(int)
                    sec_r, sec_g, sec_b = int(sec_rgb[0]), int(sec_rgb[1]), int(sec_rgb[2])
                    # Only report secondary if visually distinct from primary
                    dist = ((r - sec_r)**2 + (g - sec_g)**2 + (b - sec_b)**2) ** 0.5
                    if dist > 60:
                        self._last_secondary_color = self._map_rgb_to_color_name(sec_r, sec_g, sec_b)
            except Exception:
                pass

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
        
        if mask is not None and np.sum(mask > 0) > 0:
            masked = cv2.bitwise_and(image, image, mask=mask)
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
        
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        hue_variance = np.var(hsv[:, :, 0])
        has_pattern = edge_density > 0.05 or hue_variance > 500
        
        pattern_type = "solid"
        if has_pattern:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            angle = np.arctan2(sobely, sobelx)
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
