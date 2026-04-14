# services/fabric_classifier.py
# Rule-based fabric inference from texture metrics, colour, and garment category.

class FabricClassifier:
    """
    Enhanced Logic Inference Engine (LIE) for fabric detection.
    Defaults to "Cotton" for any unmatched case.
    """

    @staticmethod
    def classify(variance: float, brightness: float, color: str, category: str) -> str:
        _DENIM_COLORS = {
            "Denim", "Light Denim", "Navy", "Blue", "Charcoal", "Ice Blue",
            "Gray", "Black", "Light Blue", "Royal Blue", "Sky Blue",
            "Slate", "Indigo", "Midnight Navy",
        }
        _DENIM_CATS = {"Pants", "Shorts", "Jacket", "Skirt", "Dress", "Jumpsuit", "Top"}

        if category in _DENIM_CATS and color in _DENIM_COLORS:
            if 150 < variance < 800 or color in ("Denim", "Light Denim"):
                return "Denim"

        if category in ("Necklace", "Ring", "Earrings", "Watch", "Jewellery"):
            if color in ("Gold", "Yellow", "Orange", "Beige", "Cream"):
                return "Gold"
            if color in ("Silver", "Gray", "White", "Platinum", "Ash"):
                return "Silver"
            if category == "Watch" and color in ("Black", "Brown", "Tan"):
                return "Leather Strap"
            return "Metal"

        if category == "Bag":
            if color in ("Brown", "Tan", "Black", "Camel", "Cognac", "Red"):
                return "Leather"
            return "Canvas" if variance > 300 else "Synthetic"

        if category in ("Pants", "Shorts"):
            return "Cotton" if variance > 200 else "Polyester"

        if category == "Skirt":
            if variance < 30 and brightness > 120:  return "Satin"
            if 300 < variance < 700:
                return "Linen" if color in ("White", "Beige", "Cream", "Olive", "Sage", "Brown", "Tan") else "Cotton"
            if variance < 100 and brightness > 150:  return "Silk"
            return "Polyester"

        if category == "Dress":
            if variance > 800:
                if color in ("Red", "Burgundy", "Navy", "Black", "Green", "Emerald", "Purple") and brightness < 120:
                    return "Velvet"
                return "Knit" if brightness < 100 else "Textured"
            if variance < 30 and brightness > 120:    return "Satin"
            if 30 < variance < 100 and brightness > 150:
                return "Chiffon" if color in ("Blush", "Lavender", "Mint", "Baby Blue", "Cream", "White") else "Silk"
            if 100 < variance < 400:
                return "Linen" if color in ("White", "Beige", "Cream", "Olive", "Sage", "Brown", "Tan", "Rust") else "Cotton"
            if 400 < variance < 700:
                return "Crepe" if brightness < 100 else "Ponte"
            return "Polyester"

        if category == "Top":
            if variance > 800:                        return "Cotton"
            if variance < 30 and brightness > 120:   return "Satin"
            if 300 < variance < 700 and color in ("White", "Beige", "Cream", "Olive"):
                return "Linen"
            return "Cotton"

        if category == "Jumpsuit":
            if variance < 30 and brightness > 120:    return "Satin"
            if variance > 500:
                return "Velvet" if brightness < 100 else "Textured"
            if 30 < variance < 200 and brightness > 150: return "Silk"
            if 200 < variance < 500:
                return "Linen" if color in ("White", "Beige", "Cream", "Olive", "Sage", "Brown", "Tan") else "Cotton"
            return "Polyester"

        if category in ("Jacket", "Coat", "Outerwear", "Blazer"):
            if variance > 400:                        return "Cotton"
            if color in ("Brown", "Tan", "Camel", "Black", "Navy"):
                return "Leather" if variance < 200 else "Suede"
            return "Polyester"

        return "Cotton"
