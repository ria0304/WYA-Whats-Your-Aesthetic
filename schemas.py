# schemas.py
# Pydantic models for request body validation.

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ====================== AUTH ======================

class UserRegister(BaseModel):
    email: str
    password: str
    full_name: str
    birthday: Optional[str] = None
    gender: Optional[str] = "Female"
    location: Optional[str] = "Global"


class UserLogin(BaseModel):
    email: str
    password: str


# ====================== WARDROBE ======================

class WardrobeItemCreate(BaseModel):
    name: str
    category: str
    color: Optional[str] = ""
    fabric: Optional[str] = ""
    image_url: Optional[str] = None
    price: Optional[float] = 0.0
    brand: Optional[str] = ""
    sustainability_score: Optional[int] = 0
    tags: Optional[List[str]] = []


class WardrobeItemUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    color: Optional[str] = None
    fabric: Optional[str] = None
    image_url: Optional[str] = None
    price: Optional[float] = None
    brand: Optional[str] = None
    sustainability_score: Optional[int] = None
    tags: Optional[List[str]] = None


# ====================== OUTFITS ======================

class OutfitItem(BaseModel):
    item_id: str
    name: Optional[str] = ""
    category: Optional[str] = ""
    color: Optional[str] = ""
    image_url: Optional[str] = None

    def dict(self, **kwargs):
        return {
            "item_id": self.item_id,
            "name": self.name,
            "category": self.category,
            "color": self.color,
            "image_url": self.image_url,
        }


class OutfitCreate(BaseModel):
    name: str
    vibe: Optional[str] = ""
    items: List[OutfitItem] = []
    created_date: Optional[str] = None


# ====================== STYLE DNA ======================

class StyleDNACreate(BaseModel):
    styles: List[str]
    comfort_level: Optional[int] = 5
    summary: Optional[str] = ""
    colors: Optional[List[str]] = []
    silhouette: Optional[str] = ""


# ====================== AI REQUESTS ======================

class WeatherRequest(BaseModel):
    city: str


class GreenAuditRequest(BaseModel):
    brand: str


class TripRequest(BaseModel):
    city: Optional[str] = "Delhi"
    duration_days: Optional[int] = 3
    vacation_type: Optional[str] = "city"


# ====================== FEATURE 1: OUTIFT SCORING ======================

class OutfitScoreRequest(BaseModel):
    outfit: Dict[str, Any]
    style_dna: Optional[Dict[str, Any]] = None
    wear_history: Optional[List[Dict[str, Any]]] = None
    color_preferences: Optional[List[str]] = None


class OutfitScoreResponse(BaseModel):
    score: float
    breakdown: Dict[str, float]
    reasoning: List[str]
    max_score: int
    rating: str


# ====================== FEATURE 2: CONTEXT-AWARE ======================

class ContextRequest(BaseModel):
    time_of_day: Optional[str] = "afternoon"
    day_of_week: Optional[str] = "weekday"
    weather: Optional[str] = "unknown"
    temperature: Optional[float] = None
    occasion: Optional[str] = "everyday"


class ContextOutfitRequest(BaseModel):
    context: ContextRequest
    limit: Optional[int] = 5


# ====================== FEATURE 3: STYLE EVOLUTION ======================

class StyleEvolutionResponse(BaseModel):
    has_evolution: bool
    trajectory: List[Dict[str, Any]]
    current_archetype: Optional[str] = None
    first_archetype: Optional[str] = None
    evolution_direction: Optional[str] = None
    total_snapshots: int
    timespan: Optional[Dict[str, str]] = None


# ====================== FEATURE 4: GAP ANALYSIS ======================

class GapAnalysisRequest(BaseModel):
    inspired_category: Optional[str] = ""
    include_shopping_links: Optional[bool] = True


class ShoppingSuggestion(BaseModel):
    amazon: Optional[str] = None
    myntra: Optional[str] = None
    search_query: Optional[str] = None
    price_range: Optional[Dict[str, Any]] = None


class GapItem(BaseModel):
    category: str
    description: str
    reason: str
    priority: str
    affiliate_query: Optional[str] = None
    affiliate_brand: Optional[str] = None
    affiliate_url: Optional[str] = None
    dna_alignment_score: Optional[float] = None
    gender: Optional[str] = None
    shopping_suggestions: Optional[ShoppingSuggestion] = None
    price_range: Optional[Dict[str, Any]] = None


class GapAnalysisResponse(BaseModel):
    gaps: List[GapItem]
    primary_aesthetic: str
    dna_alignment_score: float
    neutral_ratio: float
    pattern_ratio: float
    wardrobe_count: int
    sustainability_score: Optional[float] = None
    shopping_links_included: bool


# ====================== FEATURE 5: ANALYTICS ======================

class WearItem(BaseModel):
    item_id: str
    name: str
    wear_count: int
    price: Optional[float] = None


class CostPerWearItem(BaseModel):
    name: str
    cost_per_wear: float
    wear_count: int
    price: float


class WardrobeAnalyticsResponse(BaseModel):
    total_items: int
    total_value: float
    most_worn: List[WearItem]
    least_worn: List[WearItem]
    cost_per_wear: List[CostPerWearItem]
    category_distribution: Dict[str, int]
    color_distribution: Dict[str, int]
    sustainability_score: float
    average_wear_count: float


# ====================== FEATURE 6: FEEDBACK ======================

class FeedbackRequest(BaseModel):
    action: str
    outfit_id: Optional[str] = None
    item_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}


class FeedbackResponse(BaseModel):
    status: str
    message: str
    action: str


# ====================== FEATURE 7: WEAR LOGGING ======================

class WearLogRequest(BaseModel):
    item_id: str
    outfit_id: Optional[str] = None
    occasion: Optional[str] = None
    weather: Optional[str] = None
    temperature: Optional[float] = None
    time_of_day: Optional[str] = None
    worn_at: Optional[str] = None


# ====================== FEATURE 8: CONVERSATION ======================

class ConversationRequest(BaseModel):
    message: str
    intent: Optional[str] = None


class ConversationResponse(BaseModel):
    text: str
    intent: Optional[str] = None
    outfits: Optional[List[Dict[str, Any]]] = None
    gap_analysis: Optional[Dict[str, Any]] = None
    style_dna: Optional[Dict[str, Any]] = None
