
# WYA - What's Your Aesthetic | Tech Stack

## Frontend (Client-Side)
- **Framework**: [React 19](https://react.dev/) (via ESM)
- **Routing**: [React Router DOM v7](https://reactrouter.com/)
- **Styling**: 
  - [TailwindCSS](https://tailwindcss.com/) (CDN)
  - Custom CSS Animations
- **Icons**: [Lucide React](https://lucide.dev/)
- **Build System**: ESM Modules (No bundler required for dev, runs natively in modern browsers via import maps)

## Backend (Server-Side)
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
- **Server**: Uvicorn (ASGI) with WatchFiles for auto-reload
- **Database**: [SQLite](https://www.sqlite.org/index.html) (Local embedded DB)
- **Authentication**: JWT (JSON Web Tokens) with `python-jose` & `bcrypt`
- **External Requests**: `requests` library for API calls

## Artificial Intelligence & Computer Vision

### Deep Learning Models
- **[FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip)** (via Hugging Face Transformers) - Zero-shot fashion classification trained on 800,000+ products
- **[Segment Anything Model (SAM)](https://segment-anything.com/)** - Optional segmentation for precise garment isolation
- **PyTorch** - Deep learning framework for running CLIP and SAM

### Computer Vision (OpenCV)
- **Image Processing**: 
  - `cv2.imdecode()` - Base64 image decoding
  - `cv2.cvtColor()` - Color space conversion (BGR ↔ RGB ↔ HSV)
  - `cv2.threshold()` - Otsu's thresholding for foreground detection
  - `cv2.Canny()` - Edge detection for garment boundaries
  - `cv2.findContours()` - Garment shape analysis
  - `cv2.grabCut()` - Enhanced background removal
  - `cv2.erode()` / `cv2.dilate()` - Mask cleaning and refinement
  - `cv2.morphologyEx()` - Morphological operations for mask quality

### Color Analysis
- **[Scikit-Learn](https://scikit-learn.org/)** - K-Means clustering for dominant color extraction
- **Custom ColorMatcher** - Intelligent color harmony engine using:
  - `colorsys` (Python built-in) - HSV color space conversions
  - Delta-E calculations - Color difference measurement
  - Color wheel theory - Complementary, analogous, split-complementary matching
  - Saturation & brightness analysis - Context-aware color suggestions

### External APIs
- **[Geoapify](https://www.geoapify.com/)** - Geocoding (city → coordinates) and Places API for real local shopping discovery
- **[Open-Meteo](https://open-meteo.com/)** - Free weather API (no key required) for real-time climate data
- **Hugging Face Hub** - Model hosting for FashionCLIP

### Data Storage
- **JSON Configuration Files**:
  - `fashion_data.json` - Style suggestions by vibe (minimalist, boho, streetwear, etc.)
  - `brand_score.json` - Sustainability scores for fashion brands
  - `regional_items.json` - Region-specific limited edition items
  - `weather_codes.json` - WMO weather code mappings
  - `category_map.json` - CLIP label to category mapping
  - `color_dictionary.json` - RGB values for color names
  - `color_harmony.json` - Static color harmony rules (fallback)
  - `country_to_region.json` - Country to cultural region mapping
  - `global_chains.json` - Global brand filter list
  - `local_indicators.json` - Keywords for identifying local businesses

## Key Features

### Garment Analysis Pipeline
1. **Image Decoding** → Base64 to numpy array
2. **Mask Generation** (SAM or enhanced GrabCut)
3. **Category Classification** (FashionCLIP with aspect ratio disambiguation)
4. **Color Extraction** (K-Means with aggressive background filtering)
5. **Fabric Inference** (Rule-based on texture variance & brightness)
6. **Intelligent Color Matching** (HSV color theory with variation)

### Travel Intelligence
- **Geoapify Geocoding** - City name → coordinates
- **Geoapify Places** - Real local markets, bakeries, boutiques
- **Chain Filtering** - Global brand exclusion
- **Limited Edition Items** - Region-based curated suggestions

### Weather Intelligence
- **Open-Meteo API** - Real-time weather data
- **Dynamic Outfit Generation** - Temperature-based clothing suggestions
- **Feels-like temperature** - More accurate comfort assessment
- **Rain/Snow/Wind adjustments** - Context-aware recommendations

## Architecture Highlights
- **Lazy Loading** - SAM and FashionCLIP loaded only when needed
- **JSON Data Separation** - All configuration externalized for easy updates
- **Fallback Chains** - Multiple strategies for reliability (SAM → GrabCut → center ellipse)
- **Type Safety** - Comprehensive type hints throughout
- **Error Handling** - Graceful degradation with fallback responses

## Performance Optimizations
- **Numpy to Python conversion** - Proper JSON serialization of numpy types
- **Aggressive erosion** - Removes boundary pixels for accurate color analysis
- **Multi-cluster K-Means** - Better dominant color detection
- **Caching-ready** - External API calls designed for caching

This stack enables **real-time fashion intelligence** with **local processing** for privacy and speed, while leveraging **external APIs** for up-to-date weather and location data!
