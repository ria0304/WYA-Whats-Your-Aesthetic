# WYA – What's Your Aesthetic

A full-stack AI-powered fashion web app that helps users discover, analyze, and refine their personal style. WYA combines computer vision, style profiling, and wardrobe intelligence to deliver personalized aesthetic insights.

---

## Live Deployment

| Service | URL |
|---|---|
| Frontend | https://d1yc69o122s878.cloudfront.net |
| Backend API | http://13.201.121.83:8000 |

---

## Architecture

```
User
  ↓
CloudFront + S3
  ↓
React + TypeScript Frontend
  ↓
FastAPI Backend (Docker on EC2 t2.micro, ap-south-1)
  ↓
AWS SageMaker Endpoint (ml.m5.xlarge)
  ↓
FashionCLIP (patrickjohncyh/fashion-clip) — zero-shot image classification
```

---

## Features

- **Style Quiz** — Interactive questionnaire that maps your aesthetic DNA
- **Wardrobe / Closet** — Upload garments with AI auto-tagging (category, color, fabric, pattern)
- **AI Outfit Matcher** — Get outfit suggestions based on color harmony and your style profile
- **Style Evolution** — Track how your style changes over time
- **Green Score** — Sustainability rating for your wardrobe
- **Aesthetic Aura** — Shareable style card generated from your wardrobe
- **Vacation Shop / Curate** — Trip and weather-based outfit curation
- **Weather Styling** — Outfit recommendations based on real-time weather
- **Background Removal** — Clean garment images automatically via rembg
- **Push Notifications** — Style alerts and reminders via VAPID

---

## AI Pipeline

1. Image decode and background removal (rembg + OpenCV)
2. Garment mask extraction (GrabCut / Otsu thresholding)
3. Garment crop and zoom (removes background noise before classification)
4. Zero-shot classification via AWS SageMaker (FashionCLIP on ml.m5.xlarge)
5. Dominant color extraction via KMeans clustering (sklearn)
6. Secondary color detection (largest non-dominant cluster)
7. Texture variance + brightness analysis (OpenCV)
8. Pattern detection — striped / floral / geometric / solid (Sobel + Canny)
9. Fabric inference via rule-based classifier (category × color × texture × pattern)
10. Smart name generation — e.g. "Floral Chiffon Midi Dress", "Washed Indigo Jeans"
11. Style profile vectorization and outfit similarity matching

---

## Tech Stack

### Frontend
- React + TypeScript
- Vite
- Deployed on AWS S3 + CloudFront

### Backend
- FastAPI (Python)
- SQLite via SQLAlchemy (persisted at `/app/data/wya.db` via Docker volume)
- OpenCV + Pillow + rembg for computer vision
- scikit-learn for KMeans color clustering
- AWS SageMaker for garment classification (FashionCLIP)
- Dockerized and deployed on AWS EC2 (ap-south-1)

### AWS Infrastructure
- EC2 `i-0ee2cb7f52191f766` (t2.micro, ap-south-1) — runs Docker backend
- S3 bucket `wya-whats-your-aesthetic` + CloudFront — static frontend
- SageMaker endpoint `wya-fashionclip-serverless` on `ml.m5.xlarge` — InService
- IAM role `wya-sagemaker-role` attached via EC2 instance profile — no API keys needed
- SQLite DB persisted at `/home/ubuntu/wya-data/wya.db` via Docker volume mount

---

## Project Structure

```
WYA-Whats-Your-Aesthetic/
│
├── views/                      # React page components
│   ├── Closet.tsx              # Wardrobe upload + autotag UI
│   ├── AIMatcher.tsx           # Outfit suggestion UI
│   ├── StyleQuiz.tsx           # Aesthetic quiz
│   ├── Dashboard.tsx
│   ├── Evolution.tsx
│   ├── GreenScore.tsx
│   ├── AestheticAura.tsx
│   ├── VacationShop.tsx
│   ├── Curate.tsx
│   ├── Weather.tsx
│   ├── TheArchive.tsx
│   ├── ScanLook.tsx
│   └── Profile.tsx
│
├── services/                   # Backend service modules
│   ├── computer_vision.py      # Garment detection, masking, color, pattern
│   ├── fabric_classifier.py    # Rule-based fabric inference engine
│   ├── color_matcher.py        # Color harmony engine
│   ├── outfit_generator.py     # Outfit + gap analysis
│   ├── style_profile.py        # Style DNA extraction
│   ├── gap_analyzer.py         # Wardrobe gap detection
│   ├── brand_auditor.py        # Brand sustainability scoring
│   ├── weather_service.py      # Real-time weather + outfit pairing
│   ├── trip_curator.py         # Vacation packing curation
│   ├── email_service.py
│   └── notification_service.py
│
├── ai_model.py                 # AI orchestrator (autotag, suggestions, aura)
├── ai_matcher.py               # Advanced similarity matching engine
├── main.py                     # FastAPI app + all API routes
├── database.py                 # SQLite models + SQLAlchemy setup
├── auth_utils.py               # JWT authentication
├── schemas.py                  # Pydantic request/response schemas
├── Dockerfile                  # Docker image for backend
├── deploy_fashionclip.py       # SageMaker endpoint deployment script
├── Test_sagemaker.py           # SageMaker connectivity diagnostic script
├── requirements.txt
├── .env
└── README.md
```

---

## Run Locally

### Prerequisites
- Node.js 18+
- Python 3.10+

### Frontend Setup
```bash
npm install
npm run dev
```

### Backend Setup
```bash
pip install -r requirements.txt
cp env.example .env
# Fill in your .env values
uvicorn main:app --reload
```

### Environment Variables
See `env.example` for all required variables:
- `SECRET_KEY` — JWT secret
- `SAGEMAKER_ENDPOINT` — SageMaker endpoint name (default: `wya-fashionclip-serverless`)
- `AWS_REGION` — AWS region (default: `ap-south-1`)
- `WYA_VAPID_PRIVATE_KEY` / `WYA_VAPID_PUBLIC_KEY` — Push notifications

---

## Garment Auto-Tagging Architecture

Autotag runs a two-tier pipeline:

1. **AWS SageMaker** — FashionCLIP zero-shot classification with candidate labels. EC2 authenticates via IAM instance profile (no API keys). Returns category (e.g. Dress, Jeans, Watch).
2. **Default fallback** — Returns `"Top"` if SageMaker is unreachable.

The fabric classifier then runs locally on the EC2 container using category × color × texture × pattern rules — no additional ML inference needed.

---

## Deployment Status

| Component | Status |
|---|---|
| Frontend (S3 + CloudFront) | ✅ Live |
| Backend (Docker on EC2) | ✅ Live |
| Database (SQLite, persistent volume) | ✅ Live |
| SageMaker FashionCLIP endpoint | ✅ InService |
| Garment auto-tagging (category) |  not Working |
| Color detection (KMeans) | ✅ Working |
| Fabric classifier |  not Working |
| Background removal | ✅ Working |
| Login / wardrobe / style DNA | ✅ Working |
| Outfit matcher | ✅ Working |
| Weather styling | ✅ Working |
| Green score | ✅ Working |
| Aesthetic aura | ✅ Working |

---

## Deployment

### Backend (Docker on EC2)

```bash
# Build image
sudo docker build -t wya-backend .

# Run container with persistent DB volume
sudo docker run -d \
  --name wya \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -e SAGEMAKER_ENDPOINT=wya-fashionclip-serverless \
  -e AWS_REGION=ap-south-1 \
  -v /home/ubuntu/wya-data:/app/data \
  wya-backend

# View logs
sudo docker logs wya -f

# Rebuild after code changes
sudo docker stop wya && sudo docker rm wya
sudo docker build -t wya-backend .
sudo docker run -d --name wya --restart unless-stopped \
  -p 8000:8000 --env-file .env \
  -e SAGEMAKER_ENDPOINT=wya-fashionclip-serverless \
  -e AWS_REGION=ap-south-1 \
  -v /home/ubuntu/wya-data:/app/data \
  wya-backend

# Free disk space if build fails (t2.micro fills up fast)
sudo docker system prune -a -f
```

### Frontend (S3 + CloudFront)

```bash
npm run build
aws s3 sync dist/ s3://wya-whats-your-aesthetic --delete
```

### SageMaker Endpoint

```bash
source venv/bin/activate
python3 deploy_fashionclip.py
```

### Diagnose SageMaker Connectivity

```bash
# Run on EC2 to verify credentials + endpoint + invocation
pip3 install boto3 pillow --break-system-packages
python3 Test_sagemaker.py

# Test with a real garment image
python3 Test_sagemaker.py /path/to/garment.jpg
```
