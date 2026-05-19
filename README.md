<div align="center">

<img src="https://img.shields.io/badge/React-TypeScript-blue?style=flat-square&logo=react" />
<img src="https://img.shields.io/badge/FastAPI-Python-green?style=flat-square&logo=fastapi" />
<img src="https://img.shields.io/badge/AWS-CloudFront%20%2B%20S3%20%2B%20EC2-orange?style=flat-square&logo=amazonaws" />
<img src="https://img.shields.io/badge/AI-FashionCLIP%20%2B%20SageMaker-purple?style=flat-square" />
<img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?style=flat-square&logo=githubactions" />

# WYA — What's Your Aesthetic

**An AI-powered full-stack fashion web app** that helps users discover, analyze, and refine their personal style through computer vision, style profiling, and wardrobe intelligence.

🔗 **Live:** [dsbml6kwxecah.cloudfront.net](https://dsbml6kwxecah.cloudfront.net)

</div>

---

## Screenshots

<table>
  <tr>
    <td align="center"><b>Login</b></td>
    <td align="center"><b>Dashboard</b></td>
  </tr>
  <tr>
    <td><img src="screenshot-login.png" width="400"/></td>
    <td><img src="screenshot-dashboard.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>My Wardrobe</b></td>
    <td align="center"><b>AI Auto-Tagging</b></td>
  </tr>
  <tr>
    <td><img src="screenshot-wardrobe.png" width="400"/></td>
    <td><img src="screenshot-autotag.png" width="400"/></td>
  </tr>
</table>

---

## Architecture

```
User (browser / mobile)
  ↓
CloudFront CDN  ·  HTTPS · global edge caching
  ├── /*          →  S3 static frontend  (React + Vite)
  └── /api/*      →  EC2 FastAPI backend  (Docker · port 8000)
                          ↓
                  AWS SageMaker endpoint
                  wya-fashionclip-serverless · ml.m5.xlarge
                          ↓
                  FashionCLIP (patrickjohncyh/fashion-clip)
                  zero-shot image classification
```

**Deployment:**
- Frontend → S3 + CloudFront (HTTPS, CDN cached, global)
- Backend → Docker on EC2 `c7i-flex.large` (ap-south-1), Elastic IP `65.1.104.57`
- CI/CD → GitHub Actions (push to `main` → auto build + deploy + CloudFront invalidation)

---

## Features

| Feature | Description |
|---|---|
| 🧬 Style DNA Quiz | Interactive questionnaire that maps your personal aesthetic |
| 👗 Wardrobe / Closet | Upload garments with AI auto-tagging (category, color, fabric, pattern) |
| 🤖 AI Outfit Matcher | Outfit suggestions based on color harmony and your style profile |
| 📈 Style Evolution | Track how your aesthetic changes over time |
| 🌿 Green Score | Sustainability rating for your wardrobe |
| ✨ Aesthetic Aura | Shareable style card generated from your wardrobe |
| ✈️ Vacation Packer | Trip and weather-based outfit curation |
| 🌤️ Weather Styling | Real-time weather-based outfit recommendations |
| 🪄 Background Removal | Clean garment images automatically via `rembg` |
| 🔔 Push Notifications | Style alerts and reminders via VAPID |

---

## AI Pipeline

```
Image upload
  ↓
Background removal  (rembg + OpenCV)
  ↓
Garment mask extraction  (GrabCut / Otsu thresholding)
  ↓
Garment crop + zoom  (removes background noise pre-classification)
  ↓
✅ Zero-shot classification  →  AWS SageMaker (FashionCLIP)
✅ Dominant color extraction  →  KMeans clustering (sklearn)
✅ Secondary color detection  →  largest non-dominant cluster
✅ Texture + brightness analysis  →  OpenCV
✅ Pattern detection  →  striped / floral / geometric / solid (Sobel + Canny)
✅ Fabric inference  →  rule-based classifier (category × color × texture × pattern)
✅ Smart name generation  →  e.g. "Floral Chiffon Midi Dress", "Washed Indigo Jeans"
✅ Style profile vectorization + outfit similarity matching
```

### Garment Auto-Tagging — Two-Tier Architecture

**Tier 1 — AWS SageMaker (FashionCLIP)**
Zero-shot classification with candidate labels. EC2 authenticates via IAM instance profile (no API keys). Returns category (e.g. Dress, Jeans, Watch).

**Tier 2 — Rule-based fabric classifier**
Runs locally on the EC2 container using `category × color × texture × pattern` rules — no additional ML inference needed.

```
Image upload
  ↓
SageMaker reachable? ── YES ──→ FashionCLIP zero-shot → category
  │                                                          ↓
  NO                                             Fabric classifier (local rules)
  ↓                                                          ↓
Fallback: category = "Top" ───────────────────→ Smart name generated
```

---

## Tech Stack

**Frontend**
- React + TypeScript + Vite
- Deployed on AWS S3 + CloudFront (HTTPS)

**Backend**
- FastAPI (Python)
- SQLite via SQLAlchemy (persisted at `/app/data/wya.db` via Docker volume)
- OpenCV + Pillow + `rembg` for computer vision
- scikit-learn for KMeans color clustering
- AWS SageMaker for garment classification (FashionCLIP)
- Dockerized, deployed on AWS EC2 (ap-south-1)

**AWS Infrastructure**
- EC2 `c7i-flex.large` (ap-south-1) — Docker backend, Elastic IP `65.1.104.57`
- S3 + CloudFront — static frontend with HTTPS and CDN caching
- CloudFront `/api/*` behavior — routes backend traffic through HTTPS (no mixed content)
- SageMaker endpoint `wya-fashionclip-serverless` on `ml.m5.xlarge` — InService
- IAM role `wya-sagemaker-role` via EC2 instance profile — no API keys needed

---

## Deployment Status

| Component | Status |
|---|---|
| Frontend (S3 + CloudFront) | ✅ Live |
| Backend (Docker on EC2) | ✅ Live |
| Elastic IP (fixed, survives reboots) | ✅ `65.1.104.57` |
| HTTPS end-to-end (no mixed content) | ✅ Via CloudFront |
| Database (SQLite, persistent volume) | ✅ Live |
| SageMaker FashionCLIP endpoint | ✅ InService |
| CI/CD (GitHub Actions) | ✅ Auto-deploy on push |
| Garment auto-tagging | ✅ Working |
| Color detection (KMeans) | ✅ Working |
| Background removal | ✅ Working |
| Login / Wardrobe / Style DNA | ✅ Working |
| Outfit Matcher | ✅ Working |
| Weather Styling | ✅ Working |

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
│   └── Profile.tsx
│
├── routers/                    # FastAPI route modules
│   ├── auth_router.py          # /api/auth
│   ├── wardrobe_router.py      # /api/wardrobe
│   ├── outfit_router.py        # /api/outfits
│   ├── ai_router.py            # /api/ai
│   ├── style_router.py         # /api/style
│   └── user_router.py          # /api/user
│
├── services/                   # Backend service modules
│   ├── computer_vision.py
│   ├── fabric_classifier.py
│   ├── color_matcher.py
│   ├── outfit_generator.py
│   ├── style_profile.py
│   └── weather_service.py
│
├── ai_model.py                 # AI orchestrator
├── main.py                     # FastAPI entry point
├── database.py                 # SQLite + SQLAlchemy
├── Dockerfile
└── .github/workflows/deploy.yml
```

---

## Run Locally

**Prerequisites:** Node.js 18+, Python 3.10+

**Frontend**
```bash
npm install
npm run dev
```

**Backend**
```bash
pip install -r requirements.txt
cp env.example .env
# Fill in your .env values
uvicorn main:app --reload
```

**Environment Variables**

| Variable | Description |
|---|---|
| `SECRET_KEY` | JWT secret |
| `SAGEMAKER_ENDPOINT` | SageMaker endpoint name |
| `AWS_REGION` | AWS region (default: `ap-south-1`) |
| `WYA_VAPID_PRIVATE_KEY` | Push notification private key |
| `WYA_VAPID_PUBLIC_KEY` | Push notification public key |

---

## Docker Deployment (EC2)

```bash
# Build and run
sudo docker build -t wya-backend .
sudo docker run -d \
  --name wya \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -e SAGEMAKER_ENDPOINT=wya-fashionclip-serverless \
  -e AWS_REGION=ap-south-1 \
  -v /home/ubuntu/wya-data:/app/data \
  wya-backend

# Logs
sudo docker logs wya -f
```

## CI/CD (GitHub Actions)

Push to `main` automatically triggers:
- `deploy-backend` — SSH into EC2, rebuild Docker image, restart container (~2m 30s)
- `deploy-frontend` — `npm run build` → S3 sync → CloudFront invalidation (~30s)
