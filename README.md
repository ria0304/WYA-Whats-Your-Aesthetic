<div align="center">

<img src="https://img.shields.io/badge/React-TypeScript-blue?style=flat-square&logo=react" />
<img src="https://img.shields.io/badge/FastAPI-Python-green?style=flat-square&logo=fastapi" />
<img src="https://img.shields.io/badge/AWS-CloudFront%20%2B%20S3%20%2B%20EC2-orange?style=flat-square&logo=amazonaws" />
<img src="https://img.shields.io/badge/AI-FashionCLIP%20%2B%20SageMaker-purple?style=flat-square" />
<img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?style=flat-square&logo=githubactions" />
<img src="https://img.shields.io/badge/Rate%20Limiting-slowapi-teal?style=flat-square" />
<img src="https://img.shields.io/badge/Tests-pytest-yellow?style=flat-square&logo=pytest" />

![CI/CD](https://github.com/ria0304/WYA-Whats-Your-Aesthetic/actions/workflows/deploy.yml/badge.svg)

# WYA — What's Your Aesthetic

**An AI-powered full-stack fashion web app** that helps users discover, analyze, and refine their personal style through computer vision, style profiling, and wardrobe intelligence.

🔗 **Live:** [dsbml6kwxecah.cloudfront.net](https://dsbml6kwxecah.cloudfront.net)

</div>

---

## Architecture

```mermaid
flowchart TD
    A["🌐 Browser / Mobile\nUser"]:::gray

    B["⚡ CloudFront CDN\nHTTPS · global edge caching · /* and /api/* routing"]:::purple

    C["🗂️ S3 Static Frontend\nReact + Vite · TypeScript"]:::teal

    D["🖥️ EC2 FastAPI Backend\nDocker · c7i-flex.large · ap-south-1"]:::blue

    DB["🗄️ SQLite Database\nPersisted via Docker volume"]:::gray

    E["🤖 AWS SageMaker\nwya-fashionclip-serverless · ml.m5.xlarge"]:::amber

    F["👗 FashionCLIP\nzero-shot image classification"]:::coral

    A --> B
    B -->|"/*"| C
    B -->|"/api/*"| D
    D --> DB
    D --> E
    E --> F

    classDef gray   fill:#e8e6e1,stroke:#9c9a92,color:#2C2C2A
    classDef purple fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef teal   fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef blue   fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    classDef amber  fill:#FAEEDA,stroke:#854F0B,color:#633806
    classDef coral  fill:#FAECE7,stroke:#993C1D,color:#712B13
```

**Deployment**
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

```mermaid
flowchart TD
    A["📤 Image upload"]:::gray
    B{"SageMaker reachable?"}:::purple
    C["FashionCLIP zero-shot\n→ category"]:::blue
    D["Fallback: category = Top"]:::coral
    E["Fabric classifier\nlocal rules"]:::teal
    F["✅ Smart name generated"]:::amber

    A --> B
    B -->|YES| C
    B -->|NO| D
    C --> E
    D --> E
    E --> F

    classDef gray   fill:#e8e6e1,stroke:#9c9a92,color:#2C2C2A
    classDef purple fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef blue   fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    classDef teal   fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef coral  fill:#FAECE7,stroke:#993C1D,color:#712B13
    classDef amber  fill:#FAEEDA,stroke:#854F0B,color:#633806
```

---

## Tech Stack

**Frontend**
- React + TypeScript + Vite
- Deployed on AWS S3 + CloudFront (HTTPS)

**Backend**
- FastAPI (Python)
- SQLite (persisted at `/app/data/wya.db` via Docker volume)
- OpenCV + Pillow + `rembg` for computer vision
- scikit-learn for KMeans color clustering
- slowapi for per-route rate limiting on AI endpoints
- AWS SageMaker for garment classification (FashionCLIP)
- Dockerized, deployed on AWS EC2 (ap-south-1)

**AWS Infrastructure**
- EC2 `c7i-flex.large` (ap-south-1) — Docker backend, Elastic IP `65.1.104.57`
- S3 + CloudFront — static frontend with HTTPS and CDN caching
- CloudFront `/api/*` behavior — routes backend traffic through HTTPS (no mixed content)
- SageMaker endpoint `wya-fashionclip-serverless` on `ml.m5.xlarge` — InService
- IAM role `wya-sagemaker-role` via EC2 instance profile — no API keys needed

---

## Rate Limiting

AI endpoints are protected with [slowapi](https://github.com/laurentS/slowapi) to prevent abuse and control SageMaker inference costs.

| Endpoint | Limit |
|---|---|
| `POST /api/ai/fabric-scan` | 10 / minute |
| `POST /api/ai/outfit-match` | 10 / minute |
| `POST /api/ai/curate-outfits` | 10 / minute |
| `POST /api/ai/gap-analysis` | 10 / minute |
| `POST /api/ai/green-audit` | 20 / minute |

Standard CRUD endpoints (`/api/wardrobe`, `/api/auth`, `/api/outfits`, etc.) are not rate limited.

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
| Rate limiting (slowapi) | ✅ AI endpoints protected |
| Health checks (`/health`, `/health/ready`) | ✅ Live |
| Automated daily backups (S3) | ✅ Running via cron |
| Server watchdog (auto-recovery) | ✅ Running via systemd |
| Garment auto-tagging (category) | ✅ Working |
| Color detection (KMeans) | ✅ Working |
| Fabric classifier | ✅ Working |
| Background removal | ✅ Working |
| Login / Wardrobe / Style DNA | ✅ Working |
| Outfit Matcher | ✅ Working |
| Weather Styling | ✅ Working |
| Green Score | ✅ Working |
| Aesthetic Aura | ✅ Working |

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
│   ├── auth_router.py          # /api/auth — login, register
│   ├── wardrobe_router.py      # /api/wardrobe — CRUD, remove-bg, archive
│   ├── outfit_router.py        # /api/outfits — save, wear tracking, history
│   ├── ai_router.py            # /api/ai — fabric-scan, outfit-match, weather, gap
│   ├── style_router.py         # /api/style — DNA, aura, evolution, dashboard
│   ├── user_router.py          # /api/user — profile, preferences, notifications
│   └── health_router.py        # /api/health — liveness, readiness, build info
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
├── tests/                      # Pytest test suite
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures, temp DB, test client
│   ├── test_auth.py            # Auth tests (15 tests)
│   ├── test_wardrobe.py        # Wardrobe CRUD tests (12 tests)
│   ├── test_health.py          # Health endpoint tests (10 tests)
│   └── test_outfits.py         # Outfit + rate limiting tests (10 tests)
│
├── ai_model.py                 # AI orchestrator (autotag, suggestions, aura)
├── ai_matcher.py               # Advanced similarity matching engine
├── logger.py                   # Centralised logging config
├── main.py                     # FastAPI entry point + router registration
├── rate_limiter.py             # slowapi limiter instance + shared rate limit config
├── database.py                 # SQLite schema + helpers
├── auth_utils.py               # JWT authentication
├── schemas.py                  # Pydantic request/response schemas
├── backup.py                   # Automatic daily S3 backup (cron job on EC2)
├── watchdog.py                 # Server watchdog — restarts container if unresponsive
├── Dockerfile                  # Docker image for backend
├── pytest.ini                  # Pytest configuration
├── .dockerignore
└── .github/workflows/deploy.yml
```

---

## Testing

**47 tests** covering auth, wardrobe CRUD, health endpoints, outfit generation, and AI rate limiting.

```bash
pip install pytest httpx
pytest
```

| File | Tests | Coverage |
|---|---|---|
| `test_auth.py` | 15 | Register, login, duplicates, missing fields, token validation |
| `test_wardrobe.py` | 12 | CRUD, auth enforcement, cross-user isolation |
| `test_health.py` | 10 | Liveness, readiness, DB check, build info |
| `test_outfits.py` | 10 | Outfit CRUD, rate limit enforcement (429) |

Tests use a temporary SQLite database — your real database is never touched.

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
| `SAGEMAKER_ENDPOINT` | SageMaker endpoint name (default: `wya-fashionclip-serverless`) |
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

# Free disk space after rebuilds
sudo docker system prune -a -f
```

---

## CI/CD (GitHub Actions)

Push to `main` automatically triggers:
1. `deploy-backend` — SSH into EC2, rebuild Docker image, restart container (~2m 30s)
2. `deploy-frontend` — `npm run build` → S3 sync → CloudFront invalidation (~30s)

Workflow file: `.github/workflows/deploy.yml`

---

## SageMaker Endpoint

```bash
source venv/bin/activate
python3 deploy_fashionclip.py
```

**Diagnose SageMaker connectivity:**
```bash
pip3 install boto3 pillow --break-system-packages
python3 Test_sagemaker.py

# Test with a real garment image
python3 Test_sagemaker.py /path/to/garment.jpg
```
