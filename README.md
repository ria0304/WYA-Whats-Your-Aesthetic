WYA – What’s Your Aesthetic
Overview

WYA (What’s Your Aesthetic) is a full-stack web application designed to help users discover, analyze, and refine their personal aesthetic style. The platform combines an interactive questionnaire with intelligent style analysis to generate personalized aesthetic insights and wardrobe recommendations.

The project follows a modern frontend–backend architecture, ensuring scalability, maintainability, and readiness for future feature expansion.

Key Features

Interactive aesthetic questionnaire

Personalized style analysis

Wardrobe and outfit categorization

Clean, responsive user interface

Modular and extensible backend architecture

Project Structure
WYA-WHAT-S--YOUR--AESTHETIC/
│
├── frontend/              # React frontend
│   ├── src/
│   ├── public/
│   └── package.json
│
├── backend/               # FastAPI backend
│   ├── ai_model.py
│   ├── auth_utils.py
│   ├── main.py
│   └── database.py
│
├── .env                   # Environment variables
├── README.md
└── requirements.txt
Future Enhancements

User authentication and profile management

AI-driven outfit and style recommendations


Run Locally
Prerequisites

Node.js

Python 3.10+

Frontend Setup

npm install

npm run dev

Backend Setup

pip install -r requirements.txt

python -m uvicorn main:app --reload
