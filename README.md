🌌 Shikha AI
Where Interview Prep Meets Intelligence
A dataset-first Interview Q&A Chatbot backend with fallback generative AI, voice evaluation, and reinforcement learning.

Flask SQLAlchemy Google Gemini Scikit-Learn Librosa Pyttsx3

✨ Features

🧠 Core Intelligence
Feature | Description
--- | ---
Dataset-First Q&A | Uses TF-IDF indexing and fuzzy matching to detect companies and provide exact, dataset-backed questions and answers.
Generative AI Fallback | Integrates with Google Gemini to dynamically generate interview questions and sample answers when the dataset falls short.
Sentiment Analysis | AI-powered sentiment scoring evaluates the emotional tone of user inputs.

🎙️ Voice & Audio Processing
Feature | Description
--- | ---
Speech-to-Text (STT) | Audio transcription using Google Web Speech API (via SpeechRecognition).
Text-to-Speech (TTS) | Offline voice synthesis using `pyttsx3` to speak out assistant responses.
Speech Prosody Analysis | Extracts audio features (pitch, energy, duration, speaking rate) using `librosa` to analyze speech quality.

📈 Interview Coaching & RL
Feature | Description
--- | ---
Automated Evaluation | Scores spoken/text answers based on semantic similarity (TF-IDF vs reference) and prosodic heuristics (speaking rate, energy).
Contextual Bandit RL | Reinforcement learning agent dynamically selects the best response tone (concise, detailed, casual, professional) based on user interaction context.
Feedback Loop | Endpoints to submit reward signals and iteratively update the bandit policy.

🛠️ Admin & Management
Feature | Description
--- | ---
Dataset Operations | Endpoints to load, index, search, and export CSV datasets (ARFF format support).
System Health | Endpoints to monitor dataset status, indexed documents, total sessions, and system health.

🏗️ Tech Stack
Shikha_AI/
├── ChatBot-main/
│   ├── app.py               # Main Flask Backend (Chat, Voice, Coach, Admin APIs)
│   ├── dataset.py           # Dataset related logic
│   ├── data.csv             # Knowledge base for interview QA
│   ├── requirements.txt     # Python dependencies
│   ├── db.sqlite3           # SQLite Database (SQLAlchemy)
│   ├── static/              # Frontend assets (HTML, CSS, JS, Images)
│   │   ├── index.html       # Main Chat Interface
│   │   ├── app.js           # Frontend logic
│   │   └── style.css        # Styling
│   └── audio_uploads/       # Temporary storage for processed audio

🚀 Getting Started
Prerequisites
- Python 3.10+
- ffmpeg (required by pydub for audio processing)
- A Google Gemini API key

1. Clone the repo
```bash
git clone <repository_url>
cd ChatBot-main
```

2. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```
API runs at http://localhost:5000
Frontend interface served at http://localhost:5000/

🔒 Environment Variables
Variable | Description | Required
--- | --- | ---
GOOGLE_API_KEY | API key for Gemini generative fallback & sentiment | Optional (Generative features disabled if empty)
GOOGLE_MODEL | Google model version (default: gemini-2.5-flash-preview-05-20) | Optional
DATABASE_URL | SQLAlchemy database URL (default: sqlite:///./db.sqlite3) | Optional
DATASET_ANSWER_THRESHOLD | Similarity threshold for dataset answers | Optional
COMPANY_FUZZY_THRESHOLD | Threshold for fuzzy company detection | Optional

📡 API Reference
Dataset & Admin
Method | Endpoint | Description
--- | --- | ---
GET | /api/dataset/status | Get dataset indexing status
POST | /api/dataset/load | Load and index `dataset.csv`
GET | /api/search_dataset | Search the indexed dataset
POST | /api/export-arff | Export dataset to ARFF format
POST | /api/reindex | Refresh TF-IDF corpus
GET | /api/history | Get chat history for a user
GET | /api/stats | Get system statistics
GET | /api/health | Check API health

Chat & Coaching
Method | Endpoint | Description
--- | --- | ---
POST | /api/chat | Main chat endpoint (dataset-first + Gemini fallback)
POST | /api/coach/get_questions | Get sample interview questions for a company
POST | /api/coach/submit_answer | Submit text/voice answer for automated evaluation
POST | /api/feedback | Submit reward feedback to update the RL bandit policy

Voice Processing
Method | Endpoint | Description
--- | --- | ---
POST | /api/voice/recognize | Upload audio for STT and prosody analysis
POST | /api/voice/speak | Synthesize TTS audio from text

🚢 Deployment
Component | Platform
--- | ---
Backend | Local / Any Python WSGI server (e.g., Gunicorn)
Database | SQLite (local)
Frontend | Served statically via Flask

🤝 Contributing
Pull requests are welcome! Please open an issue first to discuss what you'd like to change.
