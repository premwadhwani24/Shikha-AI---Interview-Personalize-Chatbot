# app.py
# -*- coding: utf-8 -*-
"""
Shikha - Dataset-first Interview Q&A Chatbot Backend (enhanced)
Features (summary):
 - Dataset-first Q&A from dataset.csv
 - TF-IDF indexing + fuzzy company detection
 - Fallback RAG + Gemini generative model
 - Sentiment analysis (Gemini structured schema fallback)
 - Voice: STT (upload audio) + basic prosody analysis + quality scoring
 - TTS: synthesize assistant responses (pyttsx3 fallback)
 - Reinforcement learning: contextual bandit guide + feedback endpoint to update policy
 - Interview coach flow: get questions, user answers (text/voice), automated evaluation & suggestions
 - Admin endpoints: load dataset, status, search, export-arff, reindex, stats, history, health
 - CLI mode
Notes:
 - This is a single-file backend. Adapt DB URL, API keys, file paths to your environment.
 - Dependencies (pip): flask flask-cors sqlalchemy pandas numpy scikit-learn requests rapidfuzz librosa soundfile pyttsx3 SpeechRecognition pydub
 - Native dependencies: ffmpeg (for pydub), optional pocketsphinx if offline STT is desired.
"""

import os
import io
import sys
import json
import uuid
import math
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2

# fuzzy match libs
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAS_RAPIDFUZZ = False

# audio + stt + tts libs
# install: pip install librosa soundfile pydub SpeechRecognition pyttsx3
try:
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    import speech_recognition as sr
    import pyttsx3
    _HAS_AUDIO_LIBS = True
except Exception:
    _HAS_AUDIO_LIBS = False

# ---------------------------
# Configuration (user provided defaults included)
# ---------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # if empty generative calls disabled
MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-preview-05-20")
GEN_API_BASE = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db.sqlite3")
STATIC_FOLDER = os.getenv("STATIC_FOLDER", "static")
DEFAULT_DATASET_PATHS = ["./data.csv", "/mnt/data/data.csv"]
DATASET_ANSWER_THRESHOLD = float(os.getenv("DATASET_ANSWER_THRESHOLD", 0.45))
COMPANY_FUZZY_THRESHOLD = float(os.getenv("COMPANY_FUZZY_THRESHOLD", 75.0))
COMPANY_FUZZY_FALLBACK = float(os.getenv("COMPANY_FUZZY_FALLBACK", 0.75))

# audio settings
AUDIO_UPLOAD_DIR = os.getenv("AUDIO_UPLOAD_DIR", "/tmp/shikha_audio")
os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("ShikhaBackend")

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="")
CORS(app)

# ---------------------------
# Database (SQLAlchemy)
# ---------------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    handle = Column(String, unique=True, index=True)
    sessions = relationship("ChatSession", back_populates="user")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String)  # "user" or "assistant"
    text = Column(Text)
    meta = Column(Text)  # JSON
    created_at = Column(DateTime, server_default=func.now())
    session = relationship("ChatSession", back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"))
    reward = Column(Integer)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class CorpusDoc(Base):
    __tablename__ = "corpus_docs"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    text = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


Index("idx_corpus_name_created", CorpusDoc.name, CorpusDoc.created_at)

Base.metadata.create_all(bind=engine)

# ---------------------------
# Utilities
# ---------------------------
def get_db():
    return SessionLocal()


def safe_json_load(s: Optional[str]) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def read_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def chunk_text(text: str, max_len: int = 1200) -> List[str]:
    text = text or ""
    if len(text) <= max_len:
        return [text]
    words = text.split()
    chunks = []
    buf = []
    cur = 0
    for w in words:
        if cur + len(w) + 1 > max_len:
            chunks.append(" ".join(buf))
            buf = [w]
            cur = len(w)
        else:
            buf.append(w)
            cur += len(w) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks

# ---------------------------
# Bandit (simple contextual)
# ---------------------------
DEFAULT_ARMS = {
    "concise": {"template": "Respond concisely, with 1-2 key points."},
    "detailed": {"template": "Provide a detailed, step-by-step response."},
    "casual": {"template": "Respond in a friendly, conversational tone."},
    "professional": {"template": "Maintain a formal, professional tone."},
}


class ContextualBandit:
    def __init__(self, arms: dict):
        self.arm_keys = list(arms.keys())
        self.k = len(self.arm_keys)
        self.epsilon = 0.1
        self.alpha = 0.15
        # small random weights for three-dim contexts
        self.weights = np.random.normal(scale=0.01, size=(self.k, 3))

    def select(self, context_vector: np.ndarray) -> str:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.arm_keys)
        scores = [np.dot(self.weights[i], context_vector) for i in range(self.k)]
        return self.arm_keys[int(np.argmax(scores))]

    def update(self, arm: str, context_vector: np.ndarray, reward: float):
        if arm not in self.arm_keys:
            return
        i = self.arm_keys.index(arm)
        pred = float(np.dot(self.weights[i], context_vector))
        err = float(reward - pred)
        self.weights[i] += self.alpha * err * context_vector


def build_features(text: str) -> np.ndarray:
    f = np.zeros(3, dtype=float)
    words = len((text or "").split())
    f[0] = 1.0 if words > 10 else 0.0
    ql = (text or "").lower()
    f[1] = 1.0 if "how to" in ql or "how do" in ql else 0.0
    f[2] = 1.0 if "explain" in ql or "what is" in ql else 0.0
    return f


BANDIT = ContextualBandit(DEFAULT_ARMS)

# ---------------------------
# RAG TF-IDF (existing corpus)
# ---------------------------
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
corpus_texts: List[str] = []
corpus_fitted = False


def refresh_corpus(db_session):
    global corpus_texts, corpus_fitted
    docs = db_session.query(CorpusDoc).order_by(CorpusDoc.id).all()
    corpus_texts = [d.text for d in docs if d.text and d.text.strip()]
    if corpus_texts:
        try:
            vectorizer.fit(corpus_texts)
            corpus_fitted = True
            logger.info("TF-IDF vectorizer fitted on %d docs", len(corpus_texts))
        except Exception as e:
            corpus_fitted = False
            logger.exception("Failed to fit TF-IDF for corpus: %s", e)
    else:
        corpus_fitted = False
        logger.info("No corpus documents to fit for RAG.")


def retrieve_contexts(query: str, top_k: int = 3, threshold: float = 0.03) -> List[str]:
    if not corpus_fitted or not corpus_texts:
        return []
    try:
        q_vec = vectorizer.transform([query])
        docs_vec = vectorizer.transform(corpus_texts)
        sims = cosine_similarity(q_vec, docs_vec)[0]
        idx = np.argsort(sims)[::-1]
        results = []
        for i in idx:
            if len(results) >= top_k:
                break
            if sims[i] >= threshold:
                results.append(corpus_texts[int(i)])
        return results
    except Exception as e:
        logger.exception("retrieve_contexts error: %s", e)
        return []

# ---------------------------
# Dataset indexing (company -> questions)
# ---------------------------
dataset_df: Optional[pd.DataFrame] = None
dataset_docs: List[str] = []
dataset_vectorizer: Optional[TfidfVectorizer] = None
dataset_doc_meta: List[Dict[str, Any]] = []
dataset_loaded = False

company_to_questions: Dict[str, List[str]] = {}
company_list_sorted: List[str] = []


def normalize_company_name(s: str) -> str:
    return str(s).strip().lower()


def build_company_questions_map(df: pd.DataFrame, company_col_index: int = 0) -> Dict[str, List[str]]:
    mapping = {}
    for _, row in df.iterrows():
        company = row.iloc[company_col_index] if len(row) > company_col_index else ""
        if pd.isna(company) or str(company).strip() == "":
            continue
        c = normalize_company_name(company)
        questions = []
        for q in row.iloc[company_col_index + 1:]:
            if pd.isna(q):
                continue
            qs = str(q).strip()
            if qs:
                questions.append(qs)
        if questions:
            mapping[c] = questions
    return mapping


def compose_dataset_doc_from_row(row: pd.Series) -> str:
    parts = []
    if len(row) > 0:
        parts.append(f"Company: {row.iloc[0]}")
    for v in row.iloc[1:]:
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).strip())
    return "\n".join(parts)


def load_and_index_dataset(path: str, truncate_existing: bool = False) -> Tuple[int, str]:
    """
    Load dataset.csv where first column is company and remaining columns are questions.
    Builds:
      - dataset_df
      - dataset_docs (text blobs)
      - dataset_vectorizer (tfidf)
      - company_to_questions map
    Returns (num_docs_indexed, message)
    """
    global dataset_df, dataset_docs, dataset_vectorizer, dataset_doc_meta, dataset_loaded, company_to_questions, company_list_sorted
    if not path or not os.path.exists(path):
        return 0, f"dataset path not found: {path}"
    try:
        df = pd.read_csv(path, header=0, dtype=str)
    except Exception as e:
        logger.exception("Failed to read dataset CSV: %s", e)
        return 0, f"failed to read CSV: {e}"

    dataset_df = df.fillna("")
    dataset_docs = []
    dataset_doc_meta = []

    for idx, row in dataset_df.iterrows():
        doc_text = compose_dataset_doc_from_row(row)
        if not doc_text.strip():
            continue
        dataset_docs.append(doc_text)
        meta = {
            "index": int(idx),
            "company": normalize_company_name(row.iloc[0]) if len(row) > 0 else "",
            "raw": row.to_dict()
        }
        dataset_doc_meta.append(meta)

    company_to_questions = build_company_questions_map(dataset_df, company_col_index=0)
    company_list_sorted = sorted(company_to_questions.keys(), key=lambda x: len(x), reverse=True)

    if not dataset_docs:
        dataset_vectorizer = None
        dataset_loaded = False
        return 0, "no valid docs in dataset"

    try:
        dv = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
        dv.fit(dataset_docs)
        dataset_vectorizer = dv
        dataset_loaded = True
        logger.info("Dataset TF-IDF fitted on %d docs", len(dataset_docs))
        return len(dataset_docs), "loaded"
    except Exception as e:
        dataset_vectorizer = None
        dataset_loaded = False
        logger.exception("Failed to build dataset TF-IDF: %s", e)
        return 0, f"vectorizer error: {e}"


def dataset_query_match(query: str, top_k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    if not dataset_loaded or not dataset_vectorizer or not dataset_docs:
        return []
    try:
        qv = dataset_vectorizer.transform([query])
        docs_v = dataset_vectorizer.transform(dataset_docs)
        sims = cosine_similarity(qv, docs_v)[0]
        idxs = np.argsort(sims)[::-1]
        results = []
        for i in idxs[:top_k]:
            results.append((float(sims[i]), dataset_doc_meta[int(i)]))
        return results
    except Exception as e:
        logger.exception("dataset_query_match error: %s", e)
        return []

# ---------------------------
# Company detection (exact + fuzzy)
# ---------------------------
def detect_company_exact(user_text: str) -> Optional[str]:
    text = user_text.lower()
    for company in company_list_sorted:
        if not company:
            continue
        # whole-word check
        if f" {company} " in f" {text} ":
            return company
        if text.startswith(company + " ") or text.endswith(" " + company) or text == company:
            return company
    return None


def detect_company_fuzzy(user_text: str, top_n: int = 1) -> Optional[str]:
    """
    Detects a company name within a longer text using fuzzy matching on individual words.
    """
    if not company_list_sorted:
        return None

    text = user_text.lower()
    exact = detect_company_exact(user_text)
    if exact:
        return exact

    user_words = [w.strip() for w in text.split() if w.strip()]

    if _HAS_RAPIDFUZZ:
        best_match = None
        best_score = 0
        for word in user_words:
            matches = rf_process.extractOne(word, company_list_sorted, scorer=rf_fuzz.ratio)
            if matches and matches[1] > best_score and matches[1] >= COMPANY_FUZZY_THRESHOLD:
                best_score = matches[1]
                best_match = matches[0]
        return best_match
    else:
        for word in user_words:
            bests = difflib.get_close_matches(word, company_list_sorted, n=1, cutoff=COMPANY_FUZZY_FALLBACK)
            if bests:
                return bests[0]
        return None

# ---------------------------
# Generative language helpers (Gemini) - fallback only
# ---------------------------
def call_generative_model(contents: List[Dict[str, Any]], timeout: int = 60) -> Dict[str, Any]:
    """
    Make POST to Google Generative endpoint - minimal wrapper.
    contents: list of role/parts dicts exactly like the Gemini v1 format.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not provided.")
    url = f"{GEN_API_BASE}?key={GOOGLE_API_KEY}"
    logger.debug("Calling Generative API: %s", url)
    resp = requests.post(url, json={"contents": contents}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_text_from_generative_response(resp_json: Dict[str, Any]) -> str:
    try:
        candidates = resp_json.get("candidates", [])
        if not candidates:
            if "output" in resp_json:
                return str(resp_json["output"])
            raise ValueError("No candidates in response.")
        cont = candidates[0].get("content", {})
        parts = cont.get("parts", [])
        if parts and isinstance(parts, list) and "text" in parts[0]:
            return parts[0]["text"]
        texts = []
        for part in parts:
            t = part.get("text")
            if t:
                texts.append(str(t))
        return "\n".join(texts)
    except Exception as e:
        logger.exception("Failed to extract text: %s", e)
    return ""


def analyze_sentiment(text: str) -> Optional[str]:
    """Uses Gemini to get sentiment of a text via JSON schema output (best-effort)."""
    if not GOOGLE_API_KEY:
        logger.warning("No API key for sentiment analysis.")
        return None

    prompt = f"Analyze the sentiment of the following text: '{text}'. Respond with a single word: 'positive', 'negative', or 'neutral'."

    contents = [{
        "role": "user",
        "parts": [{"text": prompt}]
    }]

    try:
        payload = {
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "sentiment": {"type": "STRING", "enum": ["positive", "negative", "neutral"]}
                    },
                    "propertyOrdering": ["sentiment"]
                }
            }
        }
        url = f"{GEN_API_BASE}?key={GOOGLE_API_KEY}"
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        resp_json = resp.json()
        part = resp_json.get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        json_text = part.get("text")
        if json_text:
            sentiment_data = json.loads(json_text)
            return sentiment_data.get("sentiment", "neutral")
        return "neutral"
    except Exception as e:
        logger.error("Sentiment analysis failed: %s", e)
        return "neutral"

# ---------------------------
# Audio: STT / TTS / Analysis
# ---------------------------
def save_uploaded_audio(file_storage) -> str:
    """
    Save uploaded file (Flask FileStorage) to disk and convert to wav 16k mono using pydub if required.
    Returns path to saved wav file.
    """
    if not _HAS_AUDIO_LIBS:
        raise RuntimeError("Audio libraries not available. Install librosa, pydub, soundfile, SpeechRecognition, pyttsx3.")

    filename = f"audio_{uuid.uuid4().hex}.wav"
    out_path = os.path.join(AUDIO_UPLOAD_DIR, filename)
    tmp_path = os.path.join(AUDIO_UPLOAD_DIR, f"tmp_{uuid.uuid4().hex}")
    file_storage.save(tmp_path)
    try:
        # normalize to wav 16k mono
        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(out_path, format="wav")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return out_path


def transcribe_audio_wav(wav_path: str) -> str:
    """
    Use SpeechRecognition with Google's free API as default (note: has usage limits).
    If you want offline STT, configure pocketsphinx and adjust code.
    """
    if not _HAS_AUDIO_LIBS:
        raise RuntimeError("Audio libs missing.")
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    try:
        # Uses Google Web Speech API (requires internet); for offline use pocketsphinx
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        logger.error("STT request error: %s", e)
        return ""


def synthesize_text_to_speech_bytes(text: str) -> bytes:
    """
    Use pyttsx3 to synthesize audio to a temporary file, then return bytes.
    pyttsx3 is offline but platform-dependent; for production prefer a cloud TTS with better voices.
    """
    if not _HAS_AUDIO_LIBS:
        raise RuntimeError("Audio libs missing.")
    engine = pyttsx3.init()
    tmp_file = os.path.join(AUDIO_UPLOAD_DIR, f"tts_{uuid.uuid4().hex}.wav")
    engine.save_to_file(text, tmp_file)
    engine.runAndWait()
    # read bytes
    data = open(tmp_file, "rb").read()
    try:
        os.remove(tmp_file)
    except Exception:
        pass
    return data


def analyze_speech_prosody(wav_path: str) -> Dict[str, Any]:
    """
    Extract simple prosodic features using librosa:
      - duration, speaking_rate (approx words/sec using STT), mean_energy, mean_pitch
    Returns a dictionary of features and a human-friendly evaluation.
    """
    if not _HAS_AUDIO_LIBS:
        raise RuntimeError("Audio libs missing.")
    y, sr_rate = librosa.load(wav_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr_rate)
    # Short-term energy (RMS)
    rms = librosa.feature.rms(y=y).flatten()
    mean_energy = float(np.mean(rms)) if len(rms) else 0.0
    # Estimate pitch using librosa.pyin if available (may be slow)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                                     fmax=librosa.note_to_hz('C7'))
        voiced = f0[~np.isnan(f0)]
        mean_pitch = float(np.mean(voiced)) if len(voiced) else 0.0
    except Exception:
        mean_pitch = 0.0

    return {
        "duration_sec": float(duration),
        "mean_energy": mean_energy,
        "mean_pitch_hz": mean_pitch,
    }


def evaluate_spoken_answer(transcript: str, wav_path: Optional[str] = None, reference_answer: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate the spoken answer (or text answer) and return:
      - text_score: semantic similarity against reference (if provided) using TF-IDF cosine
      - prosody_score: if wav_path provided (speaking rate, energy), measured against heuristics
      - overall_score: weighted average
      - feedback: textual suggestions
    """
    feedback = []
    text_score = None
    prosody_score = None

    if reference_answer and transcript:
        # use dataset_vectorizer if present; fallback to simple fuzzy ratio
        try:
            if dataset_vectorizer and dataset_docs:
                vec_q = dataset_vectorizer.transform([transcript])
                vec_ref = dataset_vectorizer.transform([reference_answer])
                sim = cosine_similarity(vec_q, vec_ref)[0][0]
                text_score = float(sim)
            else:
                # fallback: simple token overlap ratio
                set_a = set(transcript.lower().split())
                set_b = set(reference_answer.lower().split())
                if not set_b:
                    text_score = 0.0
                else:
                    text_score = float(len(set_a & set_b) / max(1, len(set_b)))
        except Exception as e:
            logger.exception("text scoring failed: %s", e)
            text_score = 0.0

    if wav_path:
        pros = analyze_speech_prosody(wav_path)
        duration = pros.get("duration_sec", 0.0)
        # speaking rate: approximate words / sec (use transcript)
        words = len((transcript or "").split())
        rate = words / duration if duration > 0.01 else 0.0
        prosody_score = 0.5
        # heuristics:
        # - preferred rate between 1.5 and 3.2 wps
        if 1.5 <= rate <= 3.2:
            prosody_score += 0.3
        elif 1.0 <= rate < 1.5 or 3.2 < rate <= 4.0:
            prosody_score += 0.15
        # energy heuristics
        energy = pros.get("mean_energy", 0.0)
        if energy > 0.02:
            prosody_score += 0.2
        # clamp
        prosody_score = min(1.0, prosody_score)
        # feedback
        if rate < 1.5:
            feedback.append("Try to speak a bit faster to sound more confident.")
        elif rate > 3.5:
            feedback.append("Slow down slightly to improve clarity.")
        if energy < 0.01:
            feedback.append("Increase vocal projection (speak louder).")

    # combine scores
    if text_score is None and prosody_score is None:
        overall = 0.0
    elif text_score is None:
        overall = prosody_score
    elif prosody_score is None:
        overall = text_score
    else:
        overall = 0.7 * text_score + 0.3 * prosody_score

    if overall >= 0.8:
        quality = "excellent"
    elif overall >= 0.6:
        quality = "good"
    elif overall >= 0.4:
        quality = "fair"
    else:
        quality = "needs_improvement"

    return {
        "transcript": transcript,
        "text_score": float(text_score) if text_score is not None else None,
        "prosody_score": float(prosody_score) if prosody_score is not None else None,
        "overall_score": float(overall),
        "quality": quality,
        "suggestions": feedback
    }

# ---------------------------
# Routes: static, dataset, admin, chat, voice, feedback, coach
# ---------------------------
SYSTEM_PROMPT = "You are Shikha, an expert interview coach. Provide concise, helpful interview answers tailored to the user's query."

@app.route("/", methods=["GET"])
def root():
    # serve index if present
    try:
        return send_from_directory(app.static_folder, "index.html")
    except Exception:
        return jsonify({"ok": True, "server": "Shikha backend running."})


@app.route("/api/dataset/load", methods=["POST"])
def api_dataset_load():
    if "file" in request.files:
        f = request.files["file"]
        tmp = f"/tmp/dataset-upload-{uuid.uuid4().hex}.csv"
        f.save(tmp)
        try:
            count, msg = load_and_index_dataset(tmp)
            return jsonify({"ok": True, "rows_indexed": count, "message": msg})
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
    body = request.get_json(force=True, silent=True) or {}
    path = body.get("path") or read_first_existing(DEFAULT_DATASET_PATHS)
    if not path:
        return jsonify({"ok": False, "error": "No dataset path provided or found."}), 400
    count, msg = load_and_index_dataset(path)
    if count == 0:
        return jsonify({"ok": False, "rows_indexed": 0, "message": msg}), 400
    return jsonify({"ok": True, "rows_indexed": count, "message": msg})


@app.route("/api/dataset/status", methods=["GET"])
def api_dataset_status():
    return jsonify({
        "ok": True,
        "dataset_loaded": bool(dataset_loaded),
        "num_docs": len(dataset_docs) if dataset_docs else 0,
        "num_companies": len(company_to_questions),
        "sample_companies": list(company_to_questions.keys())[:20]
    })


@app.route("/api/search_dataset", methods=["GET"])
def api_search_dataset():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing q"}), 400
    k = int(request.args.get("k", 3))
    matches = dataset_query_match(q, top_k=k)
    out = [{"score": s, "company": m.get("company"), "raw": m.get("raw")} for s, m in matches]
    return jsonify({"ok": True, "results": out})


@app.route("/api/export-arff", methods=["POST"])
def api_export_arff():
    payload = request.get_json(force=True, silent=True) or {}
    path = payload.get("path") or read_first_existing(DEFAULT_DATASET_PATHS)
    if not path:
        return jsonify({"ok": False, "error": "dataset.csv not found"}), 404
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        return jsonify({"ok": False, "error": f"read failed: {e}"}), 400

    first_col = df.columns[0]
    question_cols = df.columns.tolist()[1:]
    tbl = pd.DataFrame()
    tbl["company"] = df[first_col].astype(str)
    for i, col in enumerate(question_cols):
        tbl[f"q{i+1}_len"] = df[col].astype(str).map(len)
    tbl["Target"] = df[first_col].astype(str)

    encoders = {}
    for col in tbl.columns:
        if not pd.api.types.is_numeric_dtype(tbl[col]):
            le = LabelEncoder()
            tbl[col] = le.fit_transform(tbl[col].astype(str))
            encoders[col] = le

    k = int(payload.get("k", -1))
    if k > 0 and tbl.shape[1] > 1:
        X = tbl.iloc[:, :-1]
        y = tbl.iloc[:, -1]
        ksel = min(k, X.shape[1])
        selector = SelectKBest(score_func=chi2, k=ksel)
        X_sel = selector.fit_transform(X, y)
        selected_cols = X.columns[selector.get_support(indices=True)]
        final_df = pd.DataFrame(X_sel, columns=selected_cols)
        final_df["Target"] = y.values
    else:
        final_df = tbl.copy()

    def df_to_arff_text(df_final, relation="dataset"):
        lines = [f"@RELATION {relation}"]
        for col in df_final.columns:
            lines.append(f"@ATTRIBUTE {col} NUMERIC")
        lines.append("@DATA")
        for _, row in df_final.iterrows():
            lines.append(",".join(str(v) for v in row.values.tolist()))
        return "\n".join(lines)

    arff_text = df_to_arff_text(final_df)
    out_name = payload.get("filename", "dataset_processed.arff")
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(arff_text)

    return jsonify({"ok": True, "file": os.path.abspath(out_name), "rows": final_df.shape[0], "cols": final_df.shape[1]})


@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    db = get_db()
    try:
        refresh_corpus(db)
        return jsonify({"ok": True, "docs_indexed": len(corpus_texts)})
    except Exception as e:
        logger.exception("reindex failed")
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/history", methods=["GET"])
def api_history():
    user_handle = request.args.get("user", "guest")
    db = get_db()
    try:
        user = db.query(User).filter_by(handle=user_handle).first()
        if not user:
            return jsonify({"ok": True, "sessions": []})
        sessions = db.query(ChatSession).filter_by(user_id=user.id).order_by(ChatSession.created_at.desc()).all()
        out = []
        for s in sessions:
            msgs = db.query(Message).filter_by(session_id=s.id).order_by(Message.created_at.asc()).all()
            out.append({
                "session_id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "messages": [{"id": m.id, "role": m.role, "text": m.text, "meta": safe_json_load(m.meta), "created_at": m.created_at.isoformat() if m.created_at else None} for m in msgs]
            })
        return jsonify({"ok": True, "sessions": out})
    except Exception as e:
        logger.exception("api_history failed")
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/stats", methods=["GET"])
def api_stats():
    db = get_db()
    try:
        users = db.query(User).count()
        sessions = db.query(ChatSession).count()
        msgs = db.query(Message).count()
        docs = db.query(CorpusDoc).count()
        return jsonify({
            "ok": True,
            "users": users,
            "sessions": sessions,
            "messages": msgs,
            "corpus_docs": docs,
            "dataset_loaded": bool(dataset_loaded),
            "dataset_docs": len(dataset_docs) if dataset_docs else 0,
            "num_companies": len(company_to_questions)
        })
    finally:
        db.close()


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat(), "audio_libs": _HAS_AUDIO_LIBS})


# ---------------------------
# Chat endpoint (dataset-first logic)
# ---------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(force=True, silent=True) or {}
    user_handle = payload.get("user", "guest")
    session_id = payload.get("session_id")
    message_text = (payload.get("message") or "").strip()
    mode = payload.get("mode", "auto")

    if not message_text:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    # Perform sentiment analysis on the user's message
    user_sentiment = analyze_sentiment(message_text)

    # detect company
    company = detect_company_exact(message_text)
    if not company:
        company = detect_company_fuzzy(message_text)

    if company and company in company_to_questions:
        questions = company_to_questions.get(company, [])
        # For coaching: also return a sample answer for each question (asked by user later)
        sample_answers = []
        for q in questions:
            # Create a prompt to generate a concise sample answer (but avoid calling Gemini here to keep dataset-only)
            sample_answers.append("This is a suggested concise answer. (enable generative model to improve)")
        answer_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        db = get_db()
        try:
            session_obj = None
            if session_id:
                session_obj = db.query(ChatSession).filter_by(id=session_id).first()
            if not session_obj:
                user = db.query(User).filter_by(handle=user_handle).first()
                if not user:
                    user = User(handle=user_handle)
                    db.add(user)
                    db.commit()
                    db.refresh(user)
                session_obj = ChatSession(user_id=user.id, title=f"Session {datetime.utcnow().isoformat()}")
                db.add(session_obj)
                db.commit()
                db.refresh(session_obj)

            user_meta = {"mode": mode, "dataset_matched": True, "company": company, "sentiment": user_sentiment}
            m_user = Message(session_id=session_obj.id, role="user", text=message_text, meta=json.dumps(user_meta))
            db.add(m_user)
            db.commit()
            db.refresh(m_user)

            meta_for_bot = {"source": "dataset", "company": company}
            m_bot = Message(session_id=session_obj.id, role="assistant", text=answer_text, meta=json.dumps(meta_for_bot))
            db.add(m_bot)
            db.commit()
            db.refresh(m_bot)

            return jsonify({
                "ok": True,
                "answer": answer_text,
                "source": "dataset",
                "company": company,
                "session_id": session_obj.id,
                "message_id": m_bot.id,
                "sample_answers": sample_answers
            })
        except Exception as e:
            logger.exception("Failed to persist dataset-based reply: %s", e)
            return jsonify({"ok": True, "answer": answer_text, "note": "dataset answer returned but DB save failed"}), 200
        finally:
            db.close()

    # fallback: RAG + generative
    db = get_db()
    try:
        session_obj = None
        if session_id:
            session_obj = db.query(ChatSession).filter_by(id=session_id).first()
        if not session_obj:
            user = db.query(User).filter_by(handle=user_handle).first()
            if not user:
                user = User(handle=user_handle)
                db.add(user)
                db.commit()
                db.refresh(user)
            session_obj = ChatSession(user_id=user.id, title=f"Session {datetime.utcnow().isoformat()}")
            db.add(session_obj)
            db.commit()
            db.refresh(session_obj)

        ctx = (mode + " " + message_text) if mode and mode != "auto" else message_text
        x = build_features(ctx)
        arm = BANDIT.select(x)
        style_template = DEFAULT_ARMS.get(arm, {}).get("template", "")

        retrievals = retrieve_contexts(message_text, top_k=3, threshold=0.03)

        contents = []
        contents.append({"role": "user", "parts": [{"text": SYSTEM_PROMPT}]})
        contents.append({"role": "user", "parts": [{"text": style_template}]})
        if retrievals:
            chunks_text = "\n\n".join([f"Context {i+1}:\n{r}" for i, r in enumerate(retrievals)])
            contents.append({"role": "user", "parts": [{"text": "Relevant context:\n" + chunks_text}]})
        contents.append({"role": "user", "parts": [{"text": message_text}]})

        try:
            resp_json = call_generative_model(contents) if GOOGLE_API_KEY else {"candidates": [{"content": {"parts": [{"text": "Generative model disabled (no GOOGLE_API_KEY)."}]}}]}
            answer_text = extract_text_from_generative_response(resp_json)
            if not answer_text:
                answer_text = "Model returned empty answer."
        except Exception as e:
            logger.exception("Model call failed: %s", e)
            return jsonify({"ok": False, "error": f"Model call failed: {e}"}), 500

        user_meta = {"mode": mode, "sentiment": user_sentiment}
        m_user = Message(session_id=session_obj.id, role="user", text=message_text, meta=json.dumps(user_meta))
        db.add(m_user)
        db.commit()
        db.refresh(m_user)

        meta_for_bot = {"arm": arm, "ctx": x.tolist(), "retrieval_used": bool(retrievals)}
        m_bot = Message(session_id=session_obj.id, role="assistant", text=answer_text, meta=json.dumps(meta_for_bot))
        db.add(m_bot)
        db.commit()
        db.refresh(m_bot)

        return jsonify({
            "ok": True,
            "answer": answer_text,
            "source": "generative",
            "session_id": session_obj.id,
            "message_id": m_bot.id,
            "style_used": arm
        })
    except Exception as e:
        logger.exception("api_chat failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        db.close()

# ---------------------------
# Voice endpoints
# ---------------------------
@app.route("/api/voice/recognize", methods=["POST"])
def api_voice_recognize():
    """
    Upload an audio file (form-data 'file') and returns transcript + audio analysis.
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "Missing file"}), 400
    if not _HAS_AUDIO_LIBS:
        return jsonify({"ok": False, "error": "Server missing audio libs. Install librosa, pydub, SpeechRecognition."}), 500

    f = request.files["file"]
    try:
        wav_path = save_uploaded_audio(f)
        transcript = transcribe_audio_wav(wav_path)
        analysis = analyze_speech_prosody(wav_path)
        return jsonify({"ok": True, "transcript": transcript, "analysis": analysis, "wav_path": wav_path})
    except Exception as e:
        logger.exception("voice recognize failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/voice/speak", methods=["POST"])
def api_voice_speak():
    """
    POST JSON { text: "..." } -> returns synthesized wav audio bytes as attachment
    """
    body = request.get_json(force=True, silent=True) or {}
    text = body.get("text", "")
    if not text:
        return jsonify({"ok": False, "error": "Missing text"}), 400

    if not _HAS_AUDIO_LIBS:
        return jsonify({"ok": False, "error": "TTS unavailable (missing pyttsx3)."}), 500
    try:
        audio_bytes = synthesize_text_to_speech_bytes(text)
        return send_file(io.BytesIO(audio_bytes), mimetype="audio/wav", as_attachment=False, download_name="response.wav")
    except Exception as e:
        logger.exception("TTS failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------------------------
# Interview coach flow endpoints
# ---------------------------
@app.route("/api/coach/get_questions", methods=["POST"])
def api_coach_get_questions():
    """
    Request body:
      { user: "prem", company: "google", k: 5 }
    Returns a list of k questions (dataset-first, otherwise generated)
    """
    body = request.get_json(force=True, silent=True) or {}
    user = body.get("user", "guest")
    company = body.get("company", "").strip().lower()
    k = int(body.get("k", 5))

    if company:
        comp_norm = normalize_company_name(company)
        if comp_norm in company_to_questions:
            qs = company_to_questions.get(comp_norm, [])[:k]
            return jsonify({"ok": True, "company": comp_norm, "questions": qs})
    # fallback: generate sample questions using generative model
    prompt = f"Give me {k} common interview questions for preparing at {company or 'general tech interviews'}."
    contents = [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}, {"role": "user", "parts": [{"text": prompt}]}]
    try:
        resp_json = call_generative_model(contents) if GOOGLE_API_KEY else {"candidates": [{"content": {"parts": [{"text": "Generative disabled; provide own questions."}]}}]}
        qtext = extract_text_from_generative_response(resp_json)
        # try to split into lines
        questions = [ln.strip() for ln in qtext.splitlines() if ln.strip()][:k]
        return jsonify({"ok": True, "company": company, "questions": questions})
    except Exception as e:
        logger.exception("coach get questions failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/coach/submit_answer", methods=["POST"])
def api_coach_submit_answer():
    """
    Submit an answer (text or voice) to a question for evaluation.
    body: JSON with:
      - user (optional), session_id (optional), question_text, answer_text (optional), audio_file (multipart optional)
    If 'file' in multipart - prefer uploaded audio; otherwise use answer_text.
    Returns evaluation with scores and suggestions.
    """
    # accept multipart/form-data or json
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        # multipart: fields plus file
        user = request.form.get("user", "guest")
        session_id = request.form.get("session_id")
        question_text = request.form.get("question_text", "")
        file = request.files.get("file")
        answer_text = request.form.get("answer_text", "")
    else:
        body = request.get_json(force=True, silent=True) or {}
        user = body.get("user", "guest")
        session_id = body.get("session_id")
        question_text = body.get("question_text", "")
        answer_text = body.get("answer_text", "")
        file = None

    if not question_text:
        return jsonify({"ok": False, "error": "Missing question_text"}), 400

    wav_path = None
    transcript = answer_text
    try:
        if file:
            if not _HAS_AUDIO_LIBS:
                return jsonify({"ok": False, "error": "Server missing audio libs."}), 500
            wav_path = save_uploaded_audio(file)
            transcript = transcribe_audio_wav(wav_path)
        # Evaluate: use reference if we can create one (from dataset or generative)
        reference = None
        # if company mentioned in question_text, try to find row in dataset with same question -> use neighbor column as sample answer (best-effort)
        # Simpler: call generative model for a sample answer (short)
        try:
            prompt = f"Provide a concise model answer to the interview question: {question_text}"
            contents = [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}, {"role": "user", "parts": [{"text": prompt}],}]
            resp_json = call_generative_model(contents) if GOOGLE_API_KEY else None
            if resp_json:
                reference = extract_text_from_generative_response(resp_json)
        except Exception:
            reference = None

        eval_res = evaluate_spoken_answer(transcript, wav_path=wav_path, reference_answer=reference)
        # persist message and assistant evaluation in DB
        db = get_db()
        try:
            user_obj = db.query(User).filter_by(handle=user).first()
            if not user_obj:
                user_obj = User(handle=user)
                db.add(user_obj)
                db.commit()
                db.refresh(user_obj)
            session_obj = None
            if session_id:
                session_obj = db.query(ChatSession).filter_by(id=int(session_id)).first()
            if not session_obj:
                session_obj = ChatSession(user_id=user_obj.id, title=f"Coach {datetime.utcnow().isoformat()}")
                db.add(session_obj)
                db.commit()
                db.refresh(session_obj)
            # save user message
            user_meta = {"type": "answer_submitted", "question": question_text}
            m_user = Message(session_id=session_obj.id, role="user", text=transcript or answer_text, meta=json.dumps(user_meta))
            db.add(m_user)
            db.commit()
            db.refresh(m_user)
            # save assistant evaluation
            bot_meta = {"type": "evaluation", "result": eval_res}
            m_bot = Message(session_id=session_obj.id, role="assistant", text=json.dumps(eval_res), meta=json.dumps(bot_meta))
            db.add(m_bot)
            db.commit()
            db.refresh(m_bot)
            return jsonify({"ok": True, "evaluation": eval_res, "session_id": session_obj.id})
        finally:
            db.close()
    except Exception as e:
        logger.exception("coach submit failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------------------------
# Feedback endpoint for RL
# ---------------------------
@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """
    Accept user feedback to update the bandit policy.
    Body: { user, session_id, message_id, reward (0..1), note (optional), arm_used (optional), ctx (optional array) }
    """
    body = request.get_json(force=True, silent=True) or {}
    reward = float(body.get("reward", 0.0))
    arm = body.get("arm")
    ctx = body.get("ctx")
    user = body.get("user", "guest")
    message_id = body.get("message_id")
    note = body.get("note", "")

    if not arm:
        return jsonify({"ok": False, "error": "Missing arm"}), 400
    try:
        # build context vector if provided else use zeros
        if ctx:
            ctx_arr = np.array(ctx, dtype=float)
        else:
            ctx_arr = np.zeros(3, dtype=float)
        BANDIT.update(arm, ctx_arr, reward)
        # persist feedback to DB
        db = get_db()
        try:
            fb = Feedback(message_id=message_id or 0, reward=int(round(reward * 100)), note=note)
            db.add(fb)
            db.commit()
            db.refresh(fb)
            return jsonify({"ok": True, "updated": True})
        finally:
            db.close()
    except Exception as e:
        logger.exception("feedback failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------
# CLI interactive loop for testing
# ---------------------------
def cli_chat_loop():
    print("Shikha Dataset-Aware Chat CLI")
    print("Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        company = detect_company_exact(user_input)
        if not company:
            company = detect_company_fuzzy(user_input)
        if company and company in company_to_questions:
            print(f"\n📌 Interview Questions for {company.capitalize()}:\n")
            for i, q in enumerate(company_to_questions[company], 1):
                print(f"{i}. {q}")
            print()
        else:
            print("No company match found; fallback to generative not invoked in CLI.\n")


# ---------------------------
# Startup loader
# ---------------------------
def startup_load():
    ds_path = read_first_existing(DEFAULT_DATASET_PATHS)
    if ds_path:
        count, msg = load_and_index_dataset(ds_path)
        logger.info("Dataset loaded at startup: %s rows=%d message=%s", ds_path, count, msg)
    else:
        logger.info("No dataset.csv found at default paths. Use /api/dataset/load to upload.")


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shiksha Dataset-Aware Backend (enhanced)")
    parser.add_argument("--cli", action="store_true", help="Start in CLI chat demo mode (no server)")
    parser.add_argument("--host", default="0.0.0.0", help="Flask host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    args = parser.parse_args()

    startup_load()

    if args.cli:
        logger.info("Starting CLI mode. Dataset loaded: %s companies=%d", bool(dataset_loaded), len(company_to_questions))
        cli_chat_loop()
        sys.exit(0)

    logger.info("Starting Flask server...")
    app.run(host=args.host, port=args.port, debug=True)
