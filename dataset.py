# dataset.py
# -*- coding: utf-8 -*-
"""
Dataset + NLP utilities for the Interview Q&A Chatbot.

Exports (kept identical to app.py expectations):
- load_dataset()
- detect_company(user_input, df)
- detect_profile(user_input, company, df)
- get_questions(company, profile, df)
- evaluate_answer(user_answer, correct_answer)
- ai_answer(prompt)

Notes:
- Reads dataset from DEFAULT_DATASET_PATHS. If not found, tries to parse an
  embedded CSV sample (so devs can run quickly).
- Company & Job Role are normalized to lowercase (matching is case-insensitive).
- Question/Answer columns supported: "Question 1..10" / "Answer 1..10".
- Answer evaluation combines keyword overlap + character-level similarity.
- Gemini fallback is optional and controlled via environment variables.
"""

from __future__ import annotations

import os
import re
import csv
import io
import json
import time
import math
import difflib
import logging
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_DATASET_PATHS = [
    "./data.csv",
    "./dataset.csv",
    "./data/data.csv",
    "/mnt/data/data.csv",
]

# Prefer environment variables (do NOT hardcode secrets).
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBJ6laUKqRn1hNy3PTmB3nbDGV1mEUiBqk").strip()
MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-preview-05-20").strip()
GEN_API_BASE = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

# Maximum number of Q/A pairs per row to read
MAX_QA_PAIRS = 10

# Optional aliases to make matching more forgiving
COMPANY_ALIASES: Dict[str, List[str]] = {
    "google": ["google llc", "alphabet", "g00gle"],
    "microsoft": ["msft", "microsoft corp", "m$"],
    "capgemini": ["cap", "cap gemini"],
    "wipro": [],
    "infosys": ["infy"],
}

ROLE_ALIASES: Dict[str, List[str]] = {
    "software engineer": ["sde", "swe", "software developer", "developer"],
    "data analyst": ["analyst - data", "business analyst (data)"],
    "full stack developer": ["fullstack developer", "full-stack developer", "full stack eng"],
    "hr marketing": ["hr-marketing", "hr (marketing)"],
    "ta specialist": ["talent acquisition specialist", "talent acquisition", "ta"],
    "campus recruiter": ["university recruiter", "campus hiring"],
}

# Basic logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _normalize(s: str) -> str:
    return (s or "").strip().lower()

def _tokenize(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())

def _best_fuzzy_match(
    text: str, candidates: List[str], cutoff: float = 0.72
) -> Optional[str]:
    if not candidates:
        return None
    best = difflib.get_close_matches(text, candidates, n=1, cutoff=cutoff)
    return best[0] if best else None

def _expand_aliases(base: List[str], alias_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for item in base:
        key = _normalize(item)
        out[key] = list({key, *(alias_map.get(key, []))})
    return out

def _extract_keywords(text: str, stopwords: Optional[set] = None) -> List[str]:
    words = _tokenize(text)
    stop = stopwords or {
        "the","is","a","an","to","of","and","or","in","on","for","by",
        "be","are","was","were","with","as","that","this","it","its","at",
        "from","into","about","over","under","between"
    }
    return [w for w in words if w not in stop and len(w) > 2]

# -----------------------------------------------------------------------------
# Embedded sample (fallback) – optional, used only if file not found
# -----------------------------------------------------------------------------

_EMBEDDED_CSV = """Company,Job Role,Tech/Non-Tech,Question 1,Answer 1,Question 2,Answer 2
Google,Software Engineer,Tech,What is the difference between list and tuple in Python?,Lists are mutable whereas tuples are immutable.,Explain polymorphism in OOP.,Polymorphism allows methods to do different things based on the object it is acting upon.
Google,Data Analyst,Tech,What is normalization in databases?,Normalization reduces redundancy and improves integrity.,Explain VLOOKUP in Excel.,VLOOKUP searches for a value in the first column and returns data in the same row from another column.
Google,Full Stack Developer,Tech,What is REST API?,REST API is an architectural style for distributed systems using stateless communication.,What is the difference between SQL and NoSQL?,"SQL is relational and structured, while NoSQL is non-relational and flexible."
Google,HR Marketing,Non-Tech,What is employer branding?,Employer branding is how a company promotes itself to potential employees.,What is the STAR method in interviews?,"STAR stands for Situation, Task, Action, and Result used to answer behavioral questions."
Google,TA Specialist,Non-Tech,What is talent acquisition?,Talent acquisition focuses on finding and acquiring skilled employees.,Difference between recruitment and selection?,"Recruitment attracts candidates, selection chooses the best fit."
Google,Campus Recruiter,Non-Tech,What qualities do you look for in fresh graduates?,"Adaptability, eagerness to learn, and problem-solving skills.",How do you handle mass hiring?,"By planning structured drives, screening tests, and automated interview tools."
Microsoft,Software Engineer,Tech,What is garbage collection in Java?,It is the process of automatically freeing memory by removing unused objects.,Explain cloud computing.,"Cloud computing delivers services like storage, servers, and software over the internet."
Microsoft,Data Analyst,Tech,What is regression analysis?,Regression estimates the relationship between variables.,Explain Power BI.,Power BI is a data visualization tool used for business intelligence.
Microsoft,HR Marketing,Non-Tech,What is employee engagement?,Employee engagement is the emotional commitment of employees towards their work.,Explain workforce diversity.,"It refers to hiring a workforce with varied backgrounds, experiences, and perspectives."
Capgemini,Software Engineer,Tech,What is Agile methodology?,Agile is an iterative approach to software development and project management.,What is DevOps?,DevOps integrates development and IT operations for faster delivery.
Capgemini,Campus Recruiter,Non-Tech,How do you assess campus candidates?,"Through aptitude tests, coding rounds, and group discussions.",What is onboarding?,Onboarding is the process of integrating new hires into the organization.
Wipro,Full Stack Developer,Tech,What is MVC architecture?,MVC stands for Model-View-Controller used in web applications.,Explain microservices.,Microservices is an architectural style dividing applications into independent services.
Wipro,HR Marketing,Non-Tech,What is succession planning?,Succession planning ensures roles are filled by skilled employees in the future.,Difference between HRM and HRD?,"HRM is administrative, HRD focuses on employee growth and development."
Infosys,Software Engineer,Tech,What is an operating system?,An OS is system software that manages computer hardware and software resources.,What is Big Data?,"Big Data refers to large datasets that require advanced tools to store, process, and analyze."
Infosys,TA Specialist,Non-Tech,How do you measure recruitment success?,"By time-to-hire, quality-of-hire, and retention rates.",Explain the importance of job descriptions.,They define roles clearly and help attract the right talent.
"""

# -----------------------------------------------------------------------------
# Step 1: Load Dataset
# -----------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """
    Try to load dataset from DEFAULT_DATASET_PATHS.
    If not found, fall back to embedded CSV so the app still works.

    Returns: DataFrame with normalized 'Company' and 'Job Role' columns.
    Raises: FileNotFoundError only if embedded fallback is disabled (not here).
    """
    df: Optional[pd.DataFrame] = None

    for path in DEFAULT_DATASET_PATHS:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path).fillna("")
                logging.info(f"Loaded dataset from: {path}")
                break
            except Exception as e:
                logging.warning(f"Failed to read {path}: {e}")

    if df is None:
        # Fallback to embedded CSV to keep dev flow smooth
        logging.warning("No dataset file found. Using embedded sample CSV.")
        df = pd.read_csv(io.StringIO(_EMBEDDED_CSV)).fillna("")

    # Normalize columns (case-insensitive matching)
    if "Company" not in df.columns or "Job Role" not in df.columns:
        raise ValueError("Dataset must contain 'Company' and 'Job Role' columns.")

    df["Company"] = df["Company"].astype(str).map(_normalize)
    df["Job Role"] = df["Job Role"].astype(str).map(_normalize)

    # Normalize Q/A columns (strip)
    for i in range(1, MAX_QA_PAIRS + 1):
        qcol, acol = f"Question {i}", f"Answer {i}"
        if qcol in df.columns:
            df[qcol] = df[qcol].astype(str).fillna("").map(str).map(lambda s: s.strip())
        if acol in df.columns:
            df[acol] = df[acol].astype(str).fillna("").map(str).map(lambda s: s.strip())

    return df

# -----------------------------------------------------------------------------
# Step 2: Detect Company & Profile
# -----------------------------------------------------------------------------

def detect_company(user_input: str, df: pd.DataFrame) -> Optional[str]:
    """
    Find company mentioned in user input using:
    1) alias-aware regex word-boundary match
    2) fuzzy fallback on full text
    Returns normalized company (lowercase) or None.
    """
    text = _normalize(user_input)

    companies = sorted(df["Company"].unique())
    alias_map = _expand_aliases(companies, COMPANY_ALIASES)

    # 1) regex matches against aliases
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return canonical

    # 2) fuzzy fallback
    # try the entire text vs company names and aliases
    all_aliases = list({a for aliases in alias_map.values() for a in aliases})
    best_alias = _best_fuzzy_match(text, all_aliases, cutoff=0.82)
    if best_alias:
        # map back to canonical
        for canonical, aliases in alias_map.items():
            if best_alias in aliases:
                return canonical

    # final try against canonical names
    return _best_fuzzy_match(text, companies, cutoff=0.82)

def detect_profile(user_input: str, company: str, df: pd.DataFrame) -> Optional[str]:
    """
    From the selected company, detect a job role present in the dataset.

    Returns normalized role (lowercase) or None.
    """
    text = _normalize(user_input)
    profiles = sorted(df[df["Company"] == _normalize(company)]["Job Role"].unique())
    alias_map = _expand_aliases(profiles, ROLE_ALIASES)

    # direct alias-aware regex
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return canonical

    # substring containment (for roles that are phrases)
    for canonical in profiles:
        if canonical in text:
            return canonical

    # fuzzy fallback
    return _best_fuzzy_match(text, profiles, cutoff=0.8)

# -----------------------------------------------------------------------------
# Step 3: Get Questions & Answers
# -----------------------------------------------------------------------------

def get_questions(company: str, profile: str, df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Return list of (question, answer) pairs for a given company & role.
    Supports up to MAX_QA_PAIRS columns Question i / Answer i.
    """
    company = _normalize(company)
    profile = _normalize(profile)

    subset = df[(df["Company"] == company) & (df["Job Role"] == profile)]
    pairs: List[Tuple[str, str]] = []

    for _, row in subset.iterrows():
        for i in range(1, MAX_QA_PAIRS + 1):
            q_col, a_col = f"Question {i}", f"Answer {i}"
            if q_col in row and a_col in row:
                q, a = (row[q_col] or "").strip(), (row[a_col] or "").strip()
                if q:
                    pairs.append((q, a))
    return pairs

# -----------------------------------------------------------------------------
# Step 4: Evaluate Answer
# -----------------------------------------------------------------------------

def _similarity_score(a: str, b: str) -> float:
    """
    Character-level similarity (0..1) using difflib.SequenceMatcher
    """
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _keyword_overlap_score(user: str, gold: str) -> float:
    """
    Keyword overlap Jaccard score (0..1)
    """
    uk = set(_extract_keywords(user))
    gk = set(_extract_keywords(gold))
    if not gk:
        return 0.0
    inter = len(uk & gk)
    union = len(uk | gk)
    return inter / max(1, union)

def evaluate_answer(user_answer: str, correct_answer: str) -> Dict[str, Any]:
    """
    Blend keyword overlap with character similarity for more robust feedback.
    Returns a dict with 'score' (0..1) and 'feedback' message.
    """
    if not correct_answer:
        return {
            "score": 0.0,
            "feedback": "⚠️ No reference answer available. Focus on clarity and key points."
        }

    sim = _similarity_score(user_answer, correct_answer)
    key = _keyword_overlap_score(user_answer, correct_answer)

    # Weighted blend (tuneable)
    score = round(0.55 * sim + 0.45 * key, 4)

    # Feedback tiers
    if score >= 0.75:
        feedback = "✅ Strong answer. You covered the core ideas and terminology."
    elif score >= 0.5:
        # find missing keywords (top 3)
        gold_keys = _extract_keywords(correct_answer)
        user_keys = set(_extract_keywords(user_answer))
        missing = [k for k in gold_keys if k not in user_keys][:3]
        extra = f" Consider adding: {', '.join(missing)}." if missing else ""
        feedback = f"🟡 Partial answer. You’re on the right track.{extra}"
    else:
        feedback = f"❌ Weak answer. Suggested points to cover: {correct_answer}"

    return {"score": score, "feedback": feedback}

# -----------------------------------------------------------------------------
# Step 5: Gemini Fallback
# -----------------------------------------------------------------------------

def ai_answer(prompt: str) -> str:
    """
    Optional Gemini fallback when:
    - No company detected
    - No role detected
    - Out-of-dataset questions
    Returns a short answer string. If API is not configured, returns a guidance string.
    """
    if not prompt or not prompt.strip():
        return "⚠️ Please provide a valid prompt."

    if not GOOGLE_API_KEY:
        # Soft fallback to keep app responsive without keys
        return ("ℹ️ AI fallback not configured (missing GOOGLE_API_KEY). "
                "Please specify company and profile, or set the API key in environment variables.")

    try:
        payload = {
            "contents": [{"parts": [{"text": prompt.strip()}]}],
            # You can add safetySettings / generationConfig if needed
        }
        url = f"{GEN_API_BASE}?key={GOOGLE_API_KEY}"
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Defensive parsing
        cand = (data.get("candidates") or [{}])[0]
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        text = (parts[0].get("text") if parts else "") or ""
        text = str(text).strip()
        return text if text else "⚠️ AI Error: Empty response from model."

    except requests.exceptions.Timeout:
        return "⚠️ AI Error: Request timed out. Try again."
    except requests.exceptions.HTTPError as e:
        try:
            details = e.response.json()
        except Exception:
            details = {"status": e.response.status_code, "text": e.response.text[:200]}
        return f"⚠️ AI Error: {details}"
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# -----------------------------------------------------------------------------
# Optional helpers (not required by app.py, but handy)
# -----------------------------------------------------------------------------

def list_companies(df: pd.DataFrame) -> List[str]:
    """Return sorted unique company names (normalized)."""
    return sorted(df["Company"].unique())

def list_roles(df: pd.DataFrame, company: Optional[str] = None) -> List[str]:
    """Return sorted roles; if company provided, filter by company."""
    if company:
        return sorted(df[df["Company"] == _normalize(company)]["Job Role"].unique())
    return sorted(df["Job Role"].unique())

def has_company_and_role(df: pd.DataFrame, company: str, role: str) -> bool:
    """Quick check used by UIs."""
    c, r = _normalize(company), _normalize(role)
    return not df[(df["Company"] == c) & (df["Job Role"] == r)].empty
