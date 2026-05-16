from __future__ import annotations
import numpy as np
from typing import Dict, Any

class LinUCBBandit:
    def __init__(self, n_features: int, arms: Dict[str, Dict[str, Any]], alpha: float = 1.2):
        self.alpha = alpha
        self.arms = list(arms.keys())
        self.prompts = arms  # id -> {template, name}
        self.A = {a: np.eye(n_features) for a in self.arms}  # d x d
        self.b = {a: np.zeros((n_features, 1)) for a in self.arms}  # d x 1
        self.d = n_features

    def _theta(self, arm: str):
        A_inv = np.linalg.inv(self.A[arm])
        return (A_inv @ self.b[arm]).reshape(-1)

    def select(self, x: np.ndarray) -> str:
        # Upper Confidence Bound selection
        x = x.reshape(-1, 1)
        scores = {}
        for a in self.arms:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mu = float((theta.T @ x))
            sigma = float(np.sqrt(x.T @ A_inv @ x))
            scores[a] = mu + self.alpha * sigma
        return max(scores, key=scores.get)

    def update(self, arm: str, x: np.ndarray, reward: float):
        x = x.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x

# Simple feature builder
import re

KEYS = {
    "behavioral": ["team", "conflict", "lead", "failure", "success", "STAR"],
    "technical": ["data structure", "algorithm", "complexity", "sql", "python", "java", "machine learning"],
    "hr": ["strength", "weakness", "salary", "relocate", "notice", "why us"],
}

def build_features(query: str) -> np.ndarray:
    q = query.lower()
    L = len(q)
    f = [
        1.0,                        # bias
        min(L/300.0, 1.0),         # normalized length
        1.0 if any(k in q for k in KEYS["technical"]) else 0.0,
        1.0 if any(k in q for k in KEYS["behavioral"]) else 0.0,
        1.0 if any(k in q for k in KEYS["hr"]) else 0.0,
        float(len(re.findall(r"\?", q)) > 0),
    ]
    return np.array(f, dtype=float)

DEFAULT_ARMS = {
    "concise": {"name": "Concise", "template": "Answer clearly in 5-7 bullet points. Keep it concise and specific for an interview."},
    "structured": {"name": "Structured (STAR)", "template": "Use STAR (Situation, Task, Action, Result). Provide a crisp, interview-ready answer."},
    "coaching": {"name": "Coaching", "template": "Give the best possible answer, then add 2 improvement tips and 1 follow-up question an interviewer might ask."},
}

BANDIT = LinUCBBandit(n_features=6, arms=DEFAULT_ARMS)
