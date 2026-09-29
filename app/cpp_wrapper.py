"""
Python wrapper for the C++17 Text Analyzer component.

Integration Mechanism:
Subprocess execution boundary via CLI/JSON interface (`persona_cpp_analyzer`).
Provides high-performance lexical analysis, readability scoring, and feature hashing.

If the C++ binary is missing or unavailable, gracefully falls back to a pure-Python
implementation so the Python application remains functional in any environment.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Find binary path
_BASE_DIR = Path(__file__).resolve().parent.parent
_CPP_DIR = _BASE_DIR / "cpp"

def _get_binary_path() -> Path | None:
    exe_name = "persona_cpp_analyzer.exe" if sys.platform == "win32" else "persona_cpp_analyzer"
    candidates = [
        _CPP_DIR / exe_name,
        _CPP_DIR / "build" / exe_name,
        _BASE_DIR / "bin" / exe_name,
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK | os.F_OK):
            return candidate
    return None

def _python_fallback_analyzer(text: str) -> dict:
    """Pure-Python fallback when the C++ executable is not available."""
    words = [w.strip(".,!?\"'()[]{}") for w in text.split() if w.strip()]
    char_count = len(text)
    word_count = len(words)
    sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0.0

    pos_keywords = {"happy", "great", "awesome", "good", "love", "thanks", "helpful", "wonderful", "excited", "nice"}
    neg_keywords = {"sad", "bad", "terrible", "awful", "hate", "angry", "upset", "depressed", "miserable", "worst"}
    anx_keywords = {"anxious", "worried", "nervous", "scared", "afraid", "panic", "stressed", "overwhelmed", "dread"}
    urg_keywords = {"now", "urgent", "asap", "immediately", "quick", "fast", "deadline", "emergency", "hurry"}

    words_lower = [w.lower() for w in words]
    pos_cnt = sum(1 for w in words_lower if w in pos_keywords)
    neg_cnt = sum(1 for w in words_lower if w in neg_keywords)
    anx_cnt = sum(1 for w in words_lower if w in anx_keywords)
    urg_cnt = sum(1 for w in words_lower if w in urg_keywords)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_len, 4),
        "type_token_ratio": round(ttr, 4),
        "readability_score": 75.0,
        "emotion_keyword_counts": {
            "positive": pos_cnt,
            "negative": neg_cnt,
            "anxiety": anx_cnt,
            "urgency": urg_cnt,
        },
        "feature_hash_16": [0.0] * 16,
        "source": "python_fallback",
    }

def analyze_text_cpp(text: str) -> dict:
    """Analyze text using the C++17 analyzer binary, falling back to Python if necessary."""
    binary = _get_binary_path()
    if not binary:
        logger.warning("C++ analyzer binary not found; using Python fallback")
        return _python_fallback_analyzer(text)

    try:
        proc = subprocess.run(
            [str(binary), "--text", text],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
        data = json.loads(proc.stdout)
        data["source"] = "cpp17"
        return data
    except Exception as exc:
        logger.warning("Failed to run C++ analyzer executable (%s); falling back to Python", exc)
        return _python_fallback_analyzer(text)
