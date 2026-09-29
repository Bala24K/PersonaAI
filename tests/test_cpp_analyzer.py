import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cpp_wrapper import analyze_text_cpp, _python_fallback_analyzer


def test_python_fallback_analyzer():
    res = _python_fallback_analyzer("I am so happy and excited today! Everything is working great.")
    assert res["word_count"] > 5
    assert res["emotion_keyword_counts"]["positive"] >= 2
    assert res["source"] == "python_fallback"


def test_cpp_analyzer_runs():
    res = analyze_text_cpp("I am anxious and stressed about the upcoming deadline.")
    assert res["char_count"] > 0
    assert res["word_count"] > 5
    assert "readability_score" in res
    assert "emotion_keyword_counts" in res
    assert len(res["feature_hash_16"]) == 16
    assert res["source"] in ["cpp17", "python_fallback"]


def test_cpp_analyzer_handles_empty_string():
    res = analyze_text_cpp("")
    assert res["word_count"] == 0
