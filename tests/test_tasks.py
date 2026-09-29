import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tasks import (
    consolidate_memory,
    extract_memory_async,
    evaluate_response_async,
    export_training_data,
)


def test_consolidate_memory_task_execution():
    res = consolidate_memory("test_task_user")
    assert isinstance(res, dict)
    assert res["user_id"] == "test_task_user"
    assert res["status"] in ["no_memories", "consolidated"]


def test_extract_memory_async_task_execution():
    res = extract_memory_async("test_task_user", "I am feeling really stressed about work today.")
    assert isinstance(res, dict)
    assert res["user_id"] == "test_task_user"
    assert "cpp_text_stats" in res
    assert res["memories_added"] >= 0


def test_evaluate_response_async_task_execution():
    res = evaluate_response_async("test_task_user", "How are you?", "I am doing great!")
    assert isinstance(res, dict)
    assert "overall" in res
    assert "is_acceptable" in res


def test_export_training_data_task_execution():
    res = export_training_data()
    assert isinstance(res, dict)
    assert "sft_examples" in res
    assert "dpo_pairs" in res
