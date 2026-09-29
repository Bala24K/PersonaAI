import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app

client = TestClient(app)


def test_chat_endpoint_returns_response_and_cpp_stats():
    payload = {
        "user_id": "test_api_u1",
        "session_id": "s1",
        "message": "Hello Nova, I am feeling very happy today!",
        "show_debug": True,
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["character_name"] == "Nova"
    assert "cpp_text_stats" in data
    assert data["cpp_text_stats"]["word_count"] > 0
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-MS" in response.headers


def test_chat_endpoint_validates_empty_message():
    response = client.post("/chat", json={"user_id": "u1", "session_id": "s1", "message": "   "})
    assert response.status_code == 400


def test_task_submission_endpoints():
    # Submit async memory extraction task
    res1 = client.post("/tasks/memory-extraction", json={"user_id": "u1", "message": "I love programming."})
    assert res1.status_code == 202
    data1 = res1.json()
    assert "task_id" in data1
    assert data1["status"] == "queued"

    # Submit async evaluation task
    res2 = client.post("/tasks/evaluate", json={"user_id": "u1", "user_message": "Hi", "response": "Hello"})
    assert res2.status_code == 202
    data2 = res2.json()
    assert "task_id" in data2

    # Poll task status
    task_id = data1["task_id"]
    res3 = client.get(f"/tasks/{task_id}")
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["task_id"] == task_id
    assert "status" in data3
