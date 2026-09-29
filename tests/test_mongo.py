import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mongo_store import MongoStore


def test_mongo_store_initialization_and_fallback():
    # Store pointing to invalid host should safely handle fallback without crashing
    store = MongoStore(mongo_url="mongodb://invalid_host_12345:27017")
    assert store.is_available is False
    
    # Operations should return None or [] safely
    assert store.log_interaction_event("u1", "s1", "hi", "hello") is None
    assert store.log_async_job("task-1", "test", "u1", "SUCCESS") is None
    assert store.get_interaction_events("u1") == []
    assert store.get_async_job_log("task-1") is None


def test_mongo_store_fallback_behavior():
    store = MongoStore()
    # Even if Mongo is not running, calling methods must not raise exceptions
    event_id = store.log_interaction_event("test_user", "s1", "msg", "resp")
    if store.is_available:
        assert event_id is not None
    else:
        assert event_id is None
