import os
import pytest
from app.tasks import celery_app

@pytest.fixture(autouse=True)
def configure_test_celery(monkeypatch):
    # Force Celery to run synchronously during unit/integration test runs
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        task_store_eager_result=True,
    )
    yield
