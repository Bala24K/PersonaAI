# Running reflection as a real background job (Celery + Redis)

`REFLECTION_MODE="inline"` (the default) runs `reflection.run_reflection()`
synchronously inside the `/chat` request — simplest, and keeps
`trend_note` available in the same response the frontend reads. This
doc covers `REFLECTION_MODE="background"`, which dispatches it as an
actual Celery task instead. These are the exact commands used to
install Redis, start a real worker, and verify a task executed on a
separate process — not just an assertion that "you'd add this later."

## 1. Install Redis

```bash
sudo apt-get update
sudo apt-get install -y redis-server
```

## 2. Start it

Same caveat as Postgres (`docs/POSTGRES_SETUP.md`) — some environments
block `systemctl`/`service` from starting daemons directly:

```bash
service redis-server start
redis-cli ping   # should print PONG
```

## 3. Install Celery + the Redis client

Not in `backend/requirements.txt` by default — inline mode needs
neither, so the zero-setup default stays zero-setup:

```bash
pip install -r backend/requirements-background.txt
```

## 4. Start a worker

```bash
cd backend
celery -A app.celery_app worker --loglevel=info
```

You should see `[tasks] . app.tasks.run_reflection_task` in the startup
output — that's the task being registered. If that line is missing, the
worker won't process anything dispatched to it; it needs
`include=["app.tasks"]` on the `Celery(...)` call in `app/celery_app.py`
(already set, but worth knowing what to check if you refactor this).

## 5. Run the app in background mode

```bash
export REFLECTION_MODE=background
uvicorn app.main:app --port 8000
```

## What was actually verified, not just asserted

- `POST /chat` returned immediately with
  `"reflection": {"status": "queued", "task_id": "..."}` instead of
  blocking on the reflection computation.
- The worker log showed the task actually being received and completed
  by a genuinely separate process (`Task app.tasks.run_reflection_task[...]
  succeeded in 0.035s`).
- `GET /reflection/{task_id}` correctly returned the completed result
  (`{"status": "done", "result": {...}}`), pulled from Redis as the
  Celery result backend — not from any in-process state, so this poll
  would work correctly even behind a load balancer with multiple
  FastAPI workers.
- The 10-message trend-detection scenario (see `docs/DEMO_SCRIPT.md`)
  was re-run through the background path end-to-end and produced the
  identical trend note as inline mode — proving `reflection.py` itself
  needed zero changes to become a background task, exactly as its
  original docstring predicted.
- Confirmed `REFLECTION_MODE="inline"` (default) is completely
  unaffected — no regression, verified by re-running the flagship
  rejection example afterward.

## What this doesn't include

Just the one task (reflection). No retry policy, no rate limiting, no
dead-letter handling, no monitoring (Flower or similar) — those are the
natural next additions once there's more than one kind of background
job and actual production traffic to tune for.
