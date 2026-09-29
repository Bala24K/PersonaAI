# Running against Postgres instead of SQLite

SQLite is the zero-setup default (see main `README.md`). Swapping to
Postgres is genuinely a one-line config change — `DATABASE_URL` in
`backend/app/config.py` — but "one line" is a claim worth actually
proving rather than asserting. These are the exact commands that were
used to install Postgres, create a database, and run the full app
(chat, memory retrieval, user model, reflection) against it, verified
by querying the tables directly with `psql` afterward.

## 1. Install Postgres

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib
```

## 2. Start it

Some environments (containers especially) block `systemctl`/`service`
from actually starting daemons. If `service postgresql start` doesn't
work, `pg_ctlcluster` does:

```bash
service postgresql start || pg_ctlcluster 16 main start
pg_lsclusters   # should show status "online"
```

## 3. Create a database and app user

```bash
sudo -u postgres psql -c "CREATE USER persona_app WITH PASSWORD 'choose-a-real-password';"
sudo -u postgres psql -c "CREATE DATABASE persona_ai OWNER persona_app;"
```

## 4. Install the Python driver

Not in `backend/requirements.txt` by default (SQLite needs no driver at
all, so it stays the zero-dependency default):

```bash
pip install psycopg2-binary
```

## 5. Point the app at it

```bash
export DATABASE_URL="postgresql://persona_app:choose-a-real-password@localhost/persona_ai"
uvicorn app.main:app --port 8000
```

That's the whole change. `backend/app/database.py`'s SQLAlchemy models
are unmodified — they work against both engines because nothing in the
codebase writes raw SQLite-specific SQL.

## What was actually verified, not just asserted

Running against a real Postgres instance set up with the commands
above:
- Server started cleanly, tables were created automatically on first
  request (same `init_db()` call as the SQLite path).
- A full chat turn (`POST /chat`) wrote correctly to `messages`,
  `memories`, and `traits` — confirmed by querying those tables
  directly with `psql`, not just trusting the API's 200 response.
- Memory retrieval on a second turn correctly pulled back the episodic
  memory from the first turn, proving the Chroma vector index (which is
  a separate store from the SQL database) still integrates correctly
  regardless of which SQL backend is behind it.

## When you'd actually want this over SQLite

SQLite is fine for a single-process local demo. Move to Postgres once
you have concurrent writers (multiple uvicorn workers, multiple users
hitting the API at once) — SQLite's file-level locking becomes a
bottleneck there in a way Postgres doesn't have.
