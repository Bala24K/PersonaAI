# Demo script

Run these against a freshly-started server (`uvicorn app.main:app --port 8000`
from `backend/`) to exercise every piece of the system via the API
directly. Everything here also works through the UI at
`http://localhost:8000/ui/` — this is just the scriptable version, useful
for showing someone how it works without a screen-recording.

```bash
BASE=http://localhost:8000

# 1. Create a user
USER_ID=$(curl -s -X POST $BASE/user | python3 -c "import json,sys; print(json.load(sys.stdin)['user_id'])")
echo "User: $USER_ID"
```

## Facts + explicit preferences get remembered

```bash
curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"I've been learning Java for two years and I study CSE. I hate generic motivational quotes, just be direct with me.\"}" \
  | python3 -m json.tool
```
Check `$BASE/user/$USER_ID/memories` afterward — you should see a
semantic memory ("i've been learning java...") and a preference memory
("i hate generic motivational quotes...").

## The flagship emotional-intelligence example

```bash
curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"I got rejected again\"}" | python3 -m json.tool
```
Expected: `situation: "rejection"`, `strategy: "acknowledge_then_offer_choice"`,
`trained_model_used: true` (the V2 trained classifier contributed to
this call), and a reply that acknowledges it and asks whether they want
to vent or dig into what happened — not generic advice, and not a
canned motivational line (which the preference memory above should also
help suppress if you're using a real LLM connection).

## Flat-affect masking ("I'm fine" that isn't)

```bash
curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"yeah i am fine lol\"}" | python3 -m json.tool
```
Expected: `flat_affect_flag: true`, dominant emotion shifted to
frustration/sadness rather than taken at face value as neutral.

## Memory retrieval actually pulling context forward

```bash
CONVO=$(curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"still stressed about interviews and Java stuff\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['conversation_id'])")
```
Check the `retrieved_memories` field in the response — it should surface
the Java/CSE fact and the rejection episode from earlier in this walkthrough.

## Reflection engine: detecting an emotional trend

Send five mildly negative/neutral messages, then five clearly distressed
ones, in the same conversation — the tenth response's `reflection.trend_note`
should flag the shift:

```bash
for m in "had a decent day" "worked on my project" "watched anime" "ate lunch" "nothing much going on" \
         "feeling really stressed and anxious" "so overwhelmed, another deadline" "anxious again, ugh" \
         "everything is stressful, frustrated" "still anxious, too much"; do
  curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
    -d "{\"user_id\": \"$USER_ID\", \"conversation_id\": \"$CONVO\", \"message\": \"$m\"}" \
    -o /tmp/last_reply.json
done
python3 -c "import json; print(json.load(open('/tmp/last_reply.json'))['reflection'])"
```

## Memory survives a server restart (V2: Chroma persistence)

```bash
# Stop the server (Ctrl+C or kill the process), then start it again:
uvicorn app.main:app --port 8000 &

# Same user, no re-import needed — retrieval still works:
curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"remind me what I said about interviews\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['retrieved_memories'])"
```
In V1 this would have come back empty on first query after a restart
(the in-memory index had to be rebuilt from SQL — see
`memory/store.py::load_index_from_db`). In V2 it comes back immediately
because Chroma persisted the vectors to `data/chroma/` across the
restart.

## Per-user isolation (also worth checking directly)

```bash
USER_B=$(curl -s -X POST $BASE/user | python3 -c "import json,sys; print(json.load(sys.stdin)['user_id'])")
curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_B\", \"message\": \"tell me about interviews and java\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['retrieved_memories'])"
# Expected: [] — a brand-new user, even asking about the same topics,
# never sees $USER_ID's memories.
```

## Profile import (Phase 4 — user-authorized bulk ingestion)

```bash
curl -s -X POST $BASE/profile/import -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"text\": \"I am a computer science student.\nI love anime and I hate small talk.\nMy friend Sarah always helps me with debugging.\"}" \
  | python3 -m json.tool
```

## The person stays in control: deleting a memory

```bash
MEM_ID=$(curl -s $BASE/user/$USER_ID/memories | python3 -c "import json,sys; print(json.load(sys.stdin)['memories'][0]['id'])")
curl -s -X DELETE $BASE/user/$USER_ID/memories/$MEM_ID
```

## Feedback capture

```bash
MSG_ID=$(python3 -c "import json; print(json.load(open('/tmp/last_reply.json'))['assistant_message_id'])")
curl -s -X POST $BASE/feedback -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message_id\": \"$MSG_ID\", \"rating\": \"down\", \"tags\": [\"too_formal\"]}"
```

## Watching the user model evolve

```bash
curl -s $BASE/user/$USER_ID/model | python3 -m json.tool
```
Run this after each message above and watch `directness`, `formality`,
`slang_tendency`, etc. move — and watch `confidence` climb as
`evidence_count` grows.

## Feedback that actually changes behavior (V3)

```bash
curl -s $BASE/user/$USER_ID/model | python3 -c "
import json,sys
d = json.load(sys.stdin)
print([t for t in d['traits'] if t['name']=='formality'])
"
# note the 'value' for formality, then:

MSG_ID=$(curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"hello\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['assistant_message_id'])")

curl -s -X POST $BASE/feedback -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message_id\": \"$MSG_ID\", \"rating\": \"down\", \"tags\": [\"too_formal\"]}" \
  | python3 -m json.tool
# response includes "adjusted_traits": [{"tag": "too_formal", "trait": "formality", "nudged_toward": 0.15}]

curl -s $BASE/user/$USER_ID/model | python3 -c "
import json,sys
d = json.load(sys.stdin)
print([t for t in d['traits'] if t['name']=='formality'])
"
# 'value' should have moved down toward 0.15, 'evidence_count' incremented
```

Try tags: `too_formal`, `too_casual`, `too_verbose`, `too_short`,
`too_blunt`, `too_vague`, `too_technical`, `too_cliche`,
`too_much_humor`, `not_enough_humor` — each maps to a specific trait
(see `TAG_TRAIT_ADJUSTMENTS` in `user_model/feedback_learning.py`).
Unrecognized tags (or `wrong_memory`, which isn't wired to a trait) come
back in `unhandled_tags` rather than silently vanishing.

## LLM-based memory extraction (V3)

With `ANTHROPIC_API_KEY` set, `/chat` uses a real model call to extract
memories as structured JSON instead of regex patterns. (`/profile/import`
always uses the regex extractor regardless of mode — it's called once
per line of pasted text, and firing an LLM call per line would be slow
and expensive for text the regex patterns already handle well; see
`memory/store.py`'s docstring.) No visible API difference on `/chat` —
`retrieved_memories` and `GET /user/{id}/memories` work identically
either way — but extraction quality should noticeably improve on
messages the regex patterns don't have an explicit pattern for.

```bash
export MEMORY_EXTRACTION_MODE=heuristic   # regex only, ignores API key
export MEMORY_EXTRACTION_MODE=llm         # LLM only, returns nothing if no key/call fails
export MEMORY_EXTRACTION_MODE=auto        # default: LLM if available, else heuristic
```

## Reproducing the trained emotion classifier

```bash
cd ml
pip install -r requirements.txt
python3 training/train_emotion_classifier.py
```
Prints the full evaluation report to stdout and writes it to
`models/training_report.txt`, plus the retrained model artifact to
`models/emotion_classifier.joblib`. Copy that file over
`backend/app/emotion/trained/emotion_classifier.joblib` to have the app
use your retrained version.

## Writing-style comparison (V7)

Does the AI write like you? Send a few messages in your natural style,
then ask:

```bash
for m in "lol idk what to do tbh" "omg that is so funny!!" "ngl this is kinda wild rn"; do
  curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
    -d "{\"user_id\": \"$USER_ID\", \"message\": \"$m\"}" > /dev/null
done

curl -s $BASE/user/$USER_ID/style | python3 -m json.tool
```
Returns your style vector (9 linguistic features), the AI's style
vector computed the same way over its replies, a single 0-1 similarity
score, and the specific features driving the biggest gap.

The "Style match" panel in the UI (`http://localhost:8000/ui/`) shows
the same thing live, updating after each message.

**Expect a low score without an API key.** Running the above against
the offline template responder gives roughly 0.37 — the fallback
responder writes in a consistent neutral register and makes no attempt
to match a casual user's slang, abbreviations, or lack of
capitalization. That's the honest, expected result, not a bug: this
metric exists precisely to make that kind of gap visible and
measurable. With `ANTHROPIC_API_KEY` set, the system prompt does
instruct the model to calibrate to the user's learned traits, so the
score should be meaningfully higher — and now you can check rather
than assume.

## Personality drift detection (V8)

The pitch's own example — someone who used to want detailed answers and
now wants concise ones. Send a batch of long messages, then a batch of
terse ones:

```bash
for i in $(seq 1 12); do
  curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
    -d "{\"user_id\": \"$USER_ID\", \"message\": \"I have been thinking quite a lot about this particular problem and I wanted to explain my full reasoning to you in detail because the context really matters here and I want to make sure you understand all the nuances involved\"}" > /dev/null
done
for i in $(seq 1 12); do
  curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
    -d "{\"user_id\": \"$USER_ID\", \"message\": \"sure fine\"}" > /dev/null
done

curl -s $BASE/user/$USER_ID/drift | python3 -m json.tool
```
Expect something like `verbosity has decreased from 0.58 to 0.34
(-0.24)`. Drift also appears in every `/chat` response's
`reflection.drift` field.

To chart the shift rather than just read the summary:
```bash
curl -s $BASE/user/$USER_ID/trait-timeline/verbosity | python3 -m json.tool
```

**An empty `findings` list is a real answer, not a failure.** Drift
requires a change of at least 0.15 between window means, at least 5
post-burn-in observations per window, AND a settled recent window — so
a fresh user (under ~25 messages) or a consistent one correctly reports
nothing. The four should-stay-silent cases in
`ml/evaluation/eval_drift.py` show exactly what gets rejected and why.

## Running the evaluation harness (V6/V7/V8)

```bash
cd ml
pip install -r requirements.txt
python3 evaluation/run_all_evaluations.py
```
Runs all six evaluations (emotion ablation, memory extraction,
critic, strategy selection, style vector, drift detection) and writes `models/EVALUATION_SUMMARY.md`.
Takes about 15 seconds total, dominated by the emotion ablation's
~6,000-example test set. Each evaluation can also be run individually
(`python3 evaluation/eval_critic.py`, etc.) if you only want one.

Worth actually reading `EVALUATION_SUMMARY.md` rather than just running
it — it documents two real bugs the harness caught during development
(a situation-classification ordering bug, and a non-obvious regression
in the emotion blend on one specific benchmark) rather than presenting
only clean, favorable numbers.

## Background reflection via Celery + Redis (V4)

Full setup in `docs/BACKGROUND_JOBS.md`; short version once Redis is
running and a worker is up:

```bash
export REFLECTION_MODE=background
uvicorn app.main:app --port 8000 &

RESP=$(curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"I got rejected again\"}")
echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['reflection'])"
# {"status": "queued", "task_id": "..."}

TASK_ID=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['reflection']['task_id'])")
curl -s $BASE/reflection/$TASK_ID | python3 -m json.tool
# {"status": "done", "result": {"trend_note": null, "contradictions": []}}
```
Compare against `REFLECTION_MODE=inline` (default) — same
`reflection.run_reflection()` function either way, just synchronous vs.
dispatched.

## Agentic memory retrieval (V4)

Requires `ANTHROPIC_API_KEY` — with `MEMORY_MODE=agentic`, the model
decides whether a message needs a memory search instead of always
retrieving top-K first:

```bash
export MEMORY_MODE=agentic
uvicorn app.main:app --port 8000 &

curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"remind me what I said about my interview\"}" \
  | python3 -c "
import json,sys
d = json.load(sys.stdin)
print('memory_mode:', d['memory_mode'])
print('tool_calls:', d['tool_calls'])
print('retrieved_memories:', d['retrieved_memories'])
"
```
For a message with no personal context ("what's 2+2"), expect
`tool_calls: 0` — the model shouldn't bother searching. Without an API
key, `memory_mode` falls back to `"always"` automatically regardless of
this setting (there's no model to make the decision with).

## Response critic (V5)

Runs by default (`CRITIC_MODE=heuristic`), no API key needed to see it
detect issues — only regeneration needs a real key:

```bash
curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"I got rejected again\"}" \
  | python3 -c "
import json,sys
d = json.load(sys.stdin)
print('critique_issues:', d['critique_issues'])
print('regenerated:', d['regenerated'])
"
# A well-formed reply should show critique_issues: [] and regenerated: false
```
To see it actually catch something, disable the module's own guard
temporarily or test `critic.py` directly:
```bash
cd backend && python3 -c "
from app.critic import heuristic_critique
print(heuristic_critique('You should just apply to more places.', 'acknowledge_then_offer_choice', 'rejection', {}))
"
# ['skipped_acknowledgment_jumped_to_advice']
```
With `ANTHROPIC_API_KEY` set, a flagged reply triggers exactly one
regeneration attempt automatically — `regenerated: true` in the
response, and `reply` will be the corrected second draft, not the
flagged first one.

## Up-vote reinforcement (V5)

```bash
curl -s $BASE/user/$USER_ID/model | python3 -c "
import json,sys
print([t for t in json.load(sys.stdin)['traits'] if t['name']=='humor'])
"
# note the confidence value, then:

MSG_ID=$(curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"lol nice\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['assistant_message_id'])")

curl -s -X POST $BASE/feedback -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message_id\": \"$MSG_ID\", \"rating\": \"up\", \"tags\": [\"good_humor\"]}" \
  | python3 -m json.tool

curl -s $BASE/user/$USER_ID/model | python3 -c "
import json,sys
print([t for t in json.load(sys.stdin)['traits'] if t['name']=='humor'])
"
# 'value' should be essentially unchanged; 'confidence' and
# 'evidence_count' should both have increased.
```
Try an untagged up-vote (`"tags": []`) too — it reinforces a broader set
of core traits at a lower weight instead of one trait at a higher one.

## `wrong_memory` feedback tracing (V5)

```bash
curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"I study CSE and love anime\"}" > /dev/null

RESP=$(curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"tell me more about CSE\"}")
echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['retrieved_memories'])"
MSG_ID=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['assistant_message_id'])")

curl -s -X POST $BASE/feedback -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message_id\": \"$MSG_ID\", \"rating\": \"down\", \"tags\": [\"wrong_memory\"]}" \
  | python3 -m json.tool
# response includes "flagged_memories": ["<the exact memory id that was retrieved>"]

curl -s $BASE/user/$USER_ID/memories | python3 -c "
import json,sys
for m in json.load(sys.stdin)['memories']:
    print(m['topic'], '|', m['content'])
"
# the flagged memory's topic should now read 'needs_review'
```
