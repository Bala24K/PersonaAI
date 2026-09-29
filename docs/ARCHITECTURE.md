# Architecture

## Data flow for one message

```
 User message
      │
      ▼
┌─────────────────────────┐
│ emotion.engine.analyze  │  trained classifier (V2) blended with
│  -> emotion scores       │  lexicon (V1, fear/disgust/frustration)
│  -> situation             │  -> situation via regex patterns
│  -> strategy              │  -> strategy via lookup table
└─────────────┬────────────┘
              │
              ▼
┌─────────────────────────┐      ┌──────────────────────────┐
│ user_model.signals        │ --> │ user_model.model           │
│  message -> trait signals │      │  EWMA update per trait     │
└─────────────────────────┘      │  value + confidence stored  │
                                    └──────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│ memory.store.retrieve_    │  MEMORY_MODE="always" (default): Chroma
│ relevant                  │  vector search (V2), native per-user
│  -- OR (V4) --             │  filtering, runs every turn.
│ llm.client.generate_       │  MEMORY_MODE="agentic": skipped here —
│ with_tools                 │  the model gets a search_memory tool
│                            │  instead and decides whether to call it
│                            │  (0+ times) — see llm/tools.py.
└─────────────┬────────────┘
              │
              ▼
┌─────────────────────────┐
│ llm.prompts.build_        │  assembles system prompt from
│ system_prompt             │  traits + memories + strategy
└─────────────┬────────────┘
              │
              ▼
┌─────────────────────────┐
│ llm.client.generate       │  real Anthropic call if API key set,
│ (or generate_with_tools)  │  else deterministic template fallback
└─────────────┬────────────┘
              │
              ▼
┌─────────────────────────┐
│ llm.personalize.apply     │  trims verbosity, strips cliches,
│                            │  based on trait confidence/value
└─────────────┬────────────┘
              │
              ▼
      Stored (Message rows) + memory.store.ingest_text
      (heuristic extraction of *this* message runs here too)
              │
              ▼
      reflection.run_reflection (the "slow brain", see below) —
      REFLECTION_MODE="background" (V4) dispatches this as a Celery
      task instead of running it inline; response includes a task_id
      to poll rather than the result directly in that mode.
              │
              ▼
        Response returned to user
```

Everything above the "stored" line is `agents/orchestrator.py::handle_user_message`
— the **fast brain**. It has to finish before the user gets a reply, so
it's kept cheap (no LLM calls except the one generation call itself).

`reflection.py::run_reflection` is the **slow brain**. It currently runs
inline, synchronously, right after the fast path, which is fine for a
demo but is exactly the wrong choice for a network of real users — the
comment at the top of `reflection.py` says explicitly where a
Celery/Redis (or any task queue) hand-off belongs instead.

## Why heuristics instead of trained models, specifically

This system was built inside a sandboxed environment with two concrete
constraints that shaped V1's architecture:

1. **No network access to model-hosting domains.** The allowed domains
   are package registries (PyPI, npm, crates, GitHub) — not
   `huggingface.co` or similar. That rules out downloading any
   pretrained sentence-embedding or emotion-classification model for
   V1, which is why `memory/vectorstore.py` uses a hashing-trick
   vectorizer (needs no download, no fitting) and `emotion/lexicon.py`
   is a hand-built word list instead of e.g. a fine-tuned RoBERTa
   emotion classifier.
2. **No bundled LLM API key.** `llm/client.py` is written so the whole
   pipeline runs and is demoable with zero external calls, falling back
   to a template responder. This is a deliberate design choice, not a
   workaround to hide: it means anyone can clone this and see the full
   architecture work before deciding whether to pay for API calls.

Both constraints are handled the same way architecturally: **narrow,
swappable interfaces**. Nothing outside `memory/vectorstore.py` knows
embeddings come from a hashing trick. Nothing outside `emotion/engine.py`
knows emotion detection is lexicon-based (or, as of V2, lexicon-plus-
trained-classifier). Nothing outside `llm/client.py` knows whether a
real model or the fallback produced a reply. Upgrading any one of these
is a change to one file's internals, not a rewrite.

**A clarification on the V2 emotion classifier**: it's a real *trained*
model, but it's not a *pretrained downloaded* model — the constraint
above (no access to `huggingface.co`) still held. What changed is that
public *training data* (as opposed to pretrained *weights*) was
reachable via GitHub, which is on the allowed-domains list, so training
a classical ML model from scratch on that data was possible even though
downloading someone else's already-trained transformer wasn't. See the
roadmap's item 5 below for the actual numbers and reasoning.

## Extending further

This follows the phased roadmap from the original design doc. Status
markers below span multiple rounds of work (✅ done, 🟡 partially done,
or still open) — order is roughly by remaining leverage, not by when
each was tackled:

### 1. Real embeddings — still open, highest remaining leverage
`memory/vectorstore.py`'s `embed()` function is still the hashing-trick
vectorizer from V1. The vector *database* underneath it was upgraded to
Chroma (item 3, below), but that's a storage/search-engine upgrade, not
an embedding-quality upgrade — worth being precise that those are
separate axes. Swapping `embed()` for `sentence-transformers` or an
embeddings API (once network access to reach one exists) is still the
single highest-leverage remaining change, and nothing else in the
codebase needs to change to do it — both `ChromaVectorIndex` and
`NumpyVectorIndex` just call `embed()`.

### 2. Postgres instead of SQLite — ✅ done
Change `DATABASE_URL` in `config.py` — and this was actually run, not
just asserted as trivial. Postgres was installed in the build
environment (`apt-get install postgresql`), a database and app user
were created, and the full app (chat, memory, user model, reflection)
was run against it with `DATABASE_URL` pointed at it. Verified by
querying `messages`, `memories`, and `traits` directly with `psql`
afterward — the data was really there, not just a 200 response being
trusted. Exact reproducible commands: `docs/POSTGRES_SETUP.md`. SQLite
remains the zero-setup default; the ORM models in `database.py` needed
zero changes to work against either engine.

### 3. A real vector database — ✅ done
`memory/vectorstore.py` now runs on Chroma (`ChromaVectorIndex`),
persistent to `data/chroma/`, with native per-user metadata filtering
(`where={"user_id": ...}`) replacing the old Python-side `allowed_ids`
set intersection. Verified concretely: memories survive a full process
restart without re-ingestion from SQL, and two different users' memories
stay isolated even when their queries share vocabulary (tested directly,
not just asserted). Falls back automatically to the old in-memory
`NumpyVectorIndex` if Chroma fails to import or initialize for any
reason — see `build_vector_index()`.

### 4. LLM-based memory extraction — ✅ done
`memory/llm_extraction.py` prompts a model for structured JSON output
(type/content/importance) instead of relying only on regex.
`memory/store.py::_get_candidates()` picks between it and the original
heuristic extractor via `config.MEMORY_EXTRACTION_MODE` — "auto" (the
default) tries the LLM extractor and falls back to the heuristic one on
*any* failure (no API key, malformed JSON, non-2xx response), so the
choice never breaks the pipeline. Both extractors return the same
`CandidateMemory` objects, so `memory/store.py`'s ingestion logic itself
didn't need to change.

No live API key was available in this build environment, so the actual
model-calling path (`llm/client.py::complete()`, shared with everything
else that talks to the API) couldn't be exercised end-to-end here —
but the extraction/validation logic that consumes its output was
tested directly by mocking `complete()`'s return value across five
scenarios: well-formed JSON, JSON wrapped in markdown fences with a
preamble (defensive stripping), a response mixing one valid and one
invalid item (the invalid one is dropped, not the whole batch), a
non-JSON garbage response, and `complete()` returning `None` (no key).
All five produced the correct behavior. Since `complete()` uses the
identical request-building code as `generate()` (already exercised via
the offline fallback path throughout this project), the main untested
surface is specifically "does the real API return well-formed JSON when
asked to" — reasonable to expect given Claude's structured-output
reliability, but worth flagging as the one part of this feature that's
inference, not verification.

### 5. A trained emotion classifier — ✅ done
`emotion/trained_classifier.py` loads a real trained model (TF-IDF +
Logistic Regression — see `ml/training/train_emotion_classifier.py`)
instead of relying on the lexicon alone. Trained on ~40k labeled tweets
(a public, widely-used educational dataset — provenance and license
caveats are in the training script's docstring) mapped down to 6
well-supported classes: neutral, anxiety, joy, sadness, surprise, anger.
Evaluated on a held-out 15% test split: **40.2% accuracy / 0.333 macro
F1** (full per-class precision/recall/F1 in
`ml/models/training_report.txt`) — against a 16.7% random baseline and
~33% majority-class baseline. That's a real, modest number, honestly
reported: short informal text is a genuinely hard classification
problem, and this is a linear model on n-gram features, not a
fine-tuned transformer.

It does **not** cover fear, disgust, or frustration — no good training
data for those in this dataset — so `emotion/engine.py` blends it with
the V1 lexicon rather than replacing it (`_blend_scores()`). Building
this blend surfaced a real bug worth knowing about if you extend it
further: a naive weighted-sum blend let the trained model's confident
mass on categories it *can* predict silently outvote a clear lexicon
signal on categories it structurally can't — e.g. "I am terrified of
the exam" was landing on "neutral" because the trained model's
probability mass, spread across its 6 classes, outweighed the lexicon's
undiluted signal for "fear" once both were scaled by their blend
weights. The fix: categories outside the trained model's label space
get the lexicon's full, unscaled signal rather than being weighted down
by `LEXICON_WEIGHT` — see the docstring in `engine.py` for the full
reasoning. A second, unrelated bug was also caught in the same pass:
multi-word lexicon entries ("sick of", "tired of") had never matched
anything since V1 because the original matcher only checked single
tokens; bigram matching was added and verified against real examples.

**Why a classical model and not a fine-tuned transformer**: this
sandbox has no network access to download pretrained transformer
weights (no `huggingface.co`), so a transformer would have to train
from random initialization on 40k examples — worse accuracy, much
slower to train, much slower for real-time CPU inference, for a worse
result. TF-IDF + Logistic Regression is the right engineering choice
under this specific constraint. If you have model-hosting access,
fine-tuning `distilbert-base-uncased` (or similar) on the same mapped
dataset is the natural upgrade — several public write-ups fine-tuning
this exact upstream dataset report ~94%+ test accuracy, which is a
useful ballpark for how much headroom a transformer leaves on the table
here.

### 6. Feedback -> preference learning loop — ✅ done
`user_model/feedback_learning.py` closes both halves of this loop now.
Down-votes: tags (`too_formal`, `too_verbose`, `too_blunt`,
`too_technical`, `too_cliche`, etc.) immediately push the specific trait
they're about toward the corrected value, via the same `update_trait()`
EWMA mechanism ordinary message signals use. Verified by hand-checking
the math: a `too_formal` down-vote moved `formality` from 0.500 to
0.461, matching the expected weighted update exactly.

Up-votes (V5): reinforce the relevant trait toward its own *current*
value — a no-op on the value, a real increase in confidence — via
specific tags (`good_formality`, `good_humor`, etc., higher weight) or,
for an untagged thumbs-up, a broader reinforcement across core traits
at lower weight. The weight asymmetry is deliberate: a specific
complaint or compliment is stronger evidence about *which* trait it
concerns than an untagged rating is, the same principle `weight` already
encodes for message-signal-based updates elsewhere in the codebase.
Verified directly: confirmed a specifically-tagged up-vote leaves the
trait's value unchanged (within floating-point tolerance) while
increasing its confidence and evidence_count by exactly one, and
confirmed a specifically-reinforced trait isn't double-counted by the
generic pass in the same request.

`wrong_memory` tracing (V5): added `Message.retrieved_memory_ids_json`,
populated by the orchestrator for every assistant message in both
"always" and "agentic" memory modes. A `wrong_memory` down-vote now
looks up exactly which memories were retrieved for that specific
message and flags them (`topic="needs_review"`, the same convention
`reflection.py`'s contradiction detection uses) instead of falling
through to `unhandled_tags` with no further action. Verified
end-to-end: seeded a memory, triggered its retrieval on a later turn,
submitted `wrong_memory` feedback against that reply, and confirmed
that exact memory (and only that one) flipped to `needs_review`. Only
works for messages created after this schema addition — there's no
migration framework (Alembic or similar) in this project, so older
messages simply have no recorded IDs to trace back to; honestly
returns an empty flag list for those rather than guessing.

### 7. LoRA/QLoRA personalization — still open
Only worth doing once (1), (4), and (6) are solid and you have a real
dataset of (context, user state, good response) triples from actual
usage — otherwise there's nothing meaningfully different to fine-tune
on. The original design doc's Phase 7/8 sequencing (build the dataset
before attempting fine-tuning) still applies.

### 8. Agentic tool use — ✅ done
`llm/tools.py` defines a `search_memory` tool and `execute_tool()`
dispatcher; `llm/client.py::generate_with_tools()` handles the full
Anthropic tool-use round-trip (call the model with `tools=[...]`, if it
returns `tool_use` blocks execute them and feed `tool_result` blocks
back, loop until a final text response or `max_rounds` is hit).
`agents/orchestrator.py` picks between this and the classic
always-retrieve path via `config.MEMORY_MODE` ("always", the default,
vs. "agentic").

No live API key was available in this build environment, so the actual
model round-trip couldn't be exercised against the real API — the same
honest limitation as item 4's LLM memory extraction. What *was* tested,
by mocking `_raw_call()`'s return value while using a real SQLite
database and real Chroma-backed retrieval underneath: a single
tool-call-then-answer round trip (confirmed the correct memory came
back and got woven into the final reply); a direct answer with no tool
call; `generate_with_tools()` returning `None` when no API key is
configured (the orchestrator's fallback signal); and the worst case
where a model never stops calling tools, correctly capped at
`max_rounds` rather than looping forever. The message-construction and
response-parsing logic — the part most likely to have a subtle bug — is
fully verified; the one thing that's inference rather than verification
is whether the real API's tool-use response shape matches what the code
expects, which is a documented Anthropic API contract, not a guess.

Worth being honest about the actual tradeoff this introduces, not just
that it exists: agentic mode costs an extra model round-trip whenever a
tool is actually called (worse latency, worse cost, on those turns),
in exchange for not wasting prompt space on irrelevant retrieved
memories for messages that didn't need any. Whether that trade is worth
it depends on how often "always retrieve top-K" was actually retrieving
something irrelevant in your usage — not knowable without real traffic,
which is exactly why this is opt-in rather than a default change.

### 9. Background jobs (Celery + Redis) — ✅ done
`reflection.run_reflection()` needed zero changes to become a background
task — `app/tasks.py::run_reflection_task` wraps it in four lines,
exactly as this section originally predicted ("wrapping it as a Celery
task is mechanical"). `config.REFLECTION_MODE="background"` dispatches
it via `.delay()` instead of calling it inline; `GET /reflection/{task_id}`
polls the result back out of Redis (Celery's result backend), which
means the poll works correctly regardless of which FastAPI process
handles it — nothing is held in in-process state.

This was actually run, not just described as mechanical and left
unverified: Redis and Celery were installed in the build environment, a
real worker was started, and a dispatched task was confirmed completing
on that separate process via the worker's own log output
(`Task app.tasks.run_reflection_task[...] succeeded in 0.035s`), not
just a "the endpoint returned 200" check. One real bug was caught doing
this: a Celery worker will start, connect to the broker, and log "ready"
completely normally while never registering a task at all if the
`Celery(...)` app isn't told to `include=["app.tasks"]` — and then it
just silently never picks anything up, no error anywhere. Easy to miss
if you only check that the worker process started. Exact commands and
the full verification checklist: `docs/BACKGROUND_JOBS.md`.

`REFLECTION_MODE="inline"` remains the default specifically because it
keeps `trend_note` available in the same `/chat` response the frontend
already reads — switching the default to "background" would be a
breaking API change for anyone using the UI as-is, not just an internal
implementation detail, so it stays opt-in until there's an actual need
(concurrent users) to justify that tradeoff.

### 10. Self-correction / response critic (design doc Phase 12) — ✅ done
`critic.py::heuristic_critique()` checks a generated reply against the
strategy it was supposed to follow, catching concrete mismatches: advice
given before any acknowledgment (for strategies that call for
acknowledgment first), a "celebrate" reply opening on a negative note, a
cliche phrase surviving despite high `cliche_aversion`, a reply too
short for a serious situation. Runs on every turn by default
(`config.CRITIC_MODE="heuristic"`), pure text matching, zero API cost.
`agents/orchestrator.py` wires it in as step 6.5, between
personalization and storage.

When it finds an issue and `ANTHROPIC_API_KEY` is configured, exactly
one regeneration attempt is made: the critique is folded into the
system prompt ("a previous draft had this problem: ... do not repeat
that") and `generate()` is called again. Never more than one attempt,
by construction (the code doesn't loop on the result of the second
call) — verified two ways: (1) mocked a bad-then-good response pair and
confirmed the final reply was the corrected second draft; (2) mocked a
response that's bad on *every* call and confirmed `generate()` was
invoked exactly twice, not more, with the final (still-flawed) reply
returned rather than the pipeline hanging or erroring. All 8 heuristic
check scenarios (4 that should flag, 4 that shouldn't) were tested
individually first, specifically to rule out false positives before
wiring in regeneration — a critic that flags correct replies is worse
than no critic, since it burns an API call and can make things worse.

The offline template responder can't usefully act on critique feedback
(no reasoning to revise), so regeneration is skipped without an API
key — the critic still runs and reports `critique_issues` in the API
response either way, which is enough to demonstrate detection working
even in fully offline mode; only the correction half needs a real model.

An optional `CRITIC_MODE="heuristic_llm"` adds `llm_critique()` — a
direct model call asking whether the reply fits the strategy and
context — when the heuristic pass finds nothing, to catch mismatches
that are structurally invisible to pattern matching (technically-
acknowledging-but-still-dismissive tone, for instance). This is opt-in
rather than default specifically because it costs one extra API call on
every single clean turn, not just the turns that need correction — a
real cost/quality tradeoff, not a strictly-better upgrade.

### 11. Personality / writing-style imitation (design doc Phase 6) — ✅ done
`user_model/style_vector.py` implements the feature set the original
pitch named explicitly — sentence length, vocabulary richness,
punctuation, emoji, slang, abbreviations, capitalization, question
frequency, response length — and `GET /user/{id}/style` does the
comparison step the pitch asked for: the person's own messages vs. the
AI's replies, as two style vectors plus a similarity score plus the
features driving the biggest gap.

Two deliberate design choices worth explaining:

**Why this is separate from `user_model/signals.py`**, which already
computes overlapping signals (slang, emoji, formality): `signals.py`
collapses each into a single 0-1 trait nudge to steer generation —
lossy on purpose, because the generation pipeline wants "how casual is
this person" not "what is their exclamation rate." StyleVector keeps
each feature as its own measured number because its job is comparing
two writers against each other, which needs the raw quantities.
Merging them would force one job to use the other's wrong
representation.

**Why normalized-mean-absolute-difference rather than cosine
similarity**: interpretability. The features have wildly different
natural scales (sentence length in the tens, question frequency in
[0,1], emoji rate per 100 words), so cosine over raw values would be
dominated by whichever feature happens to have the biggest magnitude
rather than by actual stylistic difference. Normalizing each feature
into [0,1] against fixed documented bounds and averaging the absolute
gaps gives a score you can decompose — `biggest_style_gaps()` returns
exactly which features drove it — so "82% aligned, mostly a
question-frequency mismatch" is a statement you can act on.

**An honest finding from building this**: pointed at a casual user with
the offline template responder, style similarity lands around 0.37.
The fallback responder writes in a fixed neutral register and makes no
attempt to match slang, abbreviation, or capitalization patterns.
That's the correct, expected result — and it's the point. Before this
feature, "does the AI actually write like the user" was an assumption
nobody could check; now it's a number. Whether a live model closes that
gap (the system prompt does instruct it to calibrate to learned traits)
is now measurable rather than assumed, which is precisely the step the
original pitch was asking for.

What's *not* built: actually feeding StyleVector back into generation
as a closed loop — e.g. detecting a low match and adjusting the system
prompt or triggering a regeneration, the way `critic.py` does for
strategy mismatches. That's a natural next step and deliberately not
attempted here, because without a live API key there's no way to verify
whether such a loop would actually improve the match or just add noise;
building an unverifiable feedback loop would be worse than leaving the
hook open.

### 12. Personality drift detection (design doc V4) — ✅ done
`database.py::TraitHistory` logs every trait update; `user_model/drift.py`
reads that log and reports traits whose estimate has durably shifted.
Exposed via `GET /user/{id}/drift`, `GET /user/{id}/trait-timeline/{trait}`
(for charting rather than only summarizing), and in the reflection
engine's output.

The EWMA has always *adapted* to change — that was true from V1. What
was missing is *noticing* it: the estimate could slide from 0.8 to 0.3
over weeks with nothing anywhere saying so. Drift is reported, never
auto-applied; the EWMA already handles adaptation, and silently
"correcting" a user model based on inferred drift would violate this
project's rule that the person stays in control of what the system
believes about them.

**Why three thresholds instead of a significance test**: consecutive
EWMA values are autocorrelated by construction (each is a weighted
function of its predecessor), which violates the independence
assumption a t-test needs. Such a test would report significance far
too readily and its p-value would be meaningless — statistically
rigorous-looking and actually wrong. Three interpretable, tunable bars
(magnitude, evidence count, recent-window stability) are honest about
being heuristics and can be explained to a user.

**Known limits**: needs roughly 25+ messages before any trait clears
the post-burn-in evidence bar, so fresh users always get an empty
result (correctly). `TraitHistory` grows a few rows per message with no
retention policy — fine at demo scale, needs downsampling of old rows
in production. And drift currently informs nobody but the reader: no
downstream behavior changes when it fires.

## A note on the trait/emotion math, for anyone auditing it

- **Trait updates** (`user_model/model.py::update_trait`) are an
  exponentially-weighted moving average: `new = (1-lr)*old + lr*observed`,
  where `lr` scales with how strong/explicit the signal was. Confidence
  is `1 - 1/(1 + evidence_count * weight / 4)`, which saturates toward 1
  as evidence accumulates and grows faster for high-weight (explicit)
  signals than for weak stylistic ones. This directly supports
  "personality drift" — a sustained change in behavior moves the
  estimate instead of the first impression being permanent.
- **Emotion scores** (`emotion/engine.py::analyze`) are a weighted blend
  of the trained classifier's calibrated-ish probabilities and the
  lexicon's frequency counts (negation-adjusted), renormalized to sum to
  1 — see `_blend_scores()` for the exact weighting and why categories
  outside the trained model's label space aren't diluted by it. Treat
  `confidence` in `EmotionResult` as a rough heuristic either way, not a
  statistically calibrated number — the trained model's own accuracy
  (40.2%) should keep you from over-trusting any single prediction.
- **Trend detection** (`reflection.py::detect_emotional_trend`) compares
  negative-emotion frequency between two adjacent windows of messages
  and flags a shift ≥40 percentage points. This is literally
  frequency-counting, explicitly not a trained trend model — good
  enough to demonstrate the concept from the design doc ("user seems
  increasingly stressed"), not remotely rigorous enough to base any
  real clinical or high-stakes claim on. The system's own language
  reflects this ("worth noting, not diagnosing").

## Evaluation — ✅ built (V6/V7/V8)

This section used to describe evaluation as future work ("if you want
to take this further"). As of V6, `ml/evaluation/` actually implements
it — six scripts plus a consolidated runner, all runnable in ~15
seconds total. What follows is what's actually there, not a proposal.

- **Memory** (`eval_memory_extraction.py`): precision/recall of
  `extract_memories()` against a 40-message hand-labeled test set. **100%
  precision** (zero false positives on 14 should-skip cases), **85.7%
  recall** (18/21) on should-remember cases — the 3 misses are all cases
  deliberately chosen because they paraphrase around the regex patterns'
  trigger words (e.g. "motivational quotes just don't land for me"
  instead of "I hate/dislike..."), included specifically to avoid
  reporting only favorable results.
- **Emotion** (`eval_emotion_classifier.py`): a genuine 3-arm ablation —
  lexicon-only vs. trained-only vs. blended — on the *same held-out test
  set* the classifier was trained on. The honest, non-obvious finding:
  blending is a small net regression on this specific benchmark (39.8%
  vs. 40.2% accuracy) because this benchmark's ground truth has zero
  fear/disgust/frustration examples, so the trained model can never be
  penalized for missing them while the lexicon's attempt to catch them
  can only cost accuracy here, never gain it. See the script's own
  "Findings" section (computed dynamically, not hardcoded) for the full
  per-class breakdown of where the regression concentrates.
- **Strategy appropriateness** (`eval_strategy_selection.py`): 16
  hand-labeled (message, expected_strategy) cases, **100% accuracy** —
  but getting there required finding and fixing a genuine bug first
  (see below). Still the most subjective of the four evaluations, and
  documented as such.
- **Response critic** (`eval_critic.py`): 22 hand-labeled
  (reply, strategy, situation, should_flag) cases specifically designed
  to stress-test for false positives, not just demonstrate true
  positives — every one of the four heuristic checks has both a
  triggering case and a close-but-shouldn't-trigger case. **100%
  precision/recall/specificity**, reported with an explicit caveat that
  the same person wrote both the heuristics and the test set, so this
  is a regression test confirming intended behavior, not independent
  validation.
- **Style vector** (`eval_style_vector.py`, added V7): a different
  shape of test from the other four, because there's no "ground truth
  style" to check extracted features against. Instead: 6 per-feature
  *directional* discrimination checks against hand-constructed
  formal/casual personas (does the casual persona actually score higher
  on slang_rate, lower on capitalization_rate, etc.) plus 3
  similarity-metric sanity checks — self-similarity ≈ 1.0, two distinct
  personas less similar than either is to itself, and a persona more
  similar to a close variant of itself than to a very different one.
  All 9 pass (self=1.00, close-variant=0.95, cross-style=0.43). This
  validates that the metric behaves sensibly on clearly-differentiated
  input; it does not claim the feature set captures "style" in some
  objective sense, and the report says so.
- **Drift detection** (`eval_drift.py`, added V8): 6 hand-constructed
  trait histories — 2 that should be reported as drift, 4 that should
  not. **6/6 correct.** The four should-stay-silent cases carry the
  weight here: three of them each broke an earlier version of the
  detector (see bugs 4-6 below), and a drift detector that fires
  readily is worse than none at all, since telling someone their
  personality changed every time an estimate settled erodes trust in
  everything else the system claims to know about them.

### Six real bugs this harness actually caught

Worth calling out specifically, because it's the difference between an
evaluation harness that exists for show and one that does its job:

1. **Situation-classification ordering bug.** "My grandmother passed
   away last week" was being classified as an *achievement* (→
   `celebrate`) because `SITUATION_PATTERNS`' `achievement` entry
   (`\bpassed\b`) matched before the `loss` entry (`\bpassed away\b`)
   was ever checked — Python dicts preserve insertion order, and
   `achievement` came first. Never once surfaced across roughly 30 rounds
   of manual and ad hoc testing throughout V1-V5, caught immediately by
   a 16-case hand-labeled strategy test. Fixed by reordering the dict
   *and* adding a negative lookahead (`\bpassed\b(?!\s+away)`) as an
   independent second safeguard, so the fix doesn't silently regress if
   the dict gets reordered again later.
2. **The emotion blend's per-class regression**, described above —
   not a code bug, but a quantitative finding that contradicts the
   naive assumption that "blending two signals is strictly better than
   one." It isn't, on this specific benchmark, for a specific and
   explicable reason. Worth knowing before tuning `TRAINED_WEIGHT`/
   `LEXICON_WEIGHT` any further.
3. **The harness's own packaging bug**, caught testing the harness the
   same way every other fresh-install check in this project has been
   verified: an isolated venv with *only* `ml/requirements.txt`
   installed. `eval_critic.py` failed to import — `app.critic` had a
   module-level `from app.llm.client import complete`, which pulls in
   `requests`, even though the only function the eval script actually
   calls (`heuristic_critique()`) never touches it. The same evaluation
   also surfaced that `TraitEstimate` (a plain 4-field dataclass)
   couldn't be imported without SQLAlchemy, because it lived in
   `user_model/model.py` alongside genuinely DB-dependent functions.
   Fixed by deferring the `requests`-dependent import to inside
   `llm_critique()` (the one function that needs it) and splitting
   `TraitEstimate` into its own dependency-free `user_model/types.py` —
   not by padding `ml/requirements.txt` with dependencies the eval
   scripts don't actually use, which would have hidden the coupling
   instead of fixing it.
4. **Drift detection's burn-in false positives (V8).** Every trait
   starts at exactly 0.5, so its first observations are the EWMA
   escaping that arbitrary default rather than the person changing. A
   simulated user whose messages contained no emoji at all got reported
   as "emoji_tendency decreased 0.40 → 0.17" — the estimate converging
   toward a truth it never knew, presented as a personality shift.
   Fixed by skipping the first 8 observations (with `LEARNING_RATE=0.15`
   the EWMA covers ~73% of the distance to a new level in 8 steps, so
   what remains is dominated by real signal).
5. **Drift detection missing the pitch's own example (V8).** A user
   going from long, detailed messages to terse ones had `verbosity` run
   0.51 → 0.60 → 0.30 — an unmistakable reversal — but early-vs-recent
   window means differed by only ~0.13, under the magnitude bar,
   because the early window caught the estimate still *climbing* away
   from its default before it reversed. The headline feature nearly
   shipped unable to detect the exact scenario it was built for. Fixed
   by also comparing the recent window against the most extreme
   sustained window anywhere earlier in the series.
6. **An over-correction for bug 4 that reintroduced bug 5 (V8).** The
   first filter written to suppress initialization decay compared only
   max-excursion against the final value, ignoring *direction* — which
   also matched the genuine verbosity reversal and suppressed it too.
   Fixed by checking sidedness instead: a real reversal visits both
   sides of the 0.5 default, while pure initialization decay stays
   strictly on one side the whole time. Worth recording as its own
   entry rather than folding into bug 4, because "the fix broke
   something else and the eval caught that too" is the more useful
   lesson.

Comparing "generic system prompt" vs. "this full pipeline" — the
ablation-study structure the very first version of this plan asked
for — is exactly what `eval_emotion_classifier.py`'s three arms do,
just scoped to the emotion-detection component specifically rather
than the whole pipeline end-to-end (an end-to-end version would need a
labeled set of full conversations with quality judgments, which no
public dataset reachable from this sandbox provides, and building one
by hand at meaningful scale was out of scope for this round).

### What's still not evaluated

The original Phase 13 list had a fourth item this harness doesn't
cover: **user model convergence** — does the inferred trait vector
actually match a person's real self-reported communication style after
N messages? This needs a live person to rate their own
directness/formality/humor and compare against what the system
inferred from a real conversation, which isn't something a hand-labeled
static test set can substitute for (unlike the other three, which are
all "given this input, is the output defensible" checks). Left open
here specifically because it's a different *kind* of evaluation
(requires live human subjects) rather than another script to write.
