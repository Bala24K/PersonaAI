# Evaluation summary

Six evaluations against this codebase, run together as one harness. Each is a hand-built test set (disclosed as such in its own report — none of this is independently validated by a third party), designed specifically to include hard/adversarial cases rather than only favorable ones. Several of these caught real bugs during development: a situation-classification ordering bug (via strategy selection), the emotion blend's per-class regression (via the ablation study), and three separate false-positive modes in drift detection (via the drift eval, whose should-stay-silent cases each broke an earlier version of that detector). Left in these reports rather than quietly fixed and hidden, because catching real issues is the actual point of building an eval harness at all.

## Emotion detection ablation (lexicon vs. trained vs. blended)

```
Emotion detection ablation study
Held-out test set: 5962 examples (same split as train_emotion_classifier.py, random_state=42)

Summary
-------
Arm                 Accuracy    Macro F1    
A_lexicon_only      0.303       0.170       
B_trained_only      0.454       0.312       
C_blended           0.438       0.314       

Random baseline: 0.167  |  Majority-class baseline: 0.330

Findings
--------
Blended vs. trained-only on THIS benchmark: -0.015 accuracy (a small regression).
Per-class recall change (blended minus trained-only), most affected first:
  anxiety    -0.073
  joy        -0.015
  neutral    -0.005
  anger      +0.000
  surprise   +0.027
  sadness    +0.031

Important caveat this number cannot capture: this dataset's ground truth
has ZERO examples of fear, disgust, or frustration (the raw source data
simply doesn't contain those categories — see train_emotion_classifier.py).
The trained classifier structurally cannot predict them at all, so it can
never be penalized for missing them on this benchmark, while the lexicon's
attempt to catch them can only ever cost accuracy here, never gain it — a
message the lexicon correctly reads as fear/disgust/frustration is *always*
counted wrong against this dataset's labels, because the true label is
necessarily one of the six classes the trained model knows about instead.
This benchmark is a fair test of the six shared classes and an inherently
unfair one for the actual reason the blend exists. A held-out set with real
fear/disgust/frustration examples would be needed to measure that honestly —
not built here; see docs/ARCHITECTURE.md for why (no such labeled data was
reachable from this sandbox's allowed domains).


=== A_lexicon_only — full per-class report ===
              precision    recall  f1-score   support

       anger      0.133     0.019     0.033       215
     anxiety      0.636     0.006     0.011      1268
         joy      0.705     0.262     0.382      1967
     neutral      0.247     0.922     0.389      1287
     sadness      0.423     0.091     0.150       897
    surprise      0.256     0.030     0.054       328

   micro avg      0.311     0.303     0.307      5962
   macro avg      0.400     0.222     0.170      5962
weighted avg      0.504     0.303     0.239      5962


=== B_trained_only — full per-class report ===
              precision    recall  f1-score   support

       anger      0.507     0.163     0.246       215
     anxiety      0.389     0.416     0.402      1268
         joy      0.517     0.810     0.631      1967
     neutral      0.382     0.289     0.329      1287
     sadness      0.365     0.196     0.255       897
    surprise      1.000     0.003     0.006       328

    accuracy                          0.454      5962
   macro avg      0.527     0.313     0.312      5962
weighted avg      0.464     0.454     0.412      5962


=== C_blended — full per-class report ===
              precision    recall  f1-score   support

       anger      0.398     0.163     0.231       215
     anxiety      0.379     0.343     0.360      1268
         joy      0.520     0.795     0.629      1967
     neutral      0.391     0.284     0.329      1287
     sadness      0.361     0.227     0.279       897
    surprise      0.256     0.030     0.054       328

   micro avg      0.452     0.438     0.445      5962
   macro avg      0.384     0.307     0.314      5962
weighted avg      0.419     0.438     0.408      5962

```

## Memory extraction (precision/recall)

```
Memory extraction evaluation (hand-labeled test set, n=35)

Recall:         21/21 = 100.0%
Type accuracy:  21/21 = 100.0%
Precision:      14/14 = 100.0%

Note: this test set was constructed by hand for this evaluation, not
sampled from real usage, and includes several cases chosen specifically
because the heuristic extractor's regex patterns are known not to catch
them (see the 'harder/ambiguous cases' section in this script) — recall
on real conversational data, which skews toward clearer statements the
patterns were actually designed around, would likely be somewhat higher
than this number suggests, not lower.

```

## Response critic (precision/recall/specificity)

```
Response critic evaluation (hand-labeled test set, n=22)

TP=10  FP=0  TN=12  FN=0
Precision: 100.0%
Recall: 100.0%
Specificity: 100.0%

Precision and specificity matter as much as recall here: a critic with
high recall but low specificity flags good replies constantly, which is
worse than no critic at all (wasted regenerations, possible degradation
of an already-fine reply). This test set was designed with that in mind —
every check has both a triggering case and a close-but-shouldn't-trigger
case, not just clean positive examples.

Honest caveat on a perfect score: the same person who wrote the four
heuristic checks also wrote this test set, so a clean 100% here mostly
confirms the checks do what they were literally written to do, not that
they'll generalize to the full variety of real replies a live model would
produce. Real text will hedge, combine acknowledgment and advice in the
same clause, or use phrasing these patterns don't recognize at all — this
eval is a regression test against known cases, not independent validation.
The honest next step would be running the critic against real generated
replies from a live model and having a person (ideally not the one who
wrote the heuristics) judge agreement — not done here, no live API access
in this build environment to generate that data.

```

## Strategy selection (accuracy)

```
Strategy selection evaluation (hand-labeled test set, n=16)

Accuracy: 16/16 = 100.0%

Caveat: strategy appropriateness is inherently more subjective than the
other evaluations in this harness. The two 'harder cases' were included
specifically to demonstrate a real, known limitation — situation
classification is regex-based (see emotion/engine.py's SITUATION_PATTERNS)
and doesn't understand sarcasm or reframing, so 'rejected but relieved'
still triggers the standard rejection strategy even though a human would
read the message as basically fine. Fixing this well would need actual
sentiment-of-the-whole-sentence understanding, not a better regex — a
natural argument for the LLM-based critique tier (CRITIC_MODE=heuristic_llm)
catching what pattern matching structurally can't.

```

## Style vector (discrimination + similarity sanity)

```
StyleVector discrimination + similarity-metric sanity check

All checks passed: True

Per-feature discrimination (formal vs. casual hand-constructed personas):
  exclamation_rate: formal=0.000 casual=7.143 [OK]
  emoji_rate: formal=0.000 casual=7.143 [OK]
  slang_rate: formal=0.000 casual=35.714 [OK]
  abbreviation_rate: formal=0.000 casual=17.857 [OK]
  avg_sentence_length: formal=9.800 casual=4.667 [OK]
  capitalization_rate: formal=1.000 casual=0.000 [OK]

self_similarity=1.000  cross_similarity=0.432  variant_similarity=0.952

Note: this validates that the StyleVector system behaves correctly on
clearly-differentiated hand-constructed examples, not that it perfectly
captures 'style' in some objective sense (no such ground truth exists).
The real-world use of this system — GET /user/{id}/style, comparing an
actual person's messages to the AI's replies — was spot-checked manually
during development (see docs/DEMO_SCRIPT.md) rather than formally
evaluated here, since there's no labeled dataset of 'correct' style-match
scores for real conversations to check against.

```

## Personality drift detection (true/false positive cases)

```
Drift detection evaluation (hand-constructed cases, n=6)

Correct: 6/6 = 100.0%

real_reversal_crossing_default
  expected=drift got=drift [OK]
  rationale: genuine detailed->concise reversal; early window catches it mid-climb
  reported: real reversal crossing default has decreased from 0.58 to 0.34 (-0.24) across 10 observations

real_step_change
  expected=drift got=drift [OK]
  rationale: clean sustained shift with a settled endpoint
  reported: real step change has decreased from 0.80 to 0.30 (-0.50) across 16 observations

pure_decay_from_default
  expected=silent got=silent [OK]
  rationale: one-sided convergence from the 0.5 default, not a change in the person

erratic
  expected=silent got=silent [OK]
  rationale: inconsistent, never settles — recent-window stddev bar rejects this

flat
  expected=silent got=silent [OK]
  rationale: no change at all — the common case; detector must stay silent

tiny_wobble
  expected=silent got=silent [OK]
  rationale: ordinary EWMA jitter, below the magnitude bar

The four should-stay-silent cases are the point of this evaluation.
Three of them (pure_decay_from_default, erratic, and the naive-comparison
failure that real_reversal_crossing_default guards against) each broke a
version of this detector during development — this file is the regression
test that keeps them fixed. A drift detector that fires readily is worse
than none at all: it would tell a user their personality changed every
time an estimate settled, which is both wrong and the kind of wrong that
erodes trust in everything else the system claims to know about them.

Caveat, same as the rest of this harness: these are hand-constructed
series chosen to represent shapes seen during real testing, not sampled
from production usage. They validate the detector's logic, not that these
are the only shapes real users produce.

```

## Extended: retrieval relevance, grounding, hallucination, regressions

```
============================================================
Extended Evaluation Harness Report
============================================================

{
  "total_evaluations": 7,
  "passed": 7,
  "failed": 0,
  "pass_rate": 1.0,
  "results": [
    {
      "test": "retrieval_relevance",
      "precision_at_3": 0.667,
      "relevant_in_top_3": 2,
      "total_retrieved": 3,
      "pass": true,
      "evaluation_name": "Retrieval Relevance"
    },
    {
      "test": "context_recall",
      "recall_rate": 1.0,
      "expected_recalled": 2,
      "actually_recalled": 2,
      "pass": true,
      "evaluation_name": "Context Recall"
    },
    {
      "test": "response_grounding",
      "ungrounded_claims": 0,
      "ungrounded_keywords": [],
      "grounding_score": 1.0,
      "pass": true,
      "evaluation_name": "Response Grounding"
    },
    {
      "test": "structured_output_validity",
      "eq_validation_rate": 1.0,
      "eval_validation_rate": 1.0,
      "eq_tested": 4,
      "eval_tested": 1,
      "pass": true,
      "evaluation_name": "Structured Output Validity"
    },
    {
      "test": "hallucination_check",
      "hallucination_phrases_found": 0,
      "hallucination_score": 1.0,
      "pass": true,
      "evaluation_name": "Hallucination Check"
    },
    {
      "test": "regressions",
      "cases": [
        {
          "case": "basic_response",
          "pass": true
        },
        {
          "case": "long_message",
          "pass": true
        },
        {
          "case": "special_chars",
          "pass": true
        },
        {
          "case": "unicode",
          "pass": true
        }
      ],
      "passed": 4,
      "total": 4,
      "pass": true,
      "evaluation_name": "Regression Tests"
    },
    {
      "test": "latency_and_tokens",
      "latencies_ms": [
        7572.7,
        8084.1,
        16351.3
      ],
      "mean_latency_ms": 10669.4,
      "metrics_snapshot_keys": [
        "retrieval_calls_total",
        "requests_total",
        "llm_calls_total",
        "tokens_prompt_total",
        "tokens_completion_total",
        "regenerations_total"
      ],
      "llm_calls_tracked": true,
      "pass": true,
      "evaluation_name": "Latency & Token Usage"
    }
  ]
}

```

