#!/usr/bin/env python3
"""Exports training data for Persona Fine-Tuning (Section 10 of pitch doc).

SFT dataset: (system_prompt, user_message) -> accepted response, one example per
accepted candidate. Only accepted responses are used — this is "high-quality examples
of how your character should respond," per the doc's Stage 1 description.

DPO dataset: for the same (user_id, session_id, user_message) context, if generation
required more than one attempt, pairs the best-scoring candidate (chosen) against a
worse-scoring one (rejected). This is real preference data derived from the evaluator's
own judgments, not synthetic labels — though note it inherits whatever bias the
evaluator has, which is itself heuristic/LLM-judged in this build (see app/evaluator.py).
A production version should mix in actual human preference judgments once available.

Usage:
  python training/export_dataset.py --database-url sqlite:///data/persona.db \
      --sft-out data/sft_dataset.jsonl --dpo-out data/dpo_dataset.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import DB


def export_sft(db: DB, out_path: str) -> int:
    count = 0
    with db.connect() as conn, open(out_path, "w", encoding="utf-8") as f:
        rows = conn.execute(
            "SELECT user_message, system_prompt, response, eval_json FROM candidates "
            "WHERE accepted = 1 ORDER BY id"
        ).fetchall()
        for row in rows:
            record = {
                "messages": [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_message"]},
                    {"role": "assistant", "content": row["response"]},
                ],
                "eval": json.loads(row["eval_json"]),
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    return count


def export_dpo(db: DB, out_path: str, min_gap: float = 0.08) -> int:
    """Groups candidates by (user_id, session_id, user_message) — i.e. all attempts at
    generating a response to the same message — and emits a preference pair when the
    best and worst candidates differ by at least `min_gap` in overall eval score.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT user_id, session_id, user_message, system_prompt, response, eval_json "
            "FROM candidates ORDER BY id"
        ).fetchall()
    for row in rows:
        key = (row["user_id"], row["session_id"], row["user_message"])
        groups[key].append(dict(row))

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for key, candidates in groups.items():
            if len(candidates) < 2:
                continue
            scored = [(json.loads(c["eval_json"])["overall"], c) for c in candidates]
            scored.sort(key=lambda x: -x[0])
            best_score, best = scored[0]
            worst_score, worst = scored[-1]
            if best_score - worst_score < min_gap:
                continue
            record = {
                "system": best["system_prompt"],
                "user_message": best["user_message"],
                "chosen": best["response"],
                "rejected": worst["response"],
                "chosen_score": best_score,
                "rejected_score": worst_score,
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None, help="defaults to DATABASE_URL env or local SQLite")
    parser.add_argument("--sft-out", default="data/sft_dataset.jsonl")
    parser.add_argument("--dpo-out", default="data/dpo_dataset.jsonl")
    parser.add_argument("--dpo-min-gap", type=float, default=0.08)
    args = parser.parse_args()

    db = DB(args.database_url)
    Path(args.sft_out).parent.mkdir(parents=True, exist_ok=True)

    sft_count = export_sft(db, args.sft_out)
    dpo_count = export_dpo(db, args.dpo_out, min_gap=args.dpo_min_gap)

    print(f"SFT examples written: {sft_count} -> {args.sft_out}")
    print(f"DPO preference pairs written: {dpo_count} -> {args.dpo_out}")
    if sft_count == 0:
        print("No accepted candidates found yet — run some conversations first "
              "(scripts/chat_cli.py or scripts/longitudinal_eval.py).")


if __name__ == "__main__":
    main()
