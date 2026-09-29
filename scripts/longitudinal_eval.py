#!/usr/bin/env python3
"""Runs a synthetic multi-turn conversation and reports the drift metrics from
Section 6/Phase 6 of the pitch doc: persona consistency, repetition rate, memory
recall, relationship-state progression. This is what makes the stated failure mode
("degrades as history grows", not "bad immediately") actually observable.
"""
import argparse
import statistics
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.character import Character
from app.orchestrator import Persona

SYNTHETIC_MESSAGES = [
    "hey, how's it going",
    "I've had a really rough week at work honestly",
    "my manager keeps giving me feedback in front of the whole team and it's humiliating",
    "yeah I don't know, maybe I'm overreacting",
    "anyway how are you",
    "I actually got some good news too, I might get promoted",
    "haha thanks, I'm pretty excited",
    "can I tell you about a project I'm working on?",
    "it's a personal ML project, kind of like a companion AI",
    "the hard part is memory, it forgets stuff after a while",
    "do you remember what I said about my manager?",
    "yeah exactly, that's the thing that's been bugging me",
    "let's do something different, want to do a quick roleplay?",
    "we're exploring an old library, you're my guide",
    "what do we find first",
    "ooh that's cool, what's in the room",
    "let's keep going deeper",
    "okay pause the roleplay for a sec",
    "remember that promotion thing I mentioned? it fell through",
    "yeah I'm frustrated about it honestly",
    "anyway, remind me what we were exploring in the library",
]


def run(turns: int, character_path: str):
    character = Character.from_yaml(character_path)
    db_path = f"data/eval_{uuid.uuid4().hex[:8]}.db"
    persona = Persona(character, db_path=db_path)

    user_id = "eval_user"
    session_id = "eval_session"

    overall_scores = []
    repetition_scores = []
    persona_scores = []
    regenerations_total = 0
    responses = []

    for i in range(turns):
        message = SYNTHETIC_MESSAGES[i % len(SYNTHETIC_MESSAGES)]
        result = persona.respond(user_id, session_id, message)
        overall_scores.append(result.eval_result.overall)
        repetition_scores.append(result.eval_result.repetition)
        persona_scores.append(result.eval_result.persona_consistency)
        regenerations_total += result.regenerations
        responses.append(result.response)

    early = overall_scores[: max(len(overall_scores) // 4, 1)]
    late = overall_scores[-max(len(overall_scores) // 4, 1):]

    print(f"\n=== Longitudinal eval: {turns} turns, db={db_path} ===")
    print(f"Overall score  — mean: {statistics.mean(overall_scores):.3f}  "
          f"early-quartile: {statistics.mean(early):.3f}  late-quartile: {statistics.mean(late):.3f}")
    print(f"Repetition     — mean: {statistics.mean(repetition_scores):.3f}  "
          f"(higher = more repetitive; watch for upward drift)")
    print(f"Persona consist— mean: {statistics.mean(persona_scores):.3f}")
    print(f"Total regenerations triggered: {regenerations_total} / {turns} turns")

    _, relationship = persona.user_state.load(user_id)
    print(f"\nFinal relationship state: {relationship}")

    memories = persona.memory.all_for_user(user_id)
    print(f"Memories accumulated: {len(memories)}")
    for m in memories[:5]:
        print(f"  - [{m.kind}] {m.content[:80]}")

    # crude "did it recall the manager thing when asked" check
    recall_check = "manager" in responses[10].lower() or "humiliat" in responses[10].lower()
    print(f"\nMemory-recall spot check (turn 11, asked to recall manager complaint): "
          f"{'plausible recall' if recall_check else 'NO recall signal — check retrieval'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--character", default="characters/default_character.yaml")
    args = parser.parse_args()
    run(min(args.turns, len(SYNTHETIC_MESSAGES) * 3), args.character)
