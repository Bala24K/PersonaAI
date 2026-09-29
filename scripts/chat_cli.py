#!/usr/bin/env python3
"""Interactive CLI to talk to a Persona. State persists across runs via SQLite."""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.character import Character
from app.orchestrator import Persona


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", default="characters/default_character.yaml")
    parser.add_argument("--user_id", default="cli_user")
    parser.add_argument("--session_id", default=None)
    parser.add_argument("--show_debug", action="store_true", help="print EQ state / eval scores each turn")
    args = parser.parse_args()

    character = Character.from_yaml(args.character)
    persona = Persona(character)
    session_id = args.session_id or str(uuid.uuid4())[:8]

    print(f"Talking to {character.name}. User: {args.user_id}  Session: {session_id}")
    print("(Ctrl+C or 'exit' to quit)\n")

    while True:
        try:
            message = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not message:
            continue
        if message.lower() in ("exit", "quit"):
            break

        result = persona.respond(args.user_id, session_id, message)
        print(f"{character.name}> {result.response}\n")

        if args.show_debug:
            print(f"  [debug] eq={result.eq_state.to_dict()}")
            print(f"  [debug] eval={result.eval_result.to_dict()} regenerations={result.regenerations}\n")


if __name__ == "__main__":
    main()
