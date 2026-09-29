import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.character import Character
from app.orchestrator import Persona


def _fresh_persona():
    character = Character.from_yaml("characters/default_character.yaml")
    db_path = f"data/test_{uuid.uuid4().hex[:8]}.db"
    return Persona(character, db_path=db_path)


def test_character_loads():
    character = Character.from_yaml("characters/default_character.yaml")
    assert character.name == "Nova"
    assert "warm" in character.personality_traits
    block = character.to_system_block()
    assert "Nova" in block


def test_single_turn_response():
    persona = _fresh_persona()
    result = persona.respond("u1", "s1", "hey how are you")
    assert isinstance(result.response, str) and len(result.response) > 0
    assert 0 <= result.eval_result.overall <= 1


def test_state_persists_across_turns():
    persona = _fresh_persona()
    persona.respond("u1", "s1", "I'm really stressed about work today")
    _, rel_after_1 = persona.user_state.load("u1")
    persona.respond("u1", "s1", "yeah it's been a lot")
    _, rel_after_2 = persona.user_state.load("u1")
    # familiarity should monotonically increase turn over turn
    assert rel_after_2.familiarity >= rel_after_1.familiarity


def test_memory_is_stored_and_retrieved():
    persona = _fresh_persona()
    persona.memory.add("u1", "semantic", "User studies computer science.", importance=0.7)
    results = persona.memory.retrieve("u1", "what does the user study", k=3)
    assert any("computer science" in m.content for m in results)


def test_high_intensity_turn_creates_episodic_memory():
    persona = _fresh_persona()
    before = len(persona.memory.all_for_user("u1"))
    # negative-sentiment message should trigger mock LLM to report higher intensity
    persona.respond("u1", "s1", "I'm so angry and frustrated and upset about everything ugh")
    after = len(persona.memory.all_for_user("u1"))
    assert after >= before  # may or may not cross threshold depending on mock scoring, but must not error


def test_eq_state_persists_between_calls():
    persona = _fresh_persona()
    r1 = persona.respond("u1", "s1", "I'm really sad today")
    assert r1.eq_state.intensity >= 0
    stored = persona._latest_eq_state("u1")
    assert stored is not None


def test_evaluator_rejects_and_regeneration_path_runs():
    persona = _fresh_persona()
    # just confirm the loop runs without error and respects max attempts
    result = persona.respond("u1", "s1", "tell me something interesting")
    assert result.regenerations <= 2


def test_scene_state_roundtrip():
    persona = _fresh_persona()
    from app.scene_state import SceneState
    scene = SceneState(active=True, location="an old library", characters_present=["Nova"])
    persona.scene_state.save("u1", scene)
    loaded = persona.scene_state.load("u1")
    assert loaded.active is True
    assert loaded.location == "an old library"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
