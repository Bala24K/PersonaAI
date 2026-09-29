import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.bootstrap_dataset import generate
from app.ml.eq_model import PersonaEQModel, evaluate, train_and_evaluate


def test_bootstrap_dataset_generates_labeled_examples():
    examples = generate(n_per_class=5)
    assert len(examples) > 0
    for e in examples:
        assert e.text and e.emotion and e.intent and e.need
        assert 0 <= e.intensity <= 1


def test_eq_model_trains_and_predicts():
    model, metrics = train_and_evaluate(n_per_class=20)
    assert metrics.intent_accuracy > 0.5  # should beat random guessing by a lot on templated data
    pred = model.predict("I am so frustrated with this project, nothing is going right")
    assert "emotion" in pred and "intent" in pred and "social_state" in pred
    assert 0 <= pred["intensity"] <= 1


def test_eq_model_save_and_load_roundtrip(tmp_path):
    model, _ = train_and_evaluate(n_per_class=15)
    path = str(tmp_path / "eq_model_test.pkl")
    model.save(path)
    loaded = PersonaEQModel.load(path)
    pred = loaded.predict("thanks so much, that really helped")
    assert "intent" in pred


def test_trained_eq_estimator_matches_interface(tmp_path):
    from app.eq_estimator import EQState
    from app.eq_estimator_trained import TrainedEQEstimator

    model, _ = train_and_evaluate(n_per_class=15)
    path = str(tmp_path / "eq_model_test.pkl")
    model.save(path)

    estimator = TrainedEQEstimator(model_path=path)
    prev = EQState.initial()
    new_state = estimator.update([], "I'm really anxious about tomorrow", prev)
    assert isinstance(new_state, EQState)
    assert 0 <= new_state.intensity <= 1
    assert new_state.social_state.keys() == {"openness", "trust", "irritation"}


def test_export_pipeline_produces_valid_jsonl(tmp_path):
    from app.character import Character
    from app.orchestrator import Persona
    from training.export_dataset import export_dpo, export_sft

    db_path = str(tmp_path / "export_test.db")
    persona = Persona(Character.from_yaml("characters/default_character.yaml"), db_path=db_path)
    for msg in ["hey how's it going", "I'm pretty stressed about work", "thanks for listening"]:
        persona.respond("export_test_user", "s1", msg)

    sft_path = str(tmp_path / "sft.jsonl")
    dpo_path = str(tmp_path / "dpo.jsonl")
    sft_count = export_sft(persona.db, sft_path)
    dpo_count = export_dpo(persona.db, dpo_path, min_gap=0.0)

    assert sft_count >= 1
    with open(sft_path) as f:
        for line in f:
            record = json.loads(line)
            assert "messages" in record
            roles = [m["role"] for m in record["messages"]]
            assert roles == ["system", "user", "assistant"]
    # dpo_count may be 0 if no message needed regeneration — that's valid, just check
    # the file is well-formed if anything was written
    if dpo_count > 0:
        with open(dpo_path) as f:
            for line in f:
                record = json.loads(line)
                assert "chosen" in record and "rejected" in record


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
