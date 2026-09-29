"""
Ablation study: does blending the lexicon with the trained classifier
actually help, or would either alone do just as well?

This is the concrete version of the original design doc's central
experiment ("Does persistent user modeling / does component X improve
personalized conversation?") applied to what's actually been built:
three arms, evaluated on the *exact same held-out test set* the
classifier itself was evaluated on (same CSV, same cleaning, same
label mapping, same train_test_split random_state=42 as
train_emotion_classifier.py — reconstructed here rather than saved
separately, so this script has zero dependency on training-run
artifacts beyond the dataset and the trained model file).

Arms:
  A. Lexicon only   — app.emotion.lexicon's word lists, no trained model
  B. Trained only    — the TF-IDF+LogisticRegression classifier alone
  C. Blended         — what actually ships (app.emotion.engine.analyze),
                        including the flat-affect override, exactly as
                        production behaves

All three predict into the same 6-class space the classifier was
trained on (neutral, anxiety, joy, sadness, surprise, anger) so the
comparison is apples-to-apples. Arm A will structurally never predict
fear/disgust/frustration correctly on this dataset, because this
dataset's ground truth never contains those labels (the raw CrowdFlower
categories don't have them) — that's a known, stated limitation of the
*dataset*, not a bug in the lexicon; see train_emotion_classifier.py's
docstring.

Run from ml/: python3 evaluation/eval_emotion_classifier.py
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.emotion.engine import analyze as blended_analyze  # noqa: E402
from app.emotion.engine import _detect_emotions as lexicon_detect  # noqa: E402
from app.emotion import trained_classifier  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "text_emotion.csv"
REPORT_OUT = Path(__file__).resolve().parent.parent / "models" / "ablation_report.txt"

# Must exactly match train_emotion_classifier.py's mapping so the
# reconstructed test split is identical to what the model was evaluated
# on during training.
LABEL_MAP = {
    "neutral": "neutral", "worry": "anxiety", "happiness": "joy", "love": "joy",
    "fun": "joy", "relief": "joy", "enthusiasm": "joy", "sadness": "sadness",
    "empty": "sadness", "surprise": "surprise", "hate": "anger", "anger": "anger",
}
EVAL_CLASSES = {"neutral", "anxiety", "joy", "sadness", "surprise", "anger"}

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return text.strip()


def predict_lexicon_only(text: str) -> str:
    scores = lexicon_detect(text)
    if not scores:
        return "neutral"
    top = max(scores, key=scores.get)
    return top if top in EVAL_CLASSES else "other"


def predict_trained_only(text: str) -> str:
    probs = trained_classifier.predict(text)
    if not probs:
        return "neutral"
    return max(probs, key=probs.get)


def predict_blended(text: str) -> str:
    result = blended_analyze(text)
    return result.dominant if result.dominant in EVAL_CLASSES else "other"


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["sentiment"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["text"] = df["content"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]

    # Identical split call to train_emotion_classifier.py -> identical
    # held-out test set, so arm B's numbers here should match
    # training_report.txt exactly (sanity check for the harness itself).
    _, X_test, _, y_test = train_test_split(
        df["text"], df["label"], test_size=0.15, random_state=42, stratify=df["label"]
    )
    X_test = X_test.tolist()
    y_test = y_test.tolist()
    print(f"Evaluating on {len(X_test)} held-out examples (same split used in training)\n")

    results = {}
    for arm_name, predict_fn in [
        ("A_lexicon_only", predict_lexicon_only),
        ("B_trained_only", predict_trained_only),
        ("C_blended", predict_blended),
    ]:
        t_arm = time.time()
        preds = [predict_fn(t) for t in X_test]
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro", labels=sorted(EVAL_CLASSES))
        results[arm_name] = {
            "accuracy": acc, "macro_f1": macro_f1,
            "report": classification_report(y_test, preds, labels=sorted(EVAL_CLASSES), digits=3, zero_division=0),
            "time_s": time.time() - t_arm,
        }
        print(f"[{arm_name}] accuracy={acc:.3f}  macro_f1={macro_f1:.3f}  ({time.time()-t_arm:.1f}s)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Per-class recall delta (blended vs trained-only) — computed
    # dynamically so this stays correct if blend weights change later,
    # rather than hardcoding today's specific numbers into the write-up.
    from sklearn.metrics import recall_score
    trained_preds = [predict_trained_only(t) for t in X_test]
    blended_preds = [predict_blended(t) for t in X_test]
    classes_sorted = sorted(EVAL_CLASSES)
    trained_recall = recall_score(y_test, trained_preds, labels=classes_sorted, average=None, zero_division=0)
    blended_recall = recall_score(y_test, blended_preds, labels=classes_sorted, average=None, zero_division=0)
    recall_deltas = {c: blended_recall[i] - trained_recall[i] for i, c in enumerate(classes_sorted)}

    with open(REPORT_OUT, "w") as f:
        f.write("Emotion detection ablation study\n")
        f.write(f"Held-out test set: {len(X_test)} examples (same split as train_emotion_classifier.py, random_state=42)\n\n")
        f.write("Summary\n-------\n")
        f.write(f"{'Arm':<20}{'Accuracy':<12}{'Macro F1':<12}\n")
        for arm_name, r in results.items():
            f.write(f"{arm_name:<20}{r['accuracy']:<12.3f}{r['macro_f1']:<12.3f}\n")
        f.write("\nRandom baseline: {:.3f}  |  Majority-class baseline: {:.3f}\n".format(
            1 / len(EVAL_CLASSES), pd.Series(y_test).value_counts(normalize=True).iloc[0]
        ))

        acc_delta = results["C_blended"]["accuracy"] - results["B_trained_only"]["accuracy"]
        f.write("\nFindings\n--------\n")
        f.write(
            f"Blended vs. trained-only on THIS benchmark: {acc_delta:+.3f} accuracy "
            f"({'a small regression' if acc_delta < 0 else 'a small improvement' if acc_delta > 0 else 'no change'}).\n"
        )
        f.write(
            "Per-class recall change (blended minus trained-only), most affected first:\n"
        )
        for cls, delta in sorted(recall_deltas.items(), key=lambda kv: kv[1]):
            f.write(f"  {cls:<10} {delta:+.3f}\n")
        f.write(
            "\nImportant caveat this number cannot capture: this dataset's ground truth\n"
            "has ZERO examples of fear, disgust, or frustration (the raw source data\n"
            "simply doesn't contain those categories — see train_emotion_classifier.py).\n"
            "The trained classifier structurally cannot predict them at all, so it can\n"
            "never be penalized for missing them on this benchmark, while the lexicon's\n"
            "attempt to catch them can only ever cost accuracy here, never gain it — a\n"
            "message the lexicon correctly reads as fear/disgust/frustration is *always*\n"
            "counted wrong against this dataset's labels, because the true label is\n"
            "necessarily one of the six classes the trained model knows about instead.\n"
            "This benchmark is a fair test of the six shared classes and an inherently\n"
            "unfair one for the actual reason the blend exists. A held-out set with real\n"
            "fear/disgust/frustration examples would be needed to measure that honestly —\n"
            "not built here; see docs/ARCHITECTURE.md for why (no such labeled data was\n"
            "reachable from this sandbox's allowed domains).\n"
        )

        for arm_name, r in results.items():
            f.write(f"\n\n=== {arm_name} — full per-class report ===\n")
            f.write(r["report"])

    print(f"\nSaved full report to {REPORT_OUT}")


if __name__ == "__main__":
    main()
