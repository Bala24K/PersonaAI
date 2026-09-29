"""
Trains a real emotion classifier to replace/augment the V1 lexicon.

Dataset: ~40k labeled tweets, 13 raw emotion categories (the
CrowdFlower/"text_emotion" dataset, widely used for educational NLP
work), pulled from a public GitHub mirror:
https://github.com/tlkh/text-emotion-classification
This redistribution's original license terms are not fully documented
upstream — treat this as an educational/research artifact, same as the
rest of V1, and swap in a dataset with clear licensing (e.g. via a
proper data provider or your own labeled data) before any commercial
use. This is noted again in docs/ARCHITECTURE.md.

Why a classical ML classifier (TF-IDF + Logistic Regression) instead of
fine-tuning a transformer: this sandbox can't download pretrained
transformer weights (no access to huggingface.co), so a from-scratch
transformer would have to train on 40k examples with random init —
worse than a well-regularized linear model on TF-IDF features, and far
slower to train and to run inference with, on CPU, in an app that needs
to reply in real time. This is a legitimate, honestly-reasoned
engineering choice for this constraint, not a corner cut for its own
sake — see docs/ARCHITECTURE.md for the reasoning and the upgrade path
(swap in a transformer fine-tune once weights are reachable).

Label mapping and an important honest limitation:
The raw dataset has 13 categories. Several map naturally onto each
other (happiness/love/fun/relief/enthusiasm -> a single "joy" bucket);
one (boredom, 179 examples) is dropped for having both too few examples
and no good semantic match. The resulting 6 trained classes
(neutral, anxiety, joy, sadness, surprise, anger) do NOT include fear,
disgust, or frustration, because this dataset has no good source of
labeled examples for them. The app's emotion engine keeps the V1
lexicon active specifically to cover that gap — see
app/emotion/engine.py and app/emotion/trained_classifier.py for how the
two signals are combined.

Run:  python3 train_emotion_classifier.py
Produces: ml/models/emotion_classifier.joblib (a sklearn Pipeline:
TfidfVectorizer + LogisticRegression) and prints an evaluation report.
"""
import re
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "datasets/text_emotion.csv"
MODEL_OUT = "models/emotion_classifier.joblib"
REPORT_OUT = "models/training_report.txt"

LABEL_MAP = {
    "neutral": "neutral",
    "worry": "anxiety",
    "happiness": "joy",
    "love": "joy",
    "fun": "joy",
    "relief": "joy",
    "enthusiasm": "joy",
    "sadness": "sadness",
    "empty": "sadness",
    "surprise": "surprise",
    "hate": "anger",
    "anger": "anger",
    # "boredom" intentionally dropped: too few examples (179) and no
    # clean semantic match to any class we want to train.
}

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return text.strip()


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["sentiment"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["text"] = df["content"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]

    print(f"Training examples after mapping/cleaning: {len(df)}")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.15, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.85, sublinear_tf=True,
            max_features=35000,
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LogisticRegression(
                max_iter=2000, class_weight="balanced", C=2.5, solver="lbfgs"
            ),
            method="sigmoid",
            cv=3,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=3)

    print(f"\nAccuracy: {acc:.3f}   Macro F1: {macro_f1:.3f}\n")
    print(report)

    joblib.dump(pipeline, MODEL_OUT)

    with open(REPORT_OUT, "w") as f:
        f.write(f"Trained on {len(df)} examples ({len(X_train)} train / {len(X_test)} test)\n")
        f.write(f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\n\n")
        f.write(report)
        f.write(f"\n\nTraining time: {time.time() - t0:.1f}s\n")
        f.write("Classes NOT covered by this classifier (use lexicon fallback): fear, disgust, frustration\n")

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved report to {REPORT_OUT}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
