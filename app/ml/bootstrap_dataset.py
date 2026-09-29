"""Generates a labeled synthetic training set for the EQ model.

No real user conversation data exists yet (this is a new product), so we bootstrap from
templates + controlled variation. This is explicitly a stand-in: swap this out for real
labeled conversation data (ideally human-annotated) the moment you have it — the training
script (train_eq_model.py) doesn't care where the data comes from, only that it matches
this schema.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

random.seed(42)

# Each class = (emotion_label, intent, need) with template utterances and target ranges.
# intensity/social_state targets are sampled within a class-appropriate range + noise,
# which is honest about being heuristic-derived rather than human-rated ground truth.

TEMPLATE_CLASSES = [
    dict(
        emotion="frustration", intent="venting", need="validation",
        intensity=(0.55, 0.9), openness=(0.5, 0.8), trust=(0.3, 0.6), irritation=(0.5, 0.85),
        templates=[
            "ugh, {topic} is driving me crazy today",
            "I'm so frustrated with {topic}, nothing is working",
            "honestly {topic} has been such a mess and I'm over it",
            "I can't believe {topic} again, this keeps happening",
            "{topic} is so annoying, I don't know what to do anymore",
        ],
    ),
    dict(
        emotion="sadness", intent="venting", need="validation",
        intensity=(0.5, 0.9), openness=(0.5, 0.85), trust=(0.4, 0.7), irritation=(0.05, 0.3),
        templates=[
            "I've been feeling really down about {topic} lately",
            "honestly {topic} has left me pretty sad",
            "I don't know, {topic} just makes me feel low",
            "it's been a hard week because of {topic}",
            "I feel kind of empty about {topic} right now",
        ],
    ),
    dict(
        emotion="anxiety", intent="seeking_advice", need="problem_solving",
        intensity=(0.5, 0.85), openness=(0.4, 0.7), trust=(0.3, 0.6), irritation=(0.1, 0.35),
        templates=[
            "I'm really nervous about {topic}, what should I do",
            "I can't stop worrying about {topic}",
            "{topic} is making me anxious, any advice?",
            "I keep overthinking {topic} and it's stressing me out",
            "not sure how to handle {topic}, I'm on edge about it",
        ],
    ),
    dict(
        emotion="joy", intent="sharing", need="celebration",
        intensity=(0.5, 0.9), openness=(0.6, 0.9), trust=(0.5, 0.8), irritation=(0.0, 0.1),
        templates=[
            "guess what, {topic} finally worked out!",
            "I'm so happy about {topic} right now",
            "great news about {topic}, I had to tell you",
            "{topic} went amazingly well today",
            "feeling really good about {topic}!",
        ],
    ),
    dict(
        emotion="gratitude", intent="sharing", need="celebration",
        intensity=(0.3, 0.6), openness=(0.55, 0.85), trust=(0.5, 0.85), irritation=(0.0, 0.05),
        templates=[
            "thanks for listening about {topic} earlier, it helped",
            "really appreciate you asking about {topic}",
            "that meant a lot, thanks for checking on {topic}",
        ],
    ),
    dict(
        emotion="excitement", intent="sharing", need="celebration",
        intensity=(0.6, 0.95), openness=(0.6, 0.9), trust=(0.5, 0.8), irritation=(0.0, 0.1),
        templates=[
            "I CANNOT wait for {topic}, so excited",
            "{topic} is going to be so much fun",
            "so pumped about {topic} right now",
        ],
    ),
    dict(
        emotion="anger", intent="complaining", need="validation",
        intensity=(0.6, 0.95), openness=(0.3, 0.6), trust=(0.15, 0.4), irritation=(0.65, 0.95),
        templates=[
            "I am SO mad about {topic} right now",
            "{topic} made me furious, I can't even",
            "honestly {topic} makes my blood boil",
        ],
    ),
    dict(
        emotion="boredom", intent="small_talk", need="humor",
        intensity=(0.15, 0.4), openness=(0.3, 0.6), trust=(0.3, 0.6), irritation=(0.05, 0.2),
        templates=[
            "not much going on, just thinking about {topic} I guess",
            "kind of bored, what do you think about {topic}",
            "nothing much happening, {topic} I suppose",
        ],
    ),
    dict(
        emotion="curiosity", intent="asking", need="information",
        intensity=(0.15, 0.45), openness=(0.4, 0.7), trust=(0.35, 0.6), irritation=(0.0, 0.1),
        templates=[
            "quick question, what do you know about {topic}",
            "hey, how does {topic} actually work",
            "curious about {topic}, can you explain",
        ],
    ),
    dict(
        emotion="contentment", intent="small_talk", need="information",
        intensity=(0.1, 0.35), openness=(0.4, 0.65), trust=(0.4, 0.7), irritation=(0.0, 0.1),
        templates=[
            "just relaxing, thinking about {topic}",
            "pretty good day, been doing {topic}",
            "all good here, just working on {topic}",
        ],
    ),
    dict(
        emotion="embarrassment", intent="venting", need="validation",
        intensity=(0.4, 0.75), openness=(0.45, 0.75), trust=(0.4, 0.7), irritation=(0.1, 0.3),
        templates=[
            "so embarrassing, I messed up {topic} in front of everyone",
            "I feel so awkward about {topic} now",
            "cringing about how {topic} went",
        ],
    ),
    dict(
        emotion="playfulness", intent="joking", need="humor",
        intensity=(0.2, 0.5), openness=(0.5, 0.8), trust=(0.4, 0.7), irritation=(0.0, 0.1),
        templates=[
            "lol okay but what if {topic} was actually a competitive sport",
            "haha imagine if {topic} could talk",
            "not me overthinking {topic} again lol",
        ],
    ),
    dict(
        emotion="neutral", intent="roleplay_initiation", need="engagement",
        intensity=(0.2, 0.5), openness=(0.5, 0.8), trust=(0.4, 0.7), irritation=(0.0, 0.1),
        templates=[
            "want to do a quick roleplay about {topic}?",
            "let's do a scene set in {topic}",
            "can we roleplay something involving {topic}",
        ],
    ),
]

TOPICS = [
    "work", "my manager", "school", "this project", "my family", "the move", "my health",
    "the deadline", "my friend group", "the interview", "money stuff", "the apartment search",
    "my sleep schedule", "the presentation", "my relationship", "the trip", "this app",
    "the exam", "my side project", "the weather today", "the team meeting", "my roommate",
]


@dataclass
class Example:
    text: str
    emotion: str
    intent: str
    need: str
    intensity: float
    openness: float
    trust: float
    irritation: float


def _sample_range(r: tuple[float, float]) -> float:
    return round(random.uniform(*r), 3)


def generate(n_per_class: int = 60) -> list[Example]:
    examples = []
    for cls in TEMPLATE_CLASSES:
        for _ in range(n_per_class):
            template = random.choice(cls["templates"])
            topic = random.choice(TOPICS)
            text = template.format(topic=topic)
            examples.append(
                Example(
                    text=text,
                    emotion=cls["emotion"],
                    intent=cls["intent"],
                    need=cls["need"],
                    intensity=_sample_range(cls["intensity"]),
                    openness=_sample_range(cls["openness"]),
                    trust=_sample_range(cls["trust"]),
                    irritation=_sample_range(cls["irritation"]),
                )
            )
    random.shuffle(examples)
    return examples
