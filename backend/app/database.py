"""
Structured state layer.

This is the "ordinary database" from the design doc — everything that
isn't semantic/similarity search lives here: users, conversations,
messages, memories (metadata + payload; the embedding vector itself is
kept in the vector index), personality traits, and feedback.

SQLite is used for V1 for zero-setup local development. Because we go
through SQLAlchemy's ORM rather than raw SQL, moving to Postgres later is
a one-line change in config.py.
"""
import datetime
import json
import uuid

from sqlalchemy import (
    Column, String, Float, Integer, Text, DateTime, ForeignKey, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=new_id)
    display_name = Column(String, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    conversations = relationship("Conversation", back_populates="user")
    memories = relationship("Memory", back_populates="user")
    traits = relationship("Trait", back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"))
    started_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=new_id)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    role = Column(String)  # "user" | "assistant"
    content = Column(Text)
    # Analysis attached to *user* messages by the fast-path pipeline.
    emotion_json = Column(Text, default="{}")   # {"sadness": 0.6, ...}
    situation = Column(String, default="")       # e.g. "rejection"
    strategy = Column(String, default="")        # e.g. "acknowledge_and_offer_choice"
    # Which memory IDs were actually retrieved/used for *this* message
    # (only meaningful on assistant messages). Added so a "wrong_memory"
    # feedback tag can flag the specific memory that was wrong instead
    # of a vague, unactionable complaint — see
    # user_model/feedback_learning.py and main.py's /feedback route.
    # Messages created before this column existed simply have "[]" here.
    retrieved_memory_ids_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

    def emotion(self) -> dict:
        try:
            return json.loads(self.emotion_json or "{}")
        except json.JSONDecodeError:
            return {}

    def retrieved_memory_ids(self) -> list[str]:
        try:
            return json.loads(self.retrieved_memory_ids_json or "[]")
        except json.JSONDecodeError:
            return []


class Memory(Base):
    """
    A single unit of long-term memory.

    type: "semantic" (fact) | "episodic" (event) | "preference" | "relationship"
    The embedding vector lives in the in-process vector index
    (app/memory/vectorstore.py), keyed by this row's id, not in SQL —
    that's what a real vector DB (Qdrant/Chroma) would hold in V2.
    """
    __tablename__ = "memories"
    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"))
    type = Column(String)
    topic = Column(String, default="")
    content = Column(Text)
    importance = Column(Float, default=0.5)
    source = Column(String, default="conversation")  # conversation | profile_import | explicit
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.datetime.utcnow)
    access_count = Column(Integer, default=0)

    user = relationship("User", back_populates="memories")


class Trait(Base):
    """
    One dimension of the dynamic user model (see design doc Phase 3).

    value:      current estimate, 0-1
    confidence: 0-1, grows with evidence_count
    evidence_count: how many observations informed this estimate
    """
    __tablename__ = "traits"
    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"))
    name = Column(String)
    value = Column(Float, default=0.5)
    confidence = Column(Float, default=0.0)
    evidence_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="traits")


class TraitHistory(Base):
    """
    An append-only log of trait values over time — one row per
    `update_trait()` call (design doc V4: "personality drift").

    The `Trait` table above holds only the *current* EWMA estimate,
    which means the system has always drifted silently: a person who
    shifts from wanting long detailed answers to wanting short ones
    gets their `verbosity` estimate quietly updated, with no way to
    notice or report that the shift happened. This table is what makes
    the difference between "the estimate moved" and "we can tell you
    it moved, when, and by how much."

    Kept as a separate append-only table rather than versioning the
    Trait row, because the two have genuinely different access
    patterns: generation reads only the current value on every single
    turn and should stay a fast single-row lookup, while drift
    analysis reads long ranges and runs rarely (in reflection, off the
    hot path).

    Note this grows one row per trait update, i.e. several rows per
    message. At demo scale that's nothing; at real scale you'd want a
    retention policy (downsample rows older than N days to daily
    averages) — not implemented here, and called out in
    docs/ARCHITECTURE.md rather than left as a surprise.
    """
    __tablename__ = "trait_history"
    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"))
    name = Column(String)
    value = Column(Float)
    confidence = Column(Float)
    evidence_count = Column(Integer)
    recorded_at = Column(DateTime, default=datetime.datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(String, primary_key=True, default=new_id)
    message_id = Column(String, ForeignKey("messages.id"))
    user_id = Column(String, ForeignKey("users.id"))
    rating = Column(String)  # "up" | "down"
    tags_json = Column(Text, default="[]")  # ["too_formal", "wrong_memory", ...]
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
