from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=True)
    source = Column(Text, nullable=True)
    language = Column(String(10), nullable=True)
    detected_language = Column(String(10), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    region = Column(Text, nullable=True)
    author = Column(Text, nullable=True)
    topics = Column(JSONB, nullable=True, default=list)
    document_type = Column(Text, nullable=True)
    original_filename = Column(Text, nullable=True)
    mime_type = Column(String(100), nullable=True)
    byte_size = Column(BigInteger, nullable=True)
    content_hash = Column(String(64), unique=True, nullable=True)
    ingestion_status = Column(String(20), nullable=False, default="pending")
    extracted_text = Column(Text, nullable=True)
    # "metadata" is a reserved name in SQLAlchemy's MetaData; map via column kwarg.
    document_metadata = Column("metadata", JSONB, nullable=True, default=dict)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    language = Column(String(10), nullable=True)
    source = Column(Text, nullable=True)
    title = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    document_type = Column(Text, nullable=True)
    region = Column(Text, nullable=True)
    topics = Column(JSONB, nullable=True, default=list)
    # Populated and kept current by a DB-side BEFORE INSERT OR UPDATE trigger.
    fts_vector = Column(TSVECTOR, nullable=True)
    # 768-dim vector from intfloat/multilingual-e5-base.
    embedding = Column(Vector(768), nullable=True)

    document = relationship("Document", back_populates="chunks")
    chunk_entities = relationship(
        "ChunkEntity", back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_chunk_order"),)


class Entity(Base):
    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    normalized_name = Column(Text, nullable=False)
    entity_type = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("normalized_name", "entity_type", name="uq_entity_normalized_type"),
    )

    chunk_entities = relationship("ChunkEntity", back_populates="entity")


class ChunkEntity(Base):
    __tablename__ = "chunk_entities"

    chunk_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    entity_id = Column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk = relationship("Chunk", back_populates="chunk_entities")
    entity = relationship("Entity", back_populates="chunk_entities")


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    filters = Column(JSONB, nullable=True, default=dict)
    answer = Column(Text, nullable=True)
    confidence = Column(String(20), nullable=True)
    retrieved_chunk_ids = Column(JSONB, nullable=True, default=list)
    sources = Column(JSONB, nullable=True, default=list)
    retrieval_notes = Column(JSONB, nullable=True, default=dict)
    model_name = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
