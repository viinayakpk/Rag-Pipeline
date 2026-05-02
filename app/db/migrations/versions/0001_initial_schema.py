"""Initial schema: documents, chunks, entities, chunk_entities, chat_logs.

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pgvector extension must exist before vector columns are defined.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("source", sa.Text, nullable=True),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("detected_language", sa.String(10), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("region", sa.Text, nullable=True),
        sa.Column("author", sa.Text, nullable=True),
        sa.Column("topics", postgresql.JSONB, nullable=True),
        sa.Column("document_type", sa.Text, nullable=True),
        sa.Column("original_filename", sa.Text, nullable=True),
        sa.Column("mime_type", sa.String(100), nullable=True),
        sa.Column("byte_size", sa.BigInteger(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True, unique=True),
        sa.Column(
            "ingestion_status",
            sa.String(20),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("extracted_text", sa.Text, nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
    )

    # chunks uses vector(768) which Alembic's DDL does not express natively.
    op.execute("""
        CREATE TABLE chunks (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text        TEXT NOT NULL,
            token_count INTEGER,
            language    TEXT,
            source      TEXT,
            title       TEXT,
            published_at TIMESTAMPTZ,
            document_type TEXT,
            region      TEXT,
            topics      JSONB,
            fts_vector  TSVECTOR,
            embedding   vector(768),
            CONSTRAINT uq_chunk_order UNIQUE (document_id, chunk_index)
        )
    """)

    op.create_table(
        "entities",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("normalized_name", sa.Text, nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "normalized_name", "entity_type", name="uq_entity_normalized_type"
        ),
    )

    op.create_table(
        "chunk_entities",
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )

    op.create_table(
        "chat_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("filters", postgresql.JSONB, nullable=True),
        sa.Column("answer", sa.Text, nullable=True),
        sa.Column("confidence", sa.String(20), nullable=True),
        sa.Column("retrieved_chunk_ids", postgresql.JSONB, nullable=True),
        sa.Column("sources", postgresql.JSONB, nullable=True),
        sa.Column("retrieval_notes", postgresql.JSONB, nullable=True),
        sa.Column("model_name", sa.String(100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Indexes
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_language", "chunks", ["language"])
    op.create_index("ix_chunks_source", "chunks", ["source"])
    op.create_index("ix_chunks_document_type", "chunks", ["document_type"])
    op.create_index("ix_chunks_region", "chunks", ["region"])
    op.create_index("ix_chunks_published_at", "chunks", ["published_at"])
    op.create_index("ix_chunks_fts", "chunks", ["fts_vector"], postgresql_using="gin")
    op.create_index("ix_entities_normalized_name", "entities", ["normalized_name"])
    op.create_index("ix_chunk_entities_entity_id", "chunk_entities", ["entity_id"])
    op.create_index("ix_chunk_entities_document_id", "chunk_entities", ["document_id"])
    # Enable HNSW only once the corpus is large enough to justify approximate ANN.
    # op.execute(
    #     "CREATE INDEX ix_chunks_embedding ON chunks "
    #     "USING hnsw (embedding vector_cosine_ops)"
    # )

    # Keep tsvector aligned with the chunk language when PostgreSQL supports it.
    op.execute("""
        CREATE OR REPLACE FUNCTION chunks_fts_update() RETURNS trigger AS $$
        DECLARE
            cfg REGCONFIG;
        BEGIN
            cfg := CASE NEW.language
                WHEN 'en' THEN 'english'::REGCONFIG
                WHEN 'de' THEN 'german'::REGCONFIG
                WHEN 'fr' THEN 'french'::REGCONFIG
                WHEN 'sv' THEN 'swedish'::REGCONFIG
                WHEN 'no' THEN 'norwegian'::REGCONFIG
                ELSE 'simple'::REGCONFIG
            END;
            NEW.fts_vector := to_tsvector(cfg, COALESCE(NEW.text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE TRIGGER chunks_fts_trigger
        BEFORE INSERT OR UPDATE OF text, language ON chunks
        FOR EACH ROW EXECUTE FUNCTION chunks_fts_update()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS chunks_fts_trigger ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_fts_update")
    op.drop_table("chat_logs")
    op.drop_table("chunk_entities")
    op.drop_table("entities")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
    op.execute("DROP EXTENSION IF EXISTS vector")
