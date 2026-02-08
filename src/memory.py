"""Memory Store — Postgres + pgvector + Google embeddings.

Handles all Layer 2 operations: embed, store, retrieve, search.
Used by the cognitive loop, memory gate, consolidation, and idle loop.
"""

import logging
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg
from google import genai

from .llm import retry_llm_call
from .config import RetryConfig

logger = logging.getLogger("agent.memory")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 768


class MemoryStore:
    """Async interface to agent memory backed by Postgres + pgvector."""

    def __init__(self, retry_config: RetryConfig | None = None):
        self.pool: asyncpg.Pool | None = None
        self.genai_client: genai.Client | None = None
        self.retry_config = retry_config or RetryConfig()

    async def connect(self):
        """Initialize DB pool and embedding client."""
        db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://agent:agent_secret@localhost:5432/agent_memory",
        )
        self.pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.genai_client = genai.Client(api_key=api_key)
        else:
            logger.warning("GOOGLE_API_KEY not set — embeddings unavailable")

        logger.info("Memory store connected.")

    async def close(self):
        """Close DB pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Memory store closed.")

    # --- EMBEDDINGS ---

    async def embed(self, text: str) -> list[float]:
        """Embed text using Google gemini-embedding-001 with retry."""
        if not self.genai_client:
            raise RuntimeError("Embedding client not initialized (missing API key?)")

        async def _call():
            result = await self.genai_client.aio.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=genai.types.EmbedContentConfig(
                    output_dimensionality=EMBED_DIMENSIONS,
                ),
            )
            return result.embeddings[0].values

        return await retry_llm_call(
            _call,
            config=self.retry_config,
            label="embed",
        )

    # --- STORE ---

    async def store_memory(
        self,
        content: str,
        memory_type: str = "semantic",
        source: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.5,
        importance: float = 0.5,
        evidence_count: int = 0,
        metadata: dict | None = None,
    ) -> str:
        """Embed and store a memory chunk. Returns the memory ID."""
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        embedding = await self.embed(content)
        now = datetime.now(timezone.utc)

        await self.pool.execute(
            """
            INSERT INTO memories (id, content, type, embedding, created_at, updated_at,
                                  source, tags, confidence, importance, evidence_count, metadata)
            VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            memory_id,
            content,
            memory_type,
            str(embedding),
            now,
            now,
            source,
            tags or [],
            confidence,
            importance,
            evidence_count,
            json.dumps(metadata or {}),
        )

        logger.info(f"Stored memory {memory_id}: {content[:80]}...")
        return memory_id

    async def store_insight(
        self,
        content: str,
        source_memory_ids: list[str],
        importance: float = 0.8,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a consolidated insight that supersedes source memories.

        - Creates a new high-importance insight memory
        - Links it to source memories via memory_supersedes
        - Lowers importance of source memories (they remain queryable)
        """
        # Store the insight
        insight_id = await self.store_memory(
            content=content,
            memory_type="semantic",
            source="consolidation",
            tags=tags,
            confidence=0.8,
            importance=importance,
            evidence_count=len(source_memory_ids),
            metadata=metadata,
        )

        # Link to source memories
        for source_id in source_memory_ids:
            await self.pool.execute(
                """
                INSERT INTO memory_supersedes (insight_id, source_id)
                VALUES ($1, $2) ON CONFLICT DO NOTHING
                """,
                insight_id,
                source_id,
            )

        # Lower importance of source memories (don't delete them)
        await self.pool.execute(
            """
            UPDATE memories SET importance = LEAST(importance, 0.3)
            WHERE id = ANY($1)
            """,
            source_memory_ids,
        )

        logger.info(
            f"Insight {insight_id} supersedes {len(source_memory_ids)} memories: "
            f"{content[:80]}..."
        )
        return insight_id

    # --- INTROSPECTION ---

    async def why_do_i_believe(self, memory_id: str) -> list[dict]:
        """Trace the evidence chain for a memory/insight.

        Returns the source memories that formed this belief,
        following the supersedes chain recursively.
        """
        rows = await self.pool.fetch(
            """
            WITH RECURSIVE evidence_chain AS (
                -- Start with direct sources
                SELECT s.source_id, 1 AS depth
                FROM memory_supersedes s
                WHERE s.insight_id = $1

                UNION ALL

                -- Follow chain deeper
                SELECT s.source_id, ec.depth + 1
                FROM memory_supersedes s
                JOIN evidence_chain ec ON s.insight_id = ec.source_id
                WHERE ec.depth < 5  -- max depth to prevent loops
            )
            SELECT DISTINCT m.id, m.content, m.type, m.confidence,
                   m.importance, m.created_at, m.source, m.tags,
                   ec.depth
            FROM evidence_chain ec
            JOIN memories m ON m.id = ec.source_id
            ORDER BY ec.depth, m.created_at
            """,
            memory_id,
        )
        return [dict(r) for r in rows]

    async def get_insights_for(self, source_memory_id: str) -> list[dict]:
        """Find insights that were built from a given source memory."""
        rows = await self.pool.fetch(
            """
            SELECT m.id, m.content, m.importance, m.evidence_count, m.created_at
            FROM memory_supersedes s
            JOIN memories m ON m.id = s.insight_id
            WHERE s.source_id = $1
            ORDER BY m.importance DESC
            """,
            source_memory_id,
        )
        return [dict(r) for r in rows]

    # --- RETRIEVE ---

    async def search_similar(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """Embed query and find most similar memories via pgvector cosine search."""
        query_embedding = await self.embed(query)

        rows = await self.pool.fetch(
            """
            SELECT id, content, type, confidence, importance,
                   access_count, last_accessed, tags, source, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            WHERE 1 - (embedding <=> $1::vector) > $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            str(query_embedding),
            min_similarity,
            top_k,
        )

        # Update access counts
        if rows:
            now = datetime.now(timezone.utc)
            ids = [r["id"] for r in rows]
            await self.pool.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = $1
                WHERE id = ANY($2)
                """,
                now,
                ids,
            )

        return [dict(r) for r in rows]

    async def get_memory(self, memory_id: str) -> dict | None:
        """Fetch a single memory by ID."""
        row = await self.pool.fetchrow(
            "SELECT * FROM memories WHERE id = $1", memory_id
        )
        return dict(row) if row else None

    async def get_random_memory(self) -> dict | None:
        """Pull a random memory — used by the idle loop / DMN."""
        row = await self.pool.fetchrow(
            "SELECT id, content, type, confidence, importance, tags, created_at "
            "FROM memories ORDER BY RANDOM() LIMIT 1"
        )
        return dict(row) if row else None

    async def memory_count(self) -> int:
        """Total number of stored memories."""
        return await self.pool.fetchval("SELECT COUNT(*) FROM memories")

    # --- SCRATCH BUFFER ---

    async def buffer_scratch(
        self,
        content: str,
        source: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Write to scratch buffer (entry gate temp storage)."""
        scratch_id = f"scratch_{uuid.uuid4().hex[:12]}"
        await self.pool.execute(
            """
            INSERT INTO scratch_buffer (id, content, source, tags, metadata)
            VALUES ($1, $2, $3, $4, $5)
            """,
            scratch_id,
            content,
            source,
            tags or [],
            json.dumps(metadata or {}),
        )
        return scratch_id

    async def flush_scratch(self, older_than_minutes: int = 60) -> list[dict]:
        """Retrieve and delete scratch entries older than threshold."""
        rows = await self.pool.fetch(
            """
            DELETE FROM scratch_buffer
            WHERE buffered_at < NOW() - INTERVAL '1 minute' * $1
            RETURNING *
            """,
            older_than_minutes,
        )
        return [dict(r) for r in rows]

    # --- NOVELTY CHECK ---

    async def check_novelty(self, content: str, threshold: float = 0.85) -> tuple[bool, float]:
        """Check if content is already in memory (for gate novelty scoring).

        Returns (is_novel, max_similarity).
        """
        embedding = await self.embed(content)
        row = await self.pool.fetchrow(
            """
            SELECT 1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            ORDER BY embedding <=> $1::vector
            LIMIT 1
            """,
            str(embedding),
        )

        if row is None:
            return True, 0.0

        max_sim = float(row["similarity"])
        return max_sim < threshold, max_sim

    # --- DECAY ---

    async def get_stale_memories(
        self, stale_days: int = 90, min_access_count: int = 3
    ) -> list[dict]:
        """Find memories eligible for decay."""
        rows = await self.pool.fetch(
            """
            SELECT id, content, importance, access_count, last_accessed
            FROM memories
            WHERE (last_accessed IS NULL OR last_accessed < NOW() - INTERVAL '1 day' * $1)
              AND access_count < $2
              AND importance > 0.05
            ORDER BY importance ASC
            """,
            stale_days,
            min_access_count,
        )
        return [dict(r) for r in rows]

    async def decay_memories(self, memory_ids: list[str], factor: float = 0.5):
        """Halve importance of stale memories (never delete)."""
        await self.pool.execute(
            """
            UPDATE memories
            SET importance = importance * $1, updated_at = NOW()
            WHERE id = ANY($2)
            """,
            factor,
            memory_ids,
        )
        logger.info(f"Decayed {len(memory_ids)} memories by factor {factor}")
