"""Memory Gate — Entry and Exit gates for the cognitive loop.

Entry gate: Stochastic filter on every message, buffers to scratch.
Exit gate: ACT-R adapted scoring on FIFO pruning / periodic flush.

All skip/persist probabilities are stochastic and evolvable by consolidation.
ACT-R equation structure is kept; parameters are human-calibrated starting
points that evolve through experience.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agent.gate")


# ── ENTRY GATE ──────────────────────────────────────────────────────────────


@dataclass
class EntryGateConfig:
    """Stochastic entry gate parameters. All skip rates are evolvable."""
    min_content_length: int = 10
    short_content_skip_rate: float = 0.95   # P(skip | content < min_length)
    mechanical_skip_rate: float = 0.90      # P(skip | mechanical content)
    base_buffer_rate: float = 0.99          # P(buffer | normal content)
    mechanical_prefixes: list[str] = field(default_factory=lambda: [
        "/", "[tool:", "[system:", "[error:", "```",
    ])


class EntryGate:
    """Stochastic entry gate — fires on every message, buffers to scratch.

    Not deterministic. Every skip has a stochastic floor that occasionally
    lets content through, giving the consolidation worker data on whether
    the heuristics are wrong. Skip rates evolve over time.
    """

    def __init__(self, config: EntryGateConfig | None = None):
        self.config = config or EntryGateConfig()
        self._stats = {"evaluated": 0, "buffered": 0, "skipped": 0}

    def evaluate(self, content: str, source: str = "unknown") -> tuple[bool, dict]:
        """Evaluate content for scratch buffering.

        Returns (should_buffer, metadata).
        Metadata includes decision reasoning for consolidation learning.
        """
        self._stats["evaluated"] += 1
        content_stripped = content.strip()

        metadata = {
            "source": source,
            "content_length": len(content_stripped),
            "gate_decision": None,
            "gate_reason": None,
            "skip_probability": None,
            "dice_roll": None,
        }

        # Short content — high skip probability, but stochastic
        if len(content_stripped) < self.config.min_content_length:
            return self._stochastic_decision(
                skip_rate=self.config.short_content_skip_rate,
                reason="short_content",
                metadata=metadata,
            )

        # Mechanical content — high skip probability, but stochastic
        if self._is_mechanical(content_stripped):
            return self._stochastic_decision(
                skip_rate=self.config.mechanical_skip_rate,
                reason="mechanical",
                metadata=metadata,
            )

        # Normal content — high buffer probability, but stochastic
        return self._stochastic_decision(
            skip_rate=1.0 - self.config.base_buffer_rate,
            reason="normal_content",
            metadata=metadata,
        )

    def _stochastic_decision(
        self, skip_rate: float, reason: str, metadata: dict
    ) -> tuple[bool, dict]:
        """Make a stochastic buffer/skip decision."""
        roll = random.random()
        metadata["skip_probability"] = skip_rate
        metadata["dice_roll"] = roll

        if roll < skip_rate:
            metadata["gate_decision"] = "skip"
            metadata["gate_reason"] = reason
            self._stats["skipped"] += 1
            return False, metadata
        else:
            metadata["gate_decision"] = "buffer"
            metadata["gate_reason"] = f"{reason}_stochastic_pass"
            self._stats["buffered"] += 1
            return True, metadata

    def _is_mechanical(self, content: str) -> bool:
        """Check if content looks like tool/system output."""
        for prefix in self.config.mechanical_prefixes:
            if content.startswith(prefix):
                return True
        return False

    @property
    def stats(self) -> dict:
        return dict(self._stats)


# ── EXIT GATE ───────────────────────────────────────────────────────────────


@dataclass
class ExitGateConfig:
    """ACT-R adapted exit gate parameters.

    Equation structure from ACT-R (decades validated).
    Parameter VALUES are human-calibrated starting points — evolved by
    consolidation based on outcomes.
    """
    persist_threshold: float = 0.3
    # ACT-R parameters
    decay_d: float = 0.5               # base-level decay rate
    noise_s: float = 0.25              # logistic noise spread
    # Component weights
    spreading_weight: float = 0.4       # relevance component
    novelty_weight: float = 0.3         # novelty component
    contradiction_bonus: float = 0.4    # bonus for contradicting beliefs
    # Spreading activation sub-weights
    goal_relevance_weight: float = 0.5
    identity_relevance_weight: float = 0.3
    context_relevance_weight: float = 0.2


class ExitGate:
    """ACT-R adapted exit gate — scores content for permanent persistence.

    gate_score = relevance(S_i) * novelty(1 - redundancy)
                 + contradiction_bonus + noise(epsilon)

    Fires on: FIFO pruning, periodic scratch flush.
    Does NOT fire on every message (that is the entry gate is job).
    """

    def __init__(self, config: ExitGateConfig | None = None):
        self.config = config or ExitGateConfig()
        self._stats = {"evaluated": 0, "persisted": 0, "dropped": 0}

    async def evaluate(
        self,
        content: str,
        memory_store,       # MemoryStore instance
        layers,             # LayerStore instance
        conversation_context: list[dict] | None = None,
    ) -> tuple[bool, float, dict]:
        """Score content for persistence using ACT-R adapted math.

        Returns (should_persist, score, metadata).
        Metadata logged for consolidation to learn from.
        """
        self._stats["evaluated"] += 1

        metadata = {
            "spreading_activation": 0.0,
            "novelty": 0.0,
            "redundancy": 0.0,
            "contradiction": 0.0,
            "noise": 0.0,
            "raw_score": 0.0,
            "final_score": 0.0,
            "decision": None,
        }

        # 1. SPREADING ACTIVATION — relevance to goals/values/context
        spreading = self._compute_spreading_activation(
            content, layers, conversation_context
        )
        metadata["spreading_activation"] = spreading

        # 2. NOVELTY — inverse of redundancy with existing memories
        redundancy, contradiction = await self._compute_novelty(
            content, memory_store
        )
        metadata["redundancy"] = redundancy
        metadata["contradiction"] = contradiction
        novelty = 1.0 - redundancy
        metadata["novelty"] = novelty

        # 3. STOCHASTIC NOISE — logistic distribution (ACT-R standard)
        noise = self._logistic_noise()
        metadata["noise"] = noise

        # 4. GATE SCORE = relevance * novelty + contradiction + noise
        c = self.config
        raw_score = (
            (c.spreading_weight * spreading) * (c.novelty_weight + novelty)
            + (c.contradiction_bonus * contradiction)
            + noise
        )
        metadata["raw_score"] = raw_score

        # Sigmoid normalization to 0-1
        score = 1.0 / (1.0 + math.exp(-raw_score * 3.0))
        metadata["final_score"] = score

        # 5. PERSIST DECISION
        should_persist = score >= self.config.persist_threshold
        metadata["decision"] = "persist" if should_persist else "drop"

        if should_persist:
            self._stats["persisted"] += 1
        else:
            self._stats["dropped"] += 1

        logger.info(
            f"Exit gate: {metadata[chr(100)+ecision]} "
            f"(score={score:.3f}, relevance={spreading:.3f}, "
            f"novelty={novelty:.3f}, contradiction={contradiction:.3f})"
        )
        return should_persist, score, metadata

    # ── Spreading activation (relevance) ─────────────────────────────────

    def _compute_spreading_activation(
        self,
        content: str,
        layers,
        conversation_context: list[dict] | None,
    ) -> float:
        """S_i = sum(W_k * association(content, source_k))

        Sources: Layer 0 values, Layer 1 goals, recent conversation.
        Uses keyword overlap for v1 (embedding-based upgrade in v2).
        """
        c = self.config
        activation = 0.0

        # Goal relevance
        goals = layers.layer1.get("active_goals", [])
        if goals:
            goal_texts = [g.get("goal", "") for g in goals]
            goal_weights = [g.get("weight", 0.5) for g in goals]
            activation += c.goal_relevance_weight * self._keyword_relevance(
                content, goal_texts, goal_weights
            )

        # Identity relevance (values + beliefs)
        values = layers.layer0.get("values", [])
        beliefs = layers.layer0.get("beliefs", [])
        if values or beliefs:
            id_texts = (
                [v.get("value", "") for v in values]
                + [b.get("belief", "") for b in beliefs]
            )
            id_weights = (
                [v.get("weight", 0.5) for v in values]
                + [b.get("confidence", 0.5) for b in beliefs]
            )
            activation += c.identity_relevance_weight * self._keyword_relevance(
                content, id_texts, id_weights
            )

        # Context relevance (last 5 messages)
        if conversation_context:
            recent = conversation_context[-5:]
            ctx_texts = [m.get("content", "") for m in recent]
            activation += c.context_relevance_weight * self._keyword_relevance(
                content, ctx_texts, [1.0] * len(ctx_texts)
            )

        return min(activation, 1.0)

    # ── Novelty + contradiction ──────────────────────────────────────────

    async def _compute_novelty(
        self,
        content: str,
        memory_store,
    ) -> tuple[float, float]:
        """Compute redundancy and contradiction vs existing memories.

        Returns (redundancy 0-1, contradiction 0-1).

        Redundancy: similarity * base_level_activation of nearest neighbor.
        Contradiction: high similarity + opposing conclusion.
        """
        count = await memory_store.memory_count()
        if count == 0:
            return 0.0, 0.0   # nothing to be redundant with

        similar = await memory_store.search_similar(
            content, top_k=3, min_similarity=0.2
        )
        if not similar:
            return 0.0, 0.0

        top = similar[0]
        similarity = top.get("similarity", 0.0)
        access_count = top.get("access_count", 0)

        # Base-level activation: B_i = ln(1 + n) where n = access count
        # Simplified from full ACT-R sum-of-recencies
        base_level = math.log(1 + access_count)

        # Redundancy: high similarity AND high base-level = well-trodden ground
        redundancy = similarity * min(1.0, 0.3 + 0.15 * base_level)

        # Contradiction: high similarity but opposing content
        contradiction = 0.0
        if similarity > 0.6:
            contradiction = self._detect_contradiction(
                content, top.get("content", "")
            )

        return redundancy, contradiction

    def _detect_contradiction(
        self, new_content: str, existing_content: str
    ) -> float:
        """Detect if new content contradicts existing content.

        v1: Keyword heuristic (negation marker asymmetry).
        v2: Embedding-based semantic opposition (TODO).
        """
        negation_markers = [
            "not", "dont, doesnt", "isnt, wasnt", "wont,
            cant", "never", "no longer", "stopped", "changed",
            "actually", "instead", "wrong", "incorrect", "mistaken",
            "however", "but actually", "on the contrary", "opposite",
            "disagree", "unlike", "different from",
        ]

        new_lower = new_content.lower()
        existing_lower = existing_content.lower()

        asymmetry = 0
        for marker in negation_markers:
            in_new = marker in new_lower
            in_old = marker in existing_lower
            if in_new != in_old:
                asymmetry += 1

        return min(1.0, asymmetry * 0.15)

    # ── Utility functions ────────────────────────────────────────────────

    def _keyword_relevance(
        self,
        content: str,
        references: list[str],
        weights: list[float],
    ) -> float:
        """Weighted keyword overlap — cheap v1 relevance proxy.

        Returns score 0-1.
        Will be replaced by embedding similarity in v2.
        """
        content_words = set(content.lower().split())
        if not content_words:
            return 0.0

        total_weight = sum(weights) or 1.0
        score = 0.0

        for text, weight in zip(references, weights):
            ref_words = set(text.lower().split())
            if not ref_words:
                continue
            overlap = len(content_words & ref_words) / max(
                len(content_words), len(ref_words)
            )
            score += weight * overlap

        return min(1.0, score / total_weight)

    def _logistic_noise(self) -> float:
        """ACT-R standard noise — logistic(0, s) distribution.

        Provides stochastic floor so the gate can surprise itself.
        """
        s = self.config.noise_s
        p = random.random()
        p = max(0.001, min(0.999, p))  # avoid infinity
        return s * math.log(p / (1.0 - p))

    @property
    def stats(self) -> dict:
        return dict(self._stats)
