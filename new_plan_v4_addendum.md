# new_plan_v4 — Addendum (Session 10 Critique)

**Created:** 2026-02-10
**Context:** Session 10 critique of v4. Two clarifications needed. Everything else resolves during implementation.

---

## Addendum A: Unified Retrieval Pipeline Composition

v4 builds retrieval components as separate tasks (§2.5-2.7, §3.1-3.2). This addendum clarifies how they compose in the end state.

**Two distinct pipelines exist, for different purposes:**

### Pipeline 1: Gate (persist/drop decision)

Runs when content exits context window or scratch buffer flushes.

```
Content leaving context
  → ACT-R activation (§2.7): B_i + S_i + P_i + ε_i
  → 3×3 matrix (§2.8): relevance axis (from S_i) × novelty axis (from check_novelty)
  → Decision: persist / reinforce / buffer / drop
  → If persist: store with depth_weight Beta(1, 4), generate compressed summary
```

ACT-R is the gate's scoring engine. It answers: "is this worth keeping?"

### Pipeline 2: Retrieval (context injection)

Runs every cognitive cycle to assemble context.

```
Current attention focus (see Addendum B)
  → pgvector top-500 pre-filter by embedding similarity (REQUIRED, not optional)
  → Hybrid Search §2.5: dense (pgvector) + sparse (tsvector) + RRF on pre-filtered set
  → FlashRank §2.6: rerank top-50 hybrid results → top-20
  → Hybrid Relevance §3.1: 5-component Dirichlet-blended score per candidate
  → Dynamic Injection §3.2: injection_score = observed_weight × hybrid_relevance
  → Fill context top-down until token budget exhausted
  → Record co-access for Hebbian learning
```

Hybrid Relevance is the retrieval's scoring engine. It answers: "what should the agent think about right now?"

**FlashRank integration note:** At step 7 (before §3.1 exists), FlashRank uses `weighted_score = 0.5*RRF + 0.3*recency + 0.2*importance`. After step 13 (§3.1 implemented), `weighted_score` is replaced by the Hybrid Relevance score. The FlashRank final formula stays `0.6 * rerank_score + 0.4 * hybrid_relevance_score`.

**ACT-R does not run during retrieval. Hybrid Relevance does not run during gating.** They are separate scoring systems for separate decisions. The only shared component is spreading activation (cosine sim to L0/L1 embeddings), which both use but for different purposes.

---

## Addendum B: Attention Embedding Definition

Multiple subsystems reference "attention embedding" without defining it. One definition, used everywhere:

**The attention embedding is the embedding of the current cycle's winning input candidate** (from attention allocation §3.10).

```python
# Computed once per cognitive cycle, after attention allocation selects winner
attention_embedding = embed(winning_candidate.content, task_type="RETRIEVAL_QUERY")
```

**Used by:**
- §3.1 Hybrid Relevance: `cosine(memory_embedding, attention_embedding)` — semantic component
- §3.2 Context Inertia: `cosine(current_attention_embedding, previous_attention_embedding)` — shift detection
- §2.7 ACT-R Spreading Activation (gate): cosine sim to attention embedding as context relevance
- §5.1 Attention Centroid: recency-weighted average of recent attention embeddings (not just current)

**Lifecycle:**
- Computed once at cycle start from the winning candidate
- Stored as `previous_attention_embedding` for next cycle's inertia calculation
- Appended to a rolling window (last N cycles) for the attention centroid in §5.1

**Before §3.10 exists (steps 1-13):** The attention embedding is simply the embedding of the user message. After §3.10, it becomes the embedding of whichever input source wins attention.

---

## Session 10 Critique — Other Notes

Items confirmed as non-issues (resolved by existing v4 design or unified memory decision):

- **A1 (L0/L1 dual store):** Resolved by unified memory. Postgres is authoritative. JSON files are convenience for centroid embedding only. Bootstrap safety boundaries go directly into Postgres as `immutable=true, Beta(50, 1)`.
- **A2 (active goals undefined):** Non-issue in unified memory. No discrete "goals" category needed. `similarity_to_active_goals()` is a misleading name — should be `similarity_to_high_weight_memories()` or just semantic similarity to the attention focus against all memories weighted by depth_weight. The content carries the intent signal. The weight determines pull. Rename during implementation.
- **A3 (ID type mismatch):** Fix at schema creation — use TEXT in co-access table to match existing memories.id.
- **A5 (importance column):** Replaced by `depth_weight.center`. Migration: keep column, populate from center for backward compat, deprecate.
- **B3 (safety timing):** Move Phase A (hard caps + diminishing returns) to right after StochasticWeight creation (step 4). Trivial.

Items that are genuine open questions (already flagged in v4 §15, resolve during implementation):
- Outcome quality signal, cold-start alpha transition, co-access pruning, cumulative importance tracking.
