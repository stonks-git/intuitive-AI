# Session Handoff — 2026-02-08

**Read this file first in the next session, then read `notes.md` for full design.**

---

## What Was Built This Session

### Infrastructure (all on stonks@norisor.local via SSH)

**Machine specs:** i7-3740QM, 8GB RAM, 150GB free disk, no usable GPU (NVS 5200M, no CUDA). Everything runs CPU-only or via API.

**Postgres + pgvector** running in Docker on port 5433 (native PG17 occupies 5432):
```bash
cd ~/agent-runtime && docker compose up -d postgres
```

**Database: `agent_memory`** (user: `agent`, pass: `agent_secret`)

Tables:
- `memories` — Layer 2 storage. Columns: id, content, type (semantic/episodic/procedural), embedding vector(768), version, created_at, updated_at, access_count, last_accessed, source, tags[], confidence, importance, evidence_count, history JSONB, metadata JSONB
- `memory_supersedes` — join table for insight→source links (insight_id, source_id). Supports "why do I believe X?" introspection chains
- `scratch_buffer` — entry gate temp staging (id, content, source, tags, buffered_at, metadata)
- `consolidation_log` — audit trail for merge/promote/decay/tune operations
- `conversations` — session tracking for episodic sourcing
- `entity_relationships` — prepped for Mem0-style graph layer (entity_a, entity_b, relationship, confidence, source_memory_id)

Indexes: HNSW on embedding (m=16, ef_construction=128), plus type, created_at, last_accessed, access_count, confidence, tags (GIN), entity relationship lookups.

**Currently 5 test memories in the DB** (Hetzner preferences test data). Can be cleaned with `DELETE FROM memories; DELETE FROM memory_supersedes;` before real use.

### Source Files (~/agent-runtime/src/)

| File | Status | What it does |
|------|--------|-------------|
| `main.py` | **Live** | Entry point. Loads .env via python-dotenv. Starts 3 async loops: cognitive, consolidation, idle. Signal handlers for graceful shutdown. |
| `config.py` | **Live** | Loads runtime.yaml, containment.yaml into dataclasses. Includes RetryConfig, GateWeights, ModelsConfig, ContainmentConfig. |
| `layers.py` | **Live** | Manages Layer 0 (identity) + Layer 1 (goals) JSON files. Two-tier rendering: identity hash (~200 tokens) always injected, full identity (~2k tokens) on triggers. |
| `loop.py` | **Live** | Cognitive loop. CLI stdin/stdout. Builds system prompt from identity hash, sends conversation to Gemini 2.5 Flash Lite, gets response. Has /identity, /identity-hash, /containment, /status introspection commands. Uses retry wrapper. |
| `llm.py` | **Live** | Retry wrapper. `retry_llm_call()` (async) and `retry_llm_call_sync()`. Exponential backoff (1s→2s→4s, cap 30s) with ±50% jitter. Detects transient (429, 5xx, timeout, connection) vs permanent (401, 400) errors. Respects Retry-After headers. |
| `memory.py` | **Live** | MemoryStore class. Postgres+pgvector+Google embeddings (gemini-embedding-001, 768 dims). Methods: embed(), store_memory(), store_insight(), search_similar(), check_novelty(), buffer_scratch(), flush_scratch(), get_random_memory(), why_do_i_believe() (recursive CTE), get_insights_for(), get_stale_memories(), decay_memories(). All tested E2E. |
| `consolidation.py` | **Skeleton** | ConsolidationWorker. Runs on timer (default 60min). _run_cycle() is TODO. |
| `idle.py` | **Skeleton** | IdleLoop / DMN. Adaptive heartbeat intervals (1min→30min based on idle time). _heartbeat() is TODO. |

### Config Files (~/.agent/config/)

**runtime.yaml** — Key settings:
- system1: google / gemini-2.5-flash-lite
- system2: anthropic / claude-sonnet-4-5-20250929
- consolidation: google / gemini-2.5-flash-lite
- embeddings: google / gemini-embedding-001 / 768 dims
- storage: postgres on localhost:5432, database agent_memory (NOTE: should be 5433 — update runtime.yaml)
- retry: 3 retries, 1s base, 30s max, 0.5 jitter
- gate weights: all the placeholder values from notes.md
- consolidation: 60min base, adaptive, merge threshold 0.85, promote thresholds
- compulsion: cap 0.92, diminishing returns, dominance 0.4
- idle: adaptive intervals, goal threshold 0.5, value threshold 0.6
- metacognition: FOK thresholds, confidence thresholds, max 3 reentry loops
- escalation: 2+ triggers needed, always escalate on irreversible/identity/goal

**containment.yaml** — trust_level=1, self_spawn=false, network=whitelist, kill_switch=true

### .env file (~/agent-runtime/.env)
```
GOOGLE_API_KEY=<set>
ANTHROPIC_API_KEY=<set>
AGENT_DB_PASSWORD=agent_secret
DATABASE_URL=postgresql://agent:agent_secret@localhost:5433/agent_memory
```

### Dependencies installed (system-wide, --break-system-packages):
google-genai 1.62.0, asyncpg 0.31.0, pgvector 0.4.2, python-dotenv 1.2.1, numpy 2.4.2, pydantic 2.12.5

### What has NOT been built yet:
- Memory gate (entry/exit) is not wired into the cognitive loop
- RAG retrieval is not wired into the cognitive loop (search_similar exists but loop.py doesn't call it)
- System 2 (Sonnet 4.5) escalation not implemented
- Consolidation operations empty
- Idle loop heartbeat empty
- MemoryStore not instantiated in main.py / passed to cognitive loop
- No `__init__.py` in src/ (imports work with `python3 -m src.main` or relative imports)

---

## Task List (in dependency order)

### DONE
1. ~~Create .env with API keys~~
2. ~~Wire System 1 (Gemini 2.5 Flash Lite) in loop.py~~
3. ~~Add retry logic (src/llm.py)~~
4. ~~Set up Postgres+pgvector + Google embeddings (gemini-embedding-001)~~

### NEXT (in order)

**#5. Wire up memory gate using ACT-R activation math**
- Replace placeholder gate weights with ACT-R: base-level learning + spreading activation + partial matching + noise
- Entry gate: runs on ALL input. Skip if < 10 chars or mechanical. Buffer everything else to scratch_buffer
- Exit gate: score content leaving context window. Threshold >= 0.3 to persist
- Wire MemoryStore into main.py and pass to cognitive loop
- Paper: "Human-Like Remembering and Forgetting in LLM Agents" (ACM 2024) https://dl.acm.org/doi/10.1145/3765766.3765803
- Blocked by: nothing. Ready to start.

**#6. Wire up RAG retrieval using Stanford Generative Agents scoring**
- Currently search_similar() sorts by vector similarity only
- Need combined scoring: recency + importance + relevance (vector sim)
- Layer 1 goal weights should skew retrieval (memories related to active wants surface more easily)
- Inject top-k results (~2000 token budget) into system prompt alongside identity
- Paper: https://arxiv.org/abs/2304.03442
- Blocked by: #5 (gate needs to be storing memories first)

**#7. Wire up System 2 escalation (Claude Sonnet 4.5)**
- System 2 called AS A TOOL by System 1
- Validate design against SOFAI-LM (IBM Research) before building
- Escalation when 2+ metacognitive triggers fire, or any stakes trigger fires
- Paper: https://www.nature.com/articles/s44387-025-00027-5
- Blocked by: #5

**#8. Implement consolidation (Stanford reflection + CMA dreaming)**
- MERGE: cluster similar memories (similarity > 0.85), create insights via store_insight() — DON'T replace originals
- PROMOTE: repeated patterns up layers (5+ signals over 14+ days → Layer 1; 10+ over 30+ days → Layer 0 with approval)
- DECAY: stale memories fade (get_stale_memories + decay_memories already implemented)
- Compulsion safety checks
- Gate weight tuning
- Make introspectable (why_do_i_believe already implemented)
- References: Stanford https://arxiv.org/abs/2304.03442, CMA https://arxiv.org/abs/2601.09913
- Blocked by: #6

**#9. Implement idle loop / DMN**
- Replace random memory sampling with CMA dormant memory recovery
- Score against Layer 1 goals + Layer 0 values
- Goal connection → self-prompt → System 1
- Value connection + no pressing goals → creative impulse (butterfly)
- Reference: CMA https://arxiv.org/abs/2601.09913
- Blocked by: #6

**#10. Add Mem0-style graph layer for entity relationships**
- entity_relationships table already exists in schema
- Extract entities during consolidation, store relationships
- Use for associative retrieval beyond vector similarity
- Reference: https://github.com/mem0ai/mem0
- Blocked by: #6, #8

---

## Key Decisions Made This Session

1. **Gemini 2.5 Flash Lite** for System 1 (was 2.0 Flash)
2. **Claude Sonnet 4.5** for System 2 (was DeepSeek R1)
3. **Google gemini-embedding-001** for embeddings at 768 dims (was Ollama nomic-embed-text). Re-embed at higher dims later if needed — costs $0.75 for 100k memories
4. **Postgres + pgvector** for all memory storage (was ChromaDB + SQLite). One DB for vectors + documents + metadata. Concurrent access for 3 async loops
5. **No Ollama** — machine is too weak and API embeddings are near-free
6. **Insights don't replace originals** — store_insight() creates new memory, lowers source importance, keeps originals queryable. Mirrors human gist + episodic detail
7. **supersedes is a join table** (memory_supersedes) not a single FK — supports many-to-one for consolidation insights
8. **SOTA for each component**: ACT-R for gate, Stanford GA for retrieval, SOFAI-LM for escalation, CMA for consolidation/idle, Mem0 for graph

---

## How to Start the Agent (for testing)

```bash
ssh stonks@norisor.local
cd ~/agent-runtime

# Make sure postgres is running
docker compose up -d postgres

# Run directly (not in Docker yet — Dockerfile needs updating)
export GOOGLE_API_KEY=$(grep GOOGLE_API_KEY .env | cut -d= -f2)
export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2)
export DATABASE_URL="postgresql://agent:agent_secret@localhost:5433/agent_memory"
python3 -m src.main
```

---

## Files to Read in Next Session (in order)

1. This file (`SESSION_HANDOFF.md`) — what exists, what's next
2. `notes.md` — full design document with all architectural decisions
3. The ACT-R paper before implementing #5: https://dl.acm.org/doi/10.1145/3765766.3765803
