# Supervisor Handoff

> **READ THIS FIRST.** You are the supervisor (queen agent) for this project.
>
> **Reading order (MANDATORY):**
> 1. This file (handoff.md) - bootstrap loader
> 2. `prompts/supervisor.md` - your supervisor contract
> 3. `state/charter.json` - project constraints (MANDATORY)
> 4. `python3 taskmaster.py ready` - available tasks
> 5. Sections below - previous session context

---

## Previous Sessions

### SESSION 2026-02-11 (C+D+E) - SAFETY + CONSOLIDATION + PERIPHERALS

**STATUS:** DONE

**What was done:**
1. Tasks 18-22: escalation, System 2, reflection bank, retrieval mutation, safety ceilings
2. Tasks 23-28: two-tier consolidation engine (constant + deep)
3. Tasks 29-35: DMN idle loop, energy tracking, session restart, docs, gut feeling, bootstrap readiness, outcome tracking

---

### SESSION 2026-02-12 (F) - FRAMEWORK ADOPTION + WIRE PHASE

**STATUS:** DONE

**What was done:**
1. Adopted AI-DEV framework (taskmaster.py, state/, prompts/, KB/)
2. FW-001 DONE — framework fully adopted
3. WIRE-001 DONE — GutFeeling wired into cognitive loop
4. WIRE-002 DONE — BootstrapReadiness wired into cognitive loop
5. WIRE-003 DONE — OutcomeTracker wired into safety + consolidation

---

### SESSION 2026-02-13 (G) - PERIPHERAL ARCHITECTURE + TELEGRAM

**STATUS:** IN PROGRESS — NEED TO FINISH DEPLOY + TEST

**What was done:**
1. Cleaned norisor of all old files (src/, .git, docs). Only .env, docker-compose.yml, agent-state/ remain.
2. Fixed Docker image name: `ghcr.io/stonks-git/intuitive-ai:latest` (was typo `intuititive-identity-ai`)
3. Fixed main.py: create `~/.agent/logs/` dir on startup (FileNotFoundError)
4. **PERIPHERAL ARCHITECTURE BUILT** — the big feature this session:
   - `src/stdin_peripheral.py` NEW — stdin factored out as a peripheral
   - `src/telegram_peripheral.py` NEW — raw httpx Telegram Bot API (long polling, owner-only auth)
   - `src/loop.py` MODIFIED — replaced hardcoded stdin with unified `input_queue: asyncio.Queue`
     - All peripherals push `AttentionCandidate` objects into shared queue
     - `reply_fn` callback in candidate metadata routes responses back to correct peripheral
     - `_handle_command()` now uses `_send()` helper that routes to reply_fn or stdout
     - Removed `sys.stdin.readline` and `dmn_queue` param
   - `src/idle.py` MODIFIED — renamed `dmn_queue` → `input_queue` (DMN pushes to shared queue)
   - `src/main.py` MODIFIED — creates shared `input_queue(maxsize=50)`, wires stdin + telegram + idle
   - `src/context_assembly.py` MODIFIED — fixed pre-existing bug: `query=""` crashed embed API.
     Now passes `attention_text=winner.content` from loop through to `search_hybrid`.
5. Telegram bot created: `@alecprats_ai_bot`
   - Token + owner_id configured in norisor `.env`
   - `api.telegram.org` added to containment.yaml whitelist
6. First deploy test: agent started, Telegram connected, bot received message, won attention.
   Crashed on context_assembly empty query bug (fixed in commit 130f26e).

**WHAT REMAINS (next session must do):**
1. **Wait for CI/CD build of commit 130f26e** (was building when session ended)
2. **Pull new image on norisor**: `ssh norisor "cd ~/agent-runtime && docker pull ghcr.io/stonks-git/intuitive-ai:latest"`
3. **Restart agent**: `ssh norisor "cd ~/agent-runtime && docker compose down && docker compose up -d"`
4. **Test from Telegram**: send a message to @alecprats_ai_bot — should get a response
5. **Test introspection commands**: `/status`, `/readiness`, `/cost` from Telegram
6. If all works → mark PERIPH-001 done, update roadmap
7. Then proceed with TEST-001 (full end-to-end test)

**Commits this session:**
- `0f04799` Fix Docker image name and ensure logs directory exists
- `a5442e8` Add peripheral architecture: Telegram + stdin input, unified queue
- `130f26e` Fix empty query crash in context assembly + telegram offset

---

## What is this project?

Cognitive architecture for emergent AI identity. Three-layer memory unified into one Postgres store with continuous depth_weight (Beta distribution). Dual-process reasoning (System 1: Gemini Flash Lite, System 2: Claude Sonnet 4.5). Metacognitive monitoring. Consolidation sleep cycle. DMN idle loop. Two-centroid gut feeling model. Identity emerges from experience, not configuration. All 35 implementation plan tasks complete. Peripheral architecture built. Currently testing Telegram integration.

---

## Tasks DOING now

| Task ID | Status |
|---------|--------|
| FW-001 | done |
| WIRE-001/002/003 | done |
| PERIPH-001 (Telegram) | IN PROGRESS — code done, need to redeploy + test |
| TEST-001 | next after PERIPH-001 |

---

## What exists

### Source files (src/)

```
src/
  __init__.py              empty
  config.py                working, clean
  llm.py                   EnergyTracker class (cost tracking)
  memory.py                Full memory store (embed, search_hybrid, search_reranked, retrieval mutation, safety integration)
  safety.py                SafetyMonitor + 6 ceiling classes + OutcomeTracker
  layers.py                L0/L1 disk store + embedding cache
  stochastic.py            StochasticWeight (Beta distribution)
  activation.py            ACT-R 4-component activation equation
  metacognition.py         Composite confidence scoring
  tokens.py                Token counting utilities
  gate.py                  3x3 exit gate + stochastic entry gate
  loop.py                  Attention-agnostic cognitive loop (unified input_queue, reply_fn routing)
  main.py                  Entry point, consolidation engine, peripheral wiring, session tracking
  relevance.py             5-component hybrid relevance + Dirichlet blend
  attention.py             Salience-based attention allocation (AttentionCandidate, AttentionAllocator)
  context_assembly.py      Dynamic context injection + FIFO pruning (fixed empty query bug)
  consolidation.py         Two-tier: ConstantConsolidation + DeepConsolidation
  idle.py                  DMN with stochastic sampling, pushes to shared input_queue
  gut.py                   Two-centroid gut feeling model
  bootstrap.py             10 readiness milestones
  stdin_peripheral.py      NEW — stdin I/O as a peripheral (pushes AttentionCandidate to input_queue)
  telegram_peripheral.py   NEW — Telegram Bot API via raw httpx (long polling, owner auth, reply_fn)
```

### Peripheral Architecture (NEW)

```
                    ┌──────────────┐
                    │  input_queue  │  asyncio.Queue(maxsize=50)
                    │  (shared)     │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   StdinPeripheral   TelegramPeripheral    IdleLoop (DMN)
   (push external_user)  (push external_user)  (push internal_dmn)
        │                  │                  │
        │           reply_fn=sendMessage      │
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    cognitive_loop drains queue
                    → attention allocation
                    → context assembly
                    → LLM call
                    → reply via winner.metadata["reply_fn"]
```

Key design:
- All peripherals push `AttentionCandidate` objects into one queue
- Each candidate carries `metadata["reply_fn"]` — async callback to route response
- Telegram: reply_fn calls `sendMessage` API
- Stdin: reply_fn calls `print()`
- DMN: no reply_fn (internal thoughts log-only)
- Loop doesn't know or care where input came from

### Norisor setup (CLEAN)

```
~/agent-runtime/
  .env                 API keys + TELEGRAM_BOT_TOKEN + TELEGRAM_OWNER_ID
  docker-compose.yml   agent_001 + agent_postgres (correct image name)
  agent-state/         config/, identity/, goals/, logs/, manifest.json
```

- No src files on norisor (Docker-only deployment)
- Image: `ghcr.io/stonks-git/intuitive-ai:latest`
- Postgres data volume preserved (6 memories)
- CI/CD: push to main → GitHub Actions → build image → pull on norisor

### Connection info

- **Server:** norisor (Debian, Docker)
- **Tailscale IP:** 100.66.170.31 (hostname `norisor`)
- **SSH:** `ssh norisor` (configured in ~/.ssh/config)
- **DB:** `postgresql://agent:agent_secret@localhost:5433/agent_memory`
- **Deploy:** Push to main -> GitHub Actions -> Docker -> norisor
- **Telegram bot:** @alecprats_ai_bot (token in norisor .env)
- **Telegram owner ID:** 6639032827

---

## Docker/Prod Status

- Docker Compose on norisor: agent container (2 CPU/2GB) + postgres container (1 CPU/1GB)
- CI/CD: GitHub Actions builds on push to main (src/, Dockerfile, requirements.txt, docker-compose.yml)
- Image: ghcr.io/stonks-git/intuitive-ai:latest
- Norisor cleaned: only docker-compose.yml + .env + agent-state/ (no old src/docs)

---

## Blockers or open questions

| Blocker/Question | Status |
|------------------|--------|
| ~~GutFeeling, Bootstrap, OutcomeTracker not wired~~ | DONE |
| ~~Docker image name typo~~ | FIXED (0f04799) |
| ~~Empty query crash in context_assembly~~ | FIXED (130f26e) |
| Need to redeploy commit 130f26e and test Telegram | NEXT ACTION |
| Multimodal perception layer (images/audio in attention loop) | FUTURE — discussed architecture, not started |

---

## Useful commands (copy-paste ready)

```bash
# Validate framework state
python3 taskmaster.py validate

# Ready tasks
python3 taskmaster.py ready

# Local Python (use venv)
.venv/bin/python3 -m py_compile src/foo.py

# Deploy to norisor (Docker only)
git push origin main  # triggers CI/CD
ssh norisor "cd ~/agent-runtime && docker pull ghcr.io/stonks-git/intuitive-ai:latest && docker compose down && docker compose up -d"

# Check agent logs
ssh norisor "docker logs --tail 30 agent_001"

# Check if Telegram is connected
ssh norisor "docker logs agent_001 2>&1 | grep -i telegram"
```

---

## Key architectural decisions (resolved, don't revisit)

- Unified memory (not 3 discrete layers) -- depth_weight Beta distribution
- Identity is a rendered view of high-weight memories, not a stored artifact
- Stochastic everything -- Beta weights, Dirichlet blends, injection rolls
- ACT-R equations with human-calibrated starting points, evolved by consolidation
- Attention-agnostic loop -- all input sources feed same pipeline via unified input_queue
- Build all safety from day one, enable incrementally
- Dual-process: System 1 (Gemini Flash Lite) drives, System 2 (Claude Sonnet 4.5) escalation
- Reflection bank: System 2 corrections stored as type="correction" memories
- Peripheral architecture: any I/O source pushes AttentionCandidate into shared queue, reply_fn routes responses back
- Telegram: raw httpx (no framework dependency), long polling (no public endpoint), owner-only auth
- Multimodal future: embedding stays text-based (subconscious); LLM gets full multimodal (conscious). Content type mismatch = gut delta signal.

---

## Checklist before handoff

- [x] Updated task statuses in handoff
- [x] Completed current session section above
- [x] devlog updated
- [x] **Kept only last 3 sessions**
- [x] KB updated if code was changed

---

## Git Status

- **Branch:** main
- **Last commit:** 130f26e Fix empty query crash in context assembly + telegram offset
- **CI/CD:** Build may still be running for 130f26e — check before deploying

---

## Memory Marker

```
MEMORY_MARKER: 2026-02-13 | PERIPH-001 in progress | Telegram peripheral built, code committed, need redeploy+test | Next: pull image, docker compose up, test from Telegram
```
