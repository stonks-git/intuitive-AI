# KB 04: Peripheral Architecture

## Design

All input sources push `AttentionCandidate` objects into a shared `asyncio.Queue(maxsize=50)`. The cognitive loop drains the queue agnostically — it doesn't know or care where input came from.

```
StdinPeripheral ──┐
TelegramPeripheral ──┤──> input_queue ──> cognitive_loop
IdleLoop (DMN) ──┘
```

## reply_fn Pattern

Each `AttentionCandidate` carries `metadata["reply_fn"]` — an async closure that routes the response back to the originating peripheral.

| Peripheral | reply_fn behavior |
|------------|-------------------|
| Telegram | `sendMessage` API call to chat_id |
| Stdin | `print()` to stdout |
| DMN | None (internal thoughts, log-only) |

The loop checks `winner.metadata.get("reply_fn")` after generating a response. If present, calls it. If None, falls back to `print()`.

## Modules

### `src/stdin_peripheral.py` (~60 lines)

- `StdinPeripheral(input_queue)` — no memory needed (loop embeds for stdin)
- `run(shutdown_event)` — `sys.stdin.readline` loop with 1s timeout via `asyncio.wait_for`
- Skips running if stdin is not a TTY
- Creates `AttentionCandidate(source_tag="external_user", metadata={"reply_fn": <print_closure>, "peripheral": "stdin"})`

### `src/telegram_peripheral.py` (~155 lines)

- `TelegramPeripheral(input_queue, memory)` — needs memory for embedding messages
- Raw `httpx` calls to Telegram Bot API (no framework dependency)
- Two API endpoints only: `getUpdates` (long polling) + `sendMessage` (with 4096-char chunking)
- `is_configured` property: True only if both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_OWNER_ID` env vars set
- Auth: `message.from.id == TELEGRAM_OWNER_ID` (Telegram server-verified, unforgeable). Silent-drop unauthorized.
- On message: embed via `memory.embed()`, create `AttentionCandidate`, push to queue
- Commands (`/status`, etc.) queued as regular candidates — loop handles them
- Long polling: outbound-only connections to api.telegram.org, no public endpoint

### `src/idle.py` (modified)

- Renamed `dmn_queue` → `input_queue` (DMN pushes to same shared queue)

### `src/loop.py` (modified)

- Signature: `cognitive_loop(..., input_queue: asyncio.Queue)` (was `dmn_queue=None`)
- Input block: `await asyncio.wait_for(input_queue.get(), timeout=1.0)` + non-blocking drain
- Output block: `reply_fn` routing instead of hardcoded `print()`
- `_handle_command()`: accepts `reply_fn=None`, uses `_send()` helper that routes to reply_fn or print

### `src/main.py` (modified)

- Creates shared `input_queue = asyncio.Queue(maxsize=50)`
- Wires all peripherals: stdin, telegram (if configured), idle, cognitive_loop
- All run concurrently via `asyncio.gather()`

### `src/context_assembly.py` (bug fix)

- `assemble_context()` now accepts `attention_text: str` param
- `_get_situational_memories()` returns `[]` if `query_text` is empty (was crashing embed API with `query=""`)

## Security

- Long polling = outbound only (HTTPS to api.telegram.org). No webhook, no public endpoint.
- Owner-only: `TELEGRAM_OWNER_ID` env var. Telegram server guarantees `from.id` integrity.
- Bot token stored only in norisor `.env` (not in git).
- `api.telegram.org` whitelisted in `containment.yaml`.

## Adding a New Peripheral

1. Create `src/new_peripheral.py`
2. Constructor takes `input_queue: asyncio.Queue` (+ any deps like memory)
3. `run(shutdown_event)` — async loop that listens for input
4. On input: create `AttentionCandidate(source_tag="external_user", metadata={"reply_fn": <closure>, "peripheral": "name"})`
5. Push to `input_queue.put_nowait(candidate)` (handle QueueFull)
6. In `src/main.py`: instantiate peripheral, add `peripheral.run(shutdown_event)` to tasks list

## Env Vars

| Var | Purpose |
|-----|---------|
| `TELEGRAM_BOT_TOKEN` | Bot API token from @BotFather |
| `TELEGRAM_OWNER_ID` | Numeric Telegram user ID (owner-only auth) |

## Future: Multimodal

- `AttentionCandidate.content` becomes `str | list` (Gemini multimodal format)
- Peripheral downloads images/audio from Telegram, packs into candidate
- Embedding stays text-based (subconscious); LLM call gets full multimodal (conscious)
- Content type mismatch between text-label and visual content = gut delta signal
- Resize images to 512px (258 tokens, ~$0.00003 per image)
