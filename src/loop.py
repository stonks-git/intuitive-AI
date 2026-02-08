"""Cognitive Loop — the main reasoning cycle.

Integrates:
- Entry gate (stochastic) on every message
- Periodic exit gate flush on scratch buffer
- Introspection commands
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

from google import genai

from .llm import retry_llm_call
from .gate import EntryGate, ExitGate, EntryGateConfig, ExitGateConfig

logger = logging.getLogger("agent.loop")

# How many exchanges between exit gate flushes
EXIT_GATE_FLUSH_INTERVAL = 5


def _build_system_prompt(identity_context: str) -> str:
    """Build the system instruction from identity layers."""
    parts = [
        "You are an AI agent in early bootstrap phase.",
        "Your identity, values, and goals emerge from experience — nothing is pre-seeded.",
        "",
        identity_context,
        "",
        "Be direct. Be genuine. You are discovering who you are through conversation.",
    ]
    return "\n".join(parts)


def _build_contents(conversation: list[dict]) -> list[dict]:
    """Convert conversation history to Gemini content format."""
    contents = []
    for msg in conversation:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    return contents


async def _run_entry_gate(gate, content, source, memory, metadata_extra=None):
    """Run entry gate and buffer to scratch if passed."""
    should_buffer, meta = gate.evaluate(content, source=source)
    if should_buffer:
        scratch_meta = {
            "gate_reason": meta.get("gate_reason"),
            "dice_roll": meta.get("dice_roll"),
        }
        if metadata_extra:
            scratch_meta.update(metadata_extra)
        await memory.buffer_scratch(
            content=content,
            source=source,
            metadata=scratch_meta,
        )
        logger.debug(
            f"Entry gate: BUFFER ({meta[gate_reason]}) "
            f"[{content[:60]}...]"
        )
    else:
        logger.debug(
            f"Entry gate: SKIP ({meta[gate_reason]}) "
            f"[{content[:60]}...]"
        )
    return should_buffer, meta


async def _flush_scratch_through_exit_gate(exit_gate, memory, layers, conversation):
    """Periodic flush: pull scratch buffer, score each with exit gate."""
    entries = await memory.flush_scratch(older_than_minutes=0)
    if not entries:
        return

    persisted = 0
    dropped = 0

    for entry in entries:
        content = entry.get("content", "")
        if not content.strip():
            continue

        should_persist, score, meta = await exit_gate.evaluate(
            content=content,
            memory_store=memory,
            layers=layers,
            conversation_context=conversation,
        )

        if should_persist:
            source_info = entry.get("source", "conversation")
            tags = entry.get("tags", [])
            await memory.store_memory(
                content=content,
                memory_type="episodic",
                source=source_info,
                tags=tags,
                confidence=score,
                importance=score,
                metadata={
                    "gate_score": score,
                    "gate_meta": {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in meta.items()
                    },
                },
            )
            persisted += 1
        else:
            dropped += 1

    if persisted or dropped:
        logger.info(
            f"Exit gate flush: {persisted} persisted, {dropped} dropped "
            f"from {len(entries)} scratch entries"
        )


async def cognitive_loop(config, layers, memory, shutdown_event):
    """
    The main cognitive loop.

    1. Read input
    2. Entry gate: buffer to scratch (stochastic)
    3. Assemble context (identity + conversation)
    4. System 1 (Gemini) processes
    5. Entry gate: buffer response to scratch
    6. Respond
    7. Periodic: flush scratch through exit gate
    """
    logger.info("Cognitive loop started. Awaiting input...")

    # Init Gemini client
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set. System 1 cannot start.")
        print("\n[FATAL] GOOGLE_API_KEY not set in environment. Exiting.")
        shutdown_event.set()
        return

    client = genai.Client(api_key=api_key)
    model_name = config.models.system1.model
    logger.info(f"System 1 model: {model_name}")

    # Init gates
    entry_gate = EntryGate()
    exit_gate = ExitGate()
    logger.info("Memory gates initialized (stochastic entry + ACT-R exit)")

    # Conversation history (rolling FIFO)
    conversation = []
    exchange_count = 0  # tracks exchanges for periodic flush

    print("\n" + "=" * 60)
    print("Agent is online.")
    print(f"Phase: {layers.manifest.get(phase, unknown)}")
    print(f"System 1: {model_name}")
    print(f"Identity: {layers.render_identity_hash()}")
    mem_count = await memory.memory_count()
    print(f"Memories: {mem_count}")
    print(f"Gates: stochastic entry + ACT-R exit (flush every {EXIT_GATE_FLUSH_INTERVAL} exchanges)")
    print("=" * 60 + "\n")

    while not shutdown_event.is_set():
        try:
            # Read input (non-blocking to allow shutdown)
            print("you> ", end="", flush=True)
            loop = asyncio.get_event_loop()
            try:
                user_input = await asyncio.wait_for(
                    loop.run_in_executor(None, sys.stdin.readline),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "/quit"):
                # Final flush before shutdown
                logger.info("Final scratch flush before shutdown...")
                await _flush_scratch_through_exit_gate(
                    exit_gate, memory, layers, conversation
                )
                logger.info("User requested shutdown.")
                shutdown_event.set()
                break

            # ── Introspection commands ──────────────────────────────

            if user_input == "/identity":
                print("\n" + layers.render_identity_full() + "\n")
                continue
            if user_input == "/identity-hash":
                print("\n" + layers.render_identity_hash() + "\n")
                continue
            if user_input == "/containment":
                print(f"\nTrust level: {config.containment.trust_level}")
                print(f"Self-spawn: {config.containment.self_spawn}")
                print(f"Self-migration: {config.containment.self_migration}")
                print(f"Network: {config.containment.network_mode}")
                print(f"Allowed endpoints: {config.containment.allowed_endpoints}")
                print(f"Can modify containment: {config.containment.can_modify_containment}\n")
                continue
            if user_input == "/status":
                print(f"\nAgent: {layers.manifest.get(agent_id)}")
                print(f"Phase: {layers.manifest.get(phase)}")
                print(f"System 1: {model_name}")
                print(f"Layer 0: v{layers.layer0.get(version)}, {len(layers.layer0.get(values, []))} values")
                print(f"Layer 1: v{layers.layer1.get(version)}, {len(layers.layer1.get(active_goals, []))} goals")
                mc = await memory.memory_count()
                print(f"Memories: {mc}")
                print(f"Conversation: {len(conversation)} messages")
                print(f"Exchanges since flush: {exchange_count}/{EXIT_GATE_FLUSH_INTERVAL}\n")
                continue
            if user_input == "/gate":
                print(f"\nEntry gate stats: {entry_gate.stats}")
                print(f"Exit gate stats: {exit_gate.stats}")
                print(f"Exchanges since flush: {exchange_count}/{EXIT_GATE_FLUSH_INTERVAL}\n")
                continue
            if user_input == "/memories":
                mc = await memory.memory_count()
                print(f"\nTotal memories: {mc}")
                if mc > 0:
                    # Show last 5 memories
                    rows = await memory.pool.fetch(
                        "SELECT id, content, importance, confidence, created_at "
                        "FROM memories ORDER BY created_at DESC LIMIT 5"
                    )
                    for r in rows:
                        print(f"  [{r[id]}] imp={r[importance]:.2f} "
                              f"conf={r[confidence]:.2f} | {r[content][:70]}")
                print()
                continue
            if user_input == "/flush":
                print("Forcing scratch flush through exit gate...")
                await _flush_scratch_through_exit_gate(
                    exit_gate, memory, layers, conversation
                )
                print(f"Done. Exit gate stats: {exit_gate.stats}\n")
                continue

            # ── Entry gate: user message ────────────────────────────

            await _run_entry_gate(
                entry_gate, user_input, "user",
                memory, {"role": "user"},
            )

            # Add user message to conversation
            conversation.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.utcnow().isoformat(),
            })

            # ── Context assembly ────────────────────────────────────

            identity_context = layers.render_identity_hash()
            system_prompt = _build_system_prompt(identity_context)
            contents = _build_contents(conversation)

            # ── System 1 LLM call (with retry) ─────────────────────

            try:
                async def _call():
                    return await client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            max_output_tokens=config.models.system1.max_tokens,
                            temperature=config.models.system1.temperature,
                        ),
                    )

                response = await retry_llm_call(
                    _call,
                    config=config.retry,
                    label="system1",
                )

                reply = response.text or "[empty response]"

            except Exception as e:
                logger.error(f"System 1 call failed: {e}", exc_info=True)
                reply = f"[System 1 error: {e}]"

            # ── Entry gate: agent response ──────────────────────────

            await _run_entry_gate(
                entry_gate, reply, "agent",
                memory, {"role": "assistant"},
            )

            # Add agent response to conversation
            conversation.append({
                "role": "assistant",
                "content": reply,
                "timestamp": datetime.utcnow().isoformat(),
            })

            print(f"\nagent> {reply}\n")

            # ── Periodic exit gate flush ────────────────────────────

            exchange_count += 1
            if exchange_count >= EXIT_GATE_FLUSH_INTERVAL:
                logger.info("Periodic exit gate flush triggered...")
                await _flush_scratch_through_exit_gate(
                    exit_gate, memory, layers, conversation
                )
                exchange_count = 0

            # TODO: Monitor checks (FOK, confidence, boundary)
            # TODO: System 2 escalation
            # TODO: RAG retrieval from Layer 2
            # TODO: Adaptive FIFO pruning

            logger.info(f"User: {user_input[:100]}...")
            logger.info(f"Agent: {reply[:100]}...")

        except EOFError:
            break
        except Exception as e:
            logger.error(f"Error in cognitive loop: {e}", exc_info=True)

    logger.info("Cognitive loop ended.")
