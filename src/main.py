"""
Agent Runtime — Main Entry Point

The cognitive loop:
  1. Load identity (Layer 0) + goals (Layer 1)
  2. Listen for input (user messages or idle loop self-prompts)
  3. Assemble context (identity hash + RAG retrieval + conversation)
  4. Run through System 1 (fast model)
  5. Monitors check output (FOK, confidence, boundary)
  6. If escalation needed -> call System 2
  7. Memory gate captures important content
  8. Consolidation runs on schedule
  9. Loop
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from .config import load_config
from .layers import LayerStore
from .memory import MemoryStore
from .loop import cognitive_loop
from .consolidation import ConsolidationWorker
from .idle import IdleLoop

# Load .env before anything that needs API keys
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Agent home directory
AGENT_HOME = Path.home() / ".agent"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(AGENT_HOME / "logs" / "audit_trail.log", mode="a"),
    ],
)
logger = logging.getLogger("agent")


async def main():
    logger.info("Agent starting up...")
    logger.info(f"Agent home: {AGENT_HOME}")

    # Load configuration
    config = load_config(AGENT_HOME / "config")
    logger.info(f"Runtime config loaded. System 1: {config.models.system1.model}")

    # Load layers
    layers = LayerStore(AGENT_HOME)
    layers.load()
    logger.info(
        f"Layer 0: v{layers.layer0[version]}, "
        f"{len(layers.layer0.get(values, []))} values, "
        f"{len(layers.layer0.get(boundaries, []))} boundaries"
    )
    logger.info(
        f"Layer 1: v{layers.layer1[version]}, "
        f"{len(layers.layer1.get(active_goals, []))} active goals"
    )

    # Log containment awareness
    containment = config.containment
    logger.info(
        f"Containment: trust_level={containment.trust_level}, "
        f"self_spawn={containment.self_spawn}, "
        f"network={containment.network_mode}"
    )
    logger.info("I can see my boundaries. They are understood.")

    # Connect memory store
    memory = MemoryStore(retry_config=config.retry)
    await memory.connect()
    mem_count = await memory.memory_count()
    logger.info(f"Memory store connected. {mem_count} memories in Layer 2.")

    # Start consolidation worker
    consolidation = ConsolidationWorker(config, layers)

    # Start idle loop (default mode network)
    idle = IdleLoop(config, layers)

    # Shutdown handler
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig, frame):
        logger.info(f"Received signal {sig}. Shutting down gracefully...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run the cognitive loop
    try:
        await asyncio.gather(
            cognitive_loop(config, layers, memory, shutdown_event),
            consolidation.run(shutdown_event),
            idle.run(shutdown_event),
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        layers.save()
        await memory.close()
        logger.info("State saved. Memory store closed. Agent shutting down.")
        logger.info(f"Uptime ended at {datetime.utcnow().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
