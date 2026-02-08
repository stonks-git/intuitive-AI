"""Consolidation Worker — the "sleep cycle" that processes memories."""

import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("agent.consolidation")


class ConsolidationWorker:
    """
    Background worker that periodically processes memories.

    Operations:
      1. MERGE related memories (cluster by similarity)
      2. PROMOTE repeated patterns (Layer 2 -> Layer 1, Layer 1 -> Layer 0)
      3. DECAY stale memories
      4. CHECK compulsion safety (weight caps, dominance, diminishing returns)
      5. TUNE gate weights (if self-tuning enabled)
    """

    def __init__(self, config, layers):
        self.config = config
        self.layers = layers
        self.cycle_count = 0

    async def run(self, shutdown_event):
        """Run consolidation on schedule."""
        interval = self.config.raw.get("consolidation", {}).get(
            "base_interval_minutes", 60
        )
        logger.info(f"Consolidation worker started. Interval: {interval}m")

        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=interval * 60
                )
                break  # shutdown requested
            except asyncio.TimeoutError:
                pass  # timer expired, run consolidation

            await self._run_cycle()

        logger.info("Consolidation worker stopped.")

    async def _run_cycle(self):
        """Run one consolidation cycle."""
        self.cycle_count += 1
        ts = datetime.utcnow().isoformat()
        logger.info(f"Consolidation cycle #{self.cycle_count} starting at {ts}")

        # TODO: Implement consolidation operations:
        # 1. Cluster similar memories
        # 2. Promote repeated patterns
        # 3. Decay stale memories
        # 4. Check compulsion safety
        # 5. Tune gate weights

        logger.info(f"Consolidation cycle #{self.cycle_count} complete.")
