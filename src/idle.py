"""Idle Loop — the Default Mode Network."""

import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("agent.idle")


class IdleLoop:
    """
    The agent's resting state — not fully off, not always-on.

    Periodically surfaces random memories from Layer 2 and checks
    them against Layer 1 goals and Layer 0 values. If a connection
    is found, generates a self-prompt that may lead to spontaneous action.

    Heartbeat interval adapts to activity level:
      Post-task:     1 minute
      Idle 10 min:   5 minutes
      Idle 1 hour:  15 minutes
      Idle 4+ hours: 30 minutes
    """

    def __init__(self, config, layers):
        self.config = config
        self.layers = layers
        self.last_activity = datetime.utcnow()
        self.heartbeat_count = 0

    def _get_interval(self) -> float:
        """Adaptive heartbeat interval based on idle time."""
        idle_minutes = (datetime.utcnow() - self.last_activity).total_seconds() / 60

        intervals = self.config.raw.get("idle", {}).get("intervals", {})

        if idle_minutes < 10:
            return intervals.get("post_task_minutes", 1) * 60
        elif idle_minutes < 60:
            return intervals.get("idle_10min", 5) * 60
        elif idle_minutes < 240:
            return intervals.get("idle_1hour", 15) * 60
        else:
            return intervals.get("idle_4hours", 30) * 60

    async def run(self, shutdown_event):
        """Run the idle loop."""
        logger.info("Idle loop (DMN) started.")

        while not shutdown_event.is_set():
            interval = self._get_interval()

            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=interval
                )
                break  # shutdown requested
            except asyncio.TimeoutError:
                pass  # timer expired, heartbeat

            await self._heartbeat()

        logger.info("Idle loop stopped.")

    async def _heartbeat(self):
        """One heartbeat of the default mode network."""
        self.heartbeat_count += 1

        # TODO: Implement DMN heartbeat:
        # 1. Sample random memory from Layer 2
        # 2. Check against Layer 1 goals (wanting field)
        # 3. Check against Layer 0 values (creative impulse)
        # 4. If connection found → generate self-prompt
        # 5. Feed self-prompt to System 1

        idle_minutes = (datetime.utcnow() - self.last_activity).total_seconds() / 60
        logger.debug(
            f"DMN heartbeat #{self.heartbeat_count} "
            f"(idle: {idle_minutes:.0f}m, interval: {self._get_interval():.0f}s)"
        )

    def notify_activity(self):
        """Called when the agent processes input — resets idle timer."""
        self.last_activity = datetime.utcnow()
