"""Per-user message queue with active task cancellation.

When a user sends a new message while Claude is processing a previous one,
the active task is cancelled and re-launched with both prompts merged.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class ActiveTask:
    """Represents an in-flight Claude execution for a single user."""

    task: "asyncio.Task[Any]"
    original_prompt: str
    user_id: int
    progress_msg: Any  # telegram.Message — kept as Any to avoid import
    heartbeat: "asyncio.Task[None]"
    created_at: float = field(default_factory=time.time)


class UserMessageQueue:
    """Per-user message queue with active task cancellation.

    Thread-safe via per-user asyncio locks.  The typical flow is:

    1. Handler calls ``cancel_if_active(user_id)``
       - If a task is running it is cancelled; the original prompt is returned.
    2. Handler creates an ``asyncio.Task`` for the Claude execution.
    3. Handler calls ``register(user_id, active_task)``.
    4. When the task finishes (or is cancelled) the handler calls
       ``unregister(user_id)`` in a ``finally`` block.
    """

    def __init__(self) -> None:
        self._active: Dict[int, ActiveTask] = {}
        self._locks: Dict[int, asyncio.Lock] = {}

    def _get_lock(self, user_id: int) -> asyncio.Lock:
        """Return (or create) the per-user lock."""
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def is_busy(self, user_id: int) -> bool:
        """Return True if the user has an active Claude task."""
        return user_id in self._active

    async def cancel_if_active(self, user_id: int) -> Optional[str]:
        """Cancel the running task for *user_id* and return its original prompt.

        Returns ``None`` when no task was active.
        """
        lock = self._get_lock(user_id)
        async with lock:
            active = self._active.pop(user_id, None)
            if active is None:
                return None

            logger.info(
                "Cancelling active task for user",
                user_id=user_id,
                age_seconds=round(time.time() - active.created_at, 1),
            )

            # Cancel heartbeat first (lightweight)
            active.heartbeat.cancel()

            # Cancel the main Claude task
            active.task.cancel()

            # Wait for the task to acknowledge cancellation
            try:
                await asyncio.wait_for(
                    asyncio.shield(active.task),
                    timeout=5.0,
                )
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

            # Delete the old progress message
            try:
                await active.progress_msg.delete()
            except Exception:
                pass

            return active.original_prompt

    def register(self, user_id: int, active_task: ActiveTask) -> None:
        """Register an active task for the user (non-blocking)."""
        self._active[user_id] = active_task

    def unregister(self, user_id: int) -> None:
        """Remove the active task entry for the user."""
        self._active.pop(user_id, None)

    @staticmethod
    def merge_prompts(original: str, addition: str) -> str:
        """Merge an original prompt with additional context."""
        return (
            f"{original}\n\n"
            f"[ADDITIONAL CONTEXT added during processing]\n"
            f"{addition}"
        )

    @property
    def active_count(self) -> int:
        """Number of users with active tasks."""
        return len(self._active)
