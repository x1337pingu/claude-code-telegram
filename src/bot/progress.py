"""Progress reporter — periodic Telegram message edits during Claude operations.

Tracks elapsed time, tool usage, and provides periodic updates to the user
so they know Claude is still working rather than seeing silence after "Working...".
"""

import asyncio
import time
from typing import Any, List, Optional

import structlog
from telegram import Message
from telegram.error import BadRequest, TimedOut

from .personality import Personality

logger = structlog.get_logger()

# Throttling constants
MIN_EDIT_INTERVAL = 8  # seconds between message edits
PERIODIC_UPDATE_INTERVAL = 45  # seconds between automatic updates


class ProgressReporter:
    """Track and report progress during Claude operations."""

    def __init__(self, progress_message: Message) -> None:
        self._message = progress_message
        self._start_time = time.monotonic()
        self._last_edit_time = 0.0
        self._tools_used: List[str] = []
        self._current_tool: Optional[str] = None
        self._periodic_task: Optional[asyncio.Task[None]] = None
        self._stopped = False

    # ── Public API ───────────────────────────────────────────────────

    def start_periodic_updates(self) -> None:
        """Launch a background task that edits the progress message periodically."""
        self._periodic_task = asyncio.create_task(self._periodic_loop())

    async def on_stream_update(self, update: Any) -> None:
        """Callback for Claude streaming — extracts tool names, throttles edits."""
        if self._stopped:
            return

        # Extract tool names from stream update
        if hasattr(update, "tool_calls") and update.tool_calls:
            for tool_call in update.tool_calls:
                name = tool_call.get("name", "")
                if name:
                    self._current_tool = name
                    if name not in self._tools_used:
                        self._tools_used.append(name)

        # Throttled edit
        await self._try_edit()

    async def stop(self) -> None:
        """Cancel periodic updates."""
        self._stopped = True
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass

    def summary(self) -> str:
        """Return a short summary like '2m30s · 5 outils (Read, Write, Bash +2)'."""
        elapsed = self._elapsed_str()
        if not self._tools_used:
            return elapsed

        count = len(self._tools_used)
        tool_label = "outil" if count == 1 else "outils"

        if count <= 3:
            tool_names = ", ".join(self._tools_used)
        else:
            shown = ", ".join(self._tools_used[:3])
            tool_names = f"{shown} +{count - 3}"

        return f"{elapsed} · {count} {tool_label} ({tool_names})"

    # ── Internal ─────────────────────────────────────────────────────

    def _elapsed_str(self) -> str:
        """Format elapsed time as '45s' or '2m30s'."""
        seconds = int(time.monotonic() - self._start_time)
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        remaining = seconds % 60
        if remaining == 0:
            return f"{minutes}m"
        return f"{minutes}m{remaining:02d}s"

    async def _try_edit(self) -> None:
        """Edit the Telegram message, respecting MIN_EDIT_INTERVAL."""
        now = time.monotonic()
        if now - self._last_edit_time < MIN_EDIT_INTERVAL:
            return

        text = Personality.progress(
            elapsed=self._elapsed_str(),
            tools_count=len(self._tools_used),
            current_tool=self._current_tool,
        )

        try:
            await self._message.edit_text(text)
            self._last_edit_time = now
        except BadRequest as e:
            # "Message is not modified" — same text, ignore
            if "not modified" in str(e).lower():
                self._last_edit_time = now
            else:
                logger.debug("Failed to edit progress message", error=str(e))
        except TimedOut:
            logger.debug("Timeout editing progress message")
        except Exception as e:
            logger.debug("Unexpected error editing progress", error=str(e))

    async def _periodic_loop(self) -> None:
        """Periodically update the progress message."""
        try:
            while not self._stopped:
                await asyncio.sleep(PERIODIC_UPDATE_INTERVAL)
                if not self._stopped:
                    await self._try_edit()
        except asyncio.CancelledError:
            pass
