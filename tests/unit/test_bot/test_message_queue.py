"""Tests for per-user message queue with interrupt-and-merge."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.message_queue import ActiveTask, UserMessageQueue


@pytest.fixture
def queue() -> UserMessageQueue:
    return UserMessageQueue()


def _make_active_task(
    user_id: int = 1,
    prompt: str = "original prompt",
    *,
    task: "asyncio.Task[None] | None" = None,
) -> ActiveTask:
    """Create an ActiveTask with mocked components."""
    if task is None:
        # Create a real never-completing task so cancel() works
        loop = asyncio.get_event_loop()
        task = loop.create_task(asyncio.sleep(3600))
    return ActiveTask(
        task=task,
        original_prompt=prompt,
        user_id=user_id,
        progress_msg=MagicMock(delete=AsyncMock()),
        heartbeat=asyncio.get_event_loop().create_task(asyncio.sleep(3600)),
    )


class TestUserMessageQueue:
    """Tests for UserMessageQueue."""

    @pytest.mark.asyncio
    async def test_is_busy_false_initially(self, queue: UserMessageQueue) -> None:
        assert queue.is_busy(1) is False

    @pytest.mark.asyncio
    async def test_register_makes_busy(self, queue: UserMessageQueue) -> None:
        active = _make_active_task(user_id=1)
        queue.register(1, active)
        assert queue.is_busy(1) is True
        # Cleanup
        active.task.cancel()
        active.heartbeat.cancel()

    @pytest.mark.asyncio
    async def test_unregister_clears_busy(self, queue: UserMessageQueue) -> None:
        active = _make_active_task(user_id=1)
        queue.register(1, active)
        queue.unregister(1)
        assert queue.is_busy(1) is False
        # Cleanup
        active.task.cancel()
        active.heartbeat.cancel()

    @pytest.mark.asyncio
    async def test_unregister_noop_for_unknown_user(
        self, queue: UserMessageQueue
    ) -> None:
        queue.unregister(999)  # Should not raise

    @pytest.mark.asyncio
    async def test_cancel_if_active_returns_none_when_idle(
        self, queue: UserMessageQueue
    ) -> None:
        result = await queue.cancel_if_active(42)
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_if_active_returns_original_prompt(
        self, queue: UserMessageQueue
    ) -> None:
        active = _make_active_task(user_id=1, prompt="build the feature")
        queue.register(1, active)
        result = await queue.cancel_if_active(1)
        assert result == "build the feature"
        assert queue.is_busy(1) is False

    @pytest.mark.asyncio
    async def test_cancel_if_active_cancels_task(self, queue: UserMessageQueue) -> None:
        active = _make_active_task(user_id=1)
        queue.register(1, active)
        await queue.cancel_if_active(1)
        assert active.task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_if_active_cancels_heartbeat(
        self, queue: UserMessageQueue
    ) -> None:
        active = _make_active_task(user_id=1)
        queue.register(1, active)
        await queue.cancel_if_active(1)
        assert active.heartbeat.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_if_active_deletes_progress_msg(
        self, queue: UserMessageQueue
    ) -> None:
        active = _make_active_task(user_id=1)
        queue.register(1, active)
        await queue.cancel_if_active(1)
        active.progress_msg.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancel_if_active_handles_delete_failure(
        self, queue: UserMessageQueue
    ) -> None:
        active = _make_active_task(user_id=1)
        active.progress_msg.delete = AsyncMock(side_effect=Exception("Telegram error"))
        queue.register(1, active)
        # Should not raise
        result = await queue.cancel_if_active(1)
        assert result == "original prompt"

    @pytest.mark.asyncio
    async def test_active_count(self, queue: UserMessageQueue) -> None:
        assert queue.active_count == 0
        a1 = _make_active_task(user_id=1)
        a2 = _make_active_task(user_id=2)
        queue.register(1, a1)
        queue.register(2, a2)
        assert queue.active_count == 2
        queue.unregister(1)
        assert queue.active_count == 1
        # Cleanup
        a1.task.cancel()
        a1.heartbeat.cancel()
        a2.task.cancel()
        a2.heartbeat.cancel()

    @pytest.mark.asyncio
    async def test_different_users_independent(self, queue: UserMessageQueue) -> None:
        a1 = _make_active_task(user_id=1, prompt="user1 prompt")
        a2 = _make_active_task(user_id=2, prompt="user2 prompt")
        queue.register(1, a1)
        queue.register(2, a2)

        result = await queue.cancel_if_active(1)
        assert result == "user1 prompt"
        assert queue.is_busy(1) is False
        assert queue.is_busy(2) is True

        # Cleanup
        a2.task.cancel()
        a2.heartbeat.cancel()


class TestMergePrompts:
    """Tests for prompt merging."""

    def test_basic_merge(self) -> None:
        result = UserMessageQueue.merge_prompts("hello", "world")
        assert "hello" in result
        assert "world" in result
        assert "[ADDITIONAL CONTEXT" in result

    def test_merge_preserves_original_first(self) -> None:
        result = UserMessageQueue.merge_prompts("first", "second")
        assert result.index("first") < result.index("second")

    def test_merge_with_empty_addition(self) -> None:
        result = UserMessageQueue.merge_prompts("original", "")
        assert "original" in result

    def test_merge_with_multiline(self) -> None:
        original = "line1\nline2"
        addition = "extra1\nextra2"
        result = UserMessageQueue.merge_prompts(original, addition)
        assert "line1\nline2" in result
        assert "extra1\nextra2" in result


class TestActiveTask:
    """Tests for ActiveTask dataclass."""

    @pytest.mark.asyncio
    async def test_created_at_auto_set(self) -> None:
        active = _make_active_task()
        assert active.created_at > 0
        # Cleanup
        active.task.cancel()
        active.heartbeat.cancel()

    @pytest.mark.asyncio
    async def test_fields_stored(self) -> None:
        task = asyncio.get_event_loop().create_task(asyncio.sleep(3600))
        heartbeat = asyncio.get_event_loop().create_task(asyncio.sleep(3600))
        progress = MagicMock(delete=AsyncMock())
        active = ActiveTask(
            task=task,
            original_prompt="test prompt",
            user_id=42,
            progress_msg=progress,
            heartbeat=heartbeat,
        )
        assert active.original_prompt == "test prompt"
        assert active.user_id == 42
        assert active.progress_msg is progress
        # Cleanup
        task.cancel()
        heartbeat.cancel()
