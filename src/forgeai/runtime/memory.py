"""Conversation memory for multi-turn interactions in Feature 10."""

from __future__ import annotations

import json
from pathlib import Path

from forgeai.runtime.schemas import ChatTurn


class ConversationMemory:
    """Bounded conversation history manager with persistence and query context helpers."""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._turns: list[ChatTurn] = []

    def add_user_message(self, content: str) -> ChatTurn:
        turn = ChatTurn(role="user", content=content)
        self._append(turn)
        return turn

    def add_assistant_message(
        self,
        content: str,
        sources: list[str] | None = None,
        tool_calls: list | None = None,
    ) -> ChatTurn:
        turn = ChatTurn(
            role="assistant",
            content=content,
            sources=sources or [],
            tool_calls=tool_calls or [],
        )
        self._append(turn)
        return turn

    def get_history(self) -> list[ChatTurn]:
        return list(self._turns)

    def to_llm_messages(self, max_messages: int = 10) -> list[dict[str, str]]:
        recent = self._turns[-max_messages:]
        return [{"role": t.role, "content": t.content} for t in recent]

    def last_user_topic_context(self) -> str:
        """Return prior user turn text to help resolve short follow-up questions."""
        user_turns = [t.content for t in self._turns if t.role == "user"]
        if len(user_turns) >= 2:
            return user_turns[-2]
        return ""

    def clear(self) -> None:
        self._turns.clear()

    def save(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = [t.model_dump() for t in self._turns]
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return dest

    def load(self, path: str | Path) -> None:
        src = Path(path)
        if not src.is_file():
            return
        data = json.loads(src.read_text(encoding="utf-8"))
        self._turns = [ChatTurn.model_validate(item) for item in data][-self.max_turns :]

    def _append(self, turn: ChatTurn) -> None:
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

