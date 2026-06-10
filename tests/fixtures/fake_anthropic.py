"""Minimal SDK-shaped fakes for hermetic tests (no network, no real key).

These are the generic building blocks (mirrors the sibling project's fixture in
spirit). Add task-specific response builders as the engine grows — e.g. a helper
that returns a ``FakeMessage`` whose content is a ``FakeToolUseBlock`` carrying a
legend style->system mapping, for testing ``legend.py`` (milestone M3).

Usage: inject ``FakeClient`` into an engine function's ``client=`` parameter; the
``responder`` callable receives the ``messages.create(**kwargs)`` call dict and
returns the ``FakeMessage`` to hand back (or raises, to exercise error paths).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict
    id: str = "toolu_fake"
    type: str = "tool_use"


@dataclass
class FakeMessage:
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "fake-model"
    id: str = "msg_fake"
    role: str = "assistant"
    type: str = "message"


class _FakeMessages:
    """Stand-in for ``client.messages``: records calls, returns a scripted reply."""

    def __init__(self, responder: Callable[[dict], FakeMessage]) -> None:
        self._responder = responder
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> FakeMessage:
        self.calls.append(kwargs)
        return self._responder(kwargs)


@dataclass
class FakeFileObject:
    id: str = "file_fake"


class _FakeFiles:
    """Stand-in for ``client.beta.files``: records uploads/deletes."""

    def __init__(self) -> None:
        self.uploads: list[dict] = []
        self.deleted: list[str] = []

    def upload(self, **kwargs: Any) -> FakeFileObject:
        self.uploads.append(kwargs)
        return FakeFileObject(id=f"file_fake_{len(self.uploads)}")

    def delete(self, file_id: str) -> None:
        self.deleted.append(file_id)


class _FakeBeta:
    def __init__(self) -> None:
        self.files = _FakeFiles()


class FakeClient:
    """Duck-typed Anthropic client for ``client=`` injection in tests.

    ``with_files=True`` adds a ``beta.files`` fake (upload/delete recorders)
    for code paths that use the Files API; the default omits it so fallback
    paths (clients without a Files API) stay exercised.
    """

    def __init__(self, responder: Callable[[dict], FakeMessage], *, with_files: bool = False) -> None:
        self.messages = _FakeMessages(responder)
        if with_files:
            self.beta = _FakeBeta()
