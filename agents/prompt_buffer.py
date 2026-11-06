from collections import deque
from typing import List, Optional

class Message:
    """Represents a conversation message with role, content, and optional attachment."""

    def __init__(self, role: str, content: str, attachment: Optional[object] = None):
        """Store the role, content, and optional attachment for a single message."""
        self.role = role
        self.content = content
        self.attachment = attachment

    def __repr__(self):
        """Return a string representation of the Message for debugging."""
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"


class PromptBuffer:
    def __init__(self, limits: dict[str, int]):
        """Initialize named deque buffers with the given per-key size limits."""
        self._limits = {k: int(v) for k, v in limits.items()}
        self._buffers = {k: deque(maxlen=self._limits[k]) for k in self._limits}

    def reset(self):
        """Clear all buffers, discarding every stored value."""
        for buf in self._buffers.values():
            buf.clear()

    def clear(self, key: str):
        """Clear the named buffer in place, leaving its maxlen and other buffers intact."""
        if key not in self._buffers:
            raise KeyError(f"PromptBuffer key '{key}' is not configured.")
        self._buffers[key].clear()

    def add(self, key: str, value):
        """Append a value to the named buffer, evicting the oldest entry if at capacity."""
        if key not in self._buffers:
            raise KeyError(f"PromptBuffer key '{key}' is not configured.")
        self._buffers[key].append(value)

    def get(self, key: str) -> List:
        """Return the current contents of the named buffer as a list."""
        if key not in self._buffers:
            raise KeyError(f"PromptBuffer key '{key}' is not configured.")
        return list(self._buffers[key])

    def get_all(self) -> dict[str, List]:
        """Return all buffers as a dict of key → list."""
        return {k: list(v) for k, v in self._buffers.items()}

    @property
    def limits(self) -> dict[str, int]:
        """Return a copy of the per-key size limits."""
        return dict(self._limits)