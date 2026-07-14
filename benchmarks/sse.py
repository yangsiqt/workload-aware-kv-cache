from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class SSEAccumulator:
    buffer: str = ""
    contents: list[str] = field(default_factory=list)
    done: bool = False

    def feed(self, raw: bytes) -> list[str]:
        self.buffer += raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        emitted: list[str] = []
        while "\n\n" in self.buffer:
            event, self.buffer = self.buffer.split("\n\n", 1)
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    self.done = True
                    continue
                try:
                    parsed = json.loads(data)
                    content = parsed["choices"][0].get("delta", {}).get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if content:
                    text = str(content)
                    self.contents.append(text)
                    emitted.append(text)
        return emitted

    @property
    def text(self) -> str:
        return "".join(self.contents)

    def validation_error(self) -> str | None:
        if not self.done:
            return "stream ended without [DONE]"
        if not self.contents:
            return "stream contained no non-empty content delta"
        return None
