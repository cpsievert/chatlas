import json
from dataclasses import dataclass
from typing import Any, Literal, Optional

ImageContentTypes = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
]


class Content:
    pass


@dataclass
class ContentText(Content):
    text: str


class ContentImage(Content):
    pass


@dataclass
class ContentImageRemote(ContentImage):
    url: str
    detail: str = ""

    def __str__(self):
        return f"[remote image]: {self.url}"


@dataclass
class ContentImageInline(ContentImage):
    content_type: ImageContentTypes
    data: Optional[str] = None

    def __str__(self):
        n_bytes = len(self.data) if self.data else 0
        return f"[inline image]: {self.content_type} ({n_bytes} bytes)"


@dataclass
class ContentToolRequest(Content):
    id: str
    name: str
    arguments: dict[str, Any]

    def __str__(self):
        args_str = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        return f"[tool request ({self.id})]: {self.name}({args_str})"


@dataclass
class ContentToolResult(Content):
    id: str
    value: Any = None
    error: Optional[str] = None

    def __str__(self):
        if self.error:
            return f"[tool result ({self.id})]: Error: {self.error}"
        return f"[tool result ({self.id})]: {self.value}"

    def get_final_value(self) -> Any:
        if self.error:
            return f"Tool calling failed with error: '{self.error}'"
        return str(self.value)


@dataclass
class ContentJson(Content):
    value: dict[str, Any]

    def __str__(self):
        return f"[json]: {self.value}"

    def to_json(self):
        return json.dumps(self.value)
