from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterable,
    Generic,
    Iterable,
    Literal,
    Optional,
    TypeVar,
    overload,
)

from pydantic import BaseModel

from ._tools import ToolDef
from ._turn import Turn

ChatCompletionT = TypeVar("ChatCompletionT")
ChatCompletionChunkT = TypeVar("ChatCompletionChunkT")
# A dictionary representation of a chat completion
ChatCompletionDictT = TypeVar("ChatCompletionDictT")


class Provider(
    ABC, Generic[ChatCompletionT, ChatCompletionChunkT, ChatCompletionDictT]
):
    @overload
    @abstractmethod
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]],
        kwargs: Any,
    ) -> ChatCompletionT: ...

    @overload
    @abstractmethod
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]],
        kwargs: Any,
    ) -> Iterable[ChatCompletionChunkT]: ...

    @abstractmethod
    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]],
        kwargs: Any,
    ) -> Iterable[ChatCompletionChunkT] | ChatCompletionT: ...

    @overload
    @abstractmethod
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]],
        kwargs: Any,
    ) -> ChatCompletionT: ...

    @overload
    @abstractmethod
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]],
        kwargs: Any,
    ) -> AsyncIterable[ChatCompletionChunkT]: ...

    @abstractmethod
    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]],
        kwargs: Any,
    ) -> AsyncIterable[ChatCompletionChunkT] | ChatCompletionT: ...

    @abstractmethod
    def stream_text(self, chunk: ChatCompletionChunkT) -> Optional[str]: ...

    @abstractmethod
    def stream_merge_chunks(
        self,
        completion: Optional[ChatCompletionDictT],
        chunk: ChatCompletionChunkT,
    ) -> ChatCompletionDictT: ...

    @abstractmethod
    def stream_turn(
        self,
        completion: ChatCompletionDictT,
        has_data_model: bool,
    ) -> Turn: ...

    @abstractmethod
    def value_turn(
        self,
        completion: ChatCompletionT,
        has_data_model: bool,
    ) -> Turn: ...
