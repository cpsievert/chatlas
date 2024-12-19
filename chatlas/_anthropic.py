from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast, overload

from pydantic import BaseModel

from ._chat import Chat
from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentJson,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
)
from ._logging import log_model_default
from ._provider import Provider
from ._tokens import tokens_log
from ._tools import Tool, basemodel_to_param_schema
from ._turn import Turn, normalize_turns

if TYPE_CHECKING:
    from anthropic.types import (
        Message,
        MessageParam,
        RawMessageStreamEvent,
        TextBlock,
        ToolParam,
        ToolUseBlock,
    )
    from anthropic.types.image_block_param import ImageBlockParam
    from anthropic.types.model_param import ModelParam
    from anthropic.types.text_block_param import TextBlockParam
    from anthropic.types.tool_result_block_param import ToolResultBlockParam
    from anthropic.types.tool_use_block_param import ToolUseBlockParam
    from openai.types.chat import ChatCompletionToolParam

    from .types.anthropic import ChatBedrockClientArgs, ChatClientArgs, SubmitInputArgs

    ContentBlockParam = Union[
        TextBlockParam,
        ImageBlockParam,
        ToolUseBlockParam,
        ToolResultBlockParam,
    ]
else:
    Message = object
    RawMessageStreamEvent = object


def ChatAnthropic(
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    model: "Optional[ModelParam]" = None,
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", Message]:
    """
    Chat with an Anthropic Claude model.

    [Anthropic](https://www.anthropic.com) provides a number of chat based
    models under the [Claude](https://www.anthropic.com/claude) moniker.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Note that a Claude Prop membership does not give you the ability to call
    models via the API. You will need to go to the [developer
    console](https://console.anthropic.com/account/keys) to sign up (and pay
    for) a developer account that will give you an API key that you can use with
    this package.
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatAnthropic` requires the `anthropic` package (e.g., `pip install anthropic`).
    :::

    Examples
    --------

    ```python
    import os
    from chatlas import ChatAnthropic

    chat = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch. Do
        not provide non-None values for both `turns` and `system_prompt`. Each
        message in the list should be a dictionary with at least `role` (usually
        `system`, `user`, or `assistant`, but `tool` is also possible). Normally
        there is also a `content` field, which is a string.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly
        choosing a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `ANTHROPIC_API_KEY` environment
        variable.
    max_tokens
        Maximum number of tokens to generate before stopping.
    kwargs
        Additional arguments to pass to the `anthropic.Anthropic()` client
        constructor.

    Returns
    -------
    Chat
        A Chat object.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatAnthropic(api_key="...")`)
    is the simplest way to get started, and is fine for interactive use, but is
    problematic for code that may be shared with others.

    Instead, consider using environment variables or a configuration file to manage
    your credentials. One popular way to manage credentials is to use a `.env` file
    to store your credentials, and then use the `python-dotenv` package to load them
    into your environment.

    ```shell
    pip install python-dotenv
    ```

    ```shell
    # .env
    ANTHROPIC_API_KEY=...
    ```

    ```python
    from chatlas import ChatAnthropic
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatAnthropic()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export ANTHROPIC_API_KEY=...
    ```
    """

    if model is None:
        model = log_model_default("claude-3-5-sonnet-latest")

    return Chat(
        provider=AnthropicProvider(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            kwargs=kwargs,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt,
        ),
    )


class AnthropicProvider(Provider[Message, RawMessageStreamEvent, Message]):
    def __init__(
        self,
        *,
        max_tokens: int,
        model: str,
        api_key: str | None,
        kwargs: Optional["ChatClientArgs"] = None,
    ):
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError:
            raise ImportError(
                "`ChatAnthropic()` requires the `anthropic` package. "
                "You can install it with 'pip install anthropic'."
            )

        self._model = model
        self._max_tokens = max_tokens

        kwargs_full: "ChatClientArgs" = {
            "api_key": api_key,
            **(kwargs or {}),
        }

        # TODO: worth bringing in sync types?
        self._client = Anthropic(**kwargs_full)  # type: ignore
        self._async_client = AsyncAnthropic(**kwargs_full)

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        return self._client.messages.create(**kwargs)  # type: ignore

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        return await self._async_client.messages.create(**kwargs)  # type: ignore

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ) -> "SubmitInputArgs":
        tool_schemas = [
            self._tool_schema_json(tool.schema) for tool in tools.values()
        ]

        # If data extraction is requested, add a "mock" tool with parameters inferred from the data model
        data_model_tool: Tool | None = None
        if data_model is not None:

            def _structured_tool_call(**kwargs: Any):
                """Extract structured data"""
                pass

            data_model_tool = Tool(_structured_tool_call)

            data_model_tool.schema["function"]["parameters"] = {
                "type": "object",
                "properties": {
                    "data": basemodel_to_param_schema(data_model),
                },
            }

            tool_schemas.append(self._tool_schema_json(data_model_tool.schema))

            if stream:
                stream = False
                warnings.warn(
                    "Anthropic does not support structured data extraction in streaming mode."
                )

        kwargs_full: "SubmitInputArgs" = {
            "stream": stream,
            "messages": self._as_message_params(turns),
            "model": self._model,
            "max_tokens": self._max_tokens,
            "tools": tool_schemas,
            **(kwargs or {}),
        }

        if data_model_tool:
            kwargs_full["tool_choice"] = {
                "type": "tool",
                "name": data_model_tool.name,
            }

        if "system" not in kwargs_full:
            if len(turns) > 0 and turns[0].role == "system":
                kwargs_full["system"] = turns[0].text

        return kwargs_full

    def stream_text(self, chunk) -> Optional[str]:
        if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
            return chunk.delta.text
        return None

    def stream_merge_chunks(self, completion, chunk):
        if chunk.type == "message_start":
            return chunk.message
        completion = cast("Message", completion)

        if chunk.type == "content_block_start":
            completion.content.append(chunk.content_block)
        elif chunk.type == "content_block_delta":
            this_content = completion.content[chunk.index]
            if chunk.delta.type == "text_delta":
                this_content = cast("TextBlock", this_content)
                this_content.text += chunk.delta.text
            elif chunk.delta.type == "input_json_delta":
                this_content = cast("ToolUseBlock", this_content)
                if not isinstance(this_content.input, str):
                    this_content.input = ""
                this_content.input += chunk.delta.partial_json
        elif chunk.type == "content_block_stop":
            this_content = completion.content[chunk.index]
            if this_content.type == "tool_use" and isinstance(this_content.input, str):
                try:
                    this_content.input = json.loads(this_content.input or "{}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON input: {e}")
        elif chunk.type == "message_delta":
            completion.stop_reason = chunk.delta.stop_reason
            completion.stop_sequence = chunk.delta.stop_sequence
            completion.usage.output_tokens = chunk.usage.output_tokens

        return completion

    def stream_turn(self, completion, has_data_model, stream) -> Turn:
        return self._as_turn(completion, has_data_model)

    async def stream_turn_async(self, completion, has_data_model, stream) -> Turn:
        return self._as_turn(completion, has_data_model)

    def value_turn(self, completion, has_data_model) -> Turn:
        return self._as_turn(completion, has_data_model)

    def _as_message_params(self, turns: list[Turn]) -> list["MessageParam"]:
        messages: list["MessageParam"] = []
        for turn in turns:
            if turn.role == "system":
                continue  # system prompt passed as separate arg
            if turn.role not in ["user", "assistant"]:
                raise ValueError(f"Unknown role {turn.role}")

            content = [self._as_content_block(c) for c in turn.contents]
            role = "user" if turn.role == "user" else "assistant"
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _as_content_block(content: Content) -> "ContentBlockParam":
        if isinstance(content, ContentText):
            return {"text": content.text, "type": "text"}
        elif isinstance(content, ContentJson):
            return {"text": "<structured data/>", "type": "text"}
        elif isinstance(content, ContentImageInline):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content.content_type,
                    "data": content.data or "",
                },
            }
        elif isinstance(content, ContentImageRemote):
            raise NotImplementedError(
                "Remote images aren't supported by Anthropic (Claude). "
                "Consider downloading the image and using content_image_file() instead."
            )
        elif isinstance(content, ContentToolRequest):
            return {
                "type": "tool_use",
                "id": content.id,
                "name": content.name,
                "input": content.arguments,
            }
        elif isinstance(content, ContentToolResult):
            return {
                "type": "tool_result",
                "tool_use_id": content.id,
                "content": content.get_final_value(),
                "is_error": content.error is not None,
            }
        raise ValueError(f"Unknown content type: {type(content)}")

    @staticmethod
    def _tool_schema_json(schema: "ChatCompletionToolParam") -> "ToolParam":
        fn = schema["function"]
        name = fn["name"]

        res: "ToolParam" = {
            "name": name,
            "input_schema": {
                "type": "object",
            },
        }

        if "description" in fn:
            res["description"] = fn["description"]

        if "parameters" in fn:
            res["input_schema"]["properties"] = fn["parameters"]["properties"]

        return res

    def _as_turn(self, completion: Message, has_data_model=False) -> Turn:
        contents = []
        for content in completion.content:
            if content.type == "text":
                contents.append(ContentText(content.text))
            elif content.type == "tool_use":
                if has_data_model and content.name == "_structured_tool_call":
                    if not isinstance(content.input, dict):
                        raise ValueError(
                            "Expected data extraction tool to return a dictionary."
                        )
                    if "data" not in content.input:
                        raise ValueError(
                            "Expected data extraction tool to return a 'data' field."
                        )
                    contents.append(ContentJson(content.input["data"]))
                else:
                    contents.append(
                        ContentToolRequest(
                            content.id,
                            name=content.name,
                            arguments=content.input,
                        )
                    )

        tokens = completion.usage.input_tokens, completion.usage.output_tokens

        tokens_log(self, tokens)

        return Turn(
            "assistant",
            contents,
            tokens=tokens,
            finish_reason=completion.stop_reason,
            completion=completion,
        )


def ChatBedrockAnthropic(
    *,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    aws_secret_key: Optional[str] = None,
    aws_access_key: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_profile: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    kwargs: Optional["ChatBedrockClientArgs"] = None,
) -> Chat["SubmitInputArgs", Message]:
    """
    Chat with an AWS bedrock model.

    [AWS Bedrock](https://aws.amazon.com/bedrock/) provides a number of chat
    based models, including those Anthropic's
    [Claude](https://aws.amazon.com/bedrock/claude/).

    Prerequisites
    -------------

    ::: {.callout-note}
    ## AWS credentials

    Consider using the approach outlined in this guide to manage your AWS credentials:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html>
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatBedrockAnthropic`, requires the `anthropic` package with the `bedrock` extras
    (e.g., `pip install anthropic[bedrock]`).
    :::

    Examples
    --------

    ```python
    from chatlas import ChatBedrockAnthropic

    chat = ChatBedrockAnthropic(
        aws_profile="...",
        aws_region="us-east",
        aws_secret_key="...",
        aws_access_key="...",
        aws_session_token="...",
    )
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    model
        The model to use for the chat.
    max_tokens
        Maximum number of tokens to generate before stopping.
    aws_secret_key
        The AWS secret key to use for authentication.
    aws_access_key
        The AWS access key to use for authentication.
    aws_region
        The AWS region to use. Defaults to the AWS_REGION environment variable.
        If that is not set, defaults to `'us-east-1'`.
    aws_profile
        The AWS profile to use.
    aws_session_token
        The AWS session token to use.
    base_url
        The base URL to use. Defaults to the ANTHROPIC_BEDROCK_BASE_URL
        environment variable. If that is not set, defaults to
        `f"https://bedrock-runtime.{aws_region}.amazonaws.com"`.
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch. Do
        not provide non-None values for both `turns` and `system_prompt`. Each
        message in the list should be a dictionary with at least `role` (usually
        `system`, `user`, or `assistant`, but `tool` is also possible). Normally
        there is also a `content` field, which is a string.
    kwargs
        Additional arguments to pass to the `anthropic.AnthropicBedrock()`
        client constructor.

    Returns
    -------
    Chat
        A Chat object.
    """

    if model is None:
        # Default model from https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#aws-bedrock
        model = log_model_default("anthropic.claude-3-5-sonnet-20241022-v2:0")

    return Chat(
        provider=AnthropicBedrockProvider(
            model=model,
            max_tokens=max_tokens,
            aws_secret_key=aws_secret_key,
            aws_access_key=aws_access_key,
            aws_region=aws_region,
            aws_profile=aws_profile,
            aws_session_token=aws_session_token,
            base_url=base_url,
            kwargs=kwargs,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt,
        ),
    )


class AnthropicBedrockProvider(AnthropicProvider):
    def __init__(
        self,
        *,
        model: str,
        aws_secret_key: str | None,
        aws_access_key: str | None,
        aws_region: str | None,
        aws_profile: str | None,
        aws_session_token: str | None,
        max_tokens: int,
        base_url: str | None,
        kwargs: Optional["ChatBedrockClientArgs"] = None,
    ):
        try:
            from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
        except ImportError:
            raise ImportError(
                "`ChatBedrockAnthropic()` requires the `anthropic` package. "
                "Install it with `pip install anthropic[bedrock]`."
            )

        self._model = model
        self._max_tokens = max_tokens

        kwargs_full: "ChatBedrockClientArgs" = {
            "aws_secret_key": aws_secret_key,
            "aws_access_key": aws_access_key,
            "aws_region": aws_region,
            "aws_profile": aws_profile,
            "aws_session_token": aws_session_token,
            "base_url": base_url,
            **(kwargs or {}),
        }

        self._client = AnthropicBedrock(**kwargs_full)  # type: ignore
        self._async_client = AsyncAnthropicBedrock(**kwargs_full)  # type: ignore
