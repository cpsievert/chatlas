from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, Optional, cast, overload

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
from ._merge import merge_dicts
from ._provider import Provider
from ._tokens import tokens_log
from ._tools import Tool, basemodel_to_param_schema
from ._turn import Turn, normalize_turns
from ._utils import MISSING, MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )
    from openai.types.chat.chat_completion_assistant_message_param import (
        ContentArrayOfContentPart,
    )
    from openai.types.chat.chat_completion_content_part_param import (
        ChatCompletionContentPartParam,
    )
    from openai.types.chat_model import ChatModel

    from .types.openai import ChatAzureClientArgs, ChatClientArgs, SubmitInputArgs
else:
    ChatCompletion = object
    ChatCompletionChunk = object


# The dictionary form of ChatCompletion (TODO: stronger typing)?
ChatCompletionDict = dict[str, Any]


def ChatOpenAI(
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    model: "Optional[ChatModel | str]" = None,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    seed: int | None | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with an OpenAI model.

    [OpenAI](https://openai.com/) provides a number of chat based models under
    the [ChatGPT](https://chatgpt.com) moniker.

    Prerequisites
    --------------

    ::: {.callout-note}
    ## API key

    Note that a ChatGPT Plus membership does not give you the ability to call
    models via the API. You will need to go to the [developer
    platform](https://platform.openai.com) to sign up (and pay for) a developer
    account that will give you an API key that you can use with this package.
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatOpenAI` requires the `openai` package (e.g., `pip install openai`).
    :::

    Examples
    --------
    ```python
    import os
    from chatlas import ChatOpenAI

    chat = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch. Do
        not provide non-`None` values for both `turns` and `system_prompt`. Each
        message in the list should be a dictionary with at least `role` (usually
        `system`, `user`, or `assistant`, but `tool` is also possible). Normally
        there is also a `content` field, which is a string.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly
        choosing a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `OPENAI_API_KEY` environment
        variable.
    base_url
        The base URL to the endpoint; the default uses OpenAI.
    seed
        Optional integer seed that ChatGPT uses to try and make output more
        reproducible.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client
        constructor.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatOpenAI(api_key="...")`)
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
    OPENAI_API_KEY=...
    ```

    ```python
    from chatlas import ChatOpenAI
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatOpenAI()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export OPENAI_API_KEY=...
    ```
    """
    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    if model is None:
        model = log_model_default("gpt-4o")

    return Chat(
        provider=OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            seed=seed,
            kwargs=kwargs,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt,
        ),
    )


class OpenAIProvider(Provider[ChatCompletion, ChatCompletionChunk, ChatCompletionDict]):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        seed: Optional[int] = None,
        kwargs: Optional["ChatClientArgs"] = None,
    ):
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "`ChatOpenAI()` requires the `openai` package. "
                "Install it with `pip install openai`."
            )

        self._model = model
        self._seed = seed

        kwargs_full: "ChatClientArgs" = {
            "api_key": api_key,
            "base_url": base_url,
            **(kwargs or {}),
        }

        # TODO: worth bringing in AsyncOpenAI types?
        self._client = OpenAI(**kwargs_full)  # type: ignore
        self._async_client = AsyncOpenAI(**kwargs_full)

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
        return self._client.chat.completions.create(**kwargs)  # type: ignore

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
        return await self._async_client.chat.completions.create(**kwargs)  # type: ignore

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ) -> "SubmitInputArgs":
        tool_schemas = [self._tool_schema_json(tool.schema) for tool in tools.values()]

        kwargs_full: "SubmitInputArgs" = {
            "stream": stream,
            "messages": self._as_message_param(turns),
            "model": self._model,
            **(kwargs or {}),
        }

        if self._seed is not None:
            kwargs_full["seed"] = self._seed

        if tool_schemas:
            kwargs_full["tools"] = tool_schemas

        if data_model is not None:
            params = basemodel_to_param_schema(data_model)
            params = cast(dict, params)
            params["additionalProperties"] = False
            kwargs_full["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_data",
                    "description": params.get("description", ""),
                    "schema": params,
                    "strict": True,
                },
            }
            # Apparently OpenAI gets confused if you include
            # both response_format and tools
            if "tools" in kwargs_full:
                del kwargs_full["tools"]

        if stream and "stream_options" not in kwargs_full:
            kwargs_full["stream_options"] = {"include_usage": True}

        return kwargs_full

    def stream_text(self, chunk):
        if not chunk.choices:
            return None
        return chunk.choices[0].delta.content

    def stream_merge_chunks(self, completion, chunk):
        chunkd = chunk.model_dump()
        if completion is None:
            return chunkd
        return merge_dicts(completion, chunkd)

    def stream_turn(self, completion, has_data_model, stream) -> Turn:
        from openai.types.chat import ChatCompletion

        delta = completion["choices"][0].pop("delta")  # type: ignore
        completion["choices"][0]["message"] = delta  # type: ignore
        completion = ChatCompletion.construct(**completion)
        return self._as_turn(completion, has_data_model)

    async def stream_turn_async(self, completion, has_data_model, stream):
        return self.stream_turn(completion, has_data_model, stream)

    def value_turn(self, completion, has_data_model) -> Turn:
        return self._as_turn(completion, has_data_model)

    @staticmethod
    def _as_message_param(turns: list[Turn]) -> list["ChatCompletionMessageParam"]:
        from openai.types.chat import (
            ChatCompletionAssistantMessageParam,
            ChatCompletionMessageToolCallParam,
            ChatCompletionSystemMessageParam,
            ChatCompletionToolMessageParam,
            ChatCompletionUserMessageParam,
        )

        res: list["ChatCompletionMessageParam"] = []
        for turn in turns:
            if turn.role == "system":
                res.append(
                    ChatCompletionSystemMessageParam(content=turn.text, role="system")
                )
            elif turn.role == "assistant":
                content_parts: list["ContentArrayOfContentPart"] = []
                tool_calls: list["ChatCompletionMessageToolCallParam"] = []
                for x in turn.contents:
                    if isinstance(x, ContentText):
                        content_parts.append({"type": "text", "text": x.text})
                    elif isinstance(x, ContentJson):
                        content_parts.append({"type": "text", "text": ""})
                    elif isinstance(x, ContentToolRequest):
                        tool_calls.append(
                            {
                                "id": x.id,
                                "function": {
                                    "name": x.name,
                                    "arguments": json.dumps(x.arguments),
                                },
                                "type": "function",
                            }
                        )
                    else:
                        raise ValueError(
                            f"Don't know how to handle content type {type(x)} for role='assistant'."
                        )

                # Some OpenAI-compatible models (e.g., Groq) don't work nicely with empty content
                args = {
                    "role": "assistant",
                    "content": content_parts,
                    "tool_calls": tool_calls,
                }
                if not content_parts:
                    del args["content"]
                if not tool_calls:
                    del args["tool_calls"]

                res.append(ChatCompletionAssistantMessageParam(**args))

            elif turn.role == "user":
                contents: list["ChatCompletionContentPartParam"] = []
                tool_results: list["ChatCompletionToolMessageParam"] = []
                for x in turn.contents:
                    if isinstance(x, ContentText):
                        contents.append({"type": "text", "text": x.text})
                    elif isinstance(x, ContentJson):
                        contents.append({"type": "text", "text": ""})
                    elif isinstance(x, ContentImageRemote):
                        contents.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": x.url,
                                    "detail": x.detail,
                                },
                            }
                        )
                    elif isinstance(x, ContentImageInline):
                        contents.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{x.content_type};base64,{x.data}"
                                },
                            }
                        )
                    elif isinstance(x, ContentToolResult):
                        tool_results.append(
                            ChatCompletionToolMessageParam(
                                # TODO: a tool could return an image!?!
                                content=x.get_final_value(),
                                tool_call_id=x.id,
                                role="tool",
                            )
                        )
                    else:
                        raise ValueError(
                            f"Don't know how to handle content type {type(x)} for role='user'."
                        )

                if contents:
                    res.append(
                        ChatCompletionUserMessageParam(content=contents, role="user")
                    )
                res.extend(tool_results)

            else:
                raise ValueError(f"Unknown role: {turn.role}")

        return res

    @staticmethod
    def _tool_schema_json(
        schema: "ChatCompletionToolParam",
    ) -> "ChatCompletionToolParam":
        return schema

    def _as_turn(
        self, completion: "ChatCompletion", has_data_model: bool
    ) -> Turn[ChatCompletion]:
        message = completion.choices[0].message

        contents: list[Content] = []
        if message.content is not None:
            if has_data_model:
                data = json.loads(message.content)
                contents = [ContentJson(data)]
            else:
                contents = [ContentText(message.content)]

        tool_calls = message.tool_calls

        if tool_calls is not None:
            for call in tool_calls:
                func = call.function
                if func is None:
                    continue

                args = {}
                try:
                    args = json.loads(func.arguments) if func.arguments else {}
                except json.JSONDecodeError:
                    raise ValueError(
                        f"The model's completion included a tool request ({func.name}) "
                        "with invalid JSON for input arguments: '{func.arguments}'"
                        "This can happen if the model hallucinates parameters not defined by "
                        "your function schema. Try revising your tool description and system "
                        "prompt to be more specific about the expected input arguments to this function."
                    )

                contents.append(
                    ContentToolRequest(
                        call.id,
                        name=func.name,
                        arguments=args,
                    )
                )

        usage = completion.usage
        if usage is None:
            tokens = (0, 0)
        else:
            tokens = usage.prompt_tokens, usage.completion_tokens

        # For some reason ChatGroq() includes tokens under completion.x_groq
        if usage is None and hasattr(completion, "x_groq"):
            usage = completion.x_groq["usage"]  # type: ignore
            tokens = usage["prompt_tokens"], usage["completion_tokens"]

        tokens_log(self, tokens)

        return Turn(
            "assistant",
            contents,
            tokens=tokens,
            finish_reason=completion.choices[0].finish_reason,
            completion=completion,
        )


def ChatAzureOpenAI(
    *,
    endpoint: str,
    deployment_id: str,
    api_version: str,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    seed: int | None | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatAzureClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on Azure OpenAI.

    The [Azure OpenAI server](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
    hosts a number of open source models as well as proprietary models
    from OpenAI.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## Python requirements

    `ChatAzureOpenAI` requires the `openai` package (e.g., `pip install openai`).
    :::

    Examples
    --------
    ```python
    import os
    from chatlas import ChatAzureOpenAI

    chat = ChatAzureOpenAI(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_id="REPLACE_WITH_YOUR_DEPLOYMENT_ID",
        api_version="YYYY-MM-DD",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    endpoint
        Azure OpenAI endpoint url with protocol and hostname, i.e.
        `https://{your-resource-name}.openai.azure.com`. Defaults to using the
        value of the `AZURE_OPENAI_ENDPOINT` envinronment variable.
    deployment_id
        Deployment id for the model you want to use.
    api_version
        The API version to use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `AZURE_OPENAI_API_KEY` environment
        variable.
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch.
        Do not provide non-None values for both `turns` and `system_prompt`.
        Each message in the list should be a dictionary with at least `role`
        (usually `system`, `user`, or `assistant`, but `tool` is also possible).
        Normally there is also a `content` field, which is a string.
    seed
        Optional integer seed that ChatGPT uses to try and make output more
        reproducible.
    kwargs
        Additional arguments to pass to the `openai.AzureOpenAI()` client constructor.

    Returns
    -------
    Chat
        A Chat object.
    """

    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    return Chat(
        provider=OpenAIAzureProvider(
            endpoint=endpoint,
            deployment_id=deployment_id,
            api_version=api_version,
            api_key=api_key,
            seed=seed,
            kwargs=kwargs,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt,
        ),
    )


class OpenAIAzureProvider(OpenAIProvider):
    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        deployment_id: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        seed: int | None = None,
        kwargs: Optional["ChatAzureClientArgs"] = None,
    ):
        try:
            from openai import AsyncAzureOpenAI, AzureOpenAI
        except ImportError:
            raise ImportError(
                "`ChatAzureOpenAI()` requires the `openai` package. "
                "Install it with `pip install openai`."
            )

        self._model = deployment_id
        self._seed = seed

        kwargs_full: "ChatAzureClientArgs" = {
            "azure_endpoint": endpoint,
            "azure_deployment": deployment_id,
            "api_version": api_version,
            "api_key": api_key,
            **(kwargs or {}),
        }

        self._client = AzureOpenAI(**kwargs_full)  # type: ignore
        self._async_client = AsyncAzureOpenAI(**kwargs_full)  # type: ignore


class InvalidJSONParameterWarning(RuntimeWarning):
    """
    Warning for when a tool request includes invalid JSON for input arguments.

    This is a subclass of `RuntimeWarning` and is used to indicate that a tool
    request included invalid JSON for input arguments. This can happen if the
    model hallucinates parameters not defined by your function schema.
    """

    pass
