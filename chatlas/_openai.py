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
from ._merge import merge_dicts
from ._provider import Provider
from ._tokens import tokens_log
from ._tools import ToolDef, ToolSchema, basemodel_to_tool_params
from ._turn import Turn, normalize_turns
from ._utils import MISSING, MISSING_TYPE, inform_model_default, is_testing

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

    from .types._openai_client import ProviderClientArgs
    from .types._openai_client_azure import ProviderClientArgs as AzureProviderArgs
    from .types._openai_create import ChatCompletionArgs
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
    kwargs: Optional["ProviderClientArgs"] = None,
) -> Chat["ChatCompletionArgs"]:
    """
    Chat with an OpenAI model.

    OpenAI (https://openai.com/) provides a number of chat based models under
    the ChatGPT (https://chatgpt.com) moniker.

    Note that a ChatGPT Plus membership does not give you the ability to call
    models via the API. You will need to go to the developer platform
    (https://platform.openai.com) to sign up (and pay for) a developer account
    that will give you an API key that you can use with this package.

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch.
        Do not provide non-`None` values for both `turns` and `system_prompt`.
        Each message in the list should be a dictionary with at least `role`
        (usually `system`, `user`, or `assistant`, but `tool` is also possible).
        Normally there is also a `content` field, which is a string.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly choosing
        a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `OPENAI_API_KEY` environment variable.
    base_url
        The base URL to the endpoint; the default uses OpenAI.
    seed
        Optional integer seed that ChatGPT uses to try and make output more
        reproducible.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client constructor.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Examples
    --------
    >>> from chatlas import ChatOpenAI
    >>> chat = ChatOpenAI()
    >>> chat.chat("What is the capital of France?")

    """
    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    if model is None:
        model = inform_model_default("gpt-4o-mini")

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
        kwargs: Optional["ProviderClientArgs"] = None,
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

        kwargs_full: "ProviderClientArgs" = {
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
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["ChatCompletionArgs"] = None,
    ): ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["ChatCompletionArgs"] = None,
    ): ...

    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["ChatCompletionArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        return self._client.chat.completions.create(**kwargs)  # type: ignore

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["ChatCompletionArgs"] = None,
    ): ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["ChatCompletionArgs"] = None,
    ): ...

    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["ChatCompletionArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        return await self._async_client.chat.completions.create(**kwargs)  # type: ignore

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["ChatCompletionArgs"] = None,
    ) -> "ChatCompletionArgs":
        tool_schemas = [
            self._openai_tool_schema(tool.schema) for tool in tools.values()
        ]
        # OpenAI's typing is wrong (an empty list isn't allowed)
        # If they fix it, we can remove this if statement
        if not tool_schemas:
            tool_schemas = cast(list, None)

        kwargs_full: "ChatCompletionArgs" = {
            "stream": stream,
            "messages": self._as_message_param(turns),
            "tools": tool_schemas,
            "model": self._model,
            "seed": self._seed,
            **(kwargs or {}),
        }

        if data_model is not None:
            params = basemodel_to_tool_params(data_model)
            params = cast(dict, params)
            params["additionalProperties"] = False
            kwargs_full["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_data",
                    "schema": params,
                    "strict": True,
                },
            }

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

    def stream_turn(self, completion, has_data_model) -> Turn:
        from openai.types.chat import ChatCompletion

        delta = completion["choices"][0].pop("delta")  # type: ignore
        completion["choices"][0]["message"] = delta  # type: ignore
        completion = ChatCompletion.construct(**completion)
        return self._as_turn(completion, has_data_model)

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
                            {"type": "image_url", "image_url": {"url": x.url}}
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
    def _openai_tool_schema(schema: ToolSchema) -> "ChatCompletionToolParam":
        fn = schema["function"]
        name = fn["name"]
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": fn["description"],
                "parameters": {
                    "type": "object",
                    "properties": fn["parameters"]["properties"],
                    "required": fn["parameters"]["required"],
                },
                "strict": True,
            },
        }

    def _as_turn(self, completion: "ChatCompletion", has_data_model: bool) -> Turn:
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
                # TODO: record parsing error
                args = json.loads(func.arguments) if func.arguments else {}
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

        tokens_log(self, tokens)

        return Turn(
            "assistant",
            contents,
            json_data=completion.model_dump(),
            tokens=tokens,
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
    kwargs: Optional["AzureProviderArgs"] = None,
) -> Chat["ChatCompletionArgs"]:
    """
    Chat with a model hosted on Azure OpenAI.

    The Azure OpenAI server (https://azure.microsoft.com/en-us/products/ai-services/openai-service)
    hosts a number of open source models as well as proprietary models
    from OpenAI.

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
        kwargs: Optional["AzureProviderArgs"] = None,
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

        kwargs_full: "AzureProviderArgs" = {
            "azure_endpoint": endpoint,
            "azure_deployment": deployment_id,
            "api_version": api_version,
            "api_key": api_key,
            **(kwargs or {}),
        }

        self._client = AzureOpenAI(**kwargs_full)  # type: ignore
        self._async_client = AsyncAzureOpenAI(**kwargs_full)  # type: ignore
