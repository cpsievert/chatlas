import os
from typing import TYPE_CHECKING, Optional

import requests

from ._chat import Chat
from ._openai import OpenAIProvider
from ._turn import Turn, normalize_turns

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam

    from .types.openai import ChatClientArgs


def ChatVLLM(
    *,
    base_url: str,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    seed: Optional[int] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat:
    """
    Chat with a model hosted by vLLM

    [vLLM](https://docs.vllm.ai/en/latest/) is an open source library that
    provides an efficient and convenient LLMs model server. You can use
    `ChatVLLM()` to connect to endpoints powered by vLLM.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## vLLM runtime

    `ChatVLLM` requires a vLLM server to be running somewhere (either on your
    machine or a remote server). If you want to run a vLLM server locally, see
    the [vLLM documentation](https://docs.vllm.ai/en/v0.5.3/getting_started/quickstart.html).
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatVLLM` requires the `openai` package (e.g., `pip install openai`).
    :::


    Parameters
    ----------
    base_url
        A system prompt to set the behavior of the assistant.
    system_prompt
        Optional system prompt to prepend to conversation.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch. Do
        not provide non-`None` values for both `turns` and `system_prompt`. Each
        message in the list should be a dictionary with at least `role` (usually
        `system`, `user`, or `assistant`, but `tool` is also possible). Normally
        there is also a `content` field, which is a string.
    model
        Model identifier to use.
    seed
        Random seed for reproducibility.
    api_key
        API key for authentication. If not provided, the `VLLM_API_KEY` environment
        variable will be used.
    kwargs
        Additional arguments to pass to the LLM client.

    Returns:
        Chat instance configured for vLLM
    """

    if api_key is None:
        api_key = get_vllm_key()

    if model is None:
        models = get_vllm_models(base_url, api_key)
        available_models = ", ".join(models)
        raise ValueError(f"Must specify model. Available models: {available_models}")

    return Chat(
        provider=VLLMProvider(
            base_url=base_url,
            model=model,
            seed=seed,
            api_key=api_key,
            kwargs=kwargs,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt,
        ),
    )


class VLLMProvider(OpenAIProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        seed: Optional[int],
        api_key: Optional[str],
        kwargs: Optional["ChatClientArgs"],
    ):
        self.base_url = base_url
        self.model = model
        self.seed = seed
        self.api_key = api_key
        self.kwargs = kwargs

    # Just like OpenAI but no strict
    @staticmethod
    def _tool_schema_json(
        schema: "ChatCompletionToolParam",
    ) -> "ChatCompletionToolParam":
        schema["function"]["strict"] = False
        return schema


def get_vllm_key() -> str:
    key = os.getenv("VLLM_API_KEY", os.getenv("VLLM_KEY"))
    if not key:
        raise ValueError("VLLM_API_KEY environment variable not set")
    return key


def get_vllm_models(base_url: str, api_key: Optional[str] = None) -> list[str]:
    if api_key is None:
        api_key = get_vllm_key()

    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{base_url}/v1/models", headers=headers)
    response.raise_for_status()
    data = response.json()

    return [model["id"] for model in data["data"]]


# def chat_vllm_test(**kwargs) -> Chat:
#     """Create a test chat instance with default parameters."""
#     return ChatVLLM(base_url="https://llm.nrp-nautilus.io/", model="llama3", **kwargs)
