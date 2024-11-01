import pytest
from chatlas import ChatOpenAI

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
)


def test_openai_simple_request():
    chat = ChatOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.last_turn()
    assert turn is not None
    assert turn.tokens == (27, 1)


@pytest.mark.asyncio
async def test_openai_simple_streaming_request():
    chat = ChatOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in chat.submit_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)


def test_openai_respects_turns_interface():
    chat_fun = ChatOpenAI
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_openai_tool_variations():
    chat_fun = ChatOpenAI
    assert_tools_simple(chat_fun)
    assert_tools_parallel(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@pytest.mark.asyncio
async def test_openai_tool_variations_async():
    await assert_tools_async(ChatOpenAI)


def test_data_extraction():
    assert_data_extraction(ChatOpenAI)


def test_openai_images():
    chat_fun = ChatOpenAI
    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


@pytest.mark.asyncio
async def test_openai_logprobs():
    chat = ChatOpenAI()

    pieces = []
    async for x in chat.submit_async("Hi", kwargs={"logprobs": True}):
        pieces.append(x)

    turn = chat.last_turn()
    assert turn is not None
    logprobs = turn.json_data["choices"][0]["logprobs"]["content"]
    assert len(logprobs) == len(pieces)
