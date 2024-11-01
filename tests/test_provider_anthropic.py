import pytest
from chatlas import ChatAnthropic

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
    retryassert,
)


def test_anthropic_simple_request():
    chat = ChatAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5)


@pytest.mark.asyncio
async def test_anthropic_simple_streaming_request():
    chat = ChatAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in chat.submit_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)


def test_anthropic_respects_turns_interface():
    chat_fun = ChatAnthropic
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_anthropic_tool_variations():
    chat_fun = ChatAnthropic
    assert_tools_simple(chat_fun)
    assert_tools_parallel(chat_fun)

    ## Fails occassionally returning "" instead of Susan
    #def run_sequentialassert():
    #    assert_tools_sequential(chat_fun, total_calls=6)
#
    #retryassert(run_sequentialassert, retries=5)


@pytest.mark.asyncio
async def test_anthropic_tool_variations_async():
    await assert_tools_async(ChatAnthropic)


def test_data_extraction():
    assert_data_extraction(ChatAnthropic)


def test_anthropic_images():
    chat_fun = ChatAnthropic
    assert_images_inline(chat_fun)
    assert_images_remote_error(chat_fun)
