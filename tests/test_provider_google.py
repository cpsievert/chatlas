import os
import time

import pytest
from chatlas import ChatGoogle

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
)

do_test = os.getenv("TEST_GOOGLE", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping Google tests", allow_module_level=True)


def test_google_simple_request():
    chat = ChatGoogle(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.last_turn()
    assert turn is not None
    assert turn.tokens == (17, 1)


@pytest.mark.asyncio
async def test_google_simple_streaming_request():
    chat = ChatGoogle(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in chat.submit_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)


def test_google_respects_turns_interface():
    chat_fun = ChatGoogle
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_google_tool_variations():
    chat_fun = ChatGoogle
    # Avoid Google's rate limits
    time.sleep(3)
    assert_tools_simple(chat_fun, stream=False)
    time.sleep(3)
    assert_tools_parallel(chat_fun, stream=False)
    time.sleep(3)
    assert_tools_sequential(
        chat_fun,
        total_calls=6,
        stream=False,
    )
    time.sleep(3)


# TODO: this test runs fine in isolation, but fails for some reason when run with the other tests
# Seems google isn't handling async 100% correctly
# @pytest.mark.asyncio
# async def test_google_tool_variations_async():
#     await assert_tools_async(ChatGoogle, stream=False)


def test_data_extraction():
    time.sleep(3)
    assert_data_extraction(ChatGoogle)


def test_google_images():
    chat_fun = ChatGoogle
    assert_images_inline(chat_fun)
    assert_images_remote_error(chat_fun)
