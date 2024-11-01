import os

import pytest
from chatlas import ChatAzureOpenAI

do_test = os.getenv("TEST_AZURE", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping Azure tests", allow_module_level=True)


def test_azure_simple_request():
    chat = ChatAzureOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
        endpoint="https://chatlas-testing.openai.azure.com",
        deployment_id="gpt-4o-mini",
        api_version="2024-08-01-preview",
    )

    chat.chat("What is 1 + 1?")
    turn = chat.last_turn()
    assert turn is not None
    assert "2" in turn.text
    assert turn.tokens == (27, 1)


@pytest.mark.asyncio
async def test_azure_simple_request_async():
    chat = ChatAzureOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
        endpoint="https://chatlas-testing.openai.azure.com",
        deployment_id="gpt-4o-mini",
        api_version="2024-08-01-preview",
    )

    await chat.chat_async("What is 1 + 1?")
    turn = chat.last_turn()
    assert turn is not None
    assert "2" in turn.text
    assert turn.tokens == (27, 1)
