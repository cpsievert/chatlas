import re

import pytest
from chatlas import ChatOpenAI, Turn
from pydantic import BaseModel


def test_simple_batch_chat():
    chat = ChatOpenAI()
    chat.chat("What's 1 + 1. Just give me the answer, no punctuation")
    turn = chat.last_turn()
    assert turn is not None
    assert turn.text == "2"


@pytest.mark.asyncio
async def test_simple_async_batch_chat():
    chat = ChatOpenAI()
    await chat.chat_async(
        "What's 1 + 1. Just give me the answer, no punctuation",
    )
    turn = chat.last_turn()
    assert turn is not None
    assert turn.text == "2"


def test_simple_streaming_chat():
    chat = ChatOpenAI()
    res = chat.submit("""
        What are the canonical colors of the ROYGBIV rainbow?
        Put each colour on its own line. Don't use punctuation.
    """)
    chunks = [chunk for chunk in res]
    assert len(chunks) > 2
    result = "".join(chunks)
    rainbow_re = "^red *\norange *\nyellow *\ngreen *\nblue *\nindigo *\nviolet *\n?$"
    assert re.match(rainbow_re, result.lower())
    turn = chat.last_turn()
    assert turn is not None
    assert re.match(rainbow_re, turn.text.lower())


@pytest.mark.asyncio
async def test_simple_streaming_chat_async():
    chat = ChatOpenAI()
    res = chat.submit_async("""
        What are the canonical colors of the ROYGBIV rainbow?
        Put each colour on its own line. Don't use punctuation.
    """)
    chunks = [chunk async for chunk in res]
    assert len(chunks) > 2
    result = "".join(chunks)
    rainbow_re = "^red *\norange *\nyellow *\ngreen *\nblue *\nindigo *\nviolet *\n?$"
    assert re.match(rainbow_re, result.lower())
    turn = chat.last_turn()
    assert turn is not None
    assert re.match(rainbow_re, turn.text.lower())


def test_basic_print_method():
    chat = ChatOpenAI(
        system_prompt="You're a helpful assistant that returns very minimal output",
        turns=[
            Turn("user", "What's 1 + 1? What's 1 + 2?"),
            Turn("assistant", "2  3", tokens=(15, 5)),
        ],
    )
    out = repr(chat)
    assert "You're a helpful assistant" in out
    assert "What's 1 + 1? What's 1 + 2?" in out
    assert "2  3" in out


def test_extract_data():
    chat = ChatOpenAI()

    class Person(BaseModel):
        name: str
        age: int

    data = chat.extract_data("John, age 15, won first prize", data_model=Person)
    assert data == dict(name="John", age=15)


@pytest.mark.asyncio
async def test_extract_data_async():
    chat = ChatOpenAI()

    class Person(BaseModel):
        name: str
        age: int

    data = await chat.extract_data_async(
        "John, age 15, won first prize", data_model=Person
    )
    assert data == dict(name="John", age=15)


def test_last_turn_retrieval():
    chat = ChatOpenAI()
    assert chat.last_turn("user") is None
    assert chat.last_turn("assistant") is None

    chat.chat("Hi")
    user_turn = chat.last_turn("user")
    assert user_turn is not None and user_turn.role == "user"
    turn = chat.last_turn("assistant")
    assert turn is not None and turn.role == "assistant"


def test_system_prompt_retrieval():
    chat1 = ChatOpenAI()
    assert chat1.system_prompt is None
    assert chat1.last_turn("system") is None

    chat2 = ChatOpenAI(system_prompt="You are from New Zealand")
    assert chat2.system_prompt == "You are from New Zealand"
    turn = chat2.last_turn("system")
    assert turn is not None and turn.text == "You are from New Zealand"


def test_modify_system_prompt():
    chat = ChatOpenAI(
        turns=[
            Turn("user", "Hi"),
            Turn("assistant", "Hello"),
        ]
    )

    # NULL -> NULL
    chat.system_prompt = None
    assert chat.system_prompt is None

    # NULL -> string
    chat.system_prompt = "x"
    assert chat.system_prompt == "x"

    # string -> string
    chat.system_prompt = "y"
    assert chat.system_prompt == "y"

    # string -> NULL
    chat.system_prompt = None
    assert chat.system_prompt is None
