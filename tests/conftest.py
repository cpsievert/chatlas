import tempfile
from pathlib import Path
from typing import Callable

import pytest
from chatlas import Chat, ToolDef, Turn, content_image_file, content_image_url
from PIL import Image
from pydantic import BaseModel

ChatFun = Callable[..., Chat]


class ArticleSummary(BaseModel):
    """Summary of the article"""

    title: str
    author: str


article = """
# Apples are tasty

By Hadley Wickham
Apples are delicious and tasty and I like to eat them.
Except for red delicious, that is. They are NOT delicious.
"""


def retryassert(assert_func: Callable[..., None], retries=1):
    for _ in range(retries):
        try:
            return assert_func()
        except Exception:
            pass
    return assert_func()


def assert_turns_system(chat_fun: ChatFun):
    system_prompt = "Return very minimal output, AND ONLY USE UPPERCASE."

    chat = chat_fun(system_prompt=system_prompt)
    chat.chat("What is the name of Winnie the Pooh's human friend?")
    assert len(chat.turns()) == 2
    turn = chat.last_turn()
    assert turn is not None
    assert "CHRISTOPHER ROBIN" in turn.text

    chat = chat_fun(turns=[Turn("system", system_prompt)])
    chat.chat("What is the name of Winnie the Pooh's human friend?")
    assert len(chat.turns()) == 2
    turn = chat.last_turn()
    assert turn is not None
    assert "CHRISTOPHER ROBIN" in turn.text


def assert_turns_existing(chat_fun: ChatFun):
    chat = chat_fun(
        turns=[
            Turn("system", "Return very minimal output; no punctuation."),
            Turn("user", "List the names of any 8 of Santa's 9 reindeer."),
            Turn(
                "assistant",
                "Dasher, Dancer, Vixen, Comet, Cupid, Donner, Blitzen, and Rudolph.",
            ),
        ]
    )
    assert len(chat.turns()) == 2

    chat.chat("Who is the remaining one? Just give the name")
    assert len(chat.turns()) == 4
    turn = chat.last_turn()
    assert turn is not None
    assert "Prancer" in turn.text


def assert_tools_simple(chat_fun: ChatFun, stream: bool = True):
    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")
    chat.register_tool(
        ToolDef(
            lambda: "2024-01-01", name="get_date", description="Gets the current date"
        )
    )

    chat.chat("What's the current date in YMD format?", stream=stream)
    turn = chat.last_turn()
    assert turn is not None
    assert "2024-01-01" in turn.text

    chat.chat("What month is it? Provide the full name.", stream=stream)
    turn = chat.last_turn()
    assert turn is not None
    assert "January" in turn.text


async def assert_tools_async(chat_fun: ChatFun, stream: bool = True):
    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")

    async def async_mock():
        import asyncio

        await asyncio.sleep(0.1)
        return "2024-01-01"

    chat.register_tool(
        ToolDef(
            async_mock, name="get_current_date", description="Gets the current date"
        )
    )

    await chat.chat_async("What's the current date in YMD format?", stream=stream)
    turn = chat.last_turn()
    assert turn is not None
    assert "2024-01-01" in turn.text

    with pytest.raises(Exception, match="async tools in a synchronous chat"):
        chat.chat("Great. Do it again.", stream=stream)


def assert_tools_parallel(chat_fun: ChatFun, stream: bool = True):
    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")

    def favorite_color(person: str):
        return "sage green" if person == "Joe" else "red"

    chat.register_tool(
        ToolDef(
            favorite_color,
            description="Returns a person's favourite colour",
            # TODO: allow for extra arguments?
            # strict=True,
        )
    )

    chat.chat(
        """
        What are Joe and Hadley's favourite colours?
        Answer like name1: colour1, name2: colour2
    """,
        stream=stream,
    )

    assert len(chat.turns()) == 4
    turn = chat.last_turn()
    assert turn is not None
    assert "Joe: sage green" in turn.text
    assert "Hadley: red" in turn.text


def assert_tools_sequential(chat_fun: ChatFun, total_calls: int, stream: bool = True):
    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")
    chat.register_tool(
        ToolDef(
            lambda: 2024,
            name="current_year",
            description="Get the current year",
        )
    )

    def popular_name(year: int):
        """Gets the most popular name for a given year"""
        return "Susan" if year == 2024 else "I don't know"

    chat.register_tool(popular_name)

    chat.chat(
        """
        What was the most popular name this year?
        Note that you have a tool available to you to find the current year.
        """,
        stream=stream,
    )
    assert len(chat.turns()) == total_calls
    turn = chat.last_turn()
    assert turn is not None
    assert "Susan" in turn.text


def assert_data_extraction(chat_fun: ChatFun):
    chat = chat_fun()
    data = chat.extract_data(article, data_model=ArticleSummary)
    assert isinstance(data, dict)
    assert data == {"title": "Apples are tasty", "author": "Hadley Wickham"}
    # TODO: we have the same problem as elmer here.
    # See what hadley does to fix it.
    # data2 = chat.extract_data(article, data_model=ArticleSummary)
    # assert data2 == data


def assert_images_inline(chat_fun: ChatFun, stream: bool = True):
    img = Image.new("RGB", (60, 30), color="red")
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test_image.png"
        img.save(img_path)
        chat = chat_fun()
        chat.chat(
            "What's in this image?",
            content_image_file(str(img_path)),
            stream=stream,
        )

    turn = chat.last_turn()
    assert turn is not None
    assert "red" in turn.text.lower()


def assert_images_remote(chat_fun: ChatFun, stream: bool = True):
    chat = chat_fun()
    chat.chat(
        "What's in this image? (Be sure to mention the outside shape)",
        content_image_url("https://httr2.r-lib.org/logo.png"),
        stream=stream,
    )
    turn = chat.last_turn()
    assert turn is not None
    assert "hex" in turn.text.lower()
    assert "baseball" in turn.text.lower()


def assert_images_remote_error(chat_fun: ChatFun):
    chat = chat_fun()
    image_remote = content_image_url("https://httr2.r-lib.org/logo.png")

    with pytest.raises(Exception, match="Remote images aren't supported"):
        chat.chat("What's in this image?", image_remote)

    assert len(chat.turns()) == 0
