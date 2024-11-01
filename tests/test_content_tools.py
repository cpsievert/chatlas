import pytest
from chatlas import ChatOpenAI
from chatlas._content import ContentToolResult

chat = ChatOpenAI()


@pytest.mark.asyncio
async def test_invoke_tool_returns_tool_result():
    async def tool():
        return 1

    res = await chat._invoke_tool(tool, {}, id="x")
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is None
    assert res.value == 1

    res = await chat._invoke_tool(tool, {"x": 1}, id="x")
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is not None
    assert "got an unexpected keyword argument" in res.error
    assert res.value is None

    res = await chat._invoke_tool(None, {"x": 1}, id="x")
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error == "Unknown tool"
    assert res.value is None
