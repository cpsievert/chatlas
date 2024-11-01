from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
)

from pydantic import BaseModel

from ._content import Content, ContentJson, ContentToolRequest, ContentToolResult
from ._provider import Provider
from ._tools import ToolDef
from ._turn import Turn, user_turn


class AnyTypeDict(TypedDict, total=False):
    pass


ChatRequestArgsT = TypeVar("ChatRequestArgsT", bound=AnyTypeDict)


class Chat(Generic[ChatRequestArgsT]):
    """
    A chat object that can be used to interact with a language model.

    A `Chat` is an sequence of sequence of user and assistant `Turn()`s sent to
    a specific `Provider`. A `Chat` takes care of managing the state associated
    with the chat; i.e. it records the messages that you send to the server, and
    the messages that you receive back. If you register a tool (i.e. an function
    that the assistant can call on your behalf), it also takes care of the tool
    loop.

    You should generally not create this object yourself, but instead call
    `ChatOpenAI()` or friends instead.
    """

    def __init__(
        self,
        provider: Provider,
        turns: Optional[list[Turn]] = None,
    ):
        """
        Create a new chat object.

        Parameters
        ----------
        provider
            A `Provider` object.
        turns
            A list of `Turn` objects to initialize the chat with.
        """
        self.provider = provider
        self._turns = turns or []
        self.tools: dict[str, ToolDef] = {}

    def turns(
        self,
        include_system_prompt: bool = False,
    ) -> list[Turn]:
        """
        Get all the turns (i.e., message contents) in the chat.

        Parameters
        ----------
        include_system_prompt
            Whether to include the system prompt in the turns.
        """

        if not self._turns:
            return self._turns

        if not include_system_prompt and self._turns[0].role == "system":
            return self._turns[1:]
        return self._turns

    def last_turn(
        self,
        role: Literal["assistant", "user", "system"] = "assistant",
    ) -> Turn | None:
        """
        Get the last turn in the chat with a specific role.

        Parameters
        ----------
        role
            The role of the turn to return.
        """
        for turn in reversed(self._turns):
            if turn.role == role:
                return turn
        return None

    @property
    def system_prompt(self) -> str | None:
        """
        Get the system prompt for the chat.
        """
        if self._turns and self._turns[0].role == "system":
            return self._turns[0].text
        return None

    @system_prompt.setter
    def system_prompt(self, value: str | None):
        if self._turns and self._turns[0].role == "system":
            self._turns.pop(0)
        if value is not None:
            self._turns.insert(0, Turn("system", value))

    def tokens(self) -> list[tuple[int, int]]:
        """
        Get the tokens for each turn in the chat.

        Returns
        -------
        list[tuple[int, int]]
            A list of tuples, where each tuple contains the start and end token
            indices for a turn.
        """
        return [turn.tokens for turn in self._turns]

    def app(
        self,
        *,
        stream: bool = True,
        launch_browser: bool = True,
        port: int = 0,
        kwargs: Optional[ChatRequestArgsT] = None,
    ):
        """
        Enter a chat browser to interact with the LLM.

        Parameters
        ----------
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        launch_browser
            Whether to launch a browser window.
        port
            The port to run the app on (the default is 0, which will choose a random port).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.
        """

        try:
            from shiny import App, run_app, ui
        except ImportError:
            raise ImportError(
                "The `shiny` package is required for the `browser` method. "
                "Install it with `pip install shiny`."
            )

        app_ui = ui.page_fillable(
            ui.chat_ui("chat"),
            fillable_mobile=True,
        )

        def server(input):  # noqa: A002
            chat = ui.Chat(
                "chat",
                messages=[
                    {"role": turn.role, "content": turn.text} for turn in self.turns()
                ],
            )

            @chat.on_user_submit
            async def _():
                user_input = chat.user_input()
                if user_input is None:
                    return
                response = self.submit(user_input, kwargs=kwargs, stream=stream)
                await chat.append_message_stream(response)

        run_app(
            App(app_ui, server),
            launch_browser=launch_browser,
            port=port,
        )

    def console(
        self,
        *,
        stream: bool = True,
        kwargs: Optional[ChatRequestArgsT] = None,
    ):
        """
        Enter a chat console to interact with the LLM.

        Press Ctrl+C to quit.

        Parameters
        ----------
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response

        Returns
        -------
        None
        """

        print("\nEntering chat console. Press Ctrl+C to quit.\n")

        while True:
            user_input = input("?> ")
            if user_input.strip().lower() in ("exit", "exit()"):
                break
            print("")
            self.chat(user_input, stream=stream, kwargs=kwargs)
            print("")

    def chat(
        self,
        *args: Content | str,
        stream: bool = True,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> None:
        """
        Generate a response from the chat.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.
        """
        self._chat_emit(user_turn(*args), stream=stream, kwargs=kwargs)

    async def chat_async(
        self,
        *args: Content | str,
        stream: bool = True,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> None:
        """
        Generate a response from the chat asynchronously.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.
        """
        await self._chat_emit_async(user_turn(*args), stream=stream, kwargs=kwargs)

    def submit(
        self,
        *args: Content | str,
        stream: bool = True,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> Generator[str, None, None]:
        """
        Submit user input(s) to the chat.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.

        Yields
        ------
        str
            The response content.
        """
        turn = user_turn(*args)
        for chunk in self._chat_impl(turn, stream=stream, kwargs=kwargs):
            yield chunk

    async def submit_async(
        self,
        *args: Content | str,
        stream: bool = True,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Submit user input(s) to the chat asynchronously.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.

        Yields
        ------
        str
            The response content.
        """
        turn = user_turn(*args)
        async for chunk in self._chat_impl_async(turn, stream=stream, kwargs=kwargs):
            yield chunk

    def extract_data(
        self,
        *args: Content | str,
        data_model: type[BaseModel],
    ) -> dict[str, Any]:
        """
        Extract structured data from the given input.

        Parameters
        ----------
        args
            The input to extract data from.
        data_model
            A Pydantic model describing the structure of the data to extract.

        Returns
        -------
        Any
            The extracted data.
        """

        generator = self._submit_turns(
            user_turn(*args),
            data_model=data_model,
            stream=False,
        )

        for _ in generator:
            pass

        turn = self.last_turn()
        assert turn is not None

        res: list[ContentJson] = []
        for x in turn.contents:
            if isinstance(x, ContentJson):
                res.append(x)

        if len(res) != 1:
            raise ValueError(
                f"Data extraction failed: {len(res)} data results received."
            )

        json = res[0]
        return json.value

    async def extract_data_async(
        self,
        *args: Content | str,
        data_model: type[BaseModel],
    ) -> dict[str, Any]:
        """
        Extract structured data from the given input asynchronously.

        Parameters
        ----------
        args
            The input to extract data from.
        data_model
            A Pydantic model describing the structure of the data to extract.

        Returns
        -------
        Any
            The extracted data.
        """

        generator = self._submit_turns_async(
            user_turn(*args),
            data_model=data_model,
            stream=False,
        )

        async for _ in generator:
            pass

        turn = self.last_turn()
        assert turn is not None

        res: list[ContentJson] = []
        for x in turn.contents:
            if isinstance(x, ContentJson):
                res.append(x)

        if len(res) != 1:
            raise ValueError(
                f"Data extraction failed: {len(res)} data results received."
            )

        json = res[0]
        return json.value

    def register_tool(
        self,
        tool: Callable[..., Any] | Callable[..., Awaitable[Any]] | ToolDef,
    ):
        """
        Register a tool with the chat.

        Parameters
        ----------
        tool
            The tool to register. This can be a function, an async function, or a ToolDef object.
        """
        if not isinstance(tool, ToolDef):
            tool = ToolDef(tool)

        self.tools[tool.name] = tool

    def _chat_emit(
        self,
        user_turn: Turn,
        stream: bool = True,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> None:
        from rich.console import Console
        from rich.live import Live
        from rich.markdown import Markdown

        response = self._chat_impl(
            user_turn=user_turn,
            stream=stream,
            kwargs=kwargs,
        )

        console = Console()
        content = ""

        with Live(console=console, auto_refresh=False) as live:
            for part in response:
                content += part
                live.update(Markdown(content), refresh=True)

    async def _chat_emit_async(
        self,
        user_turn: Turn,
        stream: bool = True,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> None:
        from rich.console import Console
        from rich.live import Live
        from rich.markdown import Markdown

        response = self._chat_impl_async(
            user_turn=user_turn,
            stream=stream,
            kwargs=kwargs,
        )

        console = Console()
        content = ""

        with Live(console=console, auto_refresh=False) as live:
            async for part in response:
                content += part
                live.update(Markdown(content), refresh=True)

    def _chat_impl(
        self,
        user_turn: Turn,
        stream: bool,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> Generator[str, None, None]:
        user_turn_result: Turn | None = user_turn
        while user_turn_result is not None:
            for chunk in self._submit_turns(user_turn_result, stream, kwargs=kwargs):
                yield chunk
            user_turn_result = self._invoke_tools()

    async def _chat_impl_async(
        self,
        user_turn: Turn,
        stream: bool,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> AsyncGenerator[str, None]:
        user_turn_result: Turn | None = user_turn
        while user_turn_result is not None:
            async for chunk in self._submit_turns_async(
                user_turn_result, stream, kwargs=kwargs
            ):
                yield chunk
            user_turn_result = await self._invoke_tools_async()

    def _submit_turns(
        self,
        user_turn: Turn,
        stream: bool,
        data_model: type[BaseModel] | None = None,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> Generator[str, None, None]:
        if any(x._is_async for x in self.tools.values()):
            raise ValueError("Cannot use async tools in a synchronous chat")

        if stream:
            response = self.provider.chat_perform(
                stream=True,
                turns=[*self._turns, user_turn],
                tools=self.tools,
                data_model=data_model,
                kwargs=kwargs,
            )

            result = None
            for chunk in response:
                text = self.provider.stream_text(chunk)
                if text:
                    yield text
                result = self.provider.stream_merge_chunks(result, chunk)

            turn = self.provider.stream_turn(
                result, has_data_model=data_model is not None
            )

        else:
            response = self.provider.chat_perform(
                stream=False,
                turns=[*self._turns, user_turn],
                tools=self.tools,
                data_model=data_model,
                kwargs=kwargs,
            )

            turn = self.provider.value_turn(
                response, has_data_model=data_model is not None
            )
            if turn.text:
                yield turn.text

        self._turns.extend([user_turn, turn])

    async def _submit_turns_async(
        self,
        user_turn: Turn,
        stream: bool,
        data_model: type[BaseModel] | None = None,
        kwargs: Optional[ChatRequestArgsT] = None,
    ) -> AsyncGenerator[str, None]:
        if stream:
            response = await self.provider.chat_perform_async(
                stream=True,
                turns=[*self._turns, user_turn],
                tools=self.tools,
                data_model=data_model,
                kwargs=kwargs,
            )

            result = None
            async for chunk in response:
                text = self.provider.stream_text(chunk)
                if text:
                    yield text
                result = self.provider.stream_merge_chunks(result, chunk)

            turn = self.provider.stream_turn(
                result, has_data_model=data_model is not None
            )

        else:
            response = await self.provider.chat_perform_async(
                stream=False,
                turns=[*self._turns, user_turn],
                tools=self.tools,
                data_model=data_model,
                kwargs=kwargs,
            )

            turn = self.provider.value_turn(
                response, has_data_model=data_model is not None
            )
            if turn.text:
                yield turn.text

        self._turns.extend([user_turn, turn])

    def _invoke_tools(self) -> Turn | None:
        turn = self.last_turn()
        if turn is None:
            return None

        results: list[ContentToolResult] = []
        for x in turn.contents:
            if isinstance(x, ContentToolRequest):
                tool_def = self.tools.get(x.name, None)
                func = tool_def.func if tool_def is not None else None
                results.append(self._invoke_tool(func, x.arguments, x.id))

        if not results:
            return None

        return Turn("user", results)

    async def _invoke_tools_async(self) -> Turn | None:
        turn = self.last_turn()
        if turn is None:
            return None

        results: list[ContentToolResult] = []
        for x in turn.contents:
            if isinstance(x, ContentToolRequest):
                tool_def = self.tools.get(x.name, None)
                func = tool_def.func if tool_def is not None else None
                results.append(await self._invoke_tool_async(func, x.arguments, x.id))

        if not results:
            return None

        return Turn("user", results)

    @staticmethod
    def _invoke_tool(
        func: Callable[..., Any] | None,
        arguments: dict[str, Any],
        id_: str,
    ) -> ContentToolResult:
        if func is None:
            return ContentToolResult(id_, None, "Unknown tool")

        try:
            result = func(**arguments)
            return ContentToolResult(id_, result, None)
        except Exception as e:
            return ContentToolResult(id_, None, str(e))

    @staticmethod
    async def _invoke_tool_async(
        func: Callable[..., Awaitable[Any]] | None,
        arguments: dict[str, Any],
        id_: str,
    ) -> ContentToolResult:
        if func is None:
            return ContentToolResult(id_, None, "Unknown tool")

        try:
            result = await func(**arguments)
            return ContentToolResult(id_, result, None)
        except Exception as e:
            return ContentToolResult(id_, None, str(e))

    def __str__(self):
        turns = self.turns(include_system_prompt=True)
        tokens = sum(sum(turn.tokens) for turn in turns)
        output = f"<Chat turns={len(turns)} tokens={tokens}>\n"
        for turn in turns:
            output += f"--- {turn.role} ---\n"
            for content in turn.contents:
                output += f"{content}\n"
        return output

    def __repr__(self):
        return str(self)
