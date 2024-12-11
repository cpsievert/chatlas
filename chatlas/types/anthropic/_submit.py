# ---------------------------------------------------------
# Do not modify this file. It was generated by `scripts/generate_typed_dicts.py`.
# ---------------------------------------------------------


from typing import Iterable, Literal, Mapping, Optional, TypedDict, Union

import anthropic
import anthropic._types
import anthropic.types.message_param
import anthropic.types.text_block_param
import anthropic.types.tool_choice_any_param
import anthropic.types.tool_choice_auto_param
import anthropic.types.tool_choice_tool_param
import anthropic.types.tool_param


class SubmitInputArgs(TypedDict, total=False):
    max_tokens: int
    messages: Iterable[anthropic.types.message_param.MessageParam]
    model: Union[
        str,
        Literal[
            "claude-3-5-haiku-latest",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-latest",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-latest",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
        ],
    ]
    stop_sequences: Union[list[str], anthropic.NotGiven]
    stream: Union[Literal[False], Literal[True], anthropic.NotGiven]
    system: Union[
        str,
        Iterable[anthropic.types.text_block_param.TextBlockParam],
        anthropic.NotGiven,
    ]
    temperature: float | anthropic.NotGiven
    tool_choice: Union[
        anthropic.types.tool_choice_auto_param.ToolChoiceAutoParam,
        anthropic.types.tool_choice_any_param.ToolChoiceAnyParam,
        anthropic.types.tool_choice_tool_param.ToolChoiceToolParam,
        anthropic.NotGiven,
    ]
    tools: Union[Iterable[anthropic.types.tool_param.ToolParam], anthropic.NotGiven]
    top_k: int | anthropic.NotGiven
    top_p: float | anthropic.NotGiven
    extra_headers: Optional[Mapping[str, Union[str, anthropic._types.Omit]]]
    extra_query: Optional[Mapping[str, object]]
    extra_body: object | None
    timeout: float | anthropic.Timeout | None | anthropic.NotGiven