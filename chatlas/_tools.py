import inspect
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel
from typing_extensions import Literal

from . import _utils
from ._typing_extensions import Required, TypedDict, is_typeddict

__all__ = ("ToolDef",)


# TODO: maybe allow for specification of param type?
class ToolDef:
    func: Callable[..., Any] | Callable[..., Awaitable[Any]]

    def __init__(
        self,
        tool: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameter_descriptions: Optional[dict[str, str]] = None,
    ):
        self.func = tool
        self._is_async = _utils.is_async_callable(tool)
        self.schema = func_to_schema(
            tool,
            name=name,
            description=description,
            parameter_descriptions=parameter_descriptions,
        )
        self.name = self.schema["function"]["name"]


class ToolSchemaProperty(TypedDict, total=False):
    type: Required[str]
    description: Required[str]


class ToolSchemaParams(TypedDict):
    type: Literal["object"]
    properties: dict[str, ToolSchemaProperty]
    required: list[str]


class ToolSchemaFunction(TypedDict):
    name: str
    description: str
    parameters: ToolSchemaParams


class ToolSchema(TypedDict):
    type: Literal["function"]
    function: ToolSchemaFunction


def func_to_schema(
    func: Callable[..., Any] | Callable[..., Awaitable[Any]],
    name: str | None = None,
    description: str | None = None,
    parameter_descriptions: dict[str, str] | None = None,
) -> ToolSchema:
    signature = inspect.signature(func)
    required: list[str] = []

    for nm, param in signature.parameters.items():
        if param.default is param.empty and param.kind not in [
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ]:
            required.append(nm)

    annotations = get_type_hints(func, include_extras=True)

    param_desc = parameter_descriptions or {}

    params: ToolSchemaParams = {
        "type": "object",
        "properties": {
            k: type_to_json_schema(v, param_desc.get(k, None))
            for k, v in annotations.items()
            if k != "return"
        },
        "required": required,
    }

    desc = description or func.__doc__

    res: ToolSchema = {
        "type": "function",
        "function": {
            "name": name or func.__name__,
            "description": desc or "",
            "parameters": params,
        },
    }

    return res


def type_to_json_schema(
    t: type,
    desc: str | None = None,
) -> ToolSchemaProperty:
    origin = get_origin(t)
    args = get_args(t)
    if origin is Annotated:
        assert len(args) == 2
        assert desc is None or desc == ""
        assert isinstance(args[1], str)
        return type_to_json_schema(args[0], args[1])

    if origin is list:
        assert len(args) == 1
        return type_dict("array", desc, items=type_to_json_schema(args[0]))

    if origin is dict:
        assert len(args) == 2
        assert args[0] is str
        return type_dict(
            "object", desc, additionalProperties=type_to_json_schema(args[1])
        )

    if is_typeddict(t):
        annotations = get_type_hints(t, include_extras=True)
        return type_dict(
            "object",
            desc,
            properties={k: type_to_json_schema(v) for k, v in annotations.items()},
        )

    if t is dict:
        return type_dict("object", desc)
    if t is list:
        return type_dict("array", desc)
    if t is str:
        return type_dict("string", desc)
    if t is int:
        return type_dict("integer", desc)
    if t is float:
        return type_dict("number", desc)
    if t is bool:
        return type_dict("boolean", desc)
    if t is None:
        return type_dict("null", desc)
    raise ValueError(f"Unsupported type: {t}")


def type_dict(
    type_: str,
    description: str | None,
    **kwargs: Any,
) -> ToolSchemaProperty:
    res: ToolSchemaProperty = {
        "type": type_,
        "description": description or "",
        **kwargs,  # type: ignore
    }
    return res


def basemodel_to_tool_params(
    model: type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ToolSchemaParams:
    try:
        import openai
    except ImportError:
        raise ImportError(
            "The openai package is required for this functionality. "
            "Please install it with `pip install openai`."
        )

    # Lean on openai's ability to translate BaseModel.model_json_schema()
    # to a valid tool schema (this wouldn't be impossible to do ourselves,
    # but it's fair amount of logic to substitute `$refs`, etc.)
    tool = openai.pydantic_function_tool(
        model,
        name=name,
        description=description,
    )

    # Translate openai's tool schema format to our own
    fn = tool["function"]
    params: dict[str, Any] = {}
    if "parameters" in fn:
        params = fn["parameters"]

    properties: dict[str, ToolSchemaProperty] = {}
    if "properties" in params:
        properties = params["properties"]

    required: list[str] = []
    if "required" in params:
        required = params["required"]

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
