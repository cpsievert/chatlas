import functools
import inspect
import os
import warnings
from typing import Awaitable, Callable, TypeVar, cast

from ._typing_extensions import ParamSpec, TypeGuard

# Copied from shiny/_utils.py

R = TypeVar("R")  # Return type
P = ParamSpec("P")


def wrap_async(
    fn: Callable[P, R] | Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """
    Given a synchronous function that returns R, return an async function that wraps the
    original function. If the input function is already async, then return it unchanged.
    """

    if is_async_callable(fn):
        return fn

    fn = cast(Callable[P, R], fn)

    @functools.wraps(fn)
    async def fn_async(*args: P.args, **kwargs: P.kwargs) -> R:
        return fn(*args, **kwargs)

    return fn_async


def is_async_callable(
    obj: Callable[P, R] | Callable[P, Awaitable[R]],
) -> TypeGuard[Callable[P, Awaitable[R]]]:
    """
    Determine if an object is an async function.

    This is a more general version of `inspect.iscoroutinefunction()`, which only works
    on functions. This function works on any object that has a `__call__` method, such
    as a class instance.

    Returns
    -------
    :
        Returns True if `obj` is an `async def` function, or if it's an object with a
        `__call__` method which is an `async def` function.
    """
    if inspect.iscoroutinefunction(obj):
        return True
    if hasattr(obj, "__call__"):  # noqa: B004
        if inspect.iscoroutinefunction(obj.__call__):  # type: ignore
            return True

    return False


# https://docs.pytest.org/en/latest/example/simple.html#pytest-current-test-environment-variable
def is_testing():
    return os.environ.get("PYTEST_CURRENT_TEST", None) is not None


class MISSING_TYPE:
    pass


MISSING = MISSING_TYPE()


class DefaultModelWarning(Warning):
    pass


def inform_model_default(model: str, stacklevel: int = 3) -> str:
    if not is_testing():
        msg = f"Defaulting to `model = '{model}'`."
        warnings.warn(msg, DefaultModelWarning, stacklevel=stacklevel)
    return model
