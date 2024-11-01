from __future__ import annotations

import copy
from threading import Lock
from typing import TYPE_CHECKING

from ._typing_extensions import TypedDict

if TYPE_CHECKING:
    from ._provider import Provider


class TokenUsage(TypedDict):
    name: str
    input: int
    output: int


class ThreadSafeTokenCounter:
    def __init__(self):
        self._lock = Lock()
        self._tokens: dict[str, TokenUsage] = {}

    def log_tokens(self, name: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            if name not in self._tokens:
                self._tokens[name] = {
                    "name": name,
                    "input": input_tokens,
                    "output": output_tokens,
                }
            else:
                self._tokens[name]["input"] += input_tokens
                self._tokens[name]["output"] += output_tokens

    def get_usage(self) -> list[TokenUsage] | None:
        with self._lock:
            if not self._tokens:
                return None
            # Create a deep copy to avoid external modifications
            return copy.deepcopy(list(self._tokens.values()))


# Global instance
_token_counter = ThreadSafeTokenCounter()


def tokens_log(provider: "Provider", tokens: tuple[int, int]) -> None:
    """
    Log token usage for a provider in a thread-safe manner.
    """
    name = provider.__class__.__name__.replace("Provider", "")
    _token_counter.log_tokens(name, tokens[0], tokens[1])


def tokens_reset() -> None:
    """
    Reset the token usage counter
    """
    global _token_counter  # noqa: PLW0603
    _token_counter = ThreadSafeTokenCounter()


def token_usage() -> list[TokenUsage] | None:
    """
    Report on token usage in the current session

    Call this function to find out the cumulative number of tokens that you
    have sent and received in the current session.
    """
    return _token_counter.get_usage()
