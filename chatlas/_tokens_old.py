# # A duck type for tiktoken.Encoding
# class TiktokenEncoding(Protocol):
#     name: str
#
#     def encode(
#         self,
#         text: str,
#         *,
#         allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
#         disallowed_special: Union[Literal["all"], Collection[str]] = "all",
#     ) -> list[int]: ...
#
#
# # A duck type for tokenizers.Encoding
# @runtime_checkable
# class TokenizersEncoding(Protocol):
#     @property
#     def ids(self) -> list[int]: ...
#
#
# # A duck type for tokenizers.Tokenizer
# class TokenizersTokenizer(Protocol):
#     def encode(
#         self,
#         sequence: Any,
#         pair: Any = None,
#         is_pretokenized: bool = False,
#         add_special_tokens: bool = True,
#     ) -> TokenizersEncoding: ...
#
#
# TokenEncoding = Union[TiktokenEncoding, TokenizersTokenizer]
#
#
# def get_default_tokenizer() -> TokenizersTokenizer | None:
#     try:
#         from tokenizers import Tokenizer
#
#         return Tokenizer.from_pretrained("bert-base-cased")  # type: ignore
#     except Exception:
#         pass
#
#     return None


# def _get_token_count(
#     self,
#     content: str,
# ) -> int:
#     if self._tokenizer is None:
#         self._tokenizer = get_default_tokenizer()
#
#     if self._tokenizer is None:
#         raise ValueError(
#             "A tokenizer is required to impose `token_limits` on messages. "
#             "To get a generic default tokenizer, install the `tokenizers` "
#             "package (`pip install tokenizers`). "
#             "To get a more precise token count, provide a specific tokenizer "
#             "to the `Chat` constructor."
#         )
#
#     encoded = self._tokenizer.encode(content)
#     if isinstance(encoded, TokenizersEncoding):
#         return len(encoded.ids)
#     else:
#         return len(encoded)


# def _trim_messages(
#         self,
#         messages: tuple[TransformedMessage, ...],
#         token_limits: tuple[int, int],
#     ) -> tuple[TransformedMessage, ...]:

#         n_total, n_reserve = token_limits
#         if n_total <= n_reserve:
#             raise ValueError(
#                 f"Invalid token limits: {token_limits}. The 1st value must be greater "
#                 "than the 2nd value."
#             )

#         # Since don't trim system messages, 1st obtain their total token count
#         # (so we can determine how many non-system messages can fit)
#         n_system_tokens: int = 0
#         n_system_messages: int = 0
#         n_other_messages: int = 0
#         token_counts: list[int] = []
#         for m in messages:
#             content = (
#                 m.content_server if isinstance(m, TransformedMessage) else m.content
#             )
#             count = self._get_token_count(content)
#             token_counts.append(count)
#             if m.role == "system":
#                 n_system_tokens += count
#                 n_system_messages += 1
#             else:
#                 n_other_messages += 1

#         remaining_non_system_tokens = n_total - n_reserve - n_system_tokens

#         if remaining_non_system_tokens <= 0:
#             raise ValueError(
#                 f"System messages exceed `.messages(token_limits={token_limits})`. "
#                 "Consider increasing the 1st value of `token_limit` or setting it to "
#                 "`token_limit=None` to disable token limits."
#             )

#         # Now, iterate through the messages in reverse order and appending
#         # until we run out of tokens
#         messages2: list[TransformedMessage] = []
#         n_other_messages2: int = 0
#         token_counts.reverse()
#         for i, m in enumerate(reversed(messages)):
#             if m.role == "system":
#                 messages2.append(m)
#                 continue
#             remaining_non_system_tokens -= token_counts[i]
#             if remaining_non_system_tokens >= 0:
#                 messages2.append(m)
#                 n_other_messages2 += 1

#         messages2.reverse()

#         if len(messages2) == n_system_messages and n_other_messages2 > 0:
#             raise ValueError(
#                 f"Only system messages fit within `.messages(token_limits={token_limits})`. "
#                 "Consider increasing the 1st value of `token_limit` or setting it to "
#                 "`token_limit=None` to disable token limits."
#             )

#         return tuple(messages2)

#   def _trim_anthropic_messages(
#       self,
#       messages: tuple[TransformedMessage, ...],
#   ) -> tuple[TransformedMessage, ...]:

#       if any(m.role == "system" for m in messages):
#           raise ValueError(
#               "Anthropic requires a system prompt to be specified in it's `.create()` method "
#               "(not in the chat messages with `role: system`)."
#           )
#       for i, m in enumerate(messages):
#           if m.role == "user":
#               return messages[i:]

#       return ()
