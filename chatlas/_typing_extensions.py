# # Within file flags to ignore unused imports
# flake8: noqa: F401
# pyright: reportUnusedImport=false

import sys

if sys.version_info >= (3, 10):
    from typing import ParamSpec, TypeGuard, is_typeddict
else:
    from typing_extensions import ParamSpec, TypeGuard, is_typeddict

# Even though TypedDict is available in Python 3.8, because it's used with NotRequired,
# they should both come from the same typing module.
# https://peps.python.org/pep-0655/#usage-in-python-3-11
if sys.version_info >= (3, 11):
    from typing import Required, TypedDict
else:
    from typing_extensions import Required, TypedDict


# The only purpose of the following line is so that pyright will put all of the
# conditional imports into the .pyi file when generating type stubs. Without this line,
# pyright will not include the above imports in the generated .pyi file, and it will
# result in a lot of red squiggles in user code.
_: "ParamSpec | TypeGuard | is_typeddict | Required | TypedDict"  # type: ignore
