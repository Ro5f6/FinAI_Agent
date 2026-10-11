"""Test-session environment setup.

Importing anything under ``finai.config`` executes the package ``__init__``,
which builds the ``Settings`` singleton. That requires the variables with no
default, so they must exist before the first test module is imported --
otherwise collection fails, with no local ``.env`` in CI.

pytest imports this file before test modules, which makes it the right place.
Individual tests still clear and set what they need through the ``clean_env``
fixture, so no test depends on these values.
"""

import os

from dotenv import dotenv_values

# Placeholders only for variables that neither the environment nor a local
# .env provides. An exported placeholder would override .env (environment
# variables win in pydantic-settings), so live tests marked `voyage` would
# send it to the API instead of the real key.
_PLACEHOLDERS = {
    "ANTHROPIC_API_KEY": "sk-ant-collection-only",
    "SEC_IDENTITY": "Collection Only collection@example.com",
    "VOYAGE_API_KEY": "pa-collection-only",
    "FINNHUB_API_KEY": "finnhub-collection-only",
}
# Tests never send LangSmith traces, whatever the shell or .env says.
os.environ["LANGSMITH_TRACING"] = "false"

_local_env = dotenv_values(".env")
for _name, _placeholder in _PLACEHOLDERS.items():
    if not os.environ.get(_name) and not _local_env.get(_name):
        os.environ[_name] = _placeholder
