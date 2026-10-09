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

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-collection-only")
os.environ.setdefault("SEC_IDENTITY", "Collection Only collection@example.com")
os.environ.setdefault("VOYAGE_API_KEY", "pa-collection-only")
