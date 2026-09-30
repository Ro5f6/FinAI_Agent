"""Configuration package.

Import the ``settings`` singleton from here; do not construct ``Settings``
in application code and do not read the environment directly.

Constructing the singleton at import time means a misconfigured environment
fails immediately, with a ``ValidationError`` naming the missing variables,
rather than surfacing later as a 401 from the LLM provider or a block from
EDGAR. Tests instantiate ``Settings`` themselves instead of using this
singleton, so they can supply their own environment.
"""

from finai.config.settings import Settings

settings = Settings()

__all__ = ["Settings", "settings"]
