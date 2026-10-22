"""Hand the tracing settings to LangSmith (1.4).

LangSmith (0.14.7) reads its configuration from the process environment, not
from ``Settings``, and caches what it reads: ``utils.get_env_var`` and
``get_tracer_project`` are ``lru_cache``d, and the client is created once per
process. So the values must be in the environment before anything is traced;
``finai.config`` calls ``export_tracing`` right after loading the settings,
before any other module can trace.

Tracing is on only when it is enabled and a key is set: tracing with no key
would fail on every run. When it is off, ``LANGSMITH_TRACING=false`` is
exported explicitly, so a value left in the shell can't turn it back on.
"""

import os
from typing import Any


def export_tracing(settings: Any, environ: dict[str, str] | None = None) -> bool:
    """Put the LangSmith settings in the environment; return whether tracing is on."""
    environ = os.environ if environ is None else environ
    enabled = bool(settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY)
    environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY.get_secret_value()
        environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        if settings.LANGSMITH_ENDPOINT:
            environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        else:
            environ.pop("LANGSMITH_ENDPOINT", None)
    return enabled
