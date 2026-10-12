"""The planner's chat model, built from settings.

The only place a model name or model parameter appears in code: switching
models is a change to ``LLM_MODEL`` / ``LLM_EFFORT``, not to code. Only
parameters every current Claude model accepts are sent. ``temperature`` is
not: Sonnet 5 rejects sampling parameters and Sonnet 5.5 / Haiku 5.5 reject
non-default values. ``tool_choice`` is left at auto: the 5.5 models reject
forced tool choice. Thinking is left at the model's default (adaptive).

Verified offline against langchain-anthropic 1.7.5 / anthropic 1.13.0 with
``ChatAnthropic._get_request_payload`` (docs/CONTEXT.md).
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from finai.config import settings
from finai.tools.agent_tools import TOOLS

# When a safety classifier declines a request, the API re-runs it on a
# fallback model chosen by refusal category, inside the same call. Claude API
# only; Bedrock, Vertex and Foundry need client-side middleware instead.
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_chat_model() -> ChatAnthropic:
    """The planner model with this deployment's settings, without tools."""
    fallback = settings.LLM_REFUSAL_FALLBACK
    return ChatAnthropic(
        model=settings.LLM_MODEL,
        api_key=settings.ANTHROPIC_API_KEY.get_secret_value(),
        max_tokens=settings.LLM_MAX_TOKENS,
        output_config={"effort": settings.LLM_EFFORT},
        betas=[REFUSAL_FALLBACK_BETA] if fallback else None,
        model_kwargs={"fallbacks": "default"} if fallback else {},
    )


def make_planner() -> Runnable[LanguageModelInput, BaseMessage]:
    """The planner model with the agent's tools bound (1.3.7)."""
    return make_chat_model().bind_tools(TOOLS)
