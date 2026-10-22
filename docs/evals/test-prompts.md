# Agent test prompts (1.3.18-1.3.21)

The step 1.3 test prompts, run against the real agent: Claude as planner,
the real tools (Finnhub, Qdrant with the S&P 500 10-Ks, Voyage, Alpaca,
EDGAR), effort `medium`. Each is run on the development model (Haiku 5.5)
and the deployed one (Sonnet 5.5). Runner: `data/experiments/run_prompt.py`
(local); token counts are the usage Claude reports, costs use $0.10 / $0.50
(Haiku) and $2 / $10 (Sonnet) per million input / output tokens.

## 1.3.18 Price only: "What's NVDA trading at?"

Run 2026-10-22, 17:46 UTC (market open).

| | Haiku 5.5 | Sonnet 5.5 |
|---|---|---|
| Tool calls | `get_live_market_data(ticker="NVDA")` | `get_live_market_data(ticker="NVDA")` |
| Planner calls | 2 (3,650 + 3,996 input tokens) | 2 (3,650 + 3,996) |
| Output tokens | 386 (71 reasoning) | 259 |
| Time | 3.4 s | 4.2 s |
| Cost | $0.0010 | $0.0179 |

**Pass on both:** only the market-data tool is called, every figure comes
from it, and the answer cites `[Finnhub quote, 2026-10-22 17:46 UTC]` with
the quote time.

Haiku's answer:

> NVDA is trading at **$212.77** (USD), down 4.14% on the day. [Finnhub quote, 2026-10-22 17:46 UTC]
>
> - **52-week range:** $164.27 to $243.37 [Finnhub quote, 2026-10-22 17:46 UTC]
> - **Market cap:** $5,247,534,028,858 (USD) [Finnhub quote, 2026-10-22 17:46 UTC]
> - **P/E:** 26.90 on trailing-twelve-month EPS of $7.9108 (period ended 2026-07-26), and 43.44 on last fiscal-year EPS of $4.8979 (fiscal year ended 2026-01-25) [Finnhub quote, 2026-10-22 17:46 UTC]

Sonnet's answer:

> NVDA is trading at **$212.77**, down 4.14% on the day. The quote is from 2026-10-22 17:46 UTC. [Finnhub quote, 2026-10-22 17:46 UTC]
>
> - **52-week range:** $164.27 to $243.37.
> - **Market cap:** about $5.25 trillion.
> - **P/E (trailing twelve months):** 26.9, based on EPS of $7.9108 for the period ending 2026-07-26.
> - **P/E (last fiscal year):** 43.4, based on EPS of $4.8979 for the year ending 2026-01-25.

Notes:

- Sonnet rounds ("about $5.25 trillion", 43.4); Haiku copies the tool's
  figures. The verifier's number check (1.5.4) must accept rounding of a
  tool value, as D-036 anticipated.
- The first planner call is 3,650 input tokens: the system prompt and five
  tool definitions measured ~2,400 with tiktoken; the rest is Anthropic's
  tool-use overhead and tokenizer.
