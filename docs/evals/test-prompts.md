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

## 1.3.19 10-K only: "What export-control risks does NVDA list?"

Run 2026-10-22, against the S&P 500 collection (47,233 points).

| | Haiku 5.5 | Sonnet 5.5 |
|---|---|---|
| Tool calls | `query_company_10k(NVDA, section="Item 1A", k=5)` | `query_company_10k(NVDA, section="Item 1A", k=8)` |
| Search wording (the model's own) | "export control risks, restrictions on sales to China and other countries, licensing requirements" | "export controls risks China data center GPUs license requirements" |
| Passages returned | 5 (27,186 chars) | 8 (43,008 chars) |
| Planner calls | 2 (3,653 + 11,825 input tokens) | 2 (3,653 + 16,537) |
| Output tokens | 1,198 (126 reasoning) | 1,365 |
| Time | 7.7 s | 12.2 s |
| Cost | $0.0021 | $0.0540 |

**Pass on both:** only the 10-K tool is called, narrowed to Item 1A (risk
factors) without being told to; every point cites `[NVDA 10-K FY2026, Item 1A]`;
the answer names the filing and its date (FY2026, filed 2026-02-25).

**Grounding check:** each model's search was re-run and 16 specific facts
from the two answers looked for in the passages it got back: "4.5 billion",
"$60 million", "25%", "September 15, 2025", "effectively foreclosed",
"design-out", "AI Diffusion", "GAIN AI", "Hong Kong", "D:5", "15%",
"throttling", "deemed export", "Russia", "Taiwan", "Israel". All 16 are in
both models' passages: nothing in either answer comes from outside them.

Haiku's answer:

> NVIDIA's FY2026 10-K (filed 2026-02-25) lists export controls as a major risk. The main points:
>
> - **Shifting China restrictions.** Since August 2022, U.S. controls have expanded from A100 and H100 chips to H20 and other chips, with licensing needed for China and certain D:5 countries. In April 2025, a license requirement on H20 led to a $4.5 billion charge in Q1 FY2026 for excess inventory and purchase obligations. [NVDA 10-K FY2026, Item 1A]
> - **Foreclosure from China's data center market.** NVIDIA says that under current rules it cannot create a product that wins approval from both the U.S. government and the Chinese government. It states that as of the end of FY2026 it was "effectively foreclosed" from China's data center compute market. [NVDA 10-K FY2026, Item 1A]
> - **Licensing uncertainty.** Licenses have been granted, but NVIDIA says they may be temporary, come with burdensome conditions, or be delayed or denied. In August 2025, licenses allowed some H20 sales, generating about $60 million in revenue. In February 2026, a license allowed small H200 shipments to specific China-based customers, but no H200 revenue had been generated yet, and H200s would face a 25% U.S. tariff on import. [NVDA 10-K FY2026, Item 1A]
> - **Worldwide and future controls.** The AI Diffusion rule was rescinded in May 2025, and its replacement's scope is uncertain. NVIDIA warns that new controls could extend to networking products or gaming GPUs, and that proposed chip tracking or throttling requirements could create security vulnerabilities. [NVDA 10-K FY2026, Item 1A]
> - **Competitive and demand effects.** Controls may push customers toward competitors, including Chinese, European and Israeli suppliers, and may lead customers to "design out" U.S. chips. [NVDA 10-K FY2026, Item 1A]
> - **China regulatory action.** On September 15, 2025, China's antitrust regulators issued a preliminary finding that compliance with U.S. export controls, which required degraded products for China, violated the terms of China's approval of the Mellanox acquisition. Penalties or business restrictions are possible. [NVDA 10-K FY2026, Item 1A]
> - **Compliance and enforcement.** Violations, even unknowing ones, could bring penalties. Diversion reports, even unsubstantiated ones, could harm relationships and reputation. Russia sales were stopped in FY2023. [NVDA 10-K FY2026, Item 1A]
> - **Supply chain.** Export restrictions affecting Taiwan and South Korea, key suppliers, could also hurt the business. [NVDA 10-K FY2026, Item 1A]

Sonnet's answer (abridged to its headings and first points; full text in
the run log): grouped under **China market**, **Worldwide and future
controls**, and **Competitive and compliance effects**, 16 cited points, the
same facts as Haiku plus the 15% revenue expectation for licensed sales, the
GAIN AI Act, Hong Kong distribution, Country Groups D:1/D:4/D:5 and
deemed-export limits on R&D, each followed by `[NVDA 10-K FY2026, Item 1A]`.

Notes:

- The models chose their own search wording and `k`; both narrowed to
  Item 1A, as the system prompt suggests for risks.
- Sonnet asked for 8 passages, which raised its second call to 16,537 input
  tokens: most of the $0.054. A risk question costs ~$0.05 on Sonnet against
  ~$0.002 on Haiku, with no difference in grounding here.
- Citations name the section, not the passage: the verifier's citation check
  (1.5.5) can confirm the ticker, year and section were retrieved, not that
  a given sentence came from a given chunk.
