# FinAI-Agent

Autonomous portfolio and investment research agent: a LangGraph agent with RAG
over SEC 10-K filings, live market data, and a verifier that checks every
number in an answer against tool output before it is returned. It reports what
filings and prices show; it does not recommend buying or selling.

## Where numbers come from

A 10-K reaches the agent two ways, and they are trusted for different things.

| Kind of number | Source | Why |
| --- | --- | --- |
| Headline financials: revenue, margins, net income, EPS, cash flow, cash, debt | Structured XBRL statements (`get_financial_history`, 1.3.22) | Every value carries its concept, period and unit, so it cannot be attached to the wrong year |
| Breakdowns: revenue by product, segment results, debt maturities | Tables inside the 10-K text chunks (`query_company_10k`) | Not on the face statements; planned to move to XBRL dimensions (see below) |
| Live price, market cap, 52-week range | Finnhub (1.3.1) | Not in any filing |
| Returns, volatility, drawdown | Alpaca daily bars (1.3.24), computed in Python | Not in any filing |

Arithmetic is always done in Python, never by the LLM.

### The structured statements

One 10-K carries three fiscal years of income statement and cash flow
statement but only **two** balance sheets, which is all GAAP requires. Verified
on AAPL, F, MSFT, NVDA, GME, RIVN, JPM, KO, XOM and WMT, including NVDA and F,
whose Item 8 text is only a cross-reference to statements elsewhere in the
filing.

### Tables in text chunks

edgartools renders a table one row per line, with cells separated by two
spaces:

```
2025  Change  2024  Change  2023
iPhone  $  209,586  4%  $  201,183  —%  $  200,583
```

A figure is tied to its year only by its column position, so a chunk that
loses the heading rows loses the years. Chunking is therefore table-aware
(1.2.9):

- A table is never split if it fits in one chunk (about 88% do), and it keeps
  the title, unit line and caption printed above it.
- A larger table is split by rows, and every piece repeats the title, the unit,
  the heading rows and the current row-group label (such as `2024` or
  `Level 1:`).

In a trial over the ten filers above, 99% of table chunks holding figures kept
their years and about 94% their unit.

**Known limit:** the model still matches figures to years by position, and the
verifier (1.5.4) confirms that a number appears in a tool result, not that the
answer gave it the right year. An eval question asking for a breakdown figure
by year guards this.

### Planned: breakdowns from XBRL

Inline XBRL tags each figure in the financial statements and notes with its
concept, period, unit, scale and dimensions such as product or segment, for
example Apple's iPhone revenue for fiscal 2025 as
`RevenueFromContractWithCustomerExcludingAssessedTax [IPhone]`. A
breakdown-by-dimension tool (an extension of 1.3.22) would answer breakdown
questions with labelled values and let the verifier check the year as well.
Tagging is required only in the statements and notes: in the trial, 78% of
Item 8 table figures were tagged but only 48% of MD&A figures, so this
complements table-aware chunking rather than replacing it.

### Not used: OCR

EDGAR 10-Ks are HTML with exact table text. OCR (including AWS Textract) would
render that text to an image and read it back, adding misread digits and cost
without supplying the year or unit of a figure.
