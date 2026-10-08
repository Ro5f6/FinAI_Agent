# Decisions

Why FinAI-Agent is built the way it is. `PLAN.md` lists the tasks; this file
records the choices made along the way and the evidence for each.

To add one, use the next number and link the commit and plan task. If a
decision replaces an earlier one, mark the old entry "Superseded by D-0NN"
instead of deleting it.

### D-001 Python 3.13 (2026-09-29)

The plan said 3.12. All dependencies resolve the same on 3.12 and 3.13, so we
took the newer one. 3.15 doesn't resolve because torch has no wheels for it,
so `requires-python` is capped at `<3.14`.
`59c4b91` · 0.3.2

### D-002 Redis 8 instead of Redis Stack (2026-09-30)

The LangGraph checkpointer needs Redis JSON and Search. Redis 8 includes both
in core (`MODULE LIST` shows `search` and `ReJSON`), so the Redis Stack image
is unnecessary. ElastiCache and MemoryDB lack these modules, which rules them
out for the AWS phase.
`d5001dc` · 1.1.3, 4.2.10

### D-003 Healthchecks check modules, not just liveness (2026-10-02)

`redis-cli` exits 0 even on an error reply, so `PING` reports a Redis without
modules as healthy. The healthcheck greps `MODULE LIST` for ReJSON and search
instead. The Qdrant image has no curl, so its check uses bash's `/dev/tcp`.
`c3a0bd2` · 1.1.4

### D-004 Settings load at import and fail fast (2026-09-30)

One `Settings` singleton is built at import, so a missing API key fails at
startup instead of as a 401 mid-request. `extra="ignore"` is required because
pydantic-settings rejects unknown variables by default, and `.env.example`
already lists variables for later phases. Keys are `SecretStr` so they don't
appear in logs.
`6cc6681`, `5ffcac2` · 0.4

### D-005 Exclude amended 10-Ks (2026-10-02)

edgartools includes 10-K/A amendments by default (AAPL has two), so
`.latest()` could return a short amendment instead of the annual report. We
pass `amendments=False`.
`af664d2` · 1.2.3

### D-006 Live tests are opt-in and cover several companies (2026-10-02)

Tests that call the SEC are marked `network` and skipped by default, so the
normal suite runs offline. Each live test runs on several companies, because
testing only AAPL hid the bug described in D-012.
`af664d2`, `f4a0f8d`

### D-007 Archive filings gzipped, separate from the cache (2026-10-04)

We keep each filing so a cited claim can be traced to its source. Gzip
shrinks a filing 14x (AAPL: 1.52 MB to 105 KB), which would put a decade of
the S&P 500 at about 540 MB instead of 7.5 GB. Metadata is stored per filing
so a second year can't overwrite the first. The archive (`data/raw/`) is
separate from edgartools' cache (`data/edgar/`), which is what keeps re-runs
off the SEC.
`34e95ac` · 1.2.4

### D-008 Fiscal year comes from `period_of_report` (2026-10-04)

A company whose year ends in November files its 10-K the following year, so
using `filing_date` would label its chunks with the wrong year and year
filters would miss them.
`34e95ac` · 1.2.4

### D-009 Ingest several years, filter search by year (2026-10-04)

With only the latest 10-K, the agent can't answer "what changed since last
year". `.latest(n)` returns one filing per year, so ingestion takes a
`--years` option and 10-K search gets a `fiscal_year` filter. Without the
filter, chunks from different years mix and answers can't say which year a
passage is from.
`f2607f4` · 1.2.19, 1.3.2

### D-010 Add numeric analysis tools; arithmetic in Python (2026-10-04)

The original plan could report current figures but not trends. We added
`get_financial_history` and `get_price_history`, which compute growth,
margins, returns, volatility and drawdown in Python so the verifier can check
every number. One 10-K already contains three years of figures, so this needs
no extra fetching.
`da928cf` · 1.3.22–1.3.25

### D-011 Report, don't recommend (2026-10-04)

The agent reports what filings and prices show and never says to buy or
sell. A recommendation is the one kind of output the verifier can't check
against a tool result.
`da928cf` · 1.3.8

### D-012 Parse filings without `filing.obj()` (2026-10-06)

The plan extracted items with `filing.obj().get("Item 1A")`. That worked on
AAPL but raised `TypeError` for NVDA, MSFT, GME, RIVN and F, because
`filing.html()` returns None for most modern filings. We now read the HTML
with `sgml().html()` and extract items with `parse_html(...).get_sec_section()`.
It uses the same parser, gives identical text for AAPL, and works for every
company tested.
`725ada3` (first version), `f4a0f8d` (fix) · 1.2.5

### D-013 Finnhub and Alpaca instead of yfinance (2026-10-07)

yfinance scrapes Yahoo unofficially and can break without notice. Finnhub's
free tier gives real-time US quotes at 60 calls/min. Its price history seems
to be paid-only: free keys got 403 errors in issues
[#58](https://github.com/Finnhub-Stock-API/finnhub-python/issues/58) and
[#64](https://github.com/Finnhub-Stock-API/finnhub-python/issues/64) on its
Python client, though Finnhub's docs don't say so. Daily prices therefore
come from Alpaca's free plan (back to 2016, 200 calls/min). Alpha Vantage was
rejected at 25 calls a day. Task 1.3.1 tests `/stock/candle` with a free key;
if it works, we drop Alpaca.
`8d49cbc` · 1.3.1, 1.3.24

### D-014 Fetch 5 years of prices, compute 1y/3y/5y locally (2026-10-07)

Three years matches the three fiscal years in a 10-K. Five years covers the
2022 drawdown, which makes volatility and drawdown meaningful. Alpaca's data
starts in 2016, so ten years isn't available. Five years is about 1,260 rows,
one request either way. Recently listed companies such as RIVN (2021) have
less history, so the tool returns the start date it actually used.
`8d49cbc` · 1.3.24

### D-015 Find page headers by position and repetition (2026-10-07)

Extracted sections still contain page numbers, running headers and item
headings. Each company formats these differently (`Table of Contents`,
`PART II` + `Item 8`, `Item 1. Business (Continued)`, the company name), so a
list of known strings would fail on the next company. A line counts as a
header if it appears right after a page number at least 3 times and is at
the top of a page in at least 80% of its appearances. A second header line
must also appear on at least half the pages under the first. Without those
two checks, table column labels (AAPL `2025 2024 2023`) and real subheadings
(F `Ford Credit Segment`) were removed too. Tested on 10 companies, the only
words removed were headers, footers and headings. The 120 sentences split by
page breaks are rejoined.
`87249bb` · 1.2.6

### D-016 Drop sections under 1,000 characters (2026-10-07)

Some items only point elsewhere, such as NVDA's Item 8: "The information
required by this Item is set forth in…". The longest such reference we found
is 777 characters (XOM) and the shortest real section is 1,988 (MSFT Item
7A), so the cutoff is 1,000. Dropped items are reported, not silently
removed, which is how we caught D-020.
`87249bb` · 1.2.6

### D-017 Keep tables in chunks, split them table-aware (2026-10-08)

In a text chunk, a table row like `iPhone  $  209,586  4%  $  201,183` links
numbers to years only by column position. If a chunk loses the header row,
the model could report a 2024 figure as 2025. We considered three
alternatives:
- Stripping tables would lose breakdowns found nowhere else, and edgartools
  ignores `include_tables=False` for these sections anyway.
- Markdown output was worse: `$` and `%` become separate columns, so values
  shift out from under their years.
- OCR is covered in D-019.

So tables stay in the chunks:
- A table that fits in one chunk stays whole, with its title, unit and
  caption.
- A larger table is split by rows, and each piece repeats the title, unit,
  header rows and current row group.
- The cleaner keeps two spaces between cells.

In a trial on 10 companies, 88% of tables fit in one chunk, and 99% of table
chunks with numbers kept their year (about 94% kept their unit). The verifier
still only checks that a number exists, not its year, so an eval question
tests for this (6.2.1).
`32266ee` · 1.2.7, 1.2.9

### D-018 Take headline figures from XBRL; add breakdowns later (2026-10-08)

10-K numbers carry XBRL tags with their concept, period, unit and dimensions.
For example, Apple tags iPhone revenue for fiscal 2025 that way. Headline
figures come from `filing.xbrl().statements`, which worked for all 10
companies tested. Breakdowns by product or segment will be added to
`get_financial_history` (1.3.22).

We decided against labelling every table cell inside chunks for now. Tagging
is only required in the financial statements and notes: 78% of Item 8 table
numbers were tagged but only 48% in MD&A (Apple's MD&A: 0%). Matching
untagged numbers to tags elsewhere by value was unreliable: only 30% matched
exactly one tag.

The statements also corrected an assumption in the plan. A 10-K has three
years of income and cash-flow statements but only two balance sheets, so the
third year's debt is reported as unavailable.
`32266ee` · 1.2.7, 1.3.22, 1.5.4

### D-019 No OCR (2026-10-08)

OCR and AWS Textract were suggested for tables. 10-Ks are HTML with real
tables (58–679 per filing, with up to 15 images that are logos or charts).
OCR would turn exact text into images and read them back, introducing errors
without solving the real problem, which is knowing each number's year and
unit. Textract also costs $15 per 1,000 pages, about $5,600 for five years of
the S&P 500, against a $50–110 project budget. It would only make sense for
scanned documents.
1.2.7

### D-020 Parse with `form="10-K"` (2026-10-08)

JPM's and XOM's Item 7 just says "MD&A appears on pages 46–160", so D-016
dropped it and those companies had no MD&A. Parsing with
`ParserConfig(form="10-K")` makes edgartools follow the reference. JPM's
Item 7 grew from 396 to 412,342 characters and XOM's from 265 to 127,745.
The other eight companies came out unchanged.
`1e89b7f` · 1.2.5
