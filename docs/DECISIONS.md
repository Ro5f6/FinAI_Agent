# Decision log

Every design decision made while building FinAI-Agent, with the options that
were weighed and the evidence that settled it. `PLAN.md` says what to build;
this file says why it was built this way.

Most decisions here were settled by measurement rather than preference: an
API checked against the installed library version, a claim tested on several
companies instead of one, a cost estimated before committing to it. Where a
first choice turned out wrong, the entry says so.

**Adding an entry:** append the next number, keep it short, and link the
commit and plan task. A decision that replaces an earlier one marks the old
entry `Superseded by D-0NN` rather than deleting it.

## Index

| ID | Date | Decision | Status |
| --- | --- | --- | --- |
| [D-001](#d-001-python-313) | 2026-09-29 | Python 3.13 | Accepted |
| [D-002](#d-002-redis-8-not-redis-stack) | 2026-09-30 | Redis 8, not Redis Stack | Accepted |
| [D-003](#d-003-healthchecks-test-capability-not-liveness) | 2026-10-02 | Healthchecks test capability, not liveness | Accepted |
| [D-004](#d-004-configuration-fails-at-startup) | 2026-09-30 | Configuration fails at startup | Accepted |
| [D-005](#d-005-exclude-amended-10-ks) | 2026-10-02 | Exclude amended 10-Ks | Accepted |
| [D-006](#d-006-live-api-tests-are-opt-in-and-multi-company) | 2026-10-02 | Live-API tests are opt-in and multi-company | Accepted |
| [D-007](#d-007-gzipped-provenance-archive-separate-from-the-cache) | 2026-10-04 | Gzipped provenance archive, separate from the cache | Accepted |
| [D-008](#d-008-fiscal-year-comes-from-the-period-reported) | 2026-10-04 | Fiscal year comes from the period reported | Accepted |
| [D-009](#d-009-multi-year-ingestion-with-a-year-filter) | 2026-10-04 | Multi-year ingestion with a year filter | Accepted |
| [D-010](#d-010-numeric-analysis-tools-math-in-python) | 2026-10-04 | Numeric analysis tools, math in Python | Accepted |
| [D-011](#d-011-a-research-tool-not-an-adviser) | 2026-10-04 | A research tool, not an adviser | Accepted |
| [D-012](#d-012-parse-filings-without-filingobj) | 2026-10-06 | Parse filings without `filing.obj()` | Accepted |
| [D-013](#d-013-finnhub-and-alpaca-not-yfinance) | 2026-10-07 | Finnhub and Alpaca, not yfinance | Accepted |
| [D-014](#d-014-fetch-five-years-of-prices-slice-locally) | 2026-10-07 | Fetch five years of prices, slice locally | Accepted |
| [D-015](#d-015-detect-page-furniture-by-position-and-repetition) | 2026-10-07 | Detect page furniture by position and repetition | Accepted |
| [D-016](#d-016-drop-cross-reference-sections-by-length) | 2026-10-07 | Drop cross-reference sections by length | Accepted |
| [D-017](#d-017-tables-stay-in-chunks-chunked-table-aware) | 2026-10-08 | Tables stay in chunks, chunked table-aware | Accepted |
| [D-018](#d-018-figures-come-from-xbrl-breakdowns-later) | 2026-10-08 | Figures come from XBRL; breakdowns later | Accepted |
| [D-019](#d-019-no-ocr-for-tables) | 2026-10-08 | No OCR for tables | Accepted |
| [D-020](#d-020-follow-incorporated-by-reference-items) | 2026-10-08 | Follow incorporated-by-reference items | Accepted |

---

## Foundations

### D-001: Python 3.13

**Context.** The plan targeted Python 3.12.
**Options.** 3.12; 3.13; 3.15.
**Decision.** 3.13, bounded `>=3.13,<3.14`, including the Docker base image.
**Why.** The full dependency set resolves identically on 3.12 and 3.13, so
3.13 costs nothing. 3.15 does not resolve: there are no torch wheels for it.
**Consequences.** The upper bound stops an accidental upgrade to an
unresolvable version.
**Refs.** `59c4b91`, task 0.3.2.

### D-002: Redis 8, not Redis Stack

**Context.** The LangGraph checkpointer needs Redis with JSON and Search.
The plan named the Redis Stack image.
**Options.** `redis/redis-stack-server`; `redis:8`; a managed service
(ElastiCache, MemoryDB).
**Decision.** `redis:8.10.2-alpine`.
**Why.** Redis 8 ships JSON and Search in core: `MODULE LIST` reports
`search` and `ReJSON`, and `JSON.SET` plus `FT.CREATE`/`FT.SEARCH` succeed.
Redis Stack is the legacy packaging. Neither ElastiCache nor MemoryDB exposes
those modules, so either would break the checkpointer.
**Consequences.** Managed Redis is ruled out for the AWS phase (4.2.10).
**Refs.** `d5001dc`, tasks 1.1.3, 3.2.6, 4.2.10.

### D-003: Healthchecks test capability, not liveness

**Context.** Docker healthchecks gate dependent services.
**Options.** `redis-cli PING`; grep `MODULE LIST`. For Qdrant: curl `/readyz`.
**Decision.** Redis checks that `MODULE LIST` contains ReJSON and search.
Qdrant uses bash's `/dev/tcp` to request `/readyz`.
**Why.** `redis-cli` exits 0 even on an error reply, so `PING` and even
`JSON.SET` report healthy on a Redis without modules (verified on
`redis:7-alpine`). Only the module check fails when it should. The Qdrant
image has no curl, wget or nc.
**Refs.** `c3a0bd2`, task 1.1.4.

### D-004: Configuration fails at startup

**Context.** All configuration goes through one `Settings` class.
**Decision.** A module-level singleton built at import time; field names
uppercase to match `.env`; `extra="ignore"`; API keys as `SecretStr`.
**Why.** A missing key fails at startup, not mid-request with a 401.
`BaseSettings` defaults to `extra="forbid"`, which would reject variables
already listed in `.env.example` for later phases. `SecretStr` keeps keys out
of logs and reprs.
**Consequences.** Importing anything under `finai.config` needs the required
variables, so `tests/conftest.py` sets them before test collection.
**Refs.** `6cc6681`, `5ffcac2`, task 0.4.

---

## Fetching filings

### D-005: Exclude amended 10-Ks

**Context.** The ingestion pipeline needs "the latest 10-K".
**Decision.** `get_filings(form="10-K", amendments=False)`.
**Why.** edgartools defaults to `amendments=True`, which mixes 10-K/A filings
into the results; AAPL has two. `.latest()` could return an amendment, which
is often a few pages, instead of the annual report.
**Refs.** `af664d2`, task 1.2.3.

### D-006: Live-API tests are opt-in and multi-company

**Context.** Some tests must hit the SEC's servers to be meaningful.
**Decision.** Live tests are marked `network` and deselected by default.
Every live test runs over several companies, never one.
**Why.** The default suite stays offline and does not use up EDGAR's rate
limit. Testing AAPL alone hid a bug that broke every other company checked
(see D-012).
**Refs.** `af664d2`, `f4a0f8d`.

### D-007: Gzipped provenance archive, separate from the cache

**Context.** Fetched filings are kept so a cited claim can be traced to the
exact document.
**Options.** Plain HTML; gzipped HTML. One metadata file per ticker or per
filing. Archive and cache in one directory or two.
**Decision.** `data/raw/<TICKER>/<accession>.html.gz` plus a per-filing
`<accession>.meta.json` with a sha256 of the original bytes. edgartools' own
store at `data/edgar/` is the cache that keeps re-runs off the SEC.
**Why.** HTML compresses 14x (1.52 MB to 105 KB on AAPL): the S&P 500 over a
decade would be about 540 MB instead of 7.5 GB. Per-filing metadata means a
second year cannot overwrite the first. The archive exists for provenance and
the cache for speed, so they are kept apart.
**Refs.** `34e95ac`, task 1.2.4.

### D-008: Fiscal year comes from the period reported

**Context.** Every chunk is labelled with a fiscal year and searched by it.
**Options.** `filing_date`; `period_of_report`.
**Decision.** `period_of_report`.
**Why.** They differ for non-calendar year ends. A company whose year ends in
November files the following year, so `filing_date` would label the chunks
with the wrong year, and a search filtered by year would silently miss them.
**Refs.** `34e95ac`, task 1.2.4.

---

## Scope of the agent

### D-009: Multi-year ingestion with a year filter

**Context.** As planned, ingestion fetched only the latest 10-K, so the
agent could not answer "what changed since last year?".
**Decision.** Ingest the latest N 10-Ks per company (`--years`) and add a
`fiscal_year` filter to 10-K search.
**Why.** `.latest(n)` returns one filing per fiscal year, each with its own
accession, so no new API is needed. Without the filter, chunks from several
years blend together and an answer cannot say which year a passage came from.
**Refs.** `f2607f4`, tasks 1.2.19, 1.3.2.

### D-010: Numeric analysis tools, math in Python

**Context.** The plan was point-in-time: a live quote, the latest 10-K and
current portfolio value. Analysis is mostly about what changed and how fast.
**Decision.** Add `get_financial_history` (three fiscal years of
fundamentals) and `get_price_history` (return, volatility, drawdown), with
all arithmetic in Python.
**Why.** Growth rates and margins computed by the LLM cannot be trusted or
checked. Computed in Python, every figure can be verified by the verifier
(1.5.4, 1.5.6). One 10-K already carries three comparative years, so this
needs no extra fetching.
**Refs.** `da928cf`, tasks 1.3.22-1.3.25.

### D-011: A research tool, not an adviser

**Context.** A finance agent can drift from reporting into recommending.
**Decision.** The agent reports what filings and prices show and never
recommends buying or selling.
**Why.** A recommendation is the one kind of output the verifier cannot check
against any tool result. Everything else the agent says can be traced to a
filing, a price or a computation.
**Refs.** `da928cf`, task 1.3.8.

---

## Parsing and cleaning 10-Ks

### D-012: Parse filings without `filing.obj()`

**Context.** The plan extracted items with `filing.obj().get("Item 1A")`.
It worked for AAPL, the only company tested at first.
**Options.** `filing.obj()` (edgartools' `TenK`); `parse_html` with
`get_sec_section`.
**Decision.** Fetch HTML with `sgml().html()` (falling back to `html()`) and
extract items with `parse_html(...).get_sec_section("part_i_item_1a")`.
**Why.** `TenK` reads `filing.html()`, which returns None for most modern
inline-XBRL filings, so the item lookup raised `TypeError` inside edgartools
for NVDA, MSFT, GME, RIVN and F. The replacement uses the same parser,
returns identical text for AAPL, and works for every company checked.
**Lesson.** A first version that passed on one company was wrong for most.
This is why D-006 requires several companies per live test.
**Refs.** `725ada3` (first version), `f4a0f8d` (fix), task 1.2.5.

### D-015: Detect page furniture by position and repetition

**Context.** Extracted sections still contain page numbers, running headers
and the item heading, which become noise chunks and cut sentences at page
breaks.
**Options.** A list of known header strings; detection by position and
repetition.
**Decision.** Detect by position and repetition, measured on ten companies
(AAPL, F, MSFT, NVDA, GME, RIVN, JPM, KO, XOM, WMT). A header is a line that
follows a page footer at least three times, sits at the top of a page in at
least 80% of its occurrences, and, if it is a second header line, appears on
at least half of the pages under the first line.
**Why.** Every company uses different furniture (`Table of Contents`,
`PART II` + `Item 8`, `Item 1. Business (Continued)`, the company name), so a
string list breaks on the next company. Repetition alone also flagged things
that must stay: table column headings (AAPL `2025 2024 2023`) and real
subsection headings (F `Ford Credit Segment`). The two extra conditions
separate them. Page breaks cut 120 sentences across the ten filings; those
are rejoined.
**Consequences.** Verified by comparing word counts before and after
cleaning: the only words removed are headers, footers and headings.
**Refs.** `87249bb`, task 1.2.6.

### D-016: Drop cross-reference sections by length

**Context.** Some items are present but only point elsewhere, for example
NVDA's Item 8: "The information required by this Item is set forth in…".
**Decision.** After cleaning, drop sections under 1,000 characters and report
them, rather than dropping them silently.
**Why.** Cross-references reach 777 characters (XOM Item 8); the shortest
real section is 1,988 (MSFT Item 7A). The threshold sits in that gap.
Reporting dropped items exposed the problem fixed in D-020.
**Refs.** `87249bb`, task 1.2.6.

### D-020: Follow incorporated-by-reference items

**Context.** JPM and XOM print Item 7 as a pointer ("MD&A appears on pages
46-160"). Their MD&A came back as 396 and 265 characters and was dropped by
D-016, so those companies had no MD&A at all.
**Decision.** Parse with `ParserConfig(form="10-K")`.
**Why.** Without the form type, edgartools detects no sections and returns
the item text as printed. With it, the parser follows the reference: JPM
Item 7 becomes 412,342 characters after cleaning, XOM 127,745. The other
eight companies come out byte-for-byte identical.
**Consequences.** JPM's recovered pages use a page footer not seen before
(the page number at alternating ends), so the footer rule was widened.
**Refs.** `1e89b7f`, task 1.2.5.

---

## Financial tables

### D-017: Tables stay in chunks, chunked table-aware

**Context.** Tables are about 24% of Item 8 text and 8% of MD&A. Flattened to
text, a figure is tied to its year only by column position:
`iPhone  $  209,586  4%  $  201,183`. A chunk that loses the heading row loses
the years, and the model could report a 2024 figure as 2025.
**Options.** Strip tables from chunks; keep them as plain text; keep them
with table-aware chunking; render them as Markdown.
**Decision.** Keep tables in the chunks, with table-aware chunking: a table
that fits in one chunk is never split, and keeps its title, unit and caption.
A larger table is split by rows, and every piece repeats the title, unit,
heading rows and current row-group label. The cleaner keeps two-space gaps
between cells.
**Why.** Tables hold breakdowns that appear nowhere else in the structured
data (revenue by product, segment results, debt maturities), so stripping
them loses answers. edgartools cannot strip them anyway:
`text(include_tables=False)` is ignored for these sections. Markdown output
was worse: `$` and `%` become separate columns, so rows misalign under their
headings. In a trial over the ten companies, 88% of tables fit whole, and of
298 table chunks holding figures, 99% kept their year and about 94% their
unit.
**Consequences.** The model still matches figures to years by position, and
the verifier checks that a number exists, not its year. An eval question
targets this (6.2.1).
**Refs.** `32266ee`, tasks 1.2.7, 1.2.9.

### D-018: Figures come from XBRL; breakdowns later

**Context.** Companies tag the numbers in their filings with inline XBRL:
each figure carries its concept, period, unit, scale and dimensions such as
product, for example Apple's iPhone revenue for fiscal 2025.
**Options.** Ignore XBRL; use XBRL for headline statements only; also label
every table cell in the chunks with its XBRL tag; add XBRL breakdowns as a
tool.
**Decision.** Headline figures come from the structured statements
(`filing.xbrl().statements`) now. Breakdowns by dimension become an extension
of `get_financial_history` (1.3.22). Labelling table cells inside chunks is
deferred unless evals show year mistakes.
**Why.** The structured statements work for all ten companies, including the
two whose Item 8 is only a cross-reference. Tagging is only required in the
financial statements and notes: 78% of Item 8 table figures were tagged but
only 48% of MD&A figures (Apple's MD&A: 0%), so labelling chunks would cover
only part of the text and cannot replace D-017. Borrowing tags for untagged
figures by matching values is unsafe: 30% matched one tag, 30% several, 39%
none. As a tool, XBRL answers breakdown questions with labelled values and
lets the verifier check the year too.
**Consequences.** One 10-K has three years of income and cash flow statement
but only **two** balance sheets, which corrected an assumption in the plan.
The third year's cash comes from the cash flow statement; the third year's
debt is reported as unavailable.
**Refs.** `32266ee`, tasks 1.2.7, 1.3.22, 1.5.4.

### D-019: No OCR for tables

**Context.** OCR, including AWS Textract, was proposed for table extraction.
**Options.** OCR or Textract; parse the HTML directly.
**Decision.** No OCR. Tables are read from the filing's HTML.
**Why.** EDGAR 10-Ks are HTML with real tables (58 to 679 `<table>` elements
per filing, 0 to 15 images, all logos or charts). OCR would render exact text
to an image and read it back, adding misread digits, without solving the
actual problem: which year and unit a figure belongs to. Only the XBRL tags
carry that. Textract also costs $15 per 1,000 pages, about $5,600 for the
S&P 500 over five years, against a $50-110 project budget.
**Consequences.** Worth revisiting only for documents that exist as scans or
images, such as uploaded brokerage statements.
**Refs.** task 1.2.7.

---

## Market data

### D-013: Finnhub and Alpaca, not yfinance

**Context.** The plan used yfinance for live quotes and price history.
**Options.** yfinance; Finnhub; Alpaca; Twelve Data; Alpha Vantage.
**Decision.** Finnhub for live quotes (1.3.1); Alpaca for daily price history
(1.3.24).
**Why.** yfinance is an unofficial scraper that can break or be rate-limited
without notice. Finnhub's free tier gives real-time US quotes at 60
calls/min. Its price-history endpoint appears paid-only: free keys get 403 in
user reports on its official Python client (finnhub-python issues
[#58](https://github.com/Finnhub-Stock-API/finnhub-python/issues/58), 2024,
and [#64](https://github.com/Finnhub-Stock-API/finnhub-python/issues/64),
2025). Its docs pages render client-side and its API spec lists no plan
tiers, so there is no official statement to cite. Alpaca's free plan has daily bars back to
2016 at 200 calls/min. Alpha Vantage allows 25 calls a day, which one eval run
would exhaust.
**Consequences.** Two API keys and two signups. The 403 report is not an
official statement, so 1.3.1 checks `/stock/candle` with a free key; if it
returns 200, Alpaca is dropped.
**Refs.** `8d49cbc`, tasks 0.1.5, 0.1.6, 1.3.1, 1.3.24.

### D-014: Fetch five years of prices, slice locally

**Context.** `get_price_history` needs a lookback period.
**Options.** 1, 3, 5 or 10 years.
**Decision.** Always fetch 5 years of adjusted daily bars in one request, and
compute 1y, 3y and 5y figures from it (default 1y), plus returns per fiscal
year.
**Why.** Three years lines up with the three fiscal years in one 10-K. Five
reaches back past the 2022 drawdown, so volatility and drawdown mean
something. Ten is past Alpaca's 2016 limit and too old to say much about the
business today. Five years is about 1,260 rows, one request either way.
**Consequences.** Short histories (RIVN listed in 2021) yield less than five
years, so the tool returns the actual start date used.
**Refs.** `8d49cbc`, task 1.3.24.
