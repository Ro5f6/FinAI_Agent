# Build context

Carries forward what conversation history otherwise would. Facts here were
verified by running code against the installed versions, not read from docs.
`PLAN.md` holds task state and inline amendments; `git log` holds the reasoning
per change.

## State

Phase 0 complete except `[HUMAN]` items (0.1.x accounts, and the "skeleton is
pushed" half of 0.3 — there is still no git remote). Step 1.1 complete. In
step 1.2, tasks 1.2.1–1.2.6 are done; **1.2.7 is next**.

## Verified library behaviour (edgartools 5.60.0)

- `set_identity(user_identity)` lives in `edgar.settings`, writes `EDGAR_IDENTITY`.
  Process-global, safe to repeat.
- `Company.get_filings(form=...)` defaults to **`amendments=True`**, so 10-K/A
  filings are mixed in and `.latest()` can return an amendment. AAPL has two.
  Always pass `amendments=False`.
- `.latest()` returns **None** for an empty result set — it does not raise.
- An unknown ticker raises `CompanyNotFoundError` from `Company()` with a useful
  recovery hint; let it propagate rather than wrapping it.
- `.latest(n)` returns one filing per fiscal year, each with its own accession.
- **`Filing.html()` returns None for most modern filings.** For inline XBRL it
  falls through to `homepage.primary_html_document`, which is None for NVDA,
  MSFT, GME, RIVN and F. AAPL works, which is why this hid for a whole task.
  Use `fetch.filing_html()`, which prefers `sgml().html()`.
- **`filing.obj()` (`TenK`) is unusable for those same filings**: it reads
  `filing.html()`, so `get("Item 1A")` raises `TypeError` from inside
  `cross_reference_index.py`. Items come from
  `parse_html(filing_html(filing)).get_sec_section("part_i_item_1a")` instead —
  same parser, verified to return identical lengths to `TenK.get` for AAPL.
- `get_sec_section` keys need the part prefix: `part_i_item_1a`. Both
  `"risk_factors"` and `"Item 1A"` return None.
- Item 8 is often vestigial rather than absent — a cross-reference to a separate
  financial-statements index (NVDA 207 chars, F 436, against AAPL 62,223 and
  MSFT 105,683). It passes a non-blank check, so chunking needs a length floor.
- **One 10-K carries three comparative fiscal years** of
  `income_statement` / `balance_sheet` / `cash_flow_statement` (AAPL FY2025 shows
  2025, 2024, 2023). Three years of fundamentals need no extra fetching.
- Local storage works and is small: `use_local_storage(path,
  allow_network_fallback=False)` gives strict offline parsing, 1.3 MB for one
  filing. No bulk feed download unless `download_edgar_data()` is called.
- The rate limiter is `pyrate_limiter.InMemoryBucket`, **per process** (default
  9/s, SEC caps at 10/s). Parallel workers multiply the rate rather than sharing
  it — the most likely way to get IP-blocked.

## Verified 10-K layout (10 filers: AAPL F MSFT NVDA GME RIVN JPM KO XOM WMT)

- Page footers are a bare page number between blank lines (nine filers) or
  `Apple Inc. | 2025 Form 10-K | 5` (AAPL). Every bare number checked was
  blank-isolated and monotonic, so none were table cells.
- Running headers differ per filer (`Table of Contents`, `PART II` + `Item 8`,
  `Item 1. Business (Continued)`, `Part I`, the company name) and so are found
  by position and repetition, not by name. Repetition alone misfires: table
  column rows (AAPL `2025 2024 2023`) and subsection headings (F `Ford Credit
  Segment`) also start pages. `clean.find_running_headers` documents the tests.
- 120 page breaks fall mid-sentence across the ten filings; cleaning rejoins
  them, so the splitter does not cut a sentence at a page boundary.
- **JPM and XOM put Items 7, 7A and 8 outside the items**: each is a
  cross-reference (170-657 chars) to an MD&A or "Financial Section" elsewhere
  in the document. Cleaning drops them as vestigial and reports them, so those
  companies currently get no MD&A chunks. Recovering it means locating the
  referenced section, which is not yet in `PLAN.md`.
- Cross-references reach 777 chars (XOM Item 8); the shortest real section is
  1,988 (MSFT Item 7A). `MIN_SECTION_CHARS = 1_000`, measured after cleaning.

## Verified infrastructure

- Redis 8 ships JSON and Search **in core**: `redis:8.10.2-alpine` reports
  `search` and `ReJSON` in `MODULE LIST`. `redis/redis-stack-server` is legacy.
- `redis-cli` **exits 0 even for error replies**, so `redis-cli PING` and even
  `redis-cli JSON.SET` report healthy on a module-less Redis. Only grepping
  `MODULE LIST` discriminates — that is what the compose healthcheck does.
- The qdrant image has **no curl, wget or nc**, only bash. Its healthcheck uses
  bash's `/dev/tcp` against `/readyz`.
- Qdrant filtering restricts candidates but does **not** balance them: a filter
  allowing two fiscal years returned 3 chunks from one and 1 from the other.
  `POST /points/search/groups` with `group_by` gives a guaranteed allocation per
  group — verified on the REST API; confirm the Python client's method at 1.2.13.
- Persistence confirmed across `docker compose down` + `up` for both services.
  `down -v` would destroy the Redis volume.

## Decisions that depart from PLAN.md

- Python **3.13**, bounded `>=3.13,<3.14`. The full dependency set resolves
  identically on 3.12 and 3.13; 3.15 does not resolve (no torch wheels).
- Filings are archived **gzipped** (`<accession>.html.gz`): 1.52 MB → 105 KB,
  14x. The S&P 500 back a decade would be ~540 MB rather than 7.5 GB.
- Metadata is **per filing** (`<accession>.meta.json`), not one per ticker, so a
  second year cannot overwrite the first.
- `fiscal_year` comes from **`period_of_report`**, never `filing_date`. They
  diverge for non-calendar year ends and would mislabel chunks.
- `data/raw/` is a **provenance archive**; the cache that makes re-runs free is
  edgartools' own store at `data/edgar/`. They must not share a directory.
- Market data is **Finnhub** (live quotes, 1.3.1) plus **Alpaca** (daily price
  history, 1.3.24), not yfinance, which is an unofficial scraper. Finnhub's
  docs pages render client-side and its `swagger.json` carries no plan tiers,
  so "price history is paid-only" rests on free-key 403 reports on
  `/stock/candle` in finnhub-python issues #58 (2024) and #64 (Oct 2025), not
  an official statement. 1.3.1 settles it with one call; a 200 means drop
  Alpaca. Alpaca's official docs give the free plan history back to 2016,
  200 calls/min, latest 15 minutes withheld.
- Price history fetches **5 years** and slices to 1y/3y/5y: 3y matches the
  three comparative fiscal years in one 10-K, 5y reaches back past the 2022
  drawdown, and 10y is past Alpaca's 2016 floor and stale for volatility.
- This is a **research** tool: it reports what filings and prices show and never
  recommends buying or selling, because a recommendation is the one output the
  verifier in 1.5 cannot check against any tool result.

## Conventions

- Live-API tests are marked `network` and deselected by default
  (`addopts = "-m 'not network'"`). Run them with `uv run pytest -m network`.
- Parametrise live tests over several tickers, never AAPL alone.
- `Settings` fields are UPPERCASE to match `.env` and because 0.4's check reads
  `settings.QDRANT_URL`. `extra="ignore"` is load-bearing: `BaseSettings`
  defaults to `extra="forbid"`, which would reject later-phase `.env` variables.
- Tests construct `Settings(_env_file=None)` rather than using the singleton;
  `tests/conftest.py` supplies the two required variables before collection,
  because importing anything under `finai.config` builds the singleton.
