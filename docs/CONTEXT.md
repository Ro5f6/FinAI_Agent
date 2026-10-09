# Build context

Carries forward what conversation history otherwise would. Facts here were
verified by running code against the installed versions, not read from docs.
`PLAN.md` holds task state and inline amendments; `docs/DECISIONS.md` holds
the decisions and why; `git log` holds the reasoning per change.

## State

Phase 0 complete except `[HUMAN]` items (0.1.x accounts, and the "skeleton is
pushed" half of 0.3 — there is still no git remote). Step 1.1 complete. In
step 1.2, tasks 1.2.1–1.2.8 are done; **1.2.9 is next**.

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

## Verified table and XBRL behaviour (edgartools 5.60.0, ten filers)

- `filing.xbrl().statements.income_statement()` / `.balance_sheet()` /
  `.cash_flow_statement()` work for all ten filers, NVDA and F included. Income
  and cash flow carry **three** fiscal years; the balance sheet only **two**.
- `parse_html(html)` with no config detects no sections: `doc.sections` is
  empty. `ParserConfig(form="10-K")` enables them, and with it
  `get_sec_section` follows "incorporated by reference" items to the referenced
  text: JPM Item 7 goes from 396 to 419,380 chars, XOM from 265 to 132,319. NVDA
  and F are unchanged; their Item 8 stays a cross-reference.
- `Section.text(include_tables=False)` is **ignored** for these (TOC-detected)
  sections: output is identical with and without it. Tables cannot be stripped
  through edgartools.
- `Section.markdown()` is worse than text for tables: `$` and `%` become their
  own columns, so rows misalign under the headers, and punctuation is escaped
  (`10\-K`, `\.`).
- Table text is one row per line with cells joined by two or more spaces. The
  cleaner keeps two spaces so cell boundaries survive.
- `doc.xbrl_facts` returns the inline XBRL facts as a flat list, not linked to
  table cells. Linking cells to facts means parsing the HTML (lxml): each
  `ix:nonFraction` carries `name`, `contextRef` (period and dimensions),
  `unitRef`, `scale` and `sign`.
- Inline XBRL covers 78% of numeric table cells in Item 8 (91-100% for most
  filers; JPM 66%, XOM 59%) but only 48% in Item 7 (AAPL 0% to KO 75%), and 0%
  in Items 1, 1A and 7A. Percentages are not tagged. Apple prints net sales by
  category twice: the notes copy is tagged, the MD&A copy is not.
- Borrowing tags for untagged MD&A figures by matching values elsewhere in the
  filing is unsafe: 30% match one fact, 30% several, 39% none.
- Table-aware chunking trial (450 tokens, tiktoken): 88% of tables fit whole;
  of 298 table chunks holding figures, 3 lacked a year and 17 a unit, after
  attaching captions, merging blank-separated blocks and carrying row-group
  labels across pieces.

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
  cross-reference to an MD&A or "Financial Section" elsewhere in the document.
  Fixed by parsing with `ParserConfig(form="10-K")` (`parse.parse_10k`), which
  follows the reference: JPM Item 7 is 412,342 chars after cleaning, XOM
  127,745. Item 7A stays a stub for both and is dropped; its content is inside
  the MD&A. The other eight filers are byte-identical either way.
- JPM's page footer is `JPMorgan Chase & Co./2025 Form 10-K  47`, with the
  number on the outer edge, so it alternates sides; the footer rule accepts a
  number at either end of a `Form 10-K` line.
- Cross-references reach 777 chars (XOM Item 8); the shortest real section is
  1,988 (MSFT Item 7A). `MIN_SECTION_CHARS = 1_000`, measured after cleaning.

## Verified Voyage AI behaviour (voyageai 0.5.0, 2026-10-09)

- `Client(api_key=...).embed(texts, model=, input_type=, truncation=True,
  output_dimension=)`. `input_type` is `"query"`, `"document"` or None; the
  API prepends its own retrieval prompt for each, so no prefixes in our code.
- voyage-4 returns 1024 dims by default (256, 512, 2048 also offered).
  Per request: at most 1,000 texts and 320K tokens (voyage-finance-2: 120K).
- Without a payment method: 3 requests and 10K tokens per minute, enforced
  with `RateLimitError` whose message states the limit. After adding one, a
  22K-token request went through immediately. Free tokens still apply.
- `Client.count_tokens(texts, model=)` runs locally with a tokenizer
  downloaded from Hugging Face on first use. On the ten filings it counts
  8.6% more tokens than tiktoken `cl100k_base` for voyage-4 (928k vs 855k)
  and 18% more for voyage-finance-2.
- Query latency from a laptop: median 172 ms (voyage-4), 204 ms
  (voyage-finance-2) over 10 calls.
- A shell that has sourced `.env` exports `EMBEDDING_MODEL`, which overrides
  `.env` and the `Settings` default. After the switch, the terminal running
  this session still had `BAAI/bge-small-en-v1.5` and the API rejected it.

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

## Decisions

Design decisions, with the options weighed and the evidence behind each, live
in `docs/DECISIONS.md`. Record new ones there, not here.

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
