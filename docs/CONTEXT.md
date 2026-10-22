# Build context

Carries forward what conversation history otherwise would. Facts here were
verified by running code against the installed versions, not read from docs.
`PLAN.md` holds task state and inline amendments; `docs/DECISIONS.md` holds
the decisions and why; `git log` holds the reasoning per change.

## State

Phase 0 complete except `[HUMAN]` 0.1.x accounts. The repo is pushed to
github.com/Ro5f6/FinAI_Agent (public; `PLAN.md` and `docs/DECISIONS.md` are
gitignored and kept local for now).

Steps 1.1 and 1.2 complete (1.2 Done when: the supply-chain query returns
Item 1A chunks; re-ingesting 9 filings left 487 points, 0 stale). In step
1.3, tasks 1.3.1-1.3.13 are done: the agent graph compiles and runs offline
with a scripted model (diagram: `docs/agent-graph.md`), and `agent.session.ask`
runs a question on a checkpointed thread; memory verified in process (1.3.15)
and stored in Redis, surviving a process restart (1.3.16-1.3.17). **1.3.18-1.3.21
(the test prompts) need Claude, so wait for Anthropic credits.** 1.3.22 done:
`tools/financials.py` (headline figures and revenue breakdowns); 1.3.23 done
(`tools/trends.py`); 1.3.24 done (`tools/prices.py`); 1.3.25 done: `get_financial_history` and `get_price_history` are agent tools, five in all (D-036). 1.3.26 done: tickers resolve to a 10-K filer before anything is fetched or queued (`fetch.resolve_ticker`, D-037). 1.3.27 done: a 10-K search for a company not indexed queues it once (`ingestion/jobs.py`, D-038). 1.3.28 done: `python -m finai.ingestion.worker` ingests queued companies (D-039); CRCL went from queued to searchable in 7.7 s. 1.3.29 done: new companies are limited to 5 per user and 50 in total per UTC day, checked in the queue script (D-040). 1.3.30 done: on-demand ingestion is tested end to end (`tests/test_ondemand.py`: FIG queued, ingested by the worker, then searchable); it found that a missing collection crashed the first search. On-demand ingestion (1.3.26-1.3.30) is complete. 1.3.31 done: `reference/sp500-2026-10-20.csv` (503 tickers, 500 companies) and `python -m finai.ingestion.universe` resolve 501 of 503 (D-041). Searches now go by company CIK, not ticker (D-043): chunks carry `ciks`, the 640 existing points were backfilled with `python -m finai.ingestion.backfill`. 1.3.32 done: the S&P 500 is indexed, 47,233 points (see "S&P 500 ingest run"). 1.3.33 done: `python -m finai.ingestion.snapshot create|restore`; the S&P 500 collection is saved as `data/snapshots/sec_filings-...-2026-10-22-17-08-26.snapshot` (480 MB, with a `.json` manifest) and restores in 7.3 s. **Next:** 1.3.32a (17 filings extract partly or not at all), or step 1.3's test prompts: 1.3.18-1.3.20 passed on Haiku 5.5 and Sonnet 5.5, facts grounded in the retrieved passages (`docs/evals/test-prompts.md`); **step 1.3's Done-when is met** (1.3.20 calls both tools and cites 10-K sections; memory survives a restart, 1.3.17). Haiku misread the P/E direction on 1.3.20, Sonnet didn't (D-046). 1.3.21: both models decline off-topic requests, including 2.3's override attempt, with one planner call and no tools. **Step 1.3 is complete** except the amendment 1.3.32a (17 filings with partial text). Step 1.5 (verifier) started before 1.4 (D-047): 1.5.1-1.5.4 done (`agent/verification/checks.py`: the number check). **Next: 1.5.5** (citation check) or the Claude test prompts 1.3.18-1.3.21 once Anthropic credits are bought. Direction (D-042): a local MVP first, with a Chainlit chat interface (new step 2.4) calling the FastAPI endpoint, run with `docker compose up` (3.1); AWS after. Step 1.3's
Done when: memory survives a restart (met); prompt 1.3.20 calls both tools and
the answer cites 10-K sections (needs Claude). Running it with Claude waits for
Anthropic credits. 1.3.7's live call (`-m anthropic`) waits for Anthropic credits. Added outside the original order: 1.2.21, reranking.

Ingest with `python -m finai.ingestion.ingest --ticker AAPL NVDA MSFT
--years 3` (needs `docker compose up -d`). The local `sec_filings` holds
those 9 filings (FY2023-2026), 487 points.

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

## Verified multi-year ingest (edgartools 5.60.0)

- `get_filings(...).latest(1)` returns a bare `EntityFiling`; `latest(n)` for
  n > 1 an `EntityFilings`; `latest(0)` None. More than exist returns what
  exists (RIVN: 5). `get_latest_10ks` normalises to a list.
- Accession prefixes are the submitter, not the company: MSFT FY2026 is
  `0001193125-…` (a filing agent), FY2025 `0000950170-…` (another).
- AAPL/NVDA/MSFT FY2023-2026 (9 filings): 44-77 chunks each, 487 in all, 26 s.
  NVDA's Item 8 is a 154-char cross-reference every year and is dropped. No
  page numbers or bare running headers in any stored chunk. Year filter
  returns each year's own answer (NVDA data center growth: FY2026 68%,
  FY2025 142%, FY2024 217%).

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

## Verified XBRL facts for financial history (edgartools 5.60.0, ten filers)

- `xbrl.query().by_concept("us-gaap:X", exact=True).to_dataframe()` gives
  facts with `numeric_value`, `period_start`/`period_end` (durations) or
  `period_instant`, `unit_ref`, `decimals`, `is_dimensioned`, `label`.
- **`FactQuery.exclude_dimensions()` is a no-op**: AAPL revenue returned 45
  facts with and without it (iPhone, Mac, regions...). `is_dimensioned` is
  right (42 True, 3 False), so filter on it.
- **The facts' `fiscal_year` is wrong for comparative periods**: AAPL cash at
  2024-09-28 is labelled 2025, FY2023 EPS labelled 2024. Use the period end.
- `standard_concept` can name a component: JPM's `Revenue` row is principal
  transactions ($27.2B); the total is `RevenuesNetOfInterestExpense`
  ($182.45B), which has no standard concept. WMT has `Revenues` ($713.2B,
  total) and a net-sales concept ($706.4B). EPS has no standard concept;
  `EarningsPerShareDiluted` is 10/10. F's net income is only
  `NetIncomeLossAvailableToCommonStockholdersBasic`.
- Gross profit 6/10 (not F, JPM, WMT, XOM); operating income 8/10 (not JPM,
  XOM). Cash `CashCashEquivalentsRestrictedCash...` 10/10, with instants back
  to the opening balance of the earliest income year.
- The cash-flow statement rows for beginning and end cash share one concept
  and differ only by label; instants avoid that.
- Debt lines on the balance sheet (standard concepts `ShortTermDebt`,
  `CurrentPortionOfLongTermDebt`, `LongTermDebt`) are separate presented lines
  in all ten; WMT's include finance leases. JPM's long-term line is
  `LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities`.
- `facts.get_facts_with_dimensions()` adds a `dim_<prefix>_<Axis>` column
  per axis, so a fact's full dimension set is visible (the query facts have
  only `is_dimensioned`; the statement view shows one axis per row and
  mislabels some: XOM revenue members labelled "Pension Benefits - U.S.",
  AAPL/NVDA/WMT segments all "Operating segments").
- Segment facts carry `srt:ConsolidationItemsAxis =
  us-gaap:OperatingSegmentsMember` as a second dimension. Products x
  geography facts exist (KO concentrate x US) beside one-axis totals.
- Revenue breakdowns with one real axis, ten filers: geography adds up for
  AAPL NVDA MSFT F KO WMT; segments for AAPL NVDA MSFT GME RIVN WMT. Not
  adding up: nested products (AAPL, NVDA, MSFT, F, RIVN), F's two segment
  groupings, JPM segments (98%, no Corporate), KO segments (include
  intersegment sales). XOM's product members have conflicting values for one
  year and are dropped.
- `element_catalog["ko_EuropeMiddleEastAfricaMember"].labels` has the
  company's labels (standard label ends " [Member]"); `country:*` members
  aren't in it (except some filers', whose labels are upper case).
- First `filing.xbrl()` for a filing downloads it (JPM 12.5 s); cached after
  (0.6-1.4 s for any of the ten, JPM 7,928 facts).

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
- `Client(max_retries=0)` is the default: **no retries**. Above 0 it retries
  `RateLimitError`, `ServiceUnavailableError` and `Timeout` with exponential
  jitter (1-16 s). We use 5.
- The sync client keeps a `requests.Session` per thread, renewed every 180 s.
  The async client opens a new aiohttp session per request (no reuse): matters
  for the async FastAPI app in phase 2.
- AAPL's 45 chunks embed in 1-2 s with one request.
- `Client.rerank(query, documents, model, top_k=None, truncation=True)`
  returns `.results` of `(index, relevance_score, ...)` and `.total_tokens`.
  `rerank-2.5`: $0.05/M tokens, billed as query tokens x documents + document
  tokens (~30K for 30 chunks). The pricing page contradicts itself on free
  reranker tokens. Errors: `voyageai.error.{RateLimitError,
  ServiceUnavailableError, ServerError, Timeout, APIConnectionError, TryAgain}`
  are transient; `AuthenticationError`, `InvalidRequestError`,
  `MalformedRequestError` are configuration errors.
- Reranking 30 chunks: median 256 ms; 50: 409 ms; 100: 440 ms.
- Query latency from a laptop: median 172 ms (voyage-4), 204 ms
  (voyage-finance-2) over 10 calls.
- A shell that has sourced `.env` exports `EMBEDDING_MODEL`, which overrides
  `.env` and the `Settings` default. After the switch, the terminal running
  this session still had `BAAI/bge-small-en-v1.5` and the API rejected it.

## Verified planner model (langchain-anthropic 1.7.5, anthropic 1.13.0)

- `ChatAnthropic(model=, api_key=, max_tokens=, output_config={"effort": ...},
  betas=[...], model_kwargs={...})`. `ChatAnthropic._get_request_payload(
  messages, **bound.kwargs)` builds the exact request offline: with betas set
  it targets the beta endpoint; `model_kwargs` keys go top-level
  (`fallbacks: "default"`). `temperature` defaults to unset and isn't sent.
  `bind_tools` returns a `_ChatModelBinding` with `.bound` and `.kwargs`.
- Round trip, checked offline: an `AIMessage` whose content is
  `[thinking (with signature), tool_use]` followed by a `ToolMessage` becomes
  an assistant message with both blocks unchanged and a user `tool_result`
  with the same id; a `SystemMessage` becomes the top-level `system` field.
- LangChain doesn't repair a thread whose `AIMessage` has tool calls with no
  `ToolMessage` after it: the next request has `tool_use` followed by plain
  user text, which the Claude API rejects (each `tool_use` needs a
  `tool_result` in the next message). Hence `stop_at_limit`.
- **The shell exports a 17-character placeholder `ANTHROPIC_API_KEY`**,
  which overrides `.env` and gets 401 "invalid x-api-key". Live runs need
  `env -u ANTHROPIC_API_KEY -u LLM_MODEL -u EMBEDDING_MODEL -u
  LANGSMITH_API_KEY -u LANGSMITH_PROJECT` until those exports are removed.
- Measured (1.3.18): the first planner call is 3,650 input tokens on both
  Haiku 5.5 and Sonnet 5.5; a price question costs $0.0010 (Haiku) and
  $0.0179 (Sonnet). `usage_metadata.output_token_details.reasoning` reports
  reasoning tokens (Haiku 71 at effort medium, Sonnet 0).
- Model facts (Claude API reference, cached 2026-10-06): Sonnet 5.5 $2/$10
  per MTok, default effort `high`; Haiku 5.5 $0.10/$0.50, default `medium`;
  Opus 5.5 $4/$20, default `medium`. On the 5.x models `temperature` is
  rejected (non-default values on 5.5) and forced `tool_choice` any/tool
  returns 400 on the 5.5 models. Server-side refusal fallback:
  `fallbacks: "default"` + beta `server-side-fallback-2026-07-01`, Claude API
  only.

## Verified Redis checkpointer (langgraph-checkpoint-redis 0.5.2, redis 8.1.0)

- `RedisSaver(redis_url=, ttl=)`; `setup()` creates search indexes
  `checkpoint` and `checkpoint_write` and is safe to repeat. A fresh saver
  continues a thread another saver wrote. `delete_thread(thread_id)` exists.
- `ttl={"default_ttl": N, "refresh_on_read": True}`: **N is minutes** (the
  code multiplies by 60). Keys written by a saver created without `ttl` get
  no expiry (`TTL -1`), even on a thread another saver gave one.
- Two short turns: `RedisSaver` 23 keys (~36 KB: `checkpoint`,
  `checkpoint_latest`, `checkpoint_write`, `write_keys_zset`);
  `ShallowRedisSaver` 14 keys (~18 KB, latest state only, no step history).

## Verified LangGraph (langgraph 1.2.14)

- A state field without a reducer is **replaced** by each step's return; with
  `operator.add` it accumulates; `add_messages` appends and assigns IDs.
- A checkpointed graph invoked without `thread_id` raises a bare
  `KeyError: 'thread_id'`. `retrieved_docs` (an `operator.add` list) starts
  as [] on a new thread without being passed. A one-step answer saves 3
  checkpoints; `get_state` on an unknown thread returns empty values.
- `add_messages` gives each message an ID as it's added (mutating the object)
  and treats a later message with a known ID as an update, replacing the
  earlier one in place. Re-sending one message object overwrites history:
  scripted test models must build a fresh message per reply.
- With a checkpointer the state persists per `thread_id`, so `loop_count`
  must be reset by each new question's input or it carries over.
- `ToolNode` (langgraph-prebuilt 1.1.0) only runs inside a graph (outside:
  "Missing required config key"). It runs one reply's tool calls in a thread
  pool (3 x 0.5 s calls: 0.51 s), JSON-serialises dict results into the
  `ToolMessage` content and sets `name` to the tool's name, injects the run's
  config, and by default returns invalid arguments and unknown tool names as
  `status="error"` messages but re-raises any other exception.
- One real step (10-K search + quote + portfolio): 1.38 s; a 5-passage 10-K
  result is ~27K characters of tool output for the next model call.
- `InMemorySaver` is at `langgraph.checkpoint.memory`. Imports used:
  `StateGraph, START, END` from `langgraph.graph`, `add_messages` from
  `langgraph.graph.message`.
- The shell may still export `LANGSMITH_TRACING=true` after `.env` is changed;
  exported variables win, so a new terminal is needed to pick up `.env`.

## Verified LangChain tools (langchain-core 1.6.9)

- `@tool(parse_docstring=True)` moves Google-style `Args:` descriptions into
  the argument schema and keeps the summary as the description. It raises
  "Found invalid Google-Style docstring" for a function whose only parameter
  is an injected one and whose docstring has no `Args:`; use plain `@tool`.
- A parameter typed `RunnableConfig` is injected and left out of
  `tool_call_schema`. Arguments the model sends that aren't in the schema
  (`user_id`, `config`) are dropped: invoking with `{"user_id": ...}` under
  another user's config returns that user's data.
- A `Literal` argument becomes an enum in the schema; a value outside it
  fails validation before the function runs.
- Tests without a live marker can't open network connections: an autouse
  fixture in `tests/conftest.py` replaces `socket.socket.connect`. It found
  one: `QdrantClient(url=...)` starts a background version check against the
  server, so constructing a real client is a network call. DNS lookups
  (`getaddrinfo`) aren't blocked. httpx passes the guard's error through
  unwrapped.
- The shell exports `LANGSMITH_TRACING=true` with a key LangSmith rejects
  (403 on every traced call). `tests/conftest.py` forces tracing off; set it
  false in `.env` until tracing is set up in 1.4.

## Verified Finnhub behaviour (free key, 2026-10-11)

- `/quote` → `c` price, `d` change, `dp` % change, `h`/`l`/`o`, `pc` previous
  close, `t` Unix seconds (Friday's close, 20:00 UTC, on a Sunday).
  `/stock/profile2` → `name`, `currency`, `marketCapitalization` **in
  millions**. `/stock/metric?metric=all` → `metric.52WeekHigh/52WeekLow`,
  `metric.peTTM`. Auth: header `X-Finnhub-Token`. No rate-limit headers.
- **Symbols are case-sensitive**: `aapl` gets an all-zero quote.
- Unknown or delisted: 200 with an all-zero quote including `t: 0` (ZZZZ,
  TWTR), or 403 "You don't have access" (ZZZQX). Non-US listings (SHOP.TO)
  are 403 on the free plan. `BRK.B` and `BRK-B` both work.
- **Market cap is in the reporting currency**: TSM TWD, ASML EUR, BABA CNY,
  while their US quotes are USD. ETFs (SPY) have an empty profile.
- **The 52-week range is the primary listing's or main share class's**: TSM
  1375-2590 (TWD) at a $453 price, TM in JPY, NVO in DKK, SHEL in pence
  despite USD reporting, BRK.B the class A prices. Kept only if the currency
  is USD (or absent, for ETFs) and the price is within the range ±5%.
- P/E is None for loss-makers (RIVN) and ETFs.
- `peTTM` is at an older price than `/quote` (NVDA 28.6483 = 7.9108 x ~226.6,
  quote 229.28), so the tool recomputes price / `epsTTM` (and `epsAnnual`).
  `epsTTM` is the sum of the last four `series.quarterly.eps` values (NVDA
  2.46+2.39+1.76+1.30 = 7.91). price / EPS against `peTTM`: US common stocks
  1.00-1.03, GME 1.17 (price move), ASML 1.10 (EUR EPS: the exchange rate),
  SHEL 2.01 (ADR = 2 shares), TSM 0.18, TM 0.07, BRK.B 0.00 (class A EPS).
  Recomputed only for USD reporters within 1.33x; otherwise Finnhub's P/E is
  returned as company-level, and EPS is returned in its own currency
  labelled per the home-listing or main-class share (TSM: TWD 87.38). `peExclExtraAnnual` (NVDA
  274.2) is inconsistent with every other field; unused.
- `/forex/rates` and home-listing quotes (2330.TW, ASML.AS, SHEL.L) are 403
  on the free key. ECB reference rates (via api.frankfurter.app) are free and
  cover 30 currencies, not TWD. Converting EPS to USD fixes ASML (1 ADR = 1
  share: 1780.34 / (26.96 / 0.89238) = 58.9 vs Finnhub's 59.8) but not TSM
  (5 shares per ADR) or SHEL (2), and shares-per-ADR aren't in the free data.
- **`/stock/candle` returns 403 on the free key**: price history is paid-only,
  so Alpaca stays (D-013, 1.3.24).
- A quote takes ~200-360 ms (three calls); cached it takes 0.01 ms.

## Number check measured (1.5.4, 2026-10-22)

`check_numbers(draft, tool_results, question)` on real answers, each against
its own run's tool results (`data/experiments/measure_number_check.py`):

- **False alarms: 0 of 35 figures** in four correct answers (Sonnet 5.5 on
  1.3.18, 1.3.19, 1.3.20; Haiku 5.5 on 1.3.20), including Sonnet's rounding
  ("26.8", "43.3", "about $5.25 trillion") and percentages from passages.
- **Planted errors caught: 44 of 47** in a real 1.3.20 answer (each figure
  5% high, 10% low, 50% high, doubled, two digits swapped). Missed: 22% ->
  44%, 14% -> 15%, 31% -> 62%: each wrong value also appears elsewhere in
  the retrieved risk-factor passages. The check confirms a number exists in
  the results, not that it belongs to the claim; round percentages in 10-K
  text are its weak spot. Prices, P/Es, EPS and dollar amounts were all
  caught. The LLM critic (1.5.8) reads claims in context.

## Verified Qdrant snapshots (qdrant-client 1.19.1, server v1.19.1, 2026-10-22)

- `create_snapshot(collection, wait=True)` -> `SnapshotDescription(name,
  creation_time, size, checksum)`; `checksum` is the file's SHA-256 (matched
  on download). 47,233 points: 3.8 s, 479,682,048 bytes, one file.
- Download: `GET /collections/{c}/snapshots/{name}`, `application/octet-stream`,
  **no Content-Length**; streamed in 2.7 s. `delete_snapshot(c, name,
  wait=True)` removes the server's copy; `list_snapshots` is then empty.
- `recover_snapshot(collection, location=...)` takes a URL or `file://` path
  **the server fetches**, so it can't restore a file on the client machine.
  Use `POST /collections/{c}/snapshots/upload?wait=true&priority=snapshot&checksum=<sha256>`
  with multipart field `snapshot` (the generated client's
  `recover_from_uploaded_snapshot`). It creates a missing collection, carries
  over payload indexes, and with `priority=snapshot` replaces drifted data
  (4 points back to 3 in the test).
- Restoring the 480 MB snapshot into a new collection: 7.3 s; same count,
  identical top-5 search with identical scores, identical payload and vector.
- The snapshot's manifest records `embedding_model` from Settings: the
  shell's stale `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` first wrote it wrong
  (recreated with `env -u EMBEDDING_MODEL`). Remove that export from the
  shell profile.

## S&P 500 ingest run (1.3.32, 2026-10-22)

`python -m finai.ingestion.ingest --tickers-file reference/sp500-2026-10-20.csv`
on the laptop (Apple Silicon, compose Qdrant v1.19.1, voyage-4). Logs in
`data/ingest-runs/` (gitignored).

- **39.9 min** for 503 tickers: 495 filings ingested, 6 skipped as already
  stored (AAPL, MSFT, NVDA from earlier runs; GOOGL, FOXA, NWSA as their
  share-class siblings' filing, see below), 2
  failed (HONA, VYLR: no 10-K). About 4.8 s per filing; ingest alone 0.8-7.3 s.
- **46,593 new chunks**; collection 47,233 points, status green. Chunks per
  filing: median 87, max 328. The plan's estimate was ~48k (95 per filing).
- **Disk:** `data/qdrant` grew from 167 MB to 458 MB.
- **Resume verified:** rerunning the same command took 4.2 min (SEC lookups
  only): 0 ingested, 495 already stored, 8 failed, points unchanged.
- Sections kept per filing (of Items 1, 1A, 7, 7A, 8): 5 in 302, 4 in 156,
  3 in 20, 2 in 8, 1 in 3, 0 in 6. Item 8 was dropped as a cross-reference
  in 101 filings, Item 7A in 93: expected (the cleaner drops pointers,
  D-016); figures come from XBRL (1.3.22) regardless.
- **No items extracted (now `NoItemsError`, a failure, not 0 chunks):** C,
  EMR, GWW, JBHT, YUM (edgartools' section detection finds no sections in
  their layout) and WFC (report in an EX-13 exhibit). Before the fix these
  were counted "ingested" with 0 chunks; through the queue they would have
  been re-ingested on every question.
- **Thin (Item 7 and usually 8 only pointers):** BNY, IBM, PGR, USB keep
  their annual report in EX-13 (as WFC); CLX, DE, DGX, ERIE, ETN, ETR, VMC
  point to pages later in the same document, not followed. 1.3.32a.
- SBAC's Item 8 came back empty (0 chars), not a pointer.
- Share classes: the S&P file is alphabetical, so GOOG, FOX and NWS were
  ingested first and GOOGL, FOXA and NWSA skipped as the same accession.
- Warnings in the run: edgartools TOC anchor collisions (6 filings) and an
  unsupported inline-XBRL date format (2); neither lost a target item.

## Verified S&P 500 universe (2026-10-21)

- Sources: State Street's SPY daily holdings xlsx (505 rows "As of
  20-Oct-2026": 503 constituents, the fund's cash `-`, and a TPG contingent
  value right `2602335D`, identifier 436CVR021, weight 0.000003%) and
  Wikipedia's "List of S&P 500 companies" (`id="constituents"` table, 503
  rows with CIK). The two ticker sets match exactly. The xlsx needs openpyxl,
  not a project dependency: read it with `uv run --no-project --with
  openpyxl==3.1.5`. Header row is the 5th (index 4).
- 503 tickers, 500 CIKs: GOOGL/GOOG (1652044), FOXA/FOX (1754301),
  NWSA/NWS (1564708). Dotted classes: BRK.B, BF.B.
- `python -m finai.ingestion.universe reference/sp500-2026-10-20.csv`:
  319 s for 503 tickers. 501 resolve to a 10-K filer; every resolved CIK
  matches Wikipedia's. XOM resolves through its predecessor (D-037).
- No 10-K yet, both spin-offs registered on Form 10-12B: HONA (Honeywell
  Aerospace, CIK 2089271, added 2026-06-29, one 10-Q filed) and VYLR (Vylor,
  CIK 2128626, added 2026-10-01). First 10-Ks expected around Feb 2027; the
  on-demand queue picks them up once filed.

## Verified Redis queue calls (redis-py 8.1.0, Redis 8, 2026-10-17)

- `Redis.from_url(url, decode_responses=True)`; `register_script(src)`
  returns a callable taking `keys=[...]`, `args=[...]`. Inside Lua,
  `redis.call('SET', k, v, 'NX', 'EX', ttl)` is truthy when set and false
  when the key exists; the script returns the `XADD` ID or nil (None).
- `XADD key MAXLEN ~ N * field value ...` works inside the script; IDs look
  like `1792259094254-0`; `xrange` returns `[(id, {field: value})]`.
- `set(key, value, keepttl=True)` changes a value without resetting its TTL
  (the worker's status updates).
- 20 threads calling `request_ingestion` at once produced one stream entry.
- A Lua script can return a table: `return {1, id}` arrives in redis-py as
  `[1, '1792...-0']`, `{0, 'user_limit'}` as `[0, 'user_limit']`.
- A tool parameter `*, config: RunnableConfig` (keyword-only, after
  defaulted ones) is injected and hidden from the model's schema like a
  positional one; `user_id`/`config` sent as tool arguments are ignored.
- Deleting the stream key also deletes its consumer group; the worker
  recreates it at start and on `NOGROUP`.
- Consumer group: `xgroup_create(stream, group, id="0", mkstream=True)`;
  repeating it raises `ResponseError` "BUSYGROUP Consumer Group name already
  exists". `xreadgroup(group, consumer, {stream: ">"}, count=1, block=ms)`
  returns `[[stream, [(id, fields)]]]`, or `[]` when nothing arrives.
- `xautoclaim(stream, group, consumer, min_idle_time=ms, count=1)` returns
  `[next_start, [(id, fields)], [deleted_ids]]`; entries idle less than
  `min_idle_time` aren't claimed. A claim increments `times_delivered` in
  `xpending_range(stream, group, id, id, 1)` (1 -> 2).
- `xreadgroup(..., claim_min_idle_time=ms)` (Redis 8 `CLAIM`) also works on
  8.10.2: idle pending entries first, as `(id, fields, idle_ms, deliveries)`,
  then new ones as `(id, fields, 0, 0)`. Not used: see D-039.
- **redis-py 8 sets a 5 s socket timeout by default**
  (`redis/_defaults.py: DEFAULT_SOCKET_TIMEOUT = 5`; `connection_kwargs`
  shows None because the default is applied in `Connection`). An
  `XREADGROUP ... BLOCK 5000` on an empty stream therefore raised
  `redis.exceptions.TimeoutError: Timeout reading from socket`. The worker
  blocks 2 s; a test keeps the block under the default timeout.
- Worker run on CRCL (one 10-K): 153 chunks, ingest 5.9 s, process 7.7 s.
  The Voyage tokenizer is downloaded from Hugging Face on first use, so a
  worker pod needs that host too.
- `redis` was only a transitive dependency (of langgraph-checkpoint-redis);
  it is now a direct one.
- The shell exports `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, which overrides
  `.env` and makes Voyage reject live searches; run live scripts with
  `env -u EMBEDDING_MODEL -u LLM_MODEL`.

## Verified ticker resolution (edgartools 5.60.0, 2026-10-17)

- `Company(ticker)` calls `find_cik`: `find_company_cik` (bundled
  `company_tickers.parquet`, then SEC's live table only for tickers missing
  from it), then `find_mutual_fund_cik`. Lookup is case-insensitive; `.` and
  `-` are the same (`BRK.B`, `BRK-B`); `BRK/B` and `" NVDA"` (whitespace) are
  not found. Unknown: `CompanyNotFoundError` (a `LookupError`).
- Bundled table vs SEC live `company_tickers.json` (426 ms, 851 KB, 10,443
  tickers): 862 only live, 784 only bundled (mostly warrants, units, OTC),
  10 with different CIKs. Of those, XOM, NVRI, XPRO, CBAT, CLBK and GORO
  moved to a new CIK with no 10-K, while the old CIK filed one in Feb-Mar
  2026; XRPN/XRPNW are a de-SPAC (the SPAC filed the only 10-K); UROY's new
  CIK has its 10-K; REAX has none under either.
- XOM: live CIK 2115436 ExxonMobil Holdings Corp (8-K, DEFA14A, no 10-K);
  CIK 34088 Exxon Mobil Corp holds 31 10-Ks, last filed 2026-02-18.
- Fund tickers resolve through the mutual-fund list: `FB` is ProShares
  Trust (CIK 1174610), not Meta. ETFs (SPY, QQQ) are in the company table
  with no 10-K or 20-F; VFIAX only in the fund list.
- 20-F filers (TSM, SHEL, BABA, ARM, KLAR) have 0 10-Ks; recent IPOs (CRCL,
  FIG, CRWV) have their first. Old tickers (TWTR, SQ) aren't in either table.
- `resolve_ticker` takes 200-800 ms live (two `Company` loads at most); the
  XOM 10-K fetch after it took 366 ms.

## Verified Alpaca behaviour (paper-account key, 2026-10-15)

- `GET https://data.alpaca.markets/v2/stocks/{symbol}/bars` with
  `timeframe=1Day`, `start`/`end` (ISO dates, inclusive), `adjustment`,
  `feed`, `limit` (up to 10,000), `page_token`. Headers `APCA-API-KEY-ID` and
  `APCA-API-SECRET-KEY`. Bar fields: `t`, `o`, `h`, `l`, `c`, `v`, `n`, `vw`.
  `t` is New York midnight in UTC (`04:00Z` summer, `05:00Z` winter).
- Rate limit 200/min (`x-ratelimit-limit`). Five years of daily bars is one
  page (AAPL 1,264 bars).
- **`feed=sip` works on the free plan for history.** A request whose `end`
  reaches into the latest 15 minutes gets 403 "subscription does not permit
  querying recent SIP data"; with no `end` it returns today's partial bar.
  IEX is one venue: AAPL 2021-10-01 close 138.92 on IEX against 139.14 SIP.
- `adjustment`: `raw`, `split`, `all` (`bogus` → 400). NVDA around the
  2024-06-10 10:1 split: raw 1208.88 → 121.79; split 120.89 → 121.79; all
  120.54 → 121.44 (dividends too).
- Dividend adjustment moves past prices: over the year to 2026-10-14, 52-week
  low split vs all: XOM 110.385 vs 107.28 (-2.8%), T high 29.435 vs 28.42
  (-3.5%), AAPL -0.3%. One-year return split vs all: XOM 58.6% vs 63.2%, T
  -11.6% vs -7.4%.
- Symbols are case-insensitive (`aapl` works, unlike Finnhub). Share classes
  use a dot: `BRK.B` 200, `BRK/B` 404. ADRs (TSM, SHEL) are their US prices.
- Unknown symbol (ZZZZQ): 200 with `"bars": null`. Bad keys: 401 (HTML body).
- RIVN's first bar is 2021-11-10 (IPO day close).
- 10-K `period_of_report` ends (edgartools 5.60.0): AAPL late September
  (2025-09-27), NVDA late January (2026-01-25, labelled fiscal 2026), WMT
  January 31, MSFT June 30, RIVN December 31 (five 10-Ks).

## Verified hybrid search (qdrant-client 1.19.1, fastembed 0.9.0)

- Named dense vector + `sparse_vectors_config={"bm25":
  SparseVectorParams(modifier=Modifier.IDF)}`; FastEmbed `Qdrant/bm25` gives
  sparse vectors (query terms weight 1; IDF applied by Qdrant). Hybrid query:
  `query_points(prefetch=[Prefetch(query=dense, using="dense", filter=f),
  Prefetch(query=sparse, using="bm25", filter=f)],
  query=FusionQuery(fusion=Fusion.RRF))`. Works in memory. Not used in
  production (D-029).

## Verified Qdrant client (qdrant-client 1.19.1, server v1.19.1)

- **List fields:** a `FieldCondition` with `MatchValue` on a list payload
  (`ciks: [34088, 2115436]`) matches if any element equals the value, on the
  server and in `:memory:`; an `INTEGER` payload index on the field works,
  and so do facets filtered by it (they need an index on the faceted field:
  400 "No appropriate index for faceting" otherwise). `set_payload(...,
  points=Filter(...), wait=True)` adds a field to matching points without
  touching vectors. `IsEmptyCondition(is_empty=PayloadField(key="ciks"))`
  selects points lacking the field.
- Chunk IDs are `uuid5(accession:section:index)`, with no ticker: two
  share-class tickers of one company produce the same IDs for its 10-K, so
  before D-043 ingesting GOOG overwrote GOOGL's chunks.
- **A missing collection:** `facet` on the server raises
  `qdrant_client.http.exceptions.UnexpectedResponse` (404, "Collection ...
  doesn't exist!"); the in-memory client raises `ValueError` ("Collection ...
  not found"). `collection_exists` takes 1.5 ms. `indexed_years` returns []
  for a missing collection, so the first question on a fresh deployment
  queues the company instead of failing (found by the 1.3.30 test).

- Client and server versions match, so the client's version check stays quiet.
- **`QdrantClient.search` no longer exists.** Search with
  `query_points(collection_name, query=<vector>, query_filter=, limit=,
  with_payload=)`; results are in `.points`.
- Present: `collection_exists`, `create_collection(collection_name,
  vectors_config=VectorParams(size=, distance=Distance.COSINE))`,
  `create_payload_index(collection_name, field_name, field_schema=)`,
  `upsert(collection_name, points=[PointStruct(id=, vector=, payload=)])`,
  `delete(collection_name, points_selector=FilterSelector(filter=Filter(...)))`,
  `count(collection_name, count_filter=, exact=True)`.
- `QdrantClient(":memory:")` supports create/get/count/upsert, so store tests
  run offline; `-m qdrant` runs the ones that need the compose server.
  `create_collection` on an existing name raises `ValueError`.
- Payload indexes: creating the same index twice is a no-op; creating it with
  another type **replaces it silently**, so `ensure_payload_indexes` compares
  `payload_schema[field].data_type`. In-memory mode ignores payload indexes
  (it warns and `payload_schema` stays `{}`), so index tests need `-m qdrant`.
- Creating a `QdrantClient` checks the server version: 24.6 ms per call
  versus 0.9 ms reusing one client. `qdrant_client()` is cached per process.
  The REST transport is an httpx pool; for host `localhost`/`127.0.0.1` the
  client sets `max_keepalive_connections=0` itself ("may cause extra delays"),
  elsewhere (e.g. the `qdrant` Service in Kubernetes) httpx's default pool
  keeps connections alive.
- `query_points` returns a `QueryResponse`; `.points` are `ScoredPoint`s with
  `.score` (cosine similarity) and `.payload`. A filter applies even when the
  only match scores 0.
- 1.2.17 on AAPL FY2025 (45 chunks): "What are Apple's main supply chain
  risks?" filtered to AAPL returns 5 of 5 from Item 1A; #1 (0.632) holds the
  "single or limited sources" paragraph, #5 (0.566) a second mention. Query
  embedding 478 ms, search 51 ms from the laptop.
- `facet(collection_name, key, facet_filter=, limit=10, exact=)` returns
  `.hits` of `(value, count)` **ordered by count, not value**, and limits to
  10 values by default. Works in memory too. `indexed_years` sorts them.
- Unfiltered search over 3 years mixes them: AAPL's supply-chain top 5 held
  FY2025/2024/2023, two the same paragraph; MSFT "How fast is Azure
  growing?" ranked FY2024 first. With the year defaulted to the latest, AAPL
  returns 5 distinct FY2025 passages and MSFT FY2026's Azure growth sentence.
- In-memory mode supports `HasIdCondition`, `FilterSelector` deletes and
  `retrieve`, so `upsert_filing` is tested offline.
- `PayloadSchemaType` values: keyword, integer, float, geo, text, bool,
  datetime, uuid. `Distance`: Cosine, Euclid, Dot, Manhattan.

## Verified chunking behaviour (langchain-text-splitters 1.1.3, tiktoken 0.14.0)

- `RecursiveCharacterTextSplitter.from_tiktoken_encoder` defaults to the
  **`gpt2`** encoding; pass `encoding_name="cl100k_base"`. `chunk_size` and
  `chunk_overlap` go through `**kwargs` to `TextSplitter`.
- With `separators=["\n", ". ", ...]` and `keep_separator="end"` it splits at
  sentence ends and overlaps by whole sentences.
- On the ten filings at 1,000/100 tokens: 1,055 chunks, largest 999 tokens,
  710 tables of which 31 (4.4%) are split, 0.6 s for all ten. Against what each
  table had in the filing, chunks keep the year in 99.6% (680/683) and the unit
  in 98.9% (659/666). The losses are JPM tables printed across pages, where
  only the first page carries "(in millions)".
- Separators can't split text with none of them: a string with no spaces, a
  long URL, a table row longer than a chunk gave single chunks of 2,500-18,000
  tokens. A last-resort cut at token boundaries now bounds every chunk. It
  must cut on whole characters: "€", "—", "日" and emoji can span two cl100k
  tokens, and decoding half gives "\ufffd"; `decode_bytes(...).decode("utf-8")`
  raising tells a window to shrink. Real filings never trigger it: re-ingesting
  the 9 filings changed no chunk.
- "Some year in the chunk" is too weak a check. AAPL Note 4 nests `Level 1:`
  and `Level 2 (1):` inside `2025` and `2024`, and its caption names both
  years, so a continuation piece of 2024 rows passed while labelled only
  `Level 1: (continued)` (also the wrong level: `(1` read as a negative
  figure). 7 of the 9 continuation pieces in year-grouped tables (AAPL, MSFT,
  JPM x3, KO, XOM) lacked their year. Now each repeats its year and inner
  label; the live test checks each piece names the year of its first row.
- A unit check that only matches "in millions" reports 72%: many tables say
  `($M)` or "millions of dollars" (XOM), or are counts with no money unit.

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
  group — verified on the REST API. The Python client exposes it as
  `query_points_groups(collection_name, group_by, query, query_filter, limit,
  group_size, ...)`.
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
