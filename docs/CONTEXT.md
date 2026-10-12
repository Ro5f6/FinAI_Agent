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
1.3, tasks 1.3.1-1.3.9 are done (tools, state, planner model, system
prompt, planner node); **1.3.10 is next**. 1.3.7's live call (`-m anthropic`) waits for Anthropic credits. Added outside the original order: 1.2.21, reranking.

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
- Model facts (Claude API reference, cached 2026-10-06): Sonnet 5.5 $2/$10
  per MTok, default effort `high`; Haiku 5.5 $0.10/$0.50, default `medium`;
  Opus 5.5 $4/$20, default `medium`. On the 5.x models `temperature` is
  rejected (non-default values on 5.5) and forced `tool_choice` any/tool
  returns 400 on the 5.5 models. Server-side refusal fallback:
  `fallbacks: "default"` + beta `server-side-fallback-2026-07-01`, Claude API
  only.

## Verified LangGraph (langgraph 1.2.14)

- A state field without a reducer is **replaced** by each step's return; with
  `operator.add` it accumulates; `add_messages` appends and assigns IDs.
- With a checkpointer the state persists per `thread_id`, so `loop_count`
  must be reset by each new question's input or it carries over.
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

## Verified hybrid search (qdrant-client 1.19.1, fastembed 0.9.0)

- Named dense vector + `sparse_vectors_config={"bm25":
  SparseVectorParams(modifier=Modifier.IDF)}`; FastEmbed `Qdrant/bm25` gives
  sparse vectors (query terms weight 1; IDF applied by Qdrant). Hybrid query:
  `query_points(prefetch=[Prefetch(query=dense, using="dense", filter=f),
  Prefetch(query=sparse, using="bm25", filter=f)],
  query=FusionQuery(fusion=Fusion.RRF))`. Works in memory. Not used in
  production (D-029).

## Verified Qdrant client (qdrant-client 1.19.1, server v1.19.1)

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
