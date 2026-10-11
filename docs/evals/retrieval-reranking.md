# Retrieval: dense, BM25 and reranking

Run 2026-10-11. The 18 questions of the [embedding evaluation](embedding-models.md),
asked again with four ways of ranking a company's 10-K chunks. Decided: dense
search plus a reranker; BM25 deferred until the 6.2 golden set can test it.

## Method

- **Corpus:** the latest 10-K of AAPL, NVDA, MSFT, F, JPM, KO, GME, RIVN, WMT
  and XOM through the production pipeline (clean, chunk at 1,000 tokens with
  100 overlap): 1055 chunks. The embedding evaluation used an
  earlier trial chunker, so the dense baseline was measured again here.
- **Dense:** voyage-4, cosine, filtered by ticker, as in production.
- **BM25:** FastEmbed `Qdrant/bm25` sparse vectors with Qdrant's IDF modifier,
  fused with dense results by Qdrant's reciprocal rank fusion (RRF), each
  prefetch filtered by ticker.
- **Reranker:** Voyage `rerank-2.5` over the top 30
  first-stage results; results beyond those keep their first-stage order.
- **Scoring:** as in the embedding evaluation, the rank of the first chunk
  containing the answer phrase. MRR is the mean of 1/rank (0 if not found).
- **Code:** `data/experiments/hybrid/hybrid_eval.py` (kept locally, not in
  git); the production search reproduces the dense + reranker setup.

## Results

| Setup | Top 1 | Top 3 | Top 5 | MRR |
|---|---|---|---|---|
| dense | 9/18 | 14/18 | 14/18 | 0.641 |
| dense+bm25 | 10/18 | 11/18 | 14/18 | 0.635 |
| dense+rerank | 14/18 | 17/18 | 17/18 | 0.853 |
| dense+bm25+rerank | 14/18 | 17/18 | 17/18 | 0.854 |

- **BM25 adds nothing here.** It helps some questions and hurts others (GameStop
  costs: rank 9 to 18), and adds nothing on top of the reranker.
- **The reranker is the gain:** three more answers in the top 5 and five more
  ranked first. GameStop's Bitcoin loss goes from rank 14 to 1.
- **JPM capital stays missed.** Its answer ranks 31-36 of JPM's 328 chunks,
  outside the reranker's 30. With 50 or 100 candidates the reranker sees it
  and still ranks it 10th: "How well capitalized is JPMorgan?" matches many
  capital passages better than the one-line CET1 bullet.

### Reranker candidates

| Candidates | Top 5 | Top 1 | JPM capital rank | Median rerank latency |
|---|---|---|---|---|
| 30 | 17/18 | 14/18 | 36 | 256 ms |
| 50 | 17/18 | 13/18 | 10 | 409 ms |
| 100 | 17/18 | 13/18 | 10 | 440 ms |

More candidates add latency and no hits, so production uses 30. A reranked
search took 450-560 ms end to end from a laptop (embed the question, search,
rerank). Reranking 30 passages bills about 30K tokens: about $0.0015 at
$0.05 per million tokens.

## Ranks per question

Bold is in the top 5.

| # | Company | Question | dense | dense+bm25 | dense+rerank | dense+bm25+rerank |
|---|---|---|---|---|---|---|
| 1 | AAPL | What are Apple's main supply chain risks? | **1** | **1** | **1** | **1** |
| 2 | AAPL | How much revenue did iPhone generate in each of the last three years? | **1** | **1** | **1** | **1** |
| 3 | AAPL | How are new US tariffs affecting Apple? | **1** | **1** | **1** | **1** |
| 4 | NVDA | How do US restrictions on selling chips to China affect NVIDIA? | **1** | **1** | **1** | **1** |
| 5 | NVDA | How fast did NVIDIA's data center business grow? | 6 | 10 | **2** | **2** |
| 6 | MSFT | Which foreign currencies is Microsoft most exposed to? | **2** | **2** | **1** | **1** |
| 7 | MSFT | How fast is Microsoft's cloud platform growing? | **2** | **4** | **2** | **2** |
| 8 | F | How does Ford account for the cost of vehicle recalls? | **1** | **1** | **1** | **1** |
| 9 | JPM | How much did JPMorgan set aside for expected loan losses? | **3** | **5** | **1** | **1** |
| 10 | JPM | How well capitalized is JPMorgan? | 36 | 31 | 36 | 31 |
| 11 | KO | Could taxes on sweetened drinks hurt Coca-Cola? | **3** | **5** | **1** | **1** |
| 12 | KO | How does Coca-Cola get its drinks to consumers? | **1** | **1** | **1** | **1** |
| 13 | GME | Did GameStop lose money on its Bitcoin? | 14 | 12 | **1** | **1** |
| 14 | GME | Why did GameStop's operating costs fall? | 9 | 18 | **3** | **3** |
| 15 | RIVN | What is Rivian doing with Volkswagen? | **2** | **1** | **1** | **1** |
| 16 | WMT | How is Walmart growing its online sales? | **1** | **1** | **1** | **1** |
| 17 | WMT | What cybersecurity risks does Walmart face? | **1** | **1** | **1** | **1** |
| 18 | XOM | Why did Exxon's upstream earnings change in 2025? | **1** | **1** | **1** | **1** |

## Limits

- 18 questions written after reading the filings, the same set used to choose
  the embedding model. The reranker's gain is large, but a golden set (6.2)
  should confirm it and is where BM25 gets a second test.
- Scoring by answer phrase, as before; a passage that answers in other words
  doesn't count.
