# Embedding model evaluation (task 1.2.8)

Run 2026-10-09. Eight models answered the same 18 questions over the latest
10-K of ten companies. This page lists every question, the passage that
answers it, and where each model ranked that passage, so any result can be
checked by hand.

## Method

- **Corpus:** the latest 10-K of AAPL, NVDA, MSFT, F, JPM, KO, GME, RIVN, WMT
  and XOM. Items 1, 1A, 7, 7A and 8, cleaned as in 1.2.6 and chunked with the
  table-aware trial chunker from 1.2.7.
- **Chunk size:** 450 tokens for models limited to 512 tokens, 1,000 for
  long-context models (the sizes 1.2.8 specifies). That gives 2,379 and 998
  chunks.
- **Search:** each question is compared only with its own company's chunks,
  as `query_company_10k` will filter by ticker. Chunks are ranked by cosine
  similarity.
- **Scoring:** a chunk counts as the answer if it contains the answer phrase
  listed below each question. The rank is the position of the first such
  chunk. "Top 5" means the agent would have seen the answer with `k=5`.
- **MRR:** the mean of 1/rank over the 18 questions, or 0 where the answer
  isn't found. 1.0 means every answer ranked first.
- **Query prefixes:** each model gets the prefixes its authors specify: the
  bge/arctic retrieval instruction, nomic's `search_query:`/`search_document:`,
  and Voyage's `input_type`.
- **Versions:** local models ran on FastEmbed 0.9.0 (ONNX, CPU) on an Apple
  laptop; Voyage models through `voyageai` 0.5.0.

## Results

| Model | Dims | Chunk tokens | Top 1 | Top 3 | Top 5 | MRR | Memory to serve queries | Query time | Time to embed corpus |
|---|---|---|---|---|---|---|---|---|---|
| `voyage-4` | 1024 | 1000 | 9/18 | 14/18 | **15/18** | **0.66** | ~100 MB (client) | 172 ms (API) | minutes (API) |
| `voyage-4` | 1024 | 450 | 10/18 | 12/18 | **15/18** | 0.65 | ~100 MB (client) | 172 ms (API) | minutes (API) |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 1000 | 9/18 | 11/18 | 13/18 | 0.59 | 902 MB | 20 ms | 1,375 s |
| `voyage-finance-2` | 1024 | 1000 | 7/18 | 12/18 | 13/18 | 0.54 | ~100 MB (client) | 204 ms (API) | minutes (API) |
| `jinaai/jina-embeddings-v2-small-en` | 512 | 1000 | 8/18 | 10/18 | 11/18 | 0.54 | 392 MB | 2 ms | 172 s |
| `snowflake/snowflake-arctic-embed-m` | 768 | 450 | 8/18 | 9/18 | 11/18 | 0.52 | 783 MB | 15 ms | 772 s |
| `BAAI/bge-small-en-v1.5` | 384 | 450 | 7/18 | 9/18 | 10/18 | 0.48 | 379 MB | 5 ms | 240 s |
| `ibm-granite/granite-embedding-small-english-r2` | 384 | 1000 | 6/18 | 9/18 | 10/18 | 0.44 | not measured | 5 ms | 522 s |
| `BAAI/bge-base-en-v1.5` | 768 | 450 | 4/18 | 9/18 | 10/18 | 0.40 | not measured | 15 ms | 771 s |

- **Memory** is the peak resident memory of a process that loads the model
  and embeds one query, the cost inside each API pod (limit 1.5–2 GiB, 3.2.9).
  Embedding the corpus in batches peaked much higher: 2.2 GB for nomic and
  3.7 GB for granite.
- **Query time** for local models is the median single-query time on the
  laptop. For Voyage it is the median of 10 calls from the same machine,
  including the network.
- **Corpus embedding** for Voyage took minutes once a payment method was
  added. Free accounts are limited to 3 requests and 10K tokens per minute,
  which would have made it about 95 minutes per model. The free token
  allowance still applies with a payment method; this whole evaluation used
  about 3M tokens.
- **The voyage-4 450-token row is a control.** Long-context models got bigger
  chunks, which hold more text and leave fewer chunks to rank per company. On
  the same 450-token chunks as the 512-token models, voyage-4 still found 15
  of 18, so its lead comes from the model, not the chunk size.

## Ranks per question

A bold rank is in the top 5.

| # | Company | Question | bge-small | bge-base | arctic-m | nomic-v1.5 | jina-v2-small | granite-small-r2 | voyage-4 | voyage-4-450 | voyage-finance-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AAPL | What are Apple's main supply chain risks? | **3** | **2** | **2** | **1** | 7 | **3** | **1** | **1** | **3** |
| 2 | AAPL | How much revenue did iPhone generate in each of the last three years? | **1** | **1** | **1** | **1** | **1** | **1** | **1** | **1** | **1** |
| 3 | AAPL | How are new US tariffs affecting Apple? | **1** | **1** | **1** | **1** | **1** | **1** | **1** | **1** | **1** |
| 4 | NVDA | How do US restrictions on selling chips to China affect NVIDIA? | **1** | **1** | **1** | **1** | **1** | **1** | **1** | **1** | **1** |
| 5 | NVDA | How fast did NVIDIA's data center business grow? | 21 | 26 | 18 | 12 | 10 | 18 | **4** | **5** | 9 |
| 6 | MSFT | Which foreign currencies is Microsoft most exposed to? | **2** | **2** | **1** | **1** | **2** | 12 | **3** | **5** | **2** |
| 7 | MSFT | How fast is Microsoft's cloud platform growing? | 10 | 9 | 11 | **5** | 6 | 7 | **2** | **3** | **3** |
| 8 | F | How does Ford account for the cost of vehicle recalls? | 24 | 11 | 29 | **1** | **1** | **1** | **1** | 6 | **1** |
| 9 | JPM | How much did JPMorgan set aside for expected loan losses? | 8 | 14 | 17 | 40 | **3** | 20 | **1** | **5** | **5** |
| 10 | JPM | How well capitalized is JPMorgan? | 89 | 102 | 139 | 125 | 34 | 60 | 21 | 39 | 52 |
| 11 | KO | Could taxes on sweetened drinks hurt Coca-Cola? | **1** | **1** | **1** | 6 | **4** | **3** | **2** | **1** | **2** |
| 12 | KO | How does Coca-Cola get its drinks to consumers? | **1** | **2** | **1** | **1** | **1** | **1** | **1** | **1** | **2** |
| 13 | GME | Did GameStop lose money on its Bitcoin? | 18 | 19 | 25 | 9 | 20 | 16 | 22 | 20 | 17 |
| 14 | GME | Why did GameStop's operating costs fall? | 15 | 18 | 13 | **4** | 12 | 10 | 6 | **2** | 11 |
| 15 | RIVN | What is Rivian doing with Volkswagen? | **1** | **2** | **1** | **1** | **1** | **1** | **2** | **1** | **1** |
| 16 | WMT | How is Walmart growing its online sales? | 6 | **5** | **5** | **2** | 9 | **4** | **2** | **1** | 12 |
| 17 | WMT | What cybersecurity risks does Walmart face? | **1** | **2** | **1** | **1** | **1** | **3** | **1** | **1** | **1** |
| 18 | XOM | Why did Exxon's upstream earnings change in 2025? | **5** | 7 | **5** | **3** | **1** | 6 | **1** | **1** | **1** |

## Limits of this evaluation

- **18 questions** is small. A difference of one or two questions between
  models can be chance; the gap between voyage-4 (15) and the best local model
  (13) is two questions, and between voyage-4 and the 512-token models four
  or five.
- **The questions were written after reading the filings**, so they may lean
  toward wording that appears in the text. Users' questions will be vaguer.
- **Scoring is by phrase match.** A chunk that answers the question in other
  words doesn't count. A phrase that appears in several chunks (`cyberattack`
  appears in 5 of WMT's 73) is easier to hit than one that appears once.
- **Two questions defeated every model:** JPM capital (10) and GameStop's
  Bitcoin loss (13). Both answers sit in long MD&A passages dense with other
  figures. No model choice fixes them; a reranker or query rewriting might,
  which is untested.
- **One filing per company, one run.** Local results are deterministic;
  Voyage's could vary slightly between calls.

## Questions, answers and passages

Each passage is the text around the answer phrase in the 1,000-token chunk
that contains it. "Chunks containing the phrase" counts how many of the
company's chunks match; more matches make a question easier.

**1. AAPL: What are Apple's main supply chain risks?**  
Answer phrase: `single or limited sources`  
Chunks containing the phrase (1,000-token chunks): 4 of 43  
Passage (Item1):

> …r contracted overall.  Supply of Components  Although most components essential to the Company’s business are generally available from multiple sources, certain components are currently obtained from single or limited sources. The Company also competes for various components with other participants in the markets for smartphones, personal computers, tablets, wearables and accessories. Therefore, m…

**2. AAPL: How much revenue did iPhone generate in each of the last three years?**  
Answer phrase: `209,586`  
Chunks containing the phrase (1,000-token chunks): 2 of 43  
Passage (Item7):

> …iPhone, Services and Mac. Products and Services Performance The following table shows net sales by category for 2025, 2024 and 2023 (dollars in millions):  2025  Change  2024  Change  2023 iPhone  $  209,586  4%  $  201,183  —%  $  200,583 Mac  33,708  12%  29,984  2%  29,357 iPad  28,023  5%  26,694  (6)%  28,300 Wearables, Home and Accessories  35,686  (4)%  37,005  (7)%  39,845 Services (1)  10…

**3. AAPL: How are new US tariffs affecting Apple?**  
Answer phrase: `new U\.S\. Tariffs were announced`  
Chunks containing the phrase (1,000-token chunks): 1 of 43  
Passage (Item7):

> …and indirectly impacted, and could in the future materially impact, the Company’s results of operations and financial condition.  Tariffs and Other Measures  Beginning in the second quarter of 2025, new U.S. Tariffs were announced, including additional tariffs on imports from China, India, Japan, South Korea, Taiwan, Vietnam and the EU, among others. In response, several countries have imposed, o…

**4. NVDA: How do US restrictions on selling chips to China affect NVIDIA?**  
Answer phrase: `export control`  
Chunks containing the phrase (1,000-token chunks): 10 of 42  
Passage (Item1):

> …ties are subject to various laws, rules, and regulations of the United States as well as of foreign governments.  Over the past three years, we have been subject to a series of shifting and expanding export control restrictions, impacting our ability to serve customers outside the United States.  In August 2022, the U.S. government, or USG, announced export restrictions and export licensing requir…

**5. NVDA: How fast did NVIDIA's data center business grow?**  
Answer phrase: `Data Center revenue for fiscal year 2026 was up`  
Chunks containing the phrase (1,000-token chunks): 1 of 42  
Passage (Item7):

> …g income  $130,387  $81,453  Up 60% Net income  $120,067  $72,880  Up 65% Net income per diluted share  $4.90  $2.94  Up 67%  Revenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago.  Data Center revenue for fiscal year 2026 was up 68% from a year ago. The strong year-on-year growth was driven by the major platform shifts – accelerated computing and AI.  Gaming revenue for fiscal y…

**6. MSFT: Which foreign currencies is Microsoft most exposed to?**  
Answer phrase: `Principal currency exposures include`  
Chunks containing the phrase (1,000-token chunks): 1 of 68  
Passage (Item7A):

> …sets, and liabilities are exposed to foreign currency risk. We monitor our foreign currency exposures daily to maximize the economic effectiveness of our foreign currency positions, including hedges. Principal currency exposures include the Euro, Japanese yen, British pound, Canadian dollar, and Australian dollar.  Interest Rate  Securities held in our fixed-income portfolio are subject to differe…

**7. MSFT: How fast is Microsoft's cloud platform growing?**  
Answer phrase: `Azure and other cloud services revenue increased`  
Chunks containing the phrase (1,000-token chunks): 1 of 68  
Passage (Item7):

> …o $678 billion.  •Microsoft 365 Commercial cloud revenue increased 17%.  •Microsoft 365 Consumer cloud revenue increased 28%.  •LinkedIn revenue increased 11%.  •Dynamics 365 revenue increased 18%.  •Azure and other cloud services revenue increased 41%.  •Windows OEM and Devices revenue decreased slightly.  •XBOX content and services revenue decreased 5%.  •Search advertising (formerly Search and…

**8. F: How does Ford account for the cost of vehicle recalls?**  
Answer phrase: `field service actions at the time a vehicle is sold`  
Chunks containing the phrase (1,000-token chunks): 1 of 80  
Passage (Item1A):

> …nties on emissions-related components, which can impose additional obligations beyond our standard warranties and increase our costs.  We accrue the estimated cost of both base warranty coverages and field service actions at the time a vehicle is sold, and we reevaluate the adequacy of our accruals on a regular basis.  In addition, from time to time, we issue extended warranties at our expense, th…

**9. JPM: How much did JPMorgan set aside for expected loan losses?**  
Answer phrase: `provision for credit losses was \$14\.2 billion`  
Chunks containing the phrase (1,000-token chunks): 2 of 313  
Passage (Item7):

> …with an increase of $725 million in the prior year, as well as the absence of a $1.0 billion contribution of Visa shares to the JPMorgan Chase Foundation recorded in the second quarter of 2024.  •The provision for credit losses was $14.2 billion. Net charge-offs were $9.8 billion, up $1.2 billion, predominantly driven by Wholesale and Card Services. The net addition to the allowance for credit los…

**10. JPM: How well capitalized is JPMorgan?**  
Answer phrase: `CET1 capital was \$288\.5 billion`  
Chunks containing the phrase (1,000-token chunks): 1 of 313  
Passage (Item7):

> …y offset by  –a decrease in CCB primarily driven by increased customer spending.  Refer to Liquidity Risk Management on pages 100–107 for additional information.  Selected capital and other metrics  •CET1 capital was $288.5 billion, and the Standardized and Advanced CET1 ratios were 14.6% and 14.1%, respectively.  •SLR was 5.8%.  •TBVPS grew 10.5%, ending 2025 at $107.56.  •As of December 31, 2025…

**11. KO: Could taxes on sweetened drinks hurt Coca-Cola?**  
Answer phrase: `excise tax on beverages sweetened`  
Chunks containing the phrase (1,000-token chunks): 1 of 116  
Passage (Item1A):

> …beverages, packaging taxes, carbon taxes, property taxes and payroll taxes, in many of the jurisdictions in which we operate. In addition, in the past, the U.S. Congress considered imposing a federal excise tax on beverages sweetened with sugar, HFCS or other nutritive sweeteners and may consider similar proposals in the future. As federal, state and local governments in the United States and thro…

**12. KO: How does Coca-Cola get its drinks to consumers?**  
Answer phrase: `network of independent bottling partners`  
Chunks containing the phrase (1,000-token chunks): 4 of 116  
Passage (Item1):

> …ft drink brands, including Coca-Cola, Sprite, Coca-Cola Zero Sugar, Fanta and Diet Coke/Coca-Cola Light.  We make our branded beverage products available to consumers throughout the world through our network of independent bottling partners, distributors, wholesalers and retailers as well as our consolidated bottling and distribution operations. Beverages bearing trademarks owned by or licensed to…

**13. GME: Did GameStop lose money on its Bitcoin?**  
Answer phrase: `realized loss recognized upon derecognition of our digital assets`  
Chunks containing the phrase (1,000-token chunks): 1 of 66  
Passage (Item7):

> …ts and related receivables increased to $131.6 million in fiscal 2025, compared to no such loss in the prior year. This represented 3.6% of net sales in fiscal 2025. The loss reflects a $71.8 million realized loss recognized upon derecognition of our digital assets (Bitcoin) pledged as collateral in our covered call strategy, a $59.7 million unrealized loss on the resulting digital asset receivabl…

**14. GME: Why did GameStop's operating costs fall?**  
Answer phrase: `primarily due to store closures`  
Chunks containing the phrase (1,000-token chunks): 1 of 66  
Passage (Item7):

> …g expenses as part of our ongoing cost-optimization initiatives.  In addition, store-related rent and occupancy costs and depreciation expense decreased $94.7 million and $19.2 million, respectively, primarily due to store closures and international divestitures completed in recent periods. Asset Impairments  Asset impairment expense was $53.8 million in fiscal 2025, compared to $9.7 million in th…

**15. RIVN: What is Rivian doing with Volkswagen?**  
Answer phrase: `Volkswagen Group Technologies`  
Chunks containing the phrase (1,000-token chunks): 8 of 98  
Passage (Item1):

> …g recognized by customers and has helped Rivian earn some of the industry’s most coveted owner experience awards.  Our zonal network architecture and software stack serves as the basis for Rivian and Volkswagen Group Technologies, LLC (the “Joint Venture”). The Joint Venture is working to develop industry-leading software-enabled features and capabilities to address global markets and segments acr…

**16. WMT: How is Walmart growing its online sales?**  
Answer phrase: `accelerating eCommerce sales growth`  
Chunks containing the phrase (1,000-token chunks): 1 of 73  
Passage (Item7):

> …mbers however they want to shop through our omnichannel business model. This includes increasing comparable store and club sales through increasing membership at Sam's Club U.S. and through Walmart+, accelerating eCommerce sales growth and expansion of omnichannel initiatives that complement our strategy.  Comparable sales is a metric that indicates the performance of our existing stores and clubs…

**17. WMT: What cybersecurity risks does Walmart face?**  
Answer phrase: `cyberattack`  
Chunks containing the phrase (1,000-token chunks): 5 of 73  
Passage (Item1A):

> …of merchandise as a result of shrink or theft, ineffective use or misuse of AI technologies, inaccurate, biased or otherwise flawed  AI search and discovery results, or a data breach as a result of a cyberattack could quickly erode trust and confidence in our business and could result in customer dissatisfaction, consumer boycotts, workforce unrest and government investigations. These incidents ma…

**18. XOM: Why did Exxon's upstream earnings change in 2025?**  
Answer phrase: `Lower realizations decreased earnings`  
Chunks containing the phrase (1,000-token chunks): 1 of 99  
Passage (Item7):

> …ified Items (1) (Non-GAAP) United States  5,534  6,786  5,691 Non-U.S.  16,713  18,389  17,918 Total  22,247  25,175  23,609  2025 Upstream Earnings Driver Analysis (1) (millions of dollars)  Price – Lower realizations decreased earnings by $6.1 billion, primarily driven by lower crude prices as record demand was more than offset by increased industry supply.  Advantaged Volume Growth – Increased…
