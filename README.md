# NudaUI Semantic Search

Natural-language search over the [NudaUI](https://nudaui.dev) component catalog. You describe what you want ("a loader with pulsing dots", "a button that glows on hover") and it returns the components that match, ranked by semantic similarity, with their category and a link to the code.

It is built as a standalone retrieval service, separate from the NudaUI site. The catalog is the corpus; the site is untouched.

Two things drove this project. First, to build a RAG pipeline that is measured, not vibes-based: every change is evaluated against a fixed set of labeled queries so improvements are numbers, not hunches. Second, to seed a future search feature for NudaUI itself.

## How it works

Retrieval runs in two phases.

Indexing happens once. Each of the 1022 components is turned into a short text and embedded into a vector with [Voyage AI](https://voyageai.com). The vectors are stored with their metadata in a local JSON file.

Search happens on every query. The user's phrase is embedded with the same model (using `input_type="query"` rather than `"document"`, which Voyage optimizes differently), then compared against every stored vector by cosine similarity. The top matches come back.

Cosine similarity is implemented directly rather than pulled from a framework like LangChain or LlamaIndex. For a corpus this size a brute-force scan is fast enough, and writing it by hand means the retrieval path is fully understood rather than hidden behind an abstraction. pgvector is the migration path once the corpus or traffic grows; it is not needed yet.

## Results

The interesting part is what happens to retrieval quality when you change what gets embedded.

The baseline embeds each component using its name plus its category label and description. The problem: every component in a category shares that same category text, so within a category the only thing distinguishing "Pulse Dots" from "Rotating Squares" is two words in the name. The ranking inside a category suffers.

The enriched version adds a signal the baseline never had: the component's own `cssInline`, extracted straight from the source catalog. The CSS of Pulse Dots says `border-radius: 50%` with pulse keyframes; the CSS of Rotating Squares does not. That difference is what lets the model tell them apart. No LLM is involved in the enrichment, so rebuilding the index costs nothing beyond the embedding calls and has no external dependency other than Voyage.

Measured on a golden set of 45 queries:

| Metric | Baseline (name + category) | Enriched (+ CSS signal) |
|---|---|---|
| hit@1 | 66.7% | 80.0% |
| hit@3 | 86.7% | 91.1% |
| hit@5 | 93.3% | 95.6% |
| recall@5 | 54.9% | 59.4% |

hit@k is the fraction of queries where at least one correct component appears in the top k. recall@5 is the fraction of all acceptable answers that land in the top 5.

The headline is hit@1, up 13 points. That was the baseline's weak spot: it usually put a good answer somewhere in the top 5, but often not first. The CSS signal fixed the fine-grained ranking. Two categories that failed because components inside them looked identical to the retriever, loaders and skeletons, went from 67% to 100% on hit@5.

The enrichment is not free of trade-offs, and the eval caught it. Text effects regressed from 100% to 67%. Many text-effect components share CSS properties (`filter`, `opacity`, transitions) while doing visually distinct things, so adding CSS pulled some of them together instead of apart. The honest read is that the CSS signal lifts the average and fixes categories where the distinction is stylistic, but adds noise where the distinction is semantic. That is a measured trade-off, not a win to wave around.

One query fails in both versions ("menu with pill-style tabs"). A failure that survives an improvement points somewhere other than the signal it added, which makes it the natural target for a future reranking pass.

## Evaluation setup

The golden set is 45 queries spread across 15 categories, a mix of literal ("striped progress bar") and conceptual ("something to show it's loading without a percentage"). Many queries are written in Spanish against an English catalog, so the numbers also reflect cross-lingual retrieval.

Ground truth was labeled by deciding what a user would accept for each query, then finding those component IDs from the catalog, never by copying what the search returned. Labeling against the system's own output would measure the system against itself. The eval script validates every labeled ID against the catalog before scoring, so a typo or a stale ID cannot silently deflate the result.

The baseline number is frozen. Each change is a separate embeddings file measured with the same golden set, so a regression in one category is visible instead of being averaged away.

## Stack

Python, no RAG frameworks. Voyage AI (`voyage-4-lite`) for embeddings, which at this scale stays inside the free tier. Vectors stored as JSON on disk. Similarity by hand.

## Running it

```bash
pip install voyageai
export VOYAGE_API_KEY="your-key"

# baseline index (name + category)
python ingest.py            # reads catalog.json, writes embeddings.json

# enriched index (adds CSS signal)
python ingest_enriched.py   # reads catalog-full.json, writes embeddings_enriched.json

# search
python search.py "a loader with pulsing dots"

# evaluate and compare
python eval.py                          # baseline
python eval.py embeddings_enriched.json # enriched
```

`catalog.json` and `catalog-full.json` come from the NudaUI API (`/api/catalog.json` and `/api/catalog-full.json`). `list_category.py` prints the components in a category, which is how the golden set was labeled.

## Status

Done: ingestion, semantic search, a 45-query golden set, and a measured baseline-versus-enriched comparison.

Not done yet: a reranking pass to attack the text-effects noise and the pill-menu miss, migration to pgvector, an HTTP API around the retriever, and a web-facing search UI on NudaUI.

## Security

API keys live in environment variables or a `.env` file that is gitignored. Nothing secret is committed. If a key ever lands in a commit, rotate it, since git history keeps it even after deletion.

## License

MIT. The NudaUI catalog is MIT-licensed and authored by the same person as this project.