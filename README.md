# nudaui-rag

Semantic search (RAG) over the [NudaUI](https://nudaui.dev) component catalog. You describe what you want in plain language ("loader with dots", "button that glows on hover") and the system returns the catalog components that best match, with name, category, similarity score and link.

The project exists for two reasons. First, to demonstrate production-quality RAG work rather than the tutorial version of it. Second, as the seed of a future search feature for NudaUI itself.

## The corpus

NudaUI is a copy-paste CSS/JS animation library: zero dependencies, no npm install, no build step, framework-agnostic, accessibility-first, MIT licensed. It ships 1022 components across 81 categories and exposes a structured, agent-friendly catalog at `https://nudaui.dev/api/catalog.json` that AI agents can query directly.

## How it works

Two-phase RAG.

**Phase 1, indexing** (`ingest.py`, run once). Reads the catalog and, for each component, builds an enriched text combining the component name with the label and description of its category. That enrichment is the key move: names alone ("Pulse Dots", "Wave Bars") are barely distinctive, and everything within a category blurs together. Injecting the category context separates them. It's a lightweight version of the contextual embeddings idea. Each text is embedded with Voyage using `input_type="document"` and stored with its metadata in a local `embeddings.json`.

**Phase 2, search** (`search.py`, per query). The user's phrase is embedded with the same Voyage model but with `input_type="query"`, then compared against every stored vector by cosine similarity. Top-k results come back with their scores.

## Design decisions worth mentioning

Cosine similarity is implemented by hand, on purpose. No LangChain, no RAG framework. The point is to understand and be able to explain exactly what happens inside.

Color doesn't exist in the catalog data; it's a CSS variable the user changes. So a query like "yellow loader" has to understand "loader" and treat "yellow" as soft intent at most. There is no color filtering, and there can't be.

The catalog contains duplicated or near-identical components (several "Table Skeleton" variants, "DNA Helix" in more than one category). This has a real consequence for evaluation: the ground truth for each query must be a set of acceptable answers, not a single correct one.

## Stack

Python. Voyage AI for embeddings (`voyage-4-lite`; the free tier of 200M tokens is far more than enough for ~1000 components). Storage is a local `embeddings.json` for now. No RAG frameworks.

## Usage

```bash
pip install voyageai

export VOYAGE_API_KEY=your_key_here

# index the catalog (once)
python ingest.py

# search
python search.py "loader with dots"
python search.py "button that glows on hover"
```

API keys live in environment variables or a `.env` file kept out of version control. Never commit them.

## Current status

Done: full ingestion (all 1022 components embedded and stored) and working semantic search from the command line.

Not done yet: evaluation dataset (golden set) and recall metrics, reranking, migration to pgvector, answer generation with Claude, web interface.

There are no performance numbers here because there are no evals yet. Placeholder until then: [pending: recall@5 baseline].

## Roadmap

1. Ingestion. Done.
2. Semantic search. Done.
3. Evaluation: golden dataset and a recall baseline. [pending: recall@5 baseline]
4. Improvement: contextual embeddings built from the component's actual code, and/or reranking, measuring before and after.
5. API.

Further out: pgvector, generation with Claude, and a web search experience integrated into NudaUI.
