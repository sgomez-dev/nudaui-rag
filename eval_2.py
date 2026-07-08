"""
Escalon 3/4: evaluacion. Ahora recibe QUE embeddings medir, para comparar
baseline vs enriquecido con el mismo golden dataset.

Uso:
  python eval.py                          # mide embeddings.json (baseline)
  python eval.py embeddings_enriched.json # mide el enriquecido

Valida los relevant_ids contra el archivo elegido y calcula hit@1/3/5 y recall@5,
con desglose por categoria y lista de fallos.
"""

import json
import os
import sys

import voyageai

GOLDEN_PATH = "golden_dataset.json"
MODEL = "voyage-4-lite"
TOP_K = 5


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rank_ids(query_vector, records, top_k):
    scored = [(cosine_similarity(query_vector, r["embedding"]), r["id"]) for r in records]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [rid for _, rid in scored[:top_k]]


def main():
    embeddings_path = sys.argv[1] if len(sys.argv) > 1 else "embeddings.json"

    if not os.environ.get("VOYAGE_API_KEY"):
        sys.exit("Falta VOYAGE_API_KEY.")
    for path in (embeddings_path, GOLDEN_PATH):
        if not os.path.exists(path):
            sys.exit(f"No encuentro {path}.")

    with open(embeddings_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        gold = json.load(f)

    print(f"\nMidiendo: {embeddings_path}  ({len(records)} componentes)\n")

    valid_ids = {r["id"] for r in records}
    queries = gold["queries"]

    total_unknown = 0
    usable = []
    for q in queries:
        relevant = q.get("relevant_ids", [])
        known = [rid for rid in relevant if rid in valid_ids]
        unknown = [rid for rid in relevant if rid not in valid_ids]
        if unknown:
            total_unknown += len(unknown)
            print(f"  {q['id']}  ids inexistentes: {', '.join(unknown)}")
        q["_known"] = known
        if known:
            usable.append(q)

    if total_unknown:
        print(f"\nIds inexistentes ignorados: {total_unknown}")
    print(f"Busquedas usables: {len(usable)} de {len(queries)}")

    if not usable:
        sys.exit("No hay busquedas usables.")

    client = voyageai.Client()
    texts = [q["query"] for q in usable]
    embeddings = client.embed(texts, model=MODEL, input_type="query").embeddings

    hits = {1: 0, 3: 0, 5: 0}
    recall_sum = 0.0
    per_cat = {}
    misses = []

    for q, qvec in zip(usable, embeddings):
        top = rank_ids(qvec, records, TOP_K)
        relevant = set(q["_known"])
        got = {k: any(rid in relevant for rid in top[:k]) for k in (1, 3, 5)}
        for k in (1, 3, 5):
            hits[k] += int(got[k])
        recall_sum += len(relevant.intersection(top)) / len(relevant)
        cat = q.get("expected_category", "?")
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][0] += int(got[5])
        per_cat[cat][1] += 1
        if not got[5]:
            misses.append((q, top))

    n = len(usable)
    print("=" * 60)
    print(f"RESULTADOS ({n} busquedas)")
    print("=" * 60)
    print(f"  hit@1:    {hits[1] / n:.1%}")
    print(f"  hit@3:    {hits[3] / n:.1%}")
    print(f"  hit@5:    {hits[5] / n:.1%}")
    print(f"  recall@5: {recall_sum / n:.1%}")

    print("\nhit@5 por categoria:")
    for cat in sorted(per_cat):
        got, tot = per_cat[cat]
        print(f"  {cat:20} {got}/{tot}  ({got / tot:.0%})")

    if misses:
        print(f"\nBusquedas sin acierto en top 5 ({len(misses)}):")
        for q, top in misses:
            print(f"\n  {q['id']}: \"{q['query']}\"")
            print(f"    esperaba: {', '.join(q['_known'])}")
            print(f"    devolvio: {', '.join(top)}")


if __name__ == "__main__":
    main()