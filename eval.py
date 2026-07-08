"""
Escalon 3 del RAG de NudaUI: evaluacion del baseline.

Hace dos cosas, en orden:
  1. VALIDA: revisa que cada relevant_id del golden dataset exista de verdad
     en embeddings.json. Los que no existan se reportan y se ignoran, porque
     un id inexistente es imposible de acertar y ensuciaria el numero.
  2. MIDE: corre cada busqueda, mira el top 5, y calcula:
       - hit@1, hit@3, hit@5: en que fraccion de busquedas aparece al menos
         una respuesta correcta entre los primeros 1 / 3 / 5 resultados.
       - recall@5: de las respuestas correctas, que fraccion aparece en el top 5.
     Ademas desglosa hit@5 por categoria y lista las busquedas que fallaron.

Uso:
  python eval.py
"""

import json
import os
import sys

import voyageai

EMBEDDINGS_PATH = "embeddings.json"
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
    if not os.environ.get("VOYAGE_API_KEY"):
        sys.exit("Falta VOYAGE_API_KEY.")
    for path in (EMBEDDINGS_PATH, GOLDEN_PATH):
        if not os.path.exists(path):
            sys.exit(f"No encuentro {path}.")

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        gold = json.load(f)

    valid_ids = {r["id"] for r in records}
    queries = gold["queries"]

    # ---- Paso 1: validacion ----
    print("=" * 60)
    print("VALIDACION de relevant_ids contra el catalogo")
    print("=" * 60)

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

    print(f"\nIds inexistentes en total: {total_unknown}")
    print(f"Busquedas usables (con al menos 1 id valido): {len(usable)} de {len(queries)}")
    if total_unknown:
        print("\nOJO: corrige o quita esos ids con list_category.py para que el numero sea limpio.")
        print("Por ahora se ignoran y se mide solo con los ids validos.\n")

    if not usable:
        sys.exit("No hay busquedas usables. Revisa el golden dataset.")

    # ---- Paso 2: medicion ----
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

        got1 = any(rid in relevant for rid in top[:1])
        got3 = any(rid in relevant for rid in top[:3])
        got5 = any(rid in relevant for rid in top[:5])
        hits[1] += int(got1)
        hits[3] += int(got3)
        hits[5] += int(got5)

        found = len(relevant.intersection(top))
        recall_sum += found / len(relevant)

        cat = q.get("expected_category", "?")
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][0] += int(got5)
        per_cat[cat][1] += 1

        if not got5:
            misses.append((q, top))

    n = len(usable)
    print("=" * 60)
    print(f"RESULTADOS (baseline, {n} busquedas)")
    print("=" * 60)
    print(f"  hit@1:    {hits[1] / n:.1%}   (respuesta correcta en el 1er resultado)")
    print(f"  hit@3:    {hits[3] / n:.1%}   (correcta entre los 3 primeros)")
    print(f"  hit@5:    {hits[5] / n:.1%}   (correcta entre los 5 primeros)")
    print(f"  recall@5: {recall_sum / n:.1%}   (fraccion de correctas dentro del top 5)")

    print("\nhit@5 por categoria:")
    for cat in sorted(per_cat):
        got, tot = per_cat[cat]
        print(f"  {cat:20} {got}/{tot}  ({got / tot:.0%})")

    if misses:
        print(f"\nBusquedas donde NO acerto ninguna en el top 5 ({len(misses)}):")
        for q, top in misses:
            print(f"\n  {q['id']}: \"{q['query']}\"")
            print(f"    esperaba: {', '.join(q['_known'])}")
            print(f"    devolvio: {', '.join(top)}")


if __name__ == "__main__":
    main()