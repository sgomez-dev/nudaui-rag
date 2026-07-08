"""
Escalon 2 del RAG de NudaUI: busqueda.

Lee los embeddings del escalon 1, embebe la consulta que le pasas,
y devuelve los componentes mas parecidos por similitud coseno.

Uso:
  python search.py "loader con puntos"
  python search.py "boton que brilla al pasar el mouse"
"""
import json
import os
import sys
import voyageai

EMBEDDINGS_PATH = "embeddings.json"
MODEL = "voyage-4-lite"
TOP_K = 5

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def search(query, records, client, top_k):
    result = client.embed([query], model=MODEL, input_type="query")
    query_vector = result.embeddings[0]

    scored = []
    for record in records:
        score = cosine_similarity(query_vector, record["embedding"])
        scored.append((score, record))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]

def main():
    if not os.environ.get("VOYAGE_API_KEY"):
        sys.exit("Falta VOYAGE_API_KEY. Exportala antes de correr el script.")
    if not os.path.exists(EMBEDDINGS_PATH):
        sys.exit(f"Archivo embeddings.json no encontrado en {os.path.abspath(EMBEDDINGS_PATH)}")

    query = sys.argv[1]
    records = load_records(EMBEDDINGS_PATH)
    client = voyageai.Client()

    results = search(query, records, client, TOP_K)

    print(f'\nBusqueda: "{query}"\n')
    for i, (score, record) in enumerate(results, 1):
        print(f"{i}. {record['name']} ({record['category_label']})")
        print(f"  score: {score:.4f}")
        print(f"  {record['anchor']}")
        print()

if __name__ == "__main__":
    main()