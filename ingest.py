"""
Escalon 1 del RAG de NudaUI: ingesta.
 
Que hace:
  1. Lee catalog.json (guardalo en la misma carpeta que este script).
  2. Por cada componente arma un texto enriquecido con el nombre + su categoria.
     Esto mete el contexto de la categoria en cada componente, que es lo que
     hace que "Pulse Dots" y "Wave Bars" dejen de parecer lo mismo al buscador.
  3. Genera un embedding por componente con Voyage.
  4. Guarda todo (metadatos + vector) en embeddings.json.
 
Que NO hace todavia: buscar, usar Claude, levantar un servidor. Eso es despues.
"""
import json
import os
import sys
import time
import voyageai

CATALOG_PATH = "catalog.json"
OUTPUT_PATH = "embeddings.json"
MODEL = "voyage-4-lite"
BATCH_SIZE = 100

def build_text(component, category):
    """Texto que de verdad se embebe: nombre del componente + contexto de su categoría."""
    name = component["name"]
    label = category["label"]
    description = category["description"]
    return f"{name}. Categoría: {label}. {description}"

def load_components(catalog_path):
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    records = []
    for category in catalog["categories"]:
        for component in category.get("components", []):
            records.append({
                "id": component["id"],
                "name": component["name"],
                "category_id": category["id"],
                "category_label": category["label"],
                "anchor": component.get("anchor"),
                "languages": component.get("languages", []),
                "hasJS": component.get("hasJS", False),
                "text": build_text(component, category),
            })
    return records

def embed_batches(client, texts, batch_size):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        result = client.embed(batch, model=MODEL, input_type="document")
        vectors.extend(result.embeddings)
        done = min(start + batch_size, len(texts))
        print(f" embebidos {done} de {len(texts)}")
        time.sleep(0.2)
    return vectors

def main():
    if not os.environ.get("VOYAGE_API_KEY"):
        sys.exit("Falta VOYAGE_API_KEY. Exportala antes de correr el script.")
    if not os.path.exists(CATALOG_PATH):
        sys.exit(f"Archivo catalog.json no encontrado en {os.path.abspath(CATALOG_PATH)}")
    
    records = load_components(CATALOG_PATH)
    print(f"Cargados {len(records)} componentes")

    client = voyageai.Client()
    texts = [r["text"] for r in records]
    print(f"Generando embeddings para {len(texts)} componentes")
    vectors = embed_batches(client, texts, BATCH_SIZE)

    for record, vector in zip(records, vectors):
        record["embedding"] = vector
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)

    dim = len(vectors[0]) if vectors else 0
    print(f"Listo. Guardados {len(records)} componentes en {OUTPUT_PATH}. Dimensión: {dim}")

if __name__ == "__main__":
    main()