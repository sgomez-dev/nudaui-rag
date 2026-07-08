"""
Escalon 4 del RAG de NudaUI: ingesta ENRIQUECIDA (sin IA).
 
Diferencia con ingest.py (baseline):
  - Fuente: catalog-full.json (trae el codigo), no catalog.json.
  - Texto embebido: nombre + categoria + senal del CSS (cssInline).
    Ese CSS es lo que distingue "Pulse Dots" (border-radius:50%, keyframes de
    pulso) de "Rotating Squares", cosa que el baseline no podia ver.
  - Salida: embeddings_enriched.json (NUEVO). No pisa embeddings.json.
    Asi puedes comparar baseline vs enriquecido con eval.py.
 
Como conseguir catalog-full.json:
  Abre https://nudaui.dev/api/catalog-full.json en el navegador y guardalo
  como catalog-full.json en esta carpeta. (Pesa mas que el lite: trae codigo.)
"""
import json
import os
import re
import sys
import time
import voyageai

CATALOG_FULL_PATH = "catalog-full.json"
OUTPUT_PATH = "embeddings_enriched.json"
MODEL = "voyage-4-lite"
BATCH_SIZE = 100
CSS_CHAR_LIMIT = 600

def clean_css(css):
    """Compacta el cssInline: quita saltos y espacios repetidos. Es senal, no formato"""
    if not css:
        return ""
    css = re.sub(r"\s+", " ", css).strip()
    if len(css) > CSS_CHAR_LIMIT:
        css = css[:CSS_CHAR_LIMIT]
    return css

def build_text(component, category_label, category_description):
    """
    Texto enriquecido: nombre + categoria + descripcion de categoria + CSS.
    El CSS es la senal nueva respecto al baseline.
    """
    name = component["name"]
    css = clean_css(component.get("cssInline"))
    base = f"{name}. Categoria: {category_label}. {category_description}"
    if css:
        return f"{base} Estilos: {css}"
    return base

def load_components(path):
    with open(path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    records = []
    for category in catalog["categories"]:
        label = category["label"]
        description = category.get("description", "")
        for component in category.get("components", []):
            records.append({
                "id": component["id"],
                "name": component["name"],
                "category_id": category["id"],
                "category_label": label,
                "anchor": component.get("anchor"),
                "languages": component.get("languages", []),
                "hasJS": component.get("hasJS", False),
                "has_css": bool(component.get("cssInline")),
                "text": build_text(component, label, description),
            })
    return records

def embed_batches(client, texts, batch_size):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        result = client.embed(batch, model=MODEL, input_type="document")
        vectors.extend(result.embeddings)
        done = min(start + batch_size, len(texts))
        print(f"  embebidos {done} de {len(texts)}")
        time.sleep(0.2)
    return vectors

def main():
    if not os.environ.get("VOYAGE_API_KEY"):
        sys.exit("Falta VOYAGE_API_KEY")
    if not os.path.exists(CATALOG_FULL_PATH):
        sys.exit(f"No se encontró {CATALOG_FULL_PATH}. Guarda el catalogo full ahí primero.")

    records = load_components(CATALOG_FULL_PATH)
    con_css = sum(1 for r in records if r["has_css"])
    print(f"Componentes: {len(records)} | con cssInline: {con_css} | sin css: {len(records) - con_css}")

    client = voyageai.Client()
    texts = [r["text"] for r in records]
    print("Generando embeddings enriquecidos con Voyage...")
    vectors = embed_batches(client, texts, BATCH_SIZE)

    for record, vector in zip(records, vectors):
        record["embedding"] = vector

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)

    dim = len(vectors[0]) if vectors else 0
    print(f"Listo. {len(records)} componentes en {OUTPUT_PATH} (dim={dim}). Baseline intacto.")
 
 
if __name__ == "__main__":
    main()
 