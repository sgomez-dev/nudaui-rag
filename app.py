"""
Escalon 5 del RAG de NudaUI: el API.
 
Envuelve la misma logica de search.py en un servidor HTTP. Los embeddings
y el cliente de Voyage se cargan UNA vez al arrancar, no por peticion.
 
Rutas:
  GET /health            -> estado del servicio y cuantos componentes cargo
  GET /search?q=...&k=5  -> los k componentes mas parecidos a la consulta
 
Correr:
  pip install fastapi uvicorn voyageai
  export VOYAGE_API_KEY="tu-key"
  uvicorn app:app --reload
 
Probar:
  navegador -> http://127.0.0.1:8000/docs   (UI para lanzar busquedas)
  o         -> http://127.0.0.1:8000/search?q=loader con puntos
"""

import json
import os
from contextlib import asynccontextmanager
import voyageai
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

EMBEDDINGS_FILE = os.environ.get("EMBEDDINGS_FILE", "embeddings_enriched.json")
MODEL = "voyage-4-lite"
DEFAULT_K = 5
MAX_K = 20

# Estado del servicio, poblado al arrancar
state = {"records": [], "client": None}

def load_records(path):
    """Carga los embeddings y precalcula la norma de cada vector (para el coseno)"""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        vec = r["embedding"]
        r["_norm"] = sum (x * x for x in vec) ** 0.5
    return records

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("VOYAGE_API_KEY"):
        raise RuntimeError("Falta VOYAGE_API_KEY")
    if not os.path.exists(EMBEDDINGS_FILE):
        raise RuntimeError(f"No se encuentra {EMBEDDINGS_FILE}. Correr la ingesta primero.")
    state["records"] = load_records(EMBEDDINGS_FILE)
    state["client"] = voyageai.Client()
    yield
    state["records"] = []
    state["client"] = None

app = FastAPI(title="NudaUI Semantic Search", lifespan=lifespan)

# Para que la web pueda llamar al API desde el navegador.
# En produccion conviene restringir allow_origins a ["https://nudaui.dev"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

def cosine_with_norm(query_vec, query_norm, record):
    dot = sum(x * y for x, y in zip(query_vec, record["embedding"]))
    denom = query_norm * record["_norm"]
    return dot / denom if denom else 0.0

@app.get("/health")
def health():
    return{
        "status": "ok",
        "components": len(state["records"]),
        "model": MODEL,
        "source": EMBEDDINGS_FILE,
    }

@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Descripcion en lenguaje natural"),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
):
    client = state["client"]
    records = state["records"]
    if not records:
        raise HTTPException(status_code=503, detail="Indice no cargado.")

    try:
        result = client.embed([q], model=MODEL, input_type="query")
        query_vec = result.embeddings[0]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Fallo al embeder la consulta: {exc}")

    query_norm = sum(x * x for x in query_vec) ** 0.5
    scored = [(cosine_with_norm(query_vec, query_norm, r), r) for r in records]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = [
        {
            "id": r["id"],
            "name": r["name"],
            "category": r["category_label"],
            "anchor": r.get("anchor"),
            "score": round(score, 4),
        }
        for score, r in scored[:k]
    ]
    return {"query": q, "count": len(results), "results": results}