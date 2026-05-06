import json
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import subprocess
import requests

try:
    requests.get("http://localhost:11434")
except:
    subprocess.Popen(["ollama", "serve"])

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "netflix_titles.csv"
INDEX_PATH = Path(r"C:\Users\sugho\OneDrive\Desktop\ML practice\Movie recomendation\Index")
META_PATH = BASE_DIR / "metadata.json"
EMBED_MODEL = "nomic-embed-text"
EMBED_ENDPOINT = "http://localhost:11434/api/embeddings"

app = FastAPI(title="Movie Recommendation Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def txt_rep(row: dict) -> str:
    return (
        f"Type: {row.get('type', '')},\n"
        f"Title: {row.get('title', '')},\n"
        f"Director: {row.get('director', '')},\n"
        f"Cast: {row.get('cast', '')},\n"
        f"Released: {row.get('release_year', '')},\n"
        f"Genres: {row.get('listed_in', '')},\n"
        f"Description: {row.get('description', '')}"
    )


def embed_text(text: str) -> np.ndarray:
    try:
        response = requests.post(
            EMBED_ENDPOINT,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if embedding is None:
            raise ValueError("Embedding endpoint did not return an embedding")
        return np.array(embedding, dtype="float32")
    except Exception as exc:
        print("Warning: embedding failed, using fallback random vector.", exc)
        return np.random.rand(512).astype("float32")


def load_index() -> tuple[list[dict], faiss.IndexFlatL2]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index file not found at {INDEX_PATH}.\n"
            "This backend is configured to use the existing index file only."
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {META_PATH}.\n"
            "The existing index file requires metadata.json to map search results back to records."
        )

    try:
        with META_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        index = faiss.read_index(str(INDEX_PATH))
        print("Loaded existing FAISS index and metadata.")
        return metadata, index
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load existing index or metadata from {INDEX_PATH} / {META_PATH}."
        ) from exc


metadata, index = load_index()


def get_recommendations(query_embedding: np.ndarray, user_title: str = "", top_k: int = 10) -> list[dict]:
    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, top_k + 1)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        if user_title and item.get("title", "").lower() == user_title.lower():
            continue
        item["score"] = float(dist)
        results.append(item)
        if len(results) == top_k:
            break
    return results


@app.get("/", response_class=HTMLResponse)
async def read_frontend() -> HTMLResponse:
    return HTMLResponse(Path(BASE_DIR / "frontend.html").read_text(encoding="utf-8"))


@app.get("/styles.css")
async def styles() -> FileResponse:
    return FileResponse(BASE_DIR / "styles.css")


@app.post("/recommend")
async def recommend(request: Request) -> JSONResponse:
    data = await request.json()
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    user_input = {
        "type": data.get("type", ""),
        "title": data.get("title", ""),
        "director": data.get("director", ""),
        "cast": data.get("cast", ""),
        "release_year": data.get("release_year", ""),
        "listed_in": data.get("genres", ""),
        "description": data.get("description", ""),
    }

    query = txt_rep(user_input)
    query_embedding = embed_text(query)
    recommendations = get_recommendations(query_embedding, user_title=user_input.get("title", ""), top_k=10)
    return JSONResponse({"recommendations": recommendations})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
