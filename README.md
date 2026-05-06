# Movie Recommendation Engine

A semantic search-based movie recommendation system that finds similar Netflix titles using text embeddings and FAISS vector search.

## Features

- **Semantic Search**: Uses text embeddings to understand movie descriptions, genres, cast, and other metadata
- **Fast Recommendations**: Pre-built FAISS index enables instant similarity search across 8,800+ Netflix titles
- **Web Interface**: Clean HTML frontend for easy movie input and recommendation viewing
- **Exclusion Logic**: Automatically filters out the exact title match from recommendations
- **Ollama Integration**: Leverages local Ollama embeddings for privacy and offline capability

## Architecture

- **Backend** (`main.py`): FastAPI server that loads FAISS index and metadata, handles embedding requests, and serves recommendations
- **Frontend** (`frontend.html` + `styles.css`): Responsive web form for movie input with styled recommendation cards
- **Data**: 
  - `Index`: Pre-computed FAISS vector index (27MB)
  - `metadata.json`: Movie metadata for 8,807 Netflix titles
  - `netflix_titles.csv`: Original Kaggle dataset

## Setup & Usage

1. **Prerequisites**:
   - Python 3.8+
   - Ollama installed with `nomic-embed-text` model: `ollama pull nomic-embed-text`
   - Required packages: `pip install faiss-cpu fastapi uvicorn pandas numpy requests`

2. **Start the Server**:
   ```bash
   python main.py
   ```

3. **Access the App**:
   - Open `http://localhost:8000` in your browser
   - Fill in movie details (title, type, director, cast, year, genres, description)
   - Click "Find Recommendations" to get top 10 similar Netflix titles

## How It Works

1. **Index Building** (one-time): The original notebook embeds all Netflix titles using Ollama and builds a FAISS L2 distance index
2. **Query Processing**: User input is formatted into text representation and embedded using the same model
3. **Similarity Search**: FAISS finds the 11 most similar vectors, excludes any exact title match, returns top 10
4. **Results**: Recommendations include title, type, year, genres, director, cast, and description

## Performance

- **Startup**: ~2-5 seconds (loads index and metadata)
- **Query**: ~1-2 seconds per recommendation (embedding + search)
- **Memory**: ~100MB RAM usage (index + metadata)

## Customization

- Modify `txt_rep()` in `main.py` to change how movie data is formatted for embedding
- Adjust `top_k` parameter to change number of recommendations
- Update CSS in `styles.css` for different visual styling
- Use different embedding models by changing `EMBED_MODEL` in `main.py`

## Data Source

Netflix titles dataset from Kaggle (https://www.kaggle.com/datasets/shivamb/netflix-shows) containing movies and TV shows with metadata like cast, director, genres, and descriptions.