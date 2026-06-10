# app/main.py
import pickle
import os
import yaml
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from app.pipeline import TextPreprocessor

# Load configuration values globally
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

artifacts = {}
preprocessors = {
    "stem": TextPreprocessor(mode="stem"),
    "lemmatize": TextPreprocessor(mode="lemmatize")
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== INITIALIZING CONTAINER LIFESPAN SETUP VIA CONFIG ===")
    domains = ["amazon", "sentiment140"]
    
    for domain in domains:
        model_path = f"models/{domain}_best_model.pkl"
        vec_path = f"models/{domain}_vectorizer.pkl"
        
        if os.path.exists(model_path) and os.path.exists(vec_path):
            with open(model_path, "rb") as f:
                artifacts[domain] = {"model": pickle.load(f)}
            with open(vec_path, "rb") as f:
                artifacts[domain]["vectorizer"] = pickle.load(f)
            print(f"-> Production assets successfully loaded for: {domain}")
        else:
            print(f"-> ERROR: Missing weights at {model_path} or {vec_path}")
            
    # Load Production Search Pool dynamically straight out of config.yaml
    artifacts["search_corpus"] = pd.DataFrame(config["search"]["static_corpus"])
    yield
    artifacts.clear()
    print("=== SHUTTING DOWN CONTAINER LIFESPAN ===")

app = FastAPI(title="Production NLP & Search Engine Service", lifespan=lifespan)

class PredictionInput(BaseModel):
    text: str
    domain: str = "amazon"
    preprocess_mode: str = "stem"

class SearchInput(BaseModel):
    query: str
    sentiment_filter: str
    top_k: int = config["search"]["top_k"] # Default fetched directly from config file

@app.post("/predict")
def predict_sentiment(data: PredictionInput):
    domain = data.domain.lower()
    mode = data.preprocess_mode.lower()
    
    if domain not in artifacts:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' artifacts not loaded.")
        
    vectorizer = artifacts[domain]["vectorizer"]
    model = artifacts[domain]["model"]
    preprocessor = preprocessors[mode]
    
    cleaned = preprocessor.process(data.text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    
    return {
        "raw_text": data.text,
        "domain": domain,
        "predicted_sentiment": prediction
    }

@app.post("/search")
def filtered_search(data: SearchInput):
    search_corpus = artifacts.get("search_corpus")
    if search_corpus is None:
        raise HTTPException(status_code=500, detail="Search dataset index missing.")
        
    filtered_df = search_corpus[search_corpus["sentiment"] == data.sentiment_filter].reset_index(drop=True)
    if filtered_df.empty:
        return {"results": []}
        
    preprocessor = preprocessors["lemmatize"]
    filtered_tokens = [preprocessor.process(doc).split() for doc in filtered_df["text"]]
    
    bm25_engine = BM25Okapi(filtered_tokens)
    query_tokens = preprocessor.process(data.query).split()
    
    filtered_df["score"] = bm25_engine.get_scores(query_tokens)
    top_results = filtered_df.sort_values(by="score", ascending=False).head(data.top_k)
    
    return {
        "query": data.query,
        "sentiment_filter": data.sentiment_filter,
        "results": top_results[["text", "score"]].to_dict(orient="records")
    }