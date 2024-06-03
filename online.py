from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pandas as pd

from services.matching_ranking import MatchingRanking

app = FastAPI()

# 127.0.0.1:8000/static/index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/search")
async def search(q: str, d: str):
        if d == "wikir":
             return MatchingRanking.search(q, wikir_corpus, wikir_tfidf_matrix, wikir_vectorizer, 10)
        elif d == "antique":
             return MatchingRanking.search(q, antique_corpus, antique_tfidf_matrix, antique_vectorizer, 10)
        else:
            return {"error": 'Dataset must be either "wikir" or "antique".'}
    

# Load dataset indices and models
wikir_tfidf_matrix, wikir_vectorizer = MatchingRanking.load_index("wikIR1k", "01")
antique_tfidf_matrix, antique_vectorizer = MatchingRanking.load_index("antique", "02")

# Load datasets
df = pd.read_csv('Datasets/wikIR1k/documents.csv')
wikir_corpus = df.set_index('id_right')['text_right'].to_dict()

antique_corpus = {}
with open("Datasets/antique/antique-collection.txt", 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            identifier, text = line.split('\t', 1)
            antique_corpus[identifier] = text
