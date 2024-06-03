import pickle
import numpy as np
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.text_processing import TextProcessing


class MatchingRanking:
    @classmethod
    def search(cls, query: str, corpus, tfidf_matrix, vectorizer, count=10):
        # Transform query to VSM
        query = [query]
        queryVector = vectorizer.transform(query)

        # Calculate cosine similarity
        cosine_scores = cosine_similarity(queryVector, tfidf_matrix)

        # Get the indices of the documents sorted by their cosine similarity score in descending order
        sorted_indices = np.argsort(cosine_scores.flatten())[::-1]

        # Create a list of corpus keys
        keys_list = list(corpus.keys())

        # Return the top [count] relevant documents
        docs = {}
        for idx in sorted_indices[:count]:
            # Check if the cosine score is zero
            if cosine_scores[0][idx] == 0: break

            doc_id = keys_list[idx]
            docs[doc_id] = corpus[doc_id]
        
        return docs

    @classmethod
    def load_index(cls, dataset, version="00"):
        with open(f'Datasets/{dataset}/index/vectorizer{version}.pickle', 'rb') as f:
            vectorizer = pickle.load(f)

        tfidf_matrix = load_npz(f'Datasets/{dataset}/index/index{version}.npz')
        return tfidf_matrix, vectorizer