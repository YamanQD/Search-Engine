import pickle
import itertools
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_processing import TextProcessing


# Load vectorizer from a file
with open('Datasets/wikIR1k/index/01/vectorizer01.pickle', 'rb') as f:
    wikir_vectorizer = pickle.load(f)

# Load dataset index
wikir_tfidf_matrix = load_npz('Datasets/wikIR1k/index/01/wikir_index01.npz')

# Load the csv file
df = pd.read_csv('Datasets/wikIR1k/documents.csv')

# Convert the dataframe to a dictionary
wikir_corpus = df.set_index('id_right')['text_right'].to_dict()



# Load vectorizer from a file
with open('Datasets/antique/index/01/vectorizer01.pickle', 'rb') as f:
    antique_vectorizer = pickle.load(f)

# Load dataset index
antique_tfidf_matrix = load_npz('Datasets/antique/index/01/antique_index01.npz')

antique_corpus = {}
with open("Datasets/antique/antique-collection.txt", 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            identifier, text = line.split('\t', 1)
            antique_corpus[identifier] = text


class SearchEngine:
    @classmethod
    def search(cls, dataset: str, query: str, count=10):
        if dataset == "wikir":
            vectorizer = wikir_vectorizer
            tfidf_matrix = wikir_tfidf_matrix
            corpus = wikir_corpus
        elif dataset == "antique":
            vectorizer = antique_vectorizer
            tfidf_matrix = antique_tfidf_matrix
            corpus = antique_corpus
        else:
            raise NameError("Dataset must be wikir or antique")

        # Transform query to VSM
        query = [query]
        queryVector = vectorizer.transform(query)

        # Calculate cosine similarity
        cosine_scores = cosine_similarity(queryVector, tfidf_matrix)

        #TODO: Remove zero score unrelevant docs

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

# SearchEngine.search("antique", "how sun rises?", 10)