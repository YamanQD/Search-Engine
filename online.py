import pickle
import itertools
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_processing import TextProcessing


# Load vectorizer from a file
with open('vectorizer01.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

# Load dataset index
tfidf_matrix = load_npz('wikir_index01.npz')

# Load the csv file
df = pd.read_csv('Datasets/wikIR1k/documents.csv')

# Convert the dataframe to a dictionary
corpus = df.set_index('id_right')['text_right'].to_dict()

# Keep the first 1000 documents and delete the rest
# corpus = dict(itertools.islice(corpus.items(), 1000))

# df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=corpus.keys())
# print(df.to_string(max_cols=10))

class SearchEngine:
    @classmethod
    def search(cls, query: str, count=10):
        # Transform query to VSM
        query = [query]
        queryVector = vectorizer.transform(query)

        # Calculate cosine similarity
        cosine_scores = cosine_similarity(queryVector, tfidf_matrix)

        #TODO: Remove low score unrelevant docs

        # Get the indices of the documents sorted by their cosine similarity score in descending order
        sorted_indices = np.argsort(cosine_scores.flatten())[::-1]

        # Create a list of corpus keys
        keys_list = list(corpus.keys())

        # Return the top [count] relevant documents
        docs = {}
        for idx in sorted_indices[:count]:
            doc_id = keys_list[idx]
            docs[doc_id] = corpus[doc_id]
        
        return docs