import pickle
import pandas as pd

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

from text_processing import TextProcessing

class Indexing:
    @classmethod
    def generate_wikir_index(cls, version="00"):
        # Load the csv file
        df = pd.read_csv('Datasets/wikIR1k/documents.csv')
        
        # Convert the dataframe to a dictionary
        corpus = df.set_index('id_right')['text_right'].to_dict()
        
        # Create VSM Index
        vectorizer = TfidfVectorizer(preprocessor=TextProcessing.process, tokenizer=word_tokenize)
        tfidf_matrix = vectorizer.fit_transform(corpus.values())
        
        save_npz(f'Datasets/wikIR1k/index/{version}/wikir_index{version}.npz', tfidf_matrix)
        
        # Save vectorizer to a file
        with open(f'Datasets/wikIR1k/index/{version}/vectorizer{version}.pickle', 'wb') as f:
            pickle.dump(vectorizer, f)


    @classmethod
    def generate_antique_index(cls, version="00"):
        corpus = {}
        with open("Datasets/antique/antique-collection.txt", 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    identifier, text = line.split('\t', 1)
                    corpus[identifier] = text
        
        # Create VSM Index
        vectorizer = TfidfVectorizer(preprocessor=TextProcessing.process, tokenizer=word_tokenize)
        tfidf_matrix = vectorizer.fit_transform(corpus.values())
        
        save_npz(f'Datasets/antique/index/{version}/antique_index{version}.npz', tfidf_matrix)
        
        # Save vectorizer to a file
        with open(f'Datasets/antique/index/{version}/vectorizer{version}.pickle', 'wb') as f:
            pickle.dump(vectorizer, f)


# IndexingService.generate_antique_index(version="02")