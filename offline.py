import pickle
import itertools
import pandas as pd

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

from text_processing import TextProcessing

# Load the csv file
df = pd.read_csv('Datasets/wikIR1k/documents.csv')

# Convert the dataframe to a dictionary
corpus = df.set_index('id_right')['text_right'].to_dict()

# Keep the first 1000 documents and delete the rest
# corpus = dict(itertools.islice(corpus.items(), 1000))


# Print corpus after text processing
# new_corpus = {}
# for key, value in corpus.items():
# 	value = TextProcessing.process(value)
# 	new_corpus[key] = value
# print(new_corpus)


# Create VSM Index
vectorizer = TfidfVectorizer(preprocessor=TextProcessing.process, tokenizer=word_tokenize)
tfidf_matrix = vectorizer.fit_transform(corpus.values())

save_npz('wikir_index02.npz', tfidf_matrix)

# Save vectorizer to a file
with open('vectorizer02.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
