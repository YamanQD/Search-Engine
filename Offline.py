import string
from typing import List

import unicodedata
import re
import pandas as pd
import numpy as np
import contractions

from num2words import num2words
from text_to_num import alpha2digit

from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer



corpus = {
	"11": "63rd, On twenty-third, I'm used one hundred fifty six to being 1st first boat second third one two three in history class. Especially in nineteen fifty one, and that's why I'd won't don't cannot do it with my eyes closed!",
	"22": "it was used in landing craft during world war ii and is used today in private boats and training facilities the 6 71 is an inline six cylinder diesel engine the 71 refers to the displacement in cubic inches of each cylinder the firing order of the engine is 1 5 3 6 2 4 the engine s compression ratio is 18 7 1 with a 4 250 inch bore and a 5 00 inch stroke the engine weighs and is 54 inches long 29 inches wide and 41 inches tall at 2 100 revolutions per minute the engine is capable of producing 230 horse power 172 kilowatts v type versions of the 71 series were developed in 1957 the 6 71 is a two stroke engine as the engine will not naturally aspirate air is provided via a roots type blower however on the 6 71t models a turbocharger and a supercharger are utilized fuel is provided by unit injectors one per cylinder the amount of fuel injected into the engine is controlled by the engine s governor the engine cooling is via liquid in a water jacket in a boat cool external water is pumped into the engine",
	"33": "after rejecting an offer from cambridge university she moved to london in 1954 working in a mayfair advertising agency while moonlighting as a hat check girl in the night club le club contemporain while working at the royal college of art she met the painter frank bowling when he was still a student there they married in 1960 and had one son kitchen was one of the women interviewed by nell dunn in talking to women 1965 after divorcing bowling in the late sixties kitchen went on to live with and later marry the writer dulan barber continuing to write novels she also began writing non fiction with biographies of patrick geddes and gerard manley hopkins in later life she bought a house in barnwell northamptonshire which became the subject of her book of the same name she died on 23 november 2005 the novelist bessie head was a close friend the pair corresponded from 1969 until head s death in 1986 on a range of subjects including head s novel a question of power",
	"44": "mat zan coached kuala lumpur fa in 1999 and won malaysia fa cup in 1999 and charity cup in 2000 in 2001 mat zan then move as head coach of terengganu fa and he won the 2001 malaysia cup 2001 malaysia charity cup and second place in malaysia league with plus f c in 2008 he took 2nd place in premier league and automatically promoted to malaysia super league in 2012 he comeback with terengganu fa and go through to fa cup semi finals and round two in afc cup in 2015 mat zan move to melaka united and won 2015 malaysia fam league and won 2016 malaysia premier league after his contract was not renewed with melaka at the end of 2016 he was appointed as new head coach of airasia allstars fc which was renamed as petaling jaya rangers f c for the 2017 season he signed a 1 year contract with negeri sembilan form the 2019 season mat zan also coached the club side of malaysia u 21 squad harimau muda b who won plate winner singapore starhub league cup thanj nien newspaper cup vietnam 3rd place and hassanal bolkiah cup brunei 3rd place"
}

class TextProcessing:
	@classmethod
	def process(cls, txt: str):
		# Convert to lowercase
		txt = txt.lower()

	    # Remove all accented characters from a string, eg: é, Á. ó
		txt = TextProcessing.remove_accented_chars(txt)

		# Expand contractions, eg: "cannot, I'm" => "can not, I am"
		txt = TextProcessing.expand_contractions(txt)

		# Separates numbers from words or other characters
		txt = TextProcessing.sep_num_words(txt)

		# Convert words to numbers
		txt = TextProcessing.words_to_num(txt)

		# Remove punctuations
		txt = txt.translate(str.maketrans('', '', string.punctuation))

		# Tokenize into words
		words = word_tokenize(txt)

		# Correct word spelling
		# words = TextProcessing.correct_spelling(words)
		
		# Remove stopwords
		words = TextProcessing.remove_stopwords(words)
		
		# Lemmatize words
		words = TextProcessing.lemmatize(words)

		# Stem words
		# words = TextProcessing.stem(words)

		# Merge words list into a string
		txt = ' '.join(words)

	    # Remove extra whitespaces
		txt = TextProcessing.remove_extra_whitespaces(txt)

		return txt


	@classmethod
	def remove_stopwords(cls, words: List[str]) -> List[str]:
		new_words = []
		
		for word in words:
			if word not in stopwords.words('english'):
				new_words.append(word)
		
		return new_words
	
    # Remove all accented characters from a string, eg: é, Á. ó
	@classmethod
	def remove_accented_chars(cls, text):
		return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
	
	@classmethod
	def remove_extra_whitespaces(cls, text):
		return re.sub(r'^\s*|\s\s*', ' ', text).strip()

	# eg: "cannot, I'm" => "can not, I am"
	@classmethod
	def expand_contractions(cls, text):
		return contractions.fix(text)
	
	# Separates numbers from words or other characters, excluding ordinal numbers.
	@classmethod
	def sep_num_words(cls, text):
		return re.sub(r"(?<!\d)(\d+)(st|nd|rd|th)", r" \1\2", text).strip()

	# Convert words to numbers
	@classmethod
	def words_to_num(cls, text):
		return alpha2digit(text, "en", True, ordinal_threshold=0)

	# Convert Numbers to Words
	@classmethod
	def num_to_words(cls, text):
		after_spliting = text.split()

		for index in range(len(after_spliting)):
			if after_spliting[index].isdigit():
				after_spliting[index] = num2words(after_spliting[index])
		numbers_to_words = ' '.join(after_spliting)
		return numbers_to_words

	@classmethod
	def correct_spelling(cls, tokens: List[str]) -> List[str]:
		spell = SpellChecker()
		misspelled = spell.unknown(tokens)
		for i, token in enumerate(tokens):
			if token in misspelled:
				corrected = spell.correction(token)
				if corrected is not None:
					tokens[i] = corrected
		return tokens

	@classmethod
	def lemmatize(cls, words: List[str]) -> List[str]:
		def get_wordnet_pos(tag_parameter):

			tag = tag_parameter[0].upper()
			tag_dict = {"J": wordnet.ADJ,
						"N": wordnet.NOUN,
						"V": wordnet.VERB,
						"R": wordnet.ADV}
			
			return tag_dict.get(tag, wordnet.NOUN)
		
		# POS tagging
		pos_tags = pos_tag(words)

		lemmatizer = WordNetLemmatizer()
		lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]

		return lemmatized_words

	@classmethod
	def stem(cls, words: List[str]) -> List[str]:
		stemmer = PorterStemmer()
		stemmed_words = [stemmer.stem(word) for word in words]
		return stemmed_words


# Print corpus after text processing
# new_corpus = {}
# for key, value in corpus.items():
# 	value = TextProcessing.process(value)
# 	new_corpus[key] = value

# print(new_corpus)


# Create VSM Index
vectorizer = TfidfVectorizer(preprocessor=TextProcessing.process, tokenizer=word_tokenize)
tfidf_matrix = vectorizer.fit_transform(corpus.values())
df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=corpus.keys())
# print(df.to_string(max_cols=10))


# Transform query to VSM
query = ["boats"]
queryVector = vectorizer.transform(query)

# Calculate cosine similarity
cosine_scores = cosine_similarity(queryVector, tfidf_matrix)

#TODO: Remove unrelevant docs

# Get the indices of the documents sorted by their cosine similarity score in descending order
sorted_indices = np.argsort(cosine_scores.flatten())[::-1]

# Create a list of corpus keys
keys_list = list(corpus.keys())

# Print the top N most relevant documents
N = 2
for idx in sorted_indices[:N]:
    doc_id = keys_list[idx]
    print(f"\nDocument ID {doc_id}:\n{corpus[doc_id]}\n")
