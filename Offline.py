import string
from typing import List

import pandas as pd

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from spellchecker import SpellChecker

corpus = {
	"1": "it was used in landing craft during world war ii and is used today in private boats and training facilities the 6 71 is an inline six cylinder diesel engine the 71 refers to the displacement in cubic inches of each cylinder the firing order of the engine is 1 5 3 6 2 4 the engine s compression ratio is 18 7 1 with a 4 250 inch bore and a 5 00 inch stroke the engine weighs and is 54 inches long 29 inches wide and 41 inches tall at 2 100 revolutions per minute the engine is capable of producing 230 horse power 172 kilowatts v type versions of the 71 series were developed in 1957 the 6 71 is a two stroke engine as the engine will not naturally aspirate air is provided via a roots type blower however on the 6 71t models a turbocharger and a supercharger are utilized fuel is provided by unit injectors one per cylinder the amount of fuel injected into the engine is controlled by the engine s governor the engine cooling is via liquid in a water jacket in a boat cool external water is pumped into the engine",
	"2": "after rejecting an offer from cambridge university she moved to london in 1954 working in a mayfair advertising agency while moonlighting as a hat check girl in the night club le club contemporain while working at the royal college of art she met the painter frank bowling when he was still a student there they married in 1960 and had one son kitchen was one of the women interviewed by nell dunn in talking to women 1965 after divorcing bowling in the late sixties kitchen went on to live with and later marry the writer dulan barber continuing to write novels she also began writing non fiction with biographies of patrick geddes and gerard manley hopkins in later life she bought a house in barnwell northamptonshire which became the subject of her book of the same name she died on 23 november 2005 the novelist bessie head was a close friend the pair corresponded from 1969 until head s death in 1986 on a range of subjects including head s novel a question of power",
	"3": "mat zan coached kuala lumpur fa in 1999 and won malaysia fa cup in 1999 and charity cup in 2000 in 2001 mat zan then move as head coach of terengganu fa and he won the 2001 malaysia cup 2001 malaysia charity cup and second place in malaysia league with plus f c in 2008 he took 2nd place in premier league and automatically promoted to malaysia super league in 2012 he comeback with terengganu fa and go through to fa cup semi finals and round two in afc cup in 2015 mat zan move to melaka united and won 2015 malaysia fam league and won 2016 malaysia premier league after his contract was not renewed with melaka at the end of 2016 he was appointed as new head coach of airasia allstars fc which was renamed as petaling jaya rangers f c for the 2017 season he signed a 1 year contract with negeri sembilan form the 2019 season mat zan also coached the club side of malaysia u 21 squad harimau muda b who won plate winner singapore starhub league cup thanj nien newspaper cup vietnam 3rd place and hassanal bolkiah cup brunei 3rd place"
}

class TextProcessing:
	@classmethod
	def process(cls, txt: str):
		# Convert to lowercase
		txt = txt.lower()

		# Remove punctuation
		txt = txt.translate(str.maketrans('', '', string.punctuation))
		
		# Tokenize into words
		words = word_tokenize(txt)

		# Correct word spelling
		words = TextProcessing.correct_spelling(words)
		
		# Remove stopwords
		words = TextProcessing.remove_stopwords(words)
		
		# Lemmatize words
		words = TextProcessing.lemmatize(words)

		# Merge words list into a string
		txt = ' '.join(words)

		return txt


	@classmethod
	def remove_stopwords(cls, words: List[str]) -> List[str]:
		new_words = []
		
		for word in words:
			if word not in stopwords.words('english'):
				new_words.append(word)
		
		return new_words
	
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



new_corpus = {}
for key, value in corpus.items():
	value = TextProcessing.process(value)
	new_corpus[key] = value

print(new_corpus)