import pandas as pd
import numpy as np
import re
import nltk
nltk.download(['punkt','stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def replace_url(text):
    '''
    Replace url with 'urlplaceholder' in text
    INPUT:
        text: string
    OUTPUT:
        text: edited string
    '''
    detected_urls = re.findall(url_regex, text)
    # replace each url in text strings with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    return text

    def tokenize(text):
    '''
    processed text strings
    INPUT: text: string
    OUTPUT: clean_tokens  a list of processed words
    '''
    # Case Normalization
    text = text.lower()  # convert to lowercase
    # tokenize text
    tokens = word_tokenize(text)
    token_list = []
    # remove stop words
    for tok in tokens:
        if tok not in stopwords.words("english"):
            token_list.append(tok)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iritate through each token
    clean_tokens = []
    for tok in token_list:
        # lemmatize and remove leading and tailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()

        clean_tokens.append(clean_tok)
    return clean_tokens


class DisasterWordExtractor(BaseEstimator, TransformerMixin):

    def disaster_words(self, text):
        """
        INPUT: text - string, raw text data
        OUTPUT: bool -bool object, True or False
        """
        # Build a list of words that are constantly used during a disaster event
        words = ['food', 'hunger', 'hungry', 'starving', 'water', 'drink',
                 'eat', 'thrist',
                 'need', 'hospital', 'medicine', 'medicial', 'ill', 'pain',
                 'disease', 'injured', 'falling',
                 'wound', 'dying', 'death', 'dead', 'aid', 'help',
                 'assistance', 'cloth', 'cold', 'wet', 'shelter',
                 'harricane', 'earthquake', 'flood', 'live', 'alive', 'child',
                 'people', 'shortage', 'blocked',
                 'gas', 'pregnant', 'baby'
                 ]

        # lemmatize the words
        lemmatized_words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in
                            words]
        # Get the stem words of each word in  lemmatized_words
        stem_disaster_words = [PorterStemmer().stem(w) for w in
                               lemmatized_words]

        # get list of all urls using regex
        detected_urls = re.findall(url_regex, text)
        # replace each url in text strings with placeholder
        for url in detected_urls:
            text = text.replace(url, 'urlplaceholder')

        # tokenize the text
        clean_tokens = tokenize(text)
        for token in clean_tokens:
            if token in stem_disaster_words:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_disaster_word = pd.Series(X).apply(self.disaster_words)
        return pd.DataFrame(X_disaster_word)