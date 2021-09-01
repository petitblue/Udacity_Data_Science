import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from custom_extractor import DisasterWordExtractor

def load_data(database_filepath):
    '''
    Function to load data from sqlite database and set X and y for modeling preparation
    
    OUTPUT
        - X: array of messages
        - y: target dataframe
    '''
    engine = create_engine('sqlite:///DisasterRespond.db')
    df = pd.read_sql('cleaned_data',con = engine)
    # define feature and target variables X and y
    X = df['message'].values
    y = df[df.columns[4:]]
    category_names = y.columns.tolist()
    return X, y, category_names


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
    """
        Function to processed text in string format, and output a list of words
        1. repalce all urls with 'urlplaceholder'
        2. case normalization and remove punctuation
        3. tokenize text
        4. remove stop words
        5. lemmatize words
        INPUT:
            -text: string, raw text data
        OUTPUT:
            -clean_tokens, list of processed words
    
    """

    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    # replace each url in text strings with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    # Case Normalization
    text = text.lower() # convert to lowercase
    # remove puntuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    
    # tokenize text
    tokens = word_tokenize(text)
    token_list =[]
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


def build_model():
    '''
    Function that build the pipeline and the grid search parameters to
    create a classification model
    
    OUTPUT: cv:classification model
    '''
    xgb_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())])),
            ('disaster_words', DisasterWordExtractor())
        ])),
        ('clf', MultiOutputClassifier(estimator=XGBClassifier(max_depth=6)))
    ])
    parameters = parameters = { 'clf__estimator__n_estimators': [100,150]}
    # create grid search object
    model = GridSearchCV(xgb_pipeline, param_grid=parameters, scoring='recall_micro', cv=4)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the model for each category of the dataset
    INPUT: 
        -model: the classification model
        -X_test: the feature variable
        -Y_test: the target variable
        -category_names: list
    OUTPUT:
        Classification report and accuracy score
    '''
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))
    

def save_model(model, model_filepath):
    '''
    The function to save machine learning pipeline 'model' to local path
    INPUT:
        model: Machine learning pipeline
        model_filepath:  the name of the local path to save the model
    OUTPUT:
        none
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()