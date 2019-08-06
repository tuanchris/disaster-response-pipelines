# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score, make_scorer
from scipy.stats.mstats import gmean
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")
import sys



def load_data(database_filepath):
    '''
    Load cleansed data from sqlite database
    Input:
        database_filepath: file path to saved sqlite database
    Output:
        X: messages for categorization
        y: category of message
        category_names: available cateogires
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages',engine)
    X = df['message']
    Y = df.drop(['message','genre','original','id'],axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    Customize tokenization function that normalize, lemmatize and tokenize text
    Input:
        text: input messages
    Output:
        clean_words: normalized, lemmatized, and tokenized text
    '''
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    clean_words = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).lower().strip()
        clean_words.append(clean_word)

    return clean_words

def build_model():
    '''
    Build machine learning pipleines and use GridSearchCV to find the best parameters
    Output:
        cv: model with best parameters
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators':[100,200],
        'clf__estimator__min_samples_split':[2,3,4],
        'clf__estimator__criterion': ['entropy', 'gini']
        }

    cv = GridSearchCV(pipeline, param_grid=parameters,verbose = 2, n_jobs=-1)

    return cv

# def build_model():
#     pipeline = Pipeline([
#         ('vect',CountVectorizer(tokenizer=tokenize)),
#         ('tfidf',TfidfTransformer()),
#         ('clf',MultiOutputClassifier(AdaBoostClassifier()))
#     ])
#     return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print('-'*60,'\n',"Category:", category_names[i],"\n", classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of',category_names[i], accuracy_score(y_test.iloc[:, i].values, y_pred[:,i]))



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


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
