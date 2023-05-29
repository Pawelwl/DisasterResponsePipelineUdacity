import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    Load data from database and split it into X and Y variables
    
    Input:
    database_filepath: filepath to database
    
    Returns:
    X: the explanatory variable (text message)
    Y: the explained variable (categories)
    category_names: names of the categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MergedDataSet', engine)
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    X = df['message']
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    Transform text into tokens - preparation for ML modelling
    
    Input:
    text: text in string format
    
    Returns:
    clean_tokens: list of tokens - cleaned words prepared for ML modelling
    '''
    tokens = word_tokenize(text)
    
    # remove stopwords
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
    Instantiation of model classifier and preparation of ML pipeline of data transformation and modelling
      
    Returns:
    pipeline: Machine Learning pipeline transforming and modelling data
    '''
    classifier = RandomForestClassifier()
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(classifier))
    ])
    
    parameters = {
        'clf__n_estimators': [50, 100],
        'clf__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Calculate model predictions on test set and compare them with actuals using classification error metrics
    
    Input:
    model: trained classification model
    X_test: test set for X variable
    Y_test: test set for Y variable
    category_names: names of the explained variable categories
    '''
    y_pred = model.predict(X_test)
    col_num = 0

    for col in category_names:
        print(f'***** REPORT for {col} column*****')
        print(classification_report(Y_test[col], y_pred[:, col_num]))
        col_num += 1

def save_model(model, model_filepath):
    '''
    Save model in pickle format in a specified filepath
    
    Input:
    model: trained classification model
    model_filepath: filepath for model
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
