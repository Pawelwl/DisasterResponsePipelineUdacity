import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from two csv files and merge into a single DataFrame
    
    Input:
    messages_filepath: filepath to messages csv file
    categories_filepath: filepath to categories csv file
    
    Returns:
    df: DataFrame merging messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='left')
    return df
    
def clean_data(df):
    '''
    Cleans the part of dataset related ot categories. Splits one 'categories' column into separate columns each representing one category.
    
    Input:
    df: input DataFrame to be cleaned
    
    Returns:
    df: output cleaned DataFrame
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Additional check if only 0' and 1's are in the categoreis dataset
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x if x == 0 else 1)
    # drop the original categories column from `df`
    df = df.drop('categories', axis='columns')
    df = pd.merge(df, categories, left_index=True, right_index=True)
    # drop duplicates
    df = df[~df.duplicated()]
    return df

def save_data(df, database_filename):
    '''
    Creates a DataBase and saves 'df' DataFrame in it.
    
    Input:
    df: input data (DataFrame)
    database_filename: name od DataBase to be created
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MergedDataSet', engine, if_exists='replace', index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
