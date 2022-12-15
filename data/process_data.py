import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load data from filepath.
    Return merged data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Return cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.head(1)
    category_colnames =  row.applymap(lambda x: x[:-2]).iloc[0,:]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    df.drop(columns = ['categories'], inplace=True)
    df = df.join(categories)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataframe into a SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('ThuyDD7', engine, index=False, if_exists='replace')


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