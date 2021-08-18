import sys
import pandas as pd
import numpy as np
from sqlalchemy import creat_engine
import warnings
warnings.filterwarnings('ignore')  
def load_data(messages_filepath, categories_filepath):
    '''
    Build a function to load messages and categories dataset from csv files, and merge two datasets into one dataframe
    INPUT: messages_filepath, categories_filepath
    OUTPUT: df, a merged dataframe
        
    '''
    # Load messages data from csv files to a dataframe
    messages = pd.read_csv(messages_filepath,dtype=str)
     # Load categories data from csv files to a dataframe
    categories = pd.read_csv(categories_filepath,dtype=str)
    # merge two dataset messages and categories on common id
    df = pd.merge(left=messages,right=categories,how='inner',on=['id'])
    # display the first five rows of dataesets
    df.head()
    return df

def clean_data(df):
    '''
    The tasks of this function include:
    1. seperate the values in the categories column and build a dataframe of 36 columns
    2. Use the first row of categories dataframe to create column names for the categories data
    3. Rename columns of categories with new column names
    4. convert categories values to 0 or 1
        1)Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
        2)Convert column from string to numeric
        3)Check if there are values other than 0 and 1 in categories and replace the other values with 1
    5. Drop the categories column from the df dataframe 
    6. Concatenate df and categories data frames
    7. Remove duplicates
    '''
    # Split the values in the categories column on the ; character so that each value becomes a separate column
    categories = df['categories'].str.split(pat=';',expand=True)
    # select the first row of the categories dataframee
    row = categories[:1]
    # build a lambda function extract names except last two characthers 
    extract_list = lambda x:x[0][:-2]
    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(extract_list).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    # iterate the categories columns
    for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    # check if there are values other than 0 and 1 in categories
    others = []
    for col in categories.columns:
    others.append(categories[col].unique())
    # If there are values other than 0 or 1 in categories' columns, replace the other values as 1
    for col in categories.columns:
        categories.loc[(categories[col]!=1)&(categories[col]!=0)] = 1
    # drop the original categories column from `df`
    df = df.drop(['categories'],axis=1)
    # concatenate the original dataframe with the new `categories`            dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
     '''
     This function save the clean dataframe into a sqlite database
     
     '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('cleaned_data', engine, index=False,if_exists = 'replace')
    pass

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