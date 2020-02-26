import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    读取数据
    输入变量：文本路径，标签路径
    输出变量：合并后的dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages,categories,how='inner',on='id')


def clean_data(df):
    """
    清洗数据
    输入变量：目标dataframe
    输出变量：清洗完的dataframe
    """
    categories = df.categories.str.split(';',expand=True)
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(float)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    保存数据
    输入变量：目标dataframe，数据库路径
    输出变量：无
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


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