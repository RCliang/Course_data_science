import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.externals import joblib

def load_data(database_filepath):
    """
    读取数据
    输入变量：数据库地址
    输出变量：自变量，应变量，标签名
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from DisasterResponse', engine)
    X = df.message.values
    y = df.iloc[:,4:]
    categories = y.columns
    return X, y, categories

def tokenize(text):
    """
    文本清洗
    输入变量：文本
    输出变量：已处理完成的文本变量
    """
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    建立模型
    输入变量：无
    输出变量：已实例化的模型对象
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    评估模型
    输入变量：模型对象，测试集自变量，测试集应变量，标签名
    输出变量：无
    """
    y_pred = pd.DataFrame(model.predict(X_test),columns=category_names)
    for col in category_names:
        print ('classification report for ', col)
        print(classification_report(Y_test[col], y_pred[col],))

def save_model(model, model_filepath):
    """
    保存模型
    输入变量：模型对象，模型保存路径
    输出变量：无
    """
    with open(model_filepath,'wb') as fw:
	    joblib.dump(model,fw)


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