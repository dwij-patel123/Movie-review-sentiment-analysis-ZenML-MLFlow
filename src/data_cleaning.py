import logging 
from abc import ABC,abstractclassmethod
from typing import Union,Tuple,Annotated
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split



class DataStrategy(ABC):
    @abstractclassmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    # data preprocess overriding above abstract method
    def handle_data(self,df:pd.DataFrame)->pd.DataFrame:
        df = df.drop(['id'],axis=1)
        y = df['sentiment']
        lm = WordNetLemmatizer()
        corpus = []
        for index in range(0,25000):
            review = re.sub('[^a-zA-Z]',' ',df.iloc[index]['review'])
            review = review.lower()
            review = review.split()
            temp = []
            for word in review:
                if word not in set(stopwords.words('english')):
                    temp.append(lm.lemmatize(word))
            review = ' '.join(temp)
            corpus.append(review)

        cv = CountVectorizer(max_features = 2500)
        X_train = cv.fit_transform(corpus)
        X_train = X_train.toarray()
        columns = cv.get_feature_names_out()
        df = pd.DataFrame(X_train,columns=columns)
        df = df.assign(sentiment=y)
        return df

class DataDividingStrategy(DataStrategy):
    # data split method overridding above abstract class method
    def handle_data(self,df:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        try:
            y = df['sentiment']
            X = df.drop('sentiment',axis=1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
         


        