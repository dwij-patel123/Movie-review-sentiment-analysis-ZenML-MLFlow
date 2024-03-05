import logging
from abc import ABC,abstractclassmethod
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


class Model(ABC):
    @abstractclassmethod
    def train(self,X_train,y_train):
        pass

class NaiveBayesModel(Model):
    def train(self,X_train,y_train,**kwargs):
        try:
            model = MultinomialNB(**kwargs)
            model.fit(X_train,y_train)
            return model
        except Exception as e:
            logging.error(e)
            raise e
    
