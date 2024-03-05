import logging
from abc import ABC,abstractclassmethod
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix



class Evaluation:
    @abstractclassmethod
    def evaluate(self,y_pred:np.ndarray,y_true:np.ndarray):
        pass


class AccuracyScore(Evaluation):
    def evaluate(self,y_pred:np.ndarray,y_true:np.ndarray)->float:
        try:
            logging.info("evaluating accuracy score")
            accuracy = accuracy_score(y_true,y_pred)
            return accuracy
        except Exception as e:
            logging.error(e)
            raise e
        
class ConfusionMatrix(Evaluation):
    def evaluate(self,y_pred:np.ndarray,y_true:np.ndarray)->np.ndarray:
        try:
            logging.info("evaluating confusion matrix")
            confu = confusion_matrix(y_true,y_pred)
            return confu
        except Exception as e:
            logging.error(e)
            raise e