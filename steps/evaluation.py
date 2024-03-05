import logging
import pandas as pd
import numpy as np
from zenml import step
from zenml.client import Client
import mlflow
from typing import Tuple,Annotated
from sklearn.naive_bayes import MultinomialNB
from src.evaluate_model import AccuracyScore,ConfusionMatrix


experiment_tracker = Client().active_stack.experiment_tracker
@step
def evaluate_model(model:MultinomialNB,X_test:pd.DataFrame,y_test:pd.Series)->Tuple[
    Annotated[float,"accuracy score"],
    Annotated[np.ndarray,"confusion matrix"]
]:
    try:
        y_pred = model.predict(X_test)
        accuracy = AccuracyScore()
        acc = accuracy.evaluate(y_pred,y_test)
        #mlflow.log_metric("acc",acc)

        confusion = ConfusionMatrix()
        confu = confusion.evaluate(y_pred,y_test)
        
        return acc,confu
    except Exception as e:
        logging.error(e)
        raise e
