import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from zenml import step
from src.model_create import NaiveBayesModel
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from zenml.client import Client

from .config import ModelNameConfig


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker="mlflow_tracker_customer")
def train_model( 
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig)->MultinomialNB:
    try:
        if config.model_name == 'MultinomialNB':
            mlflow.sklearn.autolog()
            model = NaiveBayesModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
    except Exception as e:
        logging.error(e)
        raise e