import logging
import pandas as pd
from zenml import step
from typing import Tuple,Annotated
from src.data_cleaning import DataDividingStrategy,DataPreProcessStrategy

@step
def clean_df(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]
]:
    try:
        logging.info("cleaning data started")
        preprocess_data = DataPreProcessStrategy()
        data = preprocess_data.handle_data(df)

        datadividing_strategy = DataDividingStrategy()
        X_train, X_test, y_train, y_test = datadividing_strategy.handle_data(data)
        return  X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e

