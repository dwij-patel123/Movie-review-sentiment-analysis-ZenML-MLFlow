from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model


@pipeline()
def train_pipeline(data_path:str):
    df = ingest_df(data_path)
    X_train,X_test,y_train,y_test = clean_df(df)
    trained_model = train_model(X_train,X_test,y_train,y_test)
    acc,confu = evaluate_model(trained_model,X_test,y_test)