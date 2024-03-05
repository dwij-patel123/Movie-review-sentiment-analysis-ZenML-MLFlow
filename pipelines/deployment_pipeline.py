import numpy as np 
import pandas as pd
from zenml import pipeline, step 
from zenml.config import DockerSettings 
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT 
from zenml. integrations. constants import MLFLOW 
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import BaseParameters,Output
import logging


from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.train_model import train_model


dockersettings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy = 0.7

@step
def deployment_trigger(accuracy:float,config:DeploymentTriggerConfig):
    return accuracy >= config.min_accuracy


@pipeline(enable_cache=False,settings={"docker":dockersettings})
def continuous_deployment_pipeline(data_path:str,min_accuracy:float=0.7,workers:int=1,timeout:int=DEFAULT_SERVICE_START_STOP_TIMEOUT):
    df = ingest_df(data_path)
    X_train,X_test,y_train,y_test = clean_df(df)
    trained_model = train_model(X_train,X_test,y_train,y_test)
    acc,confu = evaluate_model(trained_model,X_test,y_test)
    deployment_decision = deployment_trigger(acc)
    mlflow_model_deployer_step(
        model=trained_model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )



