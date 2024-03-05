from pipelines.training_pipelines import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline('/Users/dwijvijaykumarpatel/zenML_test/data/labeledTrainData.tsv')
