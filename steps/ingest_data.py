import logging
import pandas as pd

from zenml import step

class IngestData:
    def __init__(self,data_path:str) -> None:
        self.data_path = data_path

    def get_data(self)-> pd.DataFrame:
        logging.info("inserting data")
        return pd.read_csv(self.data_path,sep='\t',index_col = False)

@step
def ingest_df(data_path:str)->pd.DataFrame:
    try:
        ingest_df = IngestData(data_path)
        df = ingest_df.get_data()
        return df
    except Exception as e:
        logging.info("error while logging {}".format(e))
        raise e
        


