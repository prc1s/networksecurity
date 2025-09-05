from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import pymongo
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from typing import List
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()
raw_mongodb_username =os.getenv("MONGO_DB_USERNAME")
raw_mongodb_password = os.getenv("MONGO_DB_PASSWORD")

mongodb_username =quote_plus(raw_mongodb_username)
mongodb_password = quote_plus(raw_mongodb_password)

MONGO_DB_URL=f"mongodb+srv://{mongodb_username}:{mongodb_password}@cluster0.yfmjcsk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
    
    def export_collection_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na":np.nan}, inplace=True) 
            return df
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
        
    def export_data_into_feature_store (self, dataframe:pd.DataFrame):
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_path, index=False, header=True)
            return dataframe
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
        

    def train_test_split(self, dataframe:pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.test_split_ratio
            )
            logger.info("Train, Test Split Completed")

            train_dir_path = os.path.dirname(self.data_ingestion_config.training_path_file)
            os.makedirs(train_dir_path, exist_ok=True)
            train_set.to_csv(
                self.data_ingestion_config.training_path_file, index=False, header=True
            )

            test_dir_path = os.path.dirname(self.data_ingestion_config.testing_path_file)
            os.makedirs(test_dir_path, exist_ok=True)
            test_set.to_csv(
                self.data_ingestion_config.testing_path_file, index=False, header=True
            )
            logger.info(f"<<<< Train csv Saved at {train_dir_path} >>>>")
            logger.info(f">>>> Test csv Saved at {test_dir_path} <<<<")

        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
    
    def initiate_data_ingestion(self):
        try:
            logger.info("Initiating Data Ingestion")
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.train_test_split(dataframe)
            dataingestionartifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_path_file,
                                                          test_file_path=self.data_ingestion_config.testing_path_file)
            logger.info("Data Ingestion Completed")
            return dataingestionartifact
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)