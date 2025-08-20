import os
import sys
import json
import certifi
import pandas as pd
import os
from urllib.parse import quote_plus
import numpy as np
import pymongo
from pymongo.mongo_client import MongoClient
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger

from dotenv import load_dotenv
load_dotenv()

raw_mongodb_username =os.getenv("MONGO_DB_USERNAME")
raw_mongodb_password = os.getenv("MONGO_DB_PASSWORD")

mongodb_username =quote_plus(raw_mongodb_username)
mongodb_password = quote_plus(raw_mongodb_password)

MONGO_DB_URL=f"mongodb+srv://{mongodb_username}:{mongodb_password}@cluster0.yfmjcsk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

ca = certifi.where()

class NetworkDataExctract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
        
    def csv_tojson_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
        
    def insert_data_to_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.records = records
            self.collection = collection

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]

            self.collection.insert_many(self.records)
            return (len(self.records))
        
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)            
        

if __name__ == '__main__':
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "NETWORKSECURITYAI"
    COLLECTION = "NetworkData"
    networkobj = NetworkDataExctract()
    RECORDS = networkobj.csv_tojson_converter(FILE_PATH)
    print(RECORDS)
    no_of_rec = networkobj.insert_data_to_mongodb(RECORDS,DATABASE, COLLECTION )
    print(no_of_rec)