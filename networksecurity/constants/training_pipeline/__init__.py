import os
import sys
import numpy as np
import pandas as pd

#Constants for model training
TARGET_COLUMN = "Result"
PIPELINE_NAME : str = "NetworkSecurity"
ARTIFACT_DIR : str = "Artifacts"
FILE_NAME : str = "phisingData.csv"

TRAIN_FILE_NAME : str = "train.csv"
TEST_FILE_NAME : str = "test.csv"
PREPROCESSING_OBJECT_FILE_NAME : str = "preprocessing.pkl"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_ESTIMATORS_DIR = os.path.join("saved_estimators")
ESTIMATOR_FILE_NAME = os.path.join("model.pkl")

#Data Ingestion Constants
DATA_INGESTION_COLLECTION_NAME : str = "NetworkData"
DATA_INGESTION_DATABASE_NAME : str = "NETWORKSECURITYAI"
DATA_INGESTION_DIR_NAME : str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR : str = "feature_store"
DATA_INGESTION_INGESTED_DIR : str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION : float = 0.2

#Data Validation Constants
DATA_VALIDATION_DIR_NAME : str = "data_validation"
DATA_VALIDATION_VALID_DIR : str = "validated"
DATA_VALIDATION_INVALID_DIR : str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR : str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME : str = "report.yaml"

#Data Transformaion Constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
TRANSFORMED_DATA_DIR: str = "transformed"
TRANSFORMED_OBJECT_DIR: str = "transformed_object"

#KNN Imputer To Teplace Nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values" : np.nan,
    "n_neighbors" : 8,
    "weights" : "uniform"
}

#Model Training constants
MODEL_TRAINER_DIR: str = "model_training"
TRAINED_MODEL_DIR: str = "trained_model"
TRAINED_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_ACCURACY: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05
