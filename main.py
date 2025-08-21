from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
import sys

if __name__ == '__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        logger.info("Initiating Data Ingestion")
        data_ingestion = DataIngestion(data_ingestion_config)
        dataingestionartifact = data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)