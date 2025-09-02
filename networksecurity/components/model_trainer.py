import os,sys
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.utils.main_utils.utils import save_pickle_object,load_pickle_object,load_numpy_array_data
from networksecurity.utils.main_utils.ml_utils.estimator import NetworkModel
from networksecurity.utils.main_utils.ml_utils.metric import get_classification_score

class ModelTrainer():
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_Artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_Artifact = data_transformation_Artifact
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
        
    def initiate_model_trainer(self):
        try:
            train_file_path = self.data_transformation_Artifact.transformed_train_file_path
            test_file_path = self.data_transformation_Artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)