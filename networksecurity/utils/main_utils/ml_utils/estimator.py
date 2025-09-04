import os,sys
from networksecurity.constants.training_pipeline import SAVED_ESTIMATORS_DIR, ESTIMATOR_FILE_NAME
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
class NetworkModel():
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
        
    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_prediction = self.model.predict(x_transform)
            return y_prediction
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)