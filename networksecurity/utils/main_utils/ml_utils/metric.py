import os,sys
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from sklearn.metrics import f1_score,precision_score,recall_score

def get_classification_score(true_values, predicted_values):
    try:
        logger.info("get_classification_score Function in ml_utils initiated")
        model_f1_score = f1_score(true_values, predicted_values)
        model_precision_score = precision_score(true_values, predicted_values)
        model_recall_score = recall_score(true_values, predicted_values)
        classification_metric = ClassificationMetricArtifact(
             f1_score=model_f1_score,
             precision_score=model_precision_score,
             recall_score=model_recall_score
        )
        logger.info("get_classification_score Function in ml_utils Completed")
        return classification_metric
    except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)