import os,sys
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.utils.main_utils.utils import save_pickle_object,load_pickle_object,load_numpy_array_data,evaluate_models
from networksecurity.utils.main_utils.ml_utils.estimator import NetworkModel
from networksecurity.utils.main_utils.ml_utils.metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier)

class ModelTrainer():
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_Artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_Artifact = data_transformation_Artifact
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
        
    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models={
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier()
            }

            params={
                "Decision Tree":{
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'splitter':['best', 'random'],
                    'max_features':['sqrt', 'log2', 'None']
                },

                "Random Forest":{
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'max_features':['sqrt', 'log2', 'None'],
                    'n_estimators':[8,16,32,64,128,256]
                },

                "Gradient Boosting":{
                    'loss':['log_loss', 'exponential'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample': [.6,.7,.75,.8,.85,.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['sqrt', 'log2', 'auto'],
                    'n_estimators':[8,16,32,64,128,256]
                },

                "Logistic Regression":{},

                "AdaBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }
            model_report: dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            classification_train_metrics = get_classification_score(true_values=y_train,predicted_values=y_train_pred)
            classification_test_metrics = get_classification_score(true_values=y_test,predicted_values=y_test_pred)

            preprocessor = load_pickle_object(file_path=self.data_transformation_Artifact.transformed_object_file_path)
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir,exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_pickle_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metrics,
                test_metric_artifact=classification_test_metrics
            )
            logger.info(f"Model Trainer Artifacts {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)
        
    def initiate_model_trainer(self):
        try:
            logger.info("Initiated Model Trainer")
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

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            logger.info("Model Training Completed")
            return model_trainer_artifact
        except Exception as e:
            logger.exception(NetworkSecurityException(e,sys))
            raise NetworkSecurityException(e,sys)