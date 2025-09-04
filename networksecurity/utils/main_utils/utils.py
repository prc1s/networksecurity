import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
import os,sys
import pickle 
import numpy as np
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def read_yaml_file(file_path:str) -> dict:
    try:
        logger.info("read_yaml_file Function in main_utils initiated")
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
        logger.info("Exited read_yaml_file Function in main_utils")
    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)
    
def write_yaml_file(file_path:str, content:object, replace:bool=False) -> None:
    try:
        logger.info("write_yaml_file Function in main_utils initiated")
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        logger.info("Exited write_yaml_file Function in main_utils")
    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)
    
def save_numpy_array_data(file_path:str, array:np.array):
    try:
        logger.info("save_numpy_array_data Function in main_utils initiated")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array)
        logger.info("Exited save_numpy_array_data Function in main_utils")
    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)
    
def load_numpy_array_data(file_path:str):
    try:
        logger.info("load_numpy_array_data Function in main_utils initiated")
        if not os.path.exists(file_path):
            logger.exception(f"The file path {file_path} does not exist")
            raise Exception(f"The file path {file_path} does not exist")
        with open(file_path, "rb") as file:
            print(file)
            return np.load(file_path)
        logger.info("Exited load_numpy_array_data Function in main_utils")
    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)

    
def save_pickle_object(file_path:str, obj:object):
    try:
        logger.info("save_pickle_object Function in main_utils initiated")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
        logger.info("Exited save_pickle_object Function in main_utils")
    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)

def load_pickle_object(file_path:str):
    try:
        logger.info("load_pickle_object Function in main_utils initiated")
        if not os.path.exists(file_path):
            logger.exception(f"The file path {file_path} does not exist")
            raise Exception(f"The file path {file_path} does not exist")
        with open(file_path, "rb") as file:
            print(file)
            return pickle.load(file)
        logger.info("Exited load_pickle_object Function in main_utils")
    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)

def evaluate_models(x_trian,y_train,x_test,y_test,models,params):
    try:
        report = {}

        for i in range (len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_trian,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_trian,y_train)

            y_train_pred = model.predict(x_trian)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)