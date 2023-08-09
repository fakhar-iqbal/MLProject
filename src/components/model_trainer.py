import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model,save_obj

@dataclass
class ModelTrainerConfig:
    model_train_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training and test input data')
            X_train,y_train,X_test,y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "random_forest" : RandomForestRegressor(),
                "decision tree" : DecisionTreeRegressor(),
                "gradient boosting" : GradientBoostingRegressor(),
                "linear regression" : LinearRegression(),
                "k neighbour regressor" : KNeighborsRegressor(),
                "XGBRegressor" : XGBRFRegressor(),
                "catBoostingRegressor" : CatBoostRegressor(verbose=False),
                "AdaBoostRegressor" : AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(X_train=X_train,
                                               y_train=y_train,
                                               X_test=X_test,
                                               y_test=y_test,
                                               models=models)
            
            ## to get best model score
            best_model_score = max(sorted(model_report.values()))

            ## to get the name of model from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            ## raising exception if no model is fittting well
            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info("best found model on both training and test dataset")

            ##save the best model
            save_obj(
                file_path = self.model_trainer_config.model_train_file_path,
                obj= best_model
            )

            logging.info("saved the best model into pkl file")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square,best_model_name
            

        except Exception as e:
            raise CustomException(e,sys)