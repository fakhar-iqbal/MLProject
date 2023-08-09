import sys
from src.exception import CustomException

import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger import logging
import os

from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_obj(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            num_features = ['writing_score','reading_score']
            cat_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encode",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"numerical columns: {num_features}")
            logging.info(f"categorical columns: {cat_features}")

            preprocessor = ColumnTransformer(
                [
                ("numerical_pipeline", num_pipeline,num_features),
                ("categorical_piopeline",cat_pipeline,cat_features)
                ]
            )

            return preprocessor


        except Exception as e:

            raise CustomException(e,sys)
    
    def initiaite_data_transformation(self,train_path,test_path):
        try:
            
            train_df =pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("train and test data read completely")
            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_preprocessor_obj()
            
            target_column = 'math_score'
            num_feature = ['reading_score','writing_score']

            input_features_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("applying preprocessing object on tarin and test dataframe")

            input_feature_train_array = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_array,np.array(target_feature_test_df)]
            

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)

