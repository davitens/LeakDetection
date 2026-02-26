import optuna
import pandas as pd
from src.data_loader import load_and_clean_data
from src.preprocessing import preprocess_data
from src.objective import objective
import os
from src.utils import get_best_model_oofs
import numpy as np
from tabarena.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel
from sklearn.metrics import log_loss
import time
import torch
from sklearn.model_selection import train_test_split
from tabarena.benchmark.models.ag.tabm.tabm_model import TabMModel
from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model


print("Loading data...")
X, y, cat_features, num_features = load_and_clean_data()
print("Preprocessing data...")
X = preprocess_data(X, cat_features, num_features)




X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)


# MODEL 1: TABMMODEL
start_time_1 = time.time()
hyperparameters = {
    "softmax_temperature": 0.25,
    "balance_probabilities": True,
    "inference_config/OUTLIER_REMOVAL_STD": 3,
    "preprocessing/categoricals": "onehot",
    "inference_config/POLYNOMIAL_FEATURES": "all",
    # Wrap this value in a list:
    "preprocessing/scaling": ["kdi_alpha_0.3_uni"] 
}
model_1 = RealTabPFNv25Model(hyperparameters=hyperparameters)
model_1.fit(X=X_train_numeric, y=y_train, num_gpus = 1)
y_pred_1 = model_1.predict(X_test)
end_time_1 = time.time()
elapsed_time_1 = end_time_1 - start_time_1
print(f"Elapsed time WITHOUT gpu: {elapsed_time_1:.4f} seconds, LogLoss = {log_loss(y_test,y_pred_1)}")
# 4060: Elapsed time WITHOUT gpu: 14.4308 seconds, LogLoss = 0.9317500210834317

"""
"softmax_temperature": trial.suggest_categorical("softmax_temperature", [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5]),
"balance_probabilities": trial.suggest_categorical("balance_probabilities", [True, False]),            
"outlier_removal_std": trial.suggest_categorical("outlier_std", [3, 6, 12]),
"""

from tabarena.benchmark.models.ag.tabicl.tabicl_model import TabICLModel

start_time_2 = time.time()
model_2 = TabICLModel()
model_2.fit(X=X_train, y=y_train, num_gpus = 1)
y_pred_2 = model_2.predict(X_test)
end_time_2 = time.time()
elapsed_time_2 = end_time_2 - start_time_2
print(f"Elapsed time WITHOUT gpu: {elapsed_time_2:.4f} seconds, LogLoss = {log_loss(y_test,y_pred_1)}")
# 4060: Elapsed time WITHOUT gpu: 80.0730 seconds, LogLoss = 0.9317500210834317