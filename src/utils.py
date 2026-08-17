import numpy as np
import pandas as pd
import optuna
import gc
import torch
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Assuming these are your paths
from src.models.ModelFactory import ModelFactory
from src.preprocessing import make_preprocessor

def get_best_model_oofs(model_name, best_params, X, y, resampling=None, k=5, cat_features=None, num_features=None):
    # --- FIX 1: RECONSTRUCT FULL PARAMS ---
    # Use FixedTrial to get static + suggested params
    fixed_trial = optuna.trial.FixedTrial(best_params)
    full_params = ModelFactory.get_space(fixed_trial, model_name)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
    oof_probs = np.zeros(len(X))
    


    for train_idx, val_idx in cv.split(X, y):
        sampler = None
        if resampling == "SMOTE":
            sampler = SMOTE(random_state=33, k_neighbors=k)
        elif resampling == "ADASYN":
            sampler = ADASYN(random_state=33, n_neighbors=k)
        elif resampling == "RandomUnderSampler":
            sampler = RandomUnderSampler(random_state=33)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if cat_features is not None and num_features is not None:
            preprocessor = make_preprocessor(cat_features, num_features)
            X_train = preprocessor.fit_transform(X_train)
            X_val = preprocessor.transform(X_val)

        if sampler is not None:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train

        # --- FIX 2: USE FULL PARAMS ---
        model = ModelFactory.get_model(model_name, params=full_params)

        if model_name in ["mlp", "nca", "tabm", "icl", "tabpfn"]:
            # Note: Ensure your ModelFactory correctly handles these arguments
            model.fit(X=X_res, y=y_res, num_gpus=1, num_cpus=6)
        else:
            model.fit(X_res, y_res)
        
        y_prob = model.predict_proba(X_val)
        
        # Handle probability indexing (Binary Class 1)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        
        oof_probs[val_idx] = y_prob

        # --- FIX 3: MEMORY CLEANUP PER FOLD ---
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return oof_probs
