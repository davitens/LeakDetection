import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from src.models.ModelFactory import ModelFactory 

def objective(trial, model_name, X, y, resampling=None, k=5):
    # 1. Setup Hyperparameters and Model
    space = ModelFactory.get_space(trial, model_name)
    model = ModelFactory.get_model(model_name, params=space)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=33)
    
    # Initialize Out-of-Fold (OOF) predictions array
    oof_predictions = np.zeros(len(X))
    scores = []
    
    # 2. Resampling Setup
    sampler = None
    if resampling == "SMOTE":
        sampler = SMOTE(random_state=33, k_neighbors=k)
    elif resampling == "ADASYN":
        sampler = ADASYN(random_state=33, n_neighbors=k)
    elif resampling == "RandomUnderSampler":
        sampler = RandomUnderSampler(random_state=33)

    # 3. Cross-Validation Loop
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if sampler is not None:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train
            
        # Model Training (Agnostic)
        if model_name in ["mlp", "nca", "tabm", "icl", "tabpfn"]:
            model.fit(X=X_res, y=y_res, num_gpus=1, num_cpus=6)
        else:
            model.fit(X=X_res, y=y_res)

        y_prob = model.predict_proba(X_val)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        
        # Store predictions in the global OOF array
        oof_predictions[val_idx] = y_prob
            
        y_true = y_val.to_numpy(dtype=int)
        
        # Guard against Log Loss infinity
        y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
        fold_score = log_loss(y_true, y_prob_clipped)
        scores.append(fold_score)

        trial.report(np.mean(scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # --- 4. Store Data in Optuna Internal Storage (NOT Parquet) ---
    # This keeps your format but avoids the concurrent write crash.
    # We save these as "User Attributes" which we will export later.
    trial.set_user_attr("space", space)
    trial.set_user_attr("resampling", str(resampling))
    trial.set_user_attr("predictions", oof_predictions.tolist()) # List for Parquet compatibility later
    trial.set_user_attr("k", k)

    return np.mean(scores)


