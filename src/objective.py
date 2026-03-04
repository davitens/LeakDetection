import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from src.models.ModelFactory import ModelFactory 

def objective(trial, model_name, X, y, resampling=None):
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
        sampler = SMOTE(random_state=33)
    elif resampling == "ADASYN":
        sampler = ADASYN(random_state=33)
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

    # 4. Save Trial Data to Parquet
    save_dir = "results/optuna_trials"
    save_path = os.path.join(save_dir, "all_trials_results.parquet")
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data for this trial
    # NOTE: Parquet handles dicts and lists directly!
    trial_data = {
        "trial_number": [trial.number],
        "model": [model_name],
        "hyperparameters": [space],        # Saved as a dictionary object
        "resampling": [str(resampling)],
        "predictions": [oof_predictions],   # Saved as an array/list object
        "mean_log_loss": [np.mean(scores)]
    }
    df_new = pd.DataFrame(trial_data)

    # Append to existing Parquet or create new
    if not os.path.exists(save_path):
        df_new.to_parquet(save_path, engine='pyarrow', index=False)
    else:
        # We read and concat to ensure the schema remains consistent
        df_old = pd.read_parquet(save_path, engine='pyarrow')
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_parquet(save_path, engine='pyarrow', index=False)

    return np.mean(scores)


