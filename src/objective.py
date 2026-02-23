import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from src.models.ModelFactory import ModelFactory
import optuna




def objective(trial, model_name, X, y, resampling=None):
    space = ModelFactory.get_space(trial, model_name)
    model = ModelFactory.get_model(model_name, params=space)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    sampler = None
    if resampling == "SMOTE":
        sampler = SMOTE(random_state=33)
    elif resampling == "ADASYN":
        sampler = ADASYN(random_state=33)
    elif resampling == "RandomUnderSampler":
        sampler = RandomUnderSampler(random_state=33)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if sampler is not None:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train
            
        # 4. Model Training (Model Agnostic)
        if model_name in ["mlp", "nca", "tabm", "icl", "tabpfn"]:
            model.fit(X=X_res, y=y_res, num_gpus=1, num_cpus = 6)

        else:
            model.fit(X=X_res, y=y_res)
        y_prob = model.predict_proba(X_val)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
            
        y_true = y_val.to_numpy(dtype=int)
        #y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15) ############## IM REMOVE CLIP. REMEMBER TO EVALUATE PLEASE
        fold_score = log_loss(y_true, y_prob)
        scores.append(fold_score)
        current_avg = np.mean(scores)
        trial.report(current_avg, step=fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores)