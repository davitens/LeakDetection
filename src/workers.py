import gc
import torch
import optuna
import time
import numpy as np
import traceback
from sklearn.metrics import log_loss
from src.objective import objective
from src.utils import get_best_model_oofs

def worker_task(queue, name, X, y, resampling, n_trials):
    # Initial GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start_time = time.time()
    resampling_label = str(resampling) if resampling else "None"

    try:
        # Optimization
        sampler = optuna.samplers.TPESampler(multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        
        study.optimize(
            lambda trial: objective(trial, name, X, y, resampling), 
            n_trials=n_trials,
            timeout=1800
        )

        # Generate OOF predictions
        best_oof = get_best_model_oofs(name, study.best_params, X, y, resampling)
        
        # Verification Step
        y_prob_verified = np.clip(best_oof, 1e-7, 1 - 1e-7)
        verified_logloss = log_loss(y, y_prob_verified)
        
        duration = time.time() - start_time

        # Send results back
        queue.put({
            "status": "success",
            "name": name,
            "resampling": resampling_label,
            "best_value": study.best_value,
            "verified_score": verified_logloss,
            "best_params": study.best_params,
            "best_oof": np.array(best_oof),
            "duration_seconds": round(duration, 2)
        })
        
    except Exception as e:
        queue.put({
            "status": "error",
            "name": name,
            "resampling": resampling_label,
            "message": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        # Clean up
        if 'study' in locals(): del study
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()