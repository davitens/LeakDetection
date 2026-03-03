import os
import time
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import log_loss
import multiprocessing as mp
from src.data_loader import load_and_clean_data
from src.preprocessing import preprocess_data
from src.objective import objective
from src.utils import get_best_model_oofs

# --- GPU MULTIPROCESSING SETUP ---
# This manager will live across all processes to hand out GPU IDs
manager = mp.Manager()
gpu_queue = manager.Queue()

def init_gpu_queue():
    """Fill the queue with available GPU IDs (e.g., [0, 1] for 2 GPUs)"""
    while not gpu_queue.empty():
        gpu_queue.get()
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        gpu_queue.put(i)
    return n_gpus

def gpu_wrapped_objective(trial, name, X, y, resampling):
    """
    A wrapper that assigns a specific GPU to this trial.
    Assumes the 'objective' function in your src/objective.py 
    can accept a 'device' argument.
    """
    gpu_id = gpu_queue.get()  # Grab an available GPU ID
    device = f"cuda:{gpu_id}"
    
    try:
        # Pass the specific device to your objective function
        # NOTE: You may need to update your 'objective' function signature 
        # in src/objective.py to accept and use this 'device' string.
        score = objective(trial, name, X, y, resampling, device=device)
        return score
    finally:
        # Clean up and put the GPU back for the next trial
        torch.cuda.empty_cache()
        gpu_queue.put(gpu_id)

# Suppress specific sklearn warnings
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
    module="sklearn"
)

def run_benchmark(n_trials=1, model_names=None, resampling_methods=None):
    if model_names is None:
        model_names = ["tabm"]
    if resampling_methods is None:
        resampling_methods = ["RandomUnderSampler", None, "SMOTE", "ADASYN"]

    # Initialize GPU Queue and get count for n_jobs
    n_gpus = init_gpu_queue()
    print(f"Detected {n_gpus} GPUs. Setting n_jobs={n_gpus}")

    print("Loading data...")
    X, y, cat_features, num_features = load_and_clean_data()
    
    print("Preprocessing data...")
    X = preprocess_data(X, cat_features, num_features)
    
    os.makedirs("results/oof_predictions", exist_ok=True)
    
    oof_master_df = pd.DataFrame(index=X.index)
    all_results = []
    total_start_time = time.time()

    for resampling in resampling_methods:
        for name in model_names:
            torch.cuda.empty_cache()
            gc.collect()
            model_iteration_start = time.time()
            
            print("\n" + "="*50)
            print(f"RUNNING: {name.upper()} | RESAMPLING: {resampling}")
            print("="*50)
            
            current_n_trials = 1 if name in ["tabdpt", "icl"] else n_trials
            
            study = optuna.create_study(
                direction="minimize", 
                sampler=optuna.samplers.TPESampler(multivariate=True), 
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Use the number of GPUs as n_jobs
            study.optimize(
                lambda trial: gpu_wrapped_objective(trial, name, X, y, resampling), 
                n_trials=current_n_trials,
                n_jobs=n_gpus if n_gpus > 0 else 1,
                timeout=1800
            ) 

            print(f"Best Log Loss: {study.best_value:.4f}")
            
            # OOF generation usually doesn't need parallelization across GPUs
            # so we run it on the default device or cuda:0
            print(f"Generating OOF predictions for best {name}...")
            best_oof = get_best_model_oofs(name, study.best_params, X, y, resampling)

            y_prob_verified = np.clip(best_oof, 1e-7, 1 - 1e-7)
            verified_logloss = log_loss(y, y_prob_verified)

            if abs(verified_logloss - study.best_value) > 0.01:
                print(f"⚠️ WARNING: Score mismatch detected for {name}!")            
            
            iteration_duration = time.time() - model_iteration_start
            
            # Save results
            resampling_label = str(resampling) if resampling else "None"
            col_name = f"{name}_{resampling_label}"
            oof_master_df[col_name] = best_oof

            all_results.append({
                "model": name,
                "resampling": resampling_label,
                "best_score": study.best_value,
                "best_params": study.best_params,
                "duration_minutes": round(iteration_duration / 60, 2)
            })

            del study
            gc.collect()
            
    # Final Exports
    oof_master_df["real"] = y
    oof_master_df.to_csv("results/oof_predictions/all_model_oofs.csv")
    pd.DataFrame(all_results).to_csv("benchmark_results.csv", index=False)

    print(f"\nPIPELINE COMPLETE | Total Time: {(time.time() - total_start_time)/60:.2f} minutes")

if __name__ == "__main__":
    # Ensure 'spawn' is used for CUDA safety
    mp.set_start_method('spawn', force=True)
    
    models_to_run = [
        "logistic", "rf", "et", "xgb", "lgbm", "catboost", 
        "knn", "dummy", "rusb", "mlp", "nca", "tabm", "icl", "tabpfn"
    ]

    run_benchmark(
        n_trials=20, # Set to more than 1 to actually see parallelization work
        model_names=models_to_run, 
        resampling_methods=[None]
    )