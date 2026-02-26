import os
import time
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import optuna
import multiprocessing as mp
from sklearn.metrics import log_loss
from src.data_loader import load_and_clean_data
from src.preprocessing import preprocess_data
from src.objective import objective
from src.utils import get_best_model_oofs

warnings.filterwarnings("ignore", category=FutureWarning)

def gpu_worker(gpu_id, study_name, storage_url, n_trials, name, X, y, resampling):
    """
    A standalone process that runs on a specific GPU.
    """
    # 1. Isolate this process to one specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. Load the existing study from the database
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    # 3. Run the optimization
    # Note: objective() must be compatible with 'cuda' (not hardcoded to 'cuda:0')
    study.optimize(
        lambda trial: objective(trial, name, X, y, resampling), 
        n_trials=n_trials
    )
    
    print(f"GPU {gpu_id} finished {n_trials} trials.")

def run_benchmark(n_trials=20, model_names=None, resampling_methods=None):
    if model_names is None:
        model_names = ["tabpfn"]
    if resampling_methods is None:
        resampling_methods = [None]

    # --- Setup ---
    X, y, cat_features, num_features = load_and_clean_data()
    X = preprocess_data(X, cat_features, num_features)
    os.makedirs("results/oof_predictions", exist_ok=True)
    
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPUs. Splitting {n_trials} trials across them.")

    db_path = "optuna_parallel.db"
    storage_url = f"sqlite:///{db_path}"

    for resampling in resampling_methods:
        for name in model_names:
            print(f"\nSTARTING PARALLEL RUN: {name} | {resampling}")
            
            # --- Initialize Study in DB ---
            study_name = f"{name}_{resampling}"
            
            # Clear old study if it exists to start fresh
            try:
                optuna.delete_study(study_name=study_name, storage=storage_url)
            except:
                pass
                
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(multivariate=True),
                pruner=optuna.pruners.MedianPruner()
            )

            # --- Launch Parallel Processes ---
            trials_per_gpu = n_trials // n_gpus
            processes = []
            
            for i in range(n_gpus):
                p = mp.Process(
                    target=gpu_worker,
                    args=(i, study_name, storage_url, trials_per_gpu, name, X, y, resampling)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # --- Post-Optimization (Main Process) ---
            # Reload the study to get results from all workers
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            print(f"Best Score across all GPUs: {study.best_value:.4f}")

            # Generate OOFs using the best params
            best_oof = get_best_model_oofs(name, study.best_params, X, y, resampling)
            
            # Save results (using your original logic)
            individual_df = pd.DataFrame({"prediction": best_oof, "actual": y}, index=X.index)
            resampling_label = str(resampling) if resampling else "None"
            individual_df.to_csv(f"results/oof_predictions/{name}_{resampling_label}_preds.csv")
            
            print(f"Cleanup for {name} complete.")
            gc.collect()

if __name__ == "__main__":
    # ESSENTIAL for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)
    
    run_benchmark(
        n_trials=40, # Example: 20 trials per GPU if you have 2 GPUs
        model_names=["tabpfn"],
        resampling_methods=[None]
    )