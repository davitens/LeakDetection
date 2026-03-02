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

warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

def run_worker(gpu_id, model_names, resampling_methods, n_trials):
    """
    Each worker calls this function. It assigns itself to a specific GPU
    and contributes to the shared Optuna study in the DB.
    """
    # 1. Assign GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Worker started on {device}")

    # 2. Load Data (Each process needs its own copy in memory)
    X, y, cat_features, num_features = load_and_clean_data()
    X = preprocess_data(X, cat_features, num_features)
    
    db_path = "sqlite:///benchmarks.db"

    for resampling in resampling_methods:
        for name in model_names:
            # 3. Connect to Shared Study
            # All workers point to the same study_name and storage
            study = optuna.create_study(
                study_name=f"{name}_{resampling}",
                storage=db_path,
                load_if_exists=True,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(multivariate=True),
                pruner=optuna.pruners.MedianPruner()
            )
            
            # 4. Optimize
            # Workers will cooperatively finish n_trials
            study.optimize(
                lambda trial: objective(trial, name, X, y, resampling), 
                n_trials=n_trials,
                n_jobs=1 
            ) 
            
            # Cleanup for next model
            torch.cuda.empty_cache()
            gc.collect()

def finalize_results(model_names, resampling_methods):
    """
    After all workers finish, one process runs this to compile CSVs.
    """
    print("\nAll workers finished. Finalizing results...")
    db_path = "sqlite:///benchmarks.db"
    X, y, cat_features, num_features = load_and_clean_data()
    X = preprocess_data(X, cat_features, num_features)
    
    all_results = []
    
    for resampling in resampling_methods:
        for name in model_names:
            study = optuna.load_study(study_name=f"{name}_{resampling}", storage=db_path)
            
            print(f"Processing best results for {name}...")
            best_oof = get_best_model_oofs(name, study.best_params, X, y, resampling)
            
            # Save individual OOF
            resampling_label = str(resampling) if resampling else "None"
            pd.DataFrame({"pred": best_oof, "actual": y}).to_csv(f"results/{name}_{resampling_label}_preds.csv")
            
            all_results.append({
                "model": name,
                "resampling": resampling_label,
                "best_score": study.best_value,
                "best_params": study.best_params
            })

    pd.DataFrame(all_results).to_csv("results/final_summary.csv", index=False)

if __name__ == "__main__":
    # CONFIGURATION
    NUM_GPUS = torch.cuda.device_count() # Automatically detects available GPUs
    TRIALS_PER_GPU = 5 
    MODELS = ["mlp", "nca", "tabm"]
    RESAMPLING = [None]

    os.makedirs("results/oof_predictions", exist_ok=True)

    # Launching processes
    processes = []
    for i in range(NUM_GPUS):
        p = mp.Process(target=run_worker, args=(i, MODELS, RESAMPLING, TRIALS_PER_GPU))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Consolidate results
    finalize_results(MODELS, RESAMPLING)