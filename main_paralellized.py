import os
import time
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import optuna
import multiprocessing as mp

# Ensure your local project imports still work
from src.data_loader import load_and_clean_data
from src.preprocessing import preprocess_data
from src.objective import objective
from src.utils import get_best_model_oofs

warnings.filterwarnings("ignore", category=FutureWarning)

def gpu_worker(gpu_id, study_name, storage_url, n_trials, name, X, y, resampling):
    """
    A standalone process that runs on a specific GPU.
    """
    # 1. Enable logging inside the child process so you see the 'evolution'
    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # 2. Isolate this process to one specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 3. Load the existing study
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    print(f"\n>>> [GPU {gpu_id}] Starting {n_trials} trials...")
    
    # 4. Run optimization with Progress Bar
    # Note: TQDM progress bars might stack/overlap in parallel, but will show progress
    study.optimize(
        lambda trial: objective(trial, name, X, y, resampling), 
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n<<< [GPU {gpu_id}] Finished all trials.")

def run_benchmark(n_trials=40, model_names=None, resampling_methods=None):
    if model_names is None:
        model_names = ["tabpfn"]
    if resampling_methods is None:
        resampling_methods = [None]

    # --- Setup ---
    X, y, cat_features, num_features = load_and_clean_data()
    X = preprocess_data(X, cat_features, num_features)
    os.makedirs("results/oof_predictions", exist_ok=True)
    
    n_gpus = torch.cuda.device_count()
    db_path = "optuna_parallel.db"
    storage_url = f"sqlite:///{db_path}"

    print(f"Detected {n_gpus} GPUs. Storage: {storage_url}")
    print(f"To view LIVE evolution, run this in a new terminal:")
    print(f"    optuna-dashboard {storage_url}")

    for resampling in resampling_methods:
        for name in model_names:
            study_name = f"{name}_{resampling}"
            
            # Initialize/Clear Study
            try:
                optuna.delete_study(study_name=study_name, storage=storage_url)
            except:
                pass
                
            optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True),
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

            # --- Post-Optimization ---
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            print(f"\n🏆 FINAL RESULTS FOR {name} | {resampling}")
            print(f"Best Score: {study.best_value:.4f}")

            # Generate OOFs using best params
            best_oof = get_best_model_oofs(name, study.best_params, X, y, resampling)
            
            individual_df = pd.DataFrame({"prediction": best_oof, "actual": y}, index=X.index)
            resampling_label = str(resampling) if resampling else "None"
            individual_df.to_csv(f"results/oof_predictions/{name}_{resampling_label}_preds.csv")
            
            gc.collect()

if __name__ == "__main__":
    # ESSENTIAL for CUDA + Multiprocessing
    mp.set_start_method('spawn', force=True)
    run_benchmark(model_names = ["tabpfn"],n_trials=40)