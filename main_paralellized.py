import os
import time
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import optuna
import multiprocessing as mp
from joblib import Parallel, delayed
from sklearn.metrics import log_loss

# Import your custom modules
from src.data_loader import load_and_clean_data
from src.preprocessing import preprocess_data
from src.objective import objective
from src.utils import get_best_model_oofs

warnings.filterwarnings("ignore", category=FutureWarning)

def run_single_experiment(gpu_id, name, resampling, n_trials, X, y):
    """
    Worker function: Isolated to a specific GPU.
    """
    # 1. Automatic GPU Isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Give the OS a split second to register the environment change
    time.sleep(gpu_id * 0.5) 
    
    resampling_label = str(resampling) if resampling else "None"
    print(f"🚀 [GPU {gpu_id}] Starting: {name} | Resampling: {resampling_label}")
    
    start_time = time.time()
    
    # 2. Optuna Storage 
    # Using 'heartbeat' and 'failed' trial detection for distributed stability
    db_path = "sqlite:///optuna_master_results.db"
    study_name = f"{name}_{resampling_label}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.MedianPruner()
    )

    # 3. Optimization
    current_n_trials = 1 if name in ["tabdpt", "icl"] else n_trials
    study.optimize(
        lambda trial: objective(trial, name, X, y, resampling), 
        n_trials=current_n_trials,
        timeout=1800
    )

    # 4. Generate OOFs and Save
    best_oof = get_best_model_oofs(name, study.best_params, X, y, resampling)
    y_prob_verified = np.clip(best_oof, 1e-7, 1 - 1e-7)
    verified_logloss = log_loss(y, y_prob_verified)
    
    os.makedirs("results/oof_predictions", exist_ok=True)
    pd.DataFrame({
        "prediction": best_oof,
        "actual": y 
    }, index=X.index).to_csv(f"results/oof_predictions/{name}_{resampling_label}_preds.csv")

    # Cleanup
    duration = time.time() - start_time
    del best_oof
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": name,
        "resampling": resampling_label,
        "best_score": study.best_value,
        "verified_score": verified_logloss,
        "duration_minutes": round(duration / 60, 2),
        "gpu_used": gpu_id
    }

def run_benchmark_parallel(n_trials=5, model_names=None, resampling_methods=None):
    # --- Step 1: Hardware Detection ---
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs detected. Defaulting to CPU (1 worker).")
        num_gpus = 1
    else:
        print(f"✨ Detected {num_gpus} GPUs.")

    # --- Step 2: Data Prep ---
    X, y, cat_features, num_features = load_and_clean_data()
    X = preprocess_data(X, cat_features, num_features)

    # --- Step 3: Task Queuing ---
    tasks = []
    for resampling in resampling_methods:
        for name in model_names:
            tasks.append((name, resampling))
    print(f"🚀 Starting with Model: {tasks[0][0]} using {tasks[0][1]}")
    # --- Step 4: Parallel Dispatch ---
    # We don't want to spawn more workers than we have tasks
    n_workers = min(num_gpus, len(tasks))
    print(f"🔥 Spawning {n_workers} parallel workers...")

    results = Parallel(n_jobs=n_workers, backend="multiprocessing", verbose=10)(
        delayed(run_single_experiment)(
            i % num_gpus, # This cycles tasks across GPUs: 0,1,2,3,0,1...
            task[0], 
            task[1], 
            n_trials, 
            X, y
        ) for i, task in enumerate(tasks)
    )

    # --- Step 5: Final Aggregation ---
    summary_df = pd.DataFrame(results)
    summary_df.to_csv("results/final_benchmark_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("ALL MODELS COMPLETE")
    print(summary_df[["model", "resampling", "verified_score", "duration_minutes"]])
    print("="*60)

if __name__ == "__main__":
    # REQUIRED for GPU multi-processing
    mp.set_start_method('spawn', force=True)
    
    # Example usage
    models = ["tabpfn", "xgb", "lgbm", "catboost", "mlp"]
    resampling = [None]

    run_benchmark_parallel(
        n_trials=10, 
        model_names=models, 
        resampling_methods=resampling
    )