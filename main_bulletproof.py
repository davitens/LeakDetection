import os
import time
import gc
import warnings
import traceback
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import log_loss
import multiprocessing as mp

# Assuming these are your local modules
from src.data_loader import load_and_clean_data
from src.preprocessing import preprocess_data
from src.objective import objective
from src.utils import get_best_model_oofs

# Suppress specific sklearn warnings
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
    module="sklearn"
)

def run_benchmark(n_trials=1, model_names=None, resampling_methods=None):
    """
    Runs optimization and tracks success/failure for each model configuration.
    """
    if model_names is None:
        model_names = ["tabm"]
    if resampling_methods is None:
        resampling_methods = [None, "SMOTE", "ADASYN", "RandomUnderSampler"]

    # --- Initialization ---
    print("Loading data...")
    X, y, cat_features, num_features = load_and_clean_data()
    
    print("Preprocessing data...")
    X = preprocess_data(X, cat_features, num_features)
    
    os.makedirs("results/oof_predictions", exist_ok=True)
    os.makedirs("results/optuna_trials", exist_ok=True)
    
    # Track status of every run
    run_history = []
    total_start_time = time.time()

    print(f"Starting pipeline for {len(model_names)} models...")

    # --- Main Loop ---
    for name in model_names:
        for resampling in resampling_methods:
            resampling_label = str(resampling) if resampling else "None"
            
            print("\n" + "="*60)
            print(f"RUNNING: {name.upper()} | RESAMPLING: {resampling_label}")
            print("="*60)
            
            iteration_start = time.time()
            status = "SUCCESS"
            error_details = None
            best_score = None

            try:
                # 1. Setup Optuna
                current_n_trials = 1 if name in ["tabdpt", "icl"] else n_trials
                sampler = optuna.samplers.TPESampler(multivariate=True)
                study = optuna.create_study(
                    direction="minimize", 
                    sampler=sampler,
                    study_name=f"{name}_{resampling_label}",
                    storage="sqlite:///results/optuna_trials/optuna_benchmark.db",
                    load_if_exists=True
                )
                
                # 2. Run Optimization
                study.optimize(
                    lambda trial: objective(trial, name, X, y, resampling), 
                    n_trials=current_n_trials,
                    timeout=1800
                )
                best_score = study.best_value

                # 3. Generate OOFs
                print(f"Generating OOF predictions for {name}...")
                best_oof = get_best_model_oofs(name, study.best_params, X, y, resampling)
                
                # 4. Save individual predictions
                individual_df = pd.DataFrame({
                    "prediction": best_oof,
                    "actual": y 
                }, index=X.index)
                file_path = f"results/oof_predictions/{name}_{resampling_label}_preds.csv"
                individual_df.to_csv(file_path, index=True)

            except Exception as e:
                # --- Error Capture ---
                status = "FAILED"
                error_details = traceback.format_exc()
                print(f"❌ CRITICAL FAILURE: {name} | {resampling_label}")
                print(error_details)
            
            finally:
                # --- Cleanup & Logging ---
                duration = time.time() - iteration_start
                
                # Log the result of this specific combination
                result_entry = {
                    "model": name,
                    "resampling": resampling_label,
                    "status": status,
                    "best_log_loss": best_score,
                    "duration_min": round(duration / 60, 2),
                    "error_msg": error_details.split('\n')[-2] if error_details else None
                }
                run_history.append(result_entry)
                
                # Update the master CSV after EVERY run
                pd.DataFrame(run_history).to_csv("results/pipeline_status.csv", index=False)

                # Force memory release
                if 'study' in locals(): del study
                if 'best_oof' in locals(): del best_oof
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print(f"Finished {name} - Status: {status}")

    # --- Final Summary ---
    total_duration = time.time() - total_start_time
    print("\n" + "!"*20 + " PIPELINE COMPLETE " + "!"*20)
    print(f"Total execution time: {total_duration/3600:.2f} hours")
    
    summary_df = pd.DataFrame(run_history)
    print("\nRun Summary:")
    print(summary_df[["model", "resampling", "status", "best_log_loss"]])

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    models_to_run = [
        #"logistic",
        #"rf",
        #"et",
        #"xgb",
        #"lgbm",
        #"catboost",
        #"knn",
        #"dummy",
        #"rusb",
        #"mlp",
        #"nca",
        #"tabm",
        #"icl"
    ]

    run_benchmark(
        n_trials=50, 
        model_names=models_to_run, 
        resampling_methods=[None, "SMOTE", "ADASYN", "RandomUnderSampler"]
    )