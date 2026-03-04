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

def run_benchmark(n_trials=1, model_names=None, resampling_methods=None, neightboors = [3,7,9]):
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
    
    # Path for the master results file
    master_results_path = "results/optuna_trials/all_trials_results.parquet"
    
    run_history = []
    total_start_time = time.time()

    print(f"Starting pipeline for {len(model_names)} models...")

    # --- Main Loop ---
    for name in model_names:
        for resampling in resampling_methods:
            
            if resampling in ["SMOTE", "ADASYN"]:
                k_list = neightboors
            else:
                k_list = [None] 

            for k in k_list:
                resampling_label = str(resampling) if resampling else "None"
                if k is not None:
                    resampling_label += f"_k{k}"
                
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
                    
                    # 2. Run Optimization - Parallel execution enabled
                    study.optimize(
                        lambda trial: objective(trial, name, X, y, resampling, k=k), 
                        n_trials=current_n_trials,
                        timeout=1800,
                        n_jobs=-1
                    )
                    best_score = study.best_value

                # --- REFINED: POST-OPTIMIZATION DATA EXPORT ---
                    # 1. Extract trials with the 'state' attribute
                    trials_df = study.trials_dataframe(attrs=('number', 'value', 'user_attrs', 'state'))
                    
                    # 2. FILTER: Only keep 'COMPLETE' trials (excludes PRUNED and FAIL)
                    complete_trials = trials_df[trials_df['state'].astype(str) == 'COMPLETE']
                    print(f"DEBUG: Found {len(trials_df)} total trials. {len(complete_trials)} are COMPLETE.")
                    
                    if not complete_trials.empty:
                        df_to_append = pd.DataFrame({
                            "trial_number": complete_trials["number"],
                            "model": name,
                            "hyperparameters": complete_trials["user_attrs_space"],
                            "resampling": complete_trials["user_attrs_resampling"],
                            "predictions": complete_trials["user_attrs_predictions"],
                            "mean_log_loss": complete_trials["value"],
                            "k": complete_trials["user_attrs_k"]
                        })

                        # 3. Append to master Parquet (Sequential write is safe here)
                        if not os.path.exists(master_results_path):
                            df_to_append.to_parquet(master_results_path, engine='pyarrow', index=False)
                        else:
                            df_old = pd.read_parquet(master_results_path)
                            #pd.concat([df_old, df_to_append], ignore_index=True).to_parquet(master_results_path, engine='pyarrow', index=False)
                    else:
                        print(f"⚠️ No completed trials for {name} ({resampling_label}). Skipping Parquet export.")

                except Exception as e:
                    status = "FAILED"
                    error_details = traceback.format_exc()
                    print(f"❌ CRITICAL FAILURE: {name} | {resampling_label}")
                    print(error_details)
                
                finally:
                    duration = time.time() - iteration_start
                    result_entry = {
                        "model": name,
                        "resampling": resampling_label,
                        "k_neighbors": k,
                        "status": status,
                        "best_log_loss": best_score,
                        "duration_min": round(duration / 60, 2),
                        "error_msg": error_details.split('\n')[-2] if error_details else None
                    }
                    run_history.append(result_entry)
                    pd.DataFrame(run_history).to_csv("results/pipeline_status.csv", index=False)

                    # Cleanup
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
#       "logistic",
#        "rf",
#       "et",
#        "xgb",
#        "lgbm",
#        "catboost",
#        "knn",
#        "dummy",
#        "rusb",
#        "mlp",
#        "nca",
#        "tabm",
#        "icl"
    ]

    run_benchmark(
        n_trials=50, 
        model_names=models_to_run, 
        resampling_methods=["SMOTE", "ADASYN"] #[None, "SMOTE", "ADASYN", "RandomUnderSampler"] 
    )