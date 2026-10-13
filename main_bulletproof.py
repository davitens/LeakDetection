import os
import time
import gc
import warnings
import traceback
import argparse
import pandas as pd
import torch
import optuna
import multiprocessing as mp

# Assuming these are your local modules
from src.data_loader import DEFAULT_DATA_PATH, load_and_clean_data
from src.objective import objective

#: Resampling configurations benchmarked in the paper: baseline, the two
#: synthetic oversamplers and random undersampling.
RESAMPLING_METHODS = [None, "SMOTE", "ADASYN", "RandomUnderSampler"]

#: Neighbour counts SMOTE/ADASYN are evaluated with (manuscript, Architecture).
K_NEIGHBORS = [3, 5, 7, 9]

# Suppress specific sklearn warnings
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
    module="sklearn"
)

def run_benchmark(n_trials=1, model_names=None, resampling_methods=None, neightboors=K_NEIGHBORS, run_name="all_trials_results", data_path=DEFAULT_DATA_PATH):
    """
    Runs optimization and tracks success/failure for each model configuration.
    """
    if model_names is None:
        model_names = ["tabm"]
    if resampling_methods is None:
        resampling_methods = RESAMPLING_METHODS

    # --- Initialization ---
    print(f"Loading data from {data_path}...")
    X, y, cat_features, num_features = load_and_clean_data(data_path)
    
    print("Preprocessing will be fit inside each CV fold...")
    
    os.makedirs("results/oof_predictions", exist_ok=True)
    os.makedirs("results/optuna_trials", exist_ok=True)
    
    # DYNAMIC PATH: Uses the run_name passed from the command line
    master_results_path = f"results/optuna_trials/{run_name}.parquet"
    print(f"Results will be saved to: {master_results_path}")
    
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
                    current_n_trials = 1 if name in ["tabpfn", "icl"] else n_trials
                    sampler = optuna.samplers.TPESampler(multivariate=True)
                    study = optuna.create_study(
                        direction="minimize", 
                        sampler=sampler,
                        study_name=f"{name}_{resampling_label}",
                        storage=f"sqlite:///results/optuna_trials/{run_name}_optuna_registry.db",
                        load_if_exists=True
                    )
                    existing_trial_numbers = {t.number for t in study.trials}
                    
                    # 2. Run Optimization
                    study.optimize(
                        lambda trial: objective(trial, name, X, y, resampling, k=k, cat_features=cat_features, num_features=num_features), 
                        n_trials=current_n_trials,
                        timeout=1800,
                        n_jobs=-1
                    )
                    best_score = study.best_value

                    # --- POST-OPTIMIZATION DATA EXPORT ---
                    trials_df = study.trials_dataframe(attrs=('number', 'value', 'user_attrs', 'state'))
                    
                    # FILTER: Only keep 'COMPLETE' trials
                    complete_trials = trials_df[trials_df['state'].astype(str) == 'COMPLETE']
                    complete_trials = complete_trials[~complete_trials["number"].isin(existing_trial_numbers)]
                    print(f"DEBUG: Found {len(trials_df)} total trials. {len(complete_trials)} new trials are COMPLETE.")
                    
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

                        # Append to dynamic master Parquet
                        if not os.path.exists(master_results_path):
                            df_to_append.to_parquet(master_results_path, engine='pyarrow', index=False)
                        else:
                            df_old = pd.read_parquet(master_results_path)
                            pd.concat([df_old, df_to_append], ignore_index=True).to_parquet(master_results_path, engine='pyarrow', index=False)
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
                    
                    # Save a separate status file named after the run
                    pd.DataFrame(run_history).to_csv(f"results/status_{run_name}.csv", index=False)

                    # Cleanup
                    if 'study' in locals(): del study
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

    # --- HANDLE TERMINAL ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Leak benchmark runner")
    parser.add_argument("run_name", nargs="?", default="all_trials_results",
                        help="name used for the results/*.parquet and status CSVs")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                        help="dataset CSV (the manuscript analysis used data/modified_real.csv)")
    args = parser.parse_args()

    # Paper configuration: 12 models x 4 resamplings x k in [3, 5, 7, 9].
    # The list below is a smoke subset; uncomment the rest for a full run.
    models_to_run = [
#       "logistic",
#        "rf",
#        "et",
#        "xgb",
#        "lgbm",
#        "catboost",
        "knn",
        "dummy",
        "rusb",
#        "mlp",
#        "nca",
#        "tabm",
#        "icl"
    ]

    run_benchmark(
        n_trials=50, 
        model_names=models_to_run, 
        resampling_methods=RESAMPLING_METHODS,
        run_name=args.run_name,
        data_path=args.data,
    )
