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

# Suppress specific sklearn warnings
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
    module="sklearn"
)

def run_benchmark(n_trials=1, model_names=None, resampling_methods=None):
    """
    Runs the optimization and OOF prediction pipeline for a list of models 
    and resampling techniques.
    """
    if model_names is None:
        model_names = ["tabm"]
    if resampling_methods is None:
        resampling_methods = ["RandomUnderSampler", None, "SMOTE", "ADASYN"]

    # --- Initialization ---
    print("Loading data...")
    X, y, cat_features, num_features = load_and_clean_data()
    
    print("Preprocessing data...")
    X = preprocess_data(X, cat_features, num_features)
    
    os.makedirs("results/oof_predictions", exist_ok=True)
    
    oof_master_df = pd.DataFrame(index=X.index)
    all_results = []
    total_start_time = time.time()

    print(f"Starting pipeline for models: {model_names}")

    # --- Main Loop ---
    for resampling in resampling_methods:
        for name in model_names:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            model_iteration_start = time.time()
            
            print("\n" + "="*50)
            print(f"RUNNING: {name.upper()} | RESAMPLING: {resampling}")
            print("="*50)
            
            # Adjust trials for specific heavy models if necessary
            current_n_trials = 1 if name in ["tabdpt", "icl"] else n_trials
            
            # Setup Optuna Study
            sampler = optuna.samplers.TPESampler(multivariate=True)
            pruner = optuna.pruners.MedianPruner()
            study = optuna.create_study(
                direction="minimize", 
                sampler=sampler, 
                pruner=pruner
            )
            
            # Run Optimization
            study.optimize(
                lambda trial: objective(trial, name, X, y, resampling), 
                n_trials=current_n_trials,
                n_jobs=1,
                timeout=1800
            ) 

            print(f"Best Log Loss: {study.best_value:.4f}")
            
            # Generate OOF predictions
            print(f"Generating OOF predictions for best {name}...")
            # Generate OOF predictions
            print(f"Generating OOF predictions for best {name}...")
            best_oof = get_best_model_oofs(name, study.best_params, X, y, resampling)

            # --- VERIFICATION STEP ---
            # Clip predictions to match the objective function's behavior
            y_prob_verified = np.clip(best_oof, 1e-7, 1 - 1e-7)
            verified_logloss = log_loss(y, y_prob_verified)

            print(f"Optuna Best Score: {study.best_value:.4f}")
            print(f"Verified OOF Score: {verified_logloss:.4f}")

            # Check for significant mismatch
            if abs(verified_logloss - study.best_value) > 0.01:
                print(f"⚠️ WARNING: Score mismatch detected for {name}!")            
            iteration_duration = time.time() - model_iteration_start
            
            # Save individual results
            resampling_label = str(resampling) if resampling else "None"
            individual_df = pd.DataFrame({
                "prediction": best_oof,
                "actual": y 
            }, index=X.index)
            
            file_path = f"results/oof_predictions/{name}_{resampling_label}_preds.csv"
            individual_df.to_csv(file_path, index=True)
            
            col_name = f"{name}_{resampling_label}"
            oof_master_df[col_name] = best_oof

            # Metadata tracking
            all_results.append({
                "model": name,
                "resampling": resampling_label,
                "best_score": study.best_value,
                "best_params": study.best_params,
                "duration_seconds": round(iteration_duration, 2),
                "duration_minutes": round(iteration_duration / 60, 2)
            })

            # --- GPU & Memory Cleanup ---
            del study
            del best_oof
            del individual_df
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize() 
            
            print(f"Memory cleared for {name}. Moving to next configuration...")

    # --- Final Exports ---
    oof_master_df["real"] = y
    oof_master_df.to_csv("results/oof_predictions/all_model_oofs.csv")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("results/model_run_summary.csv", index=False)
    summary_df.to_csv("benchmark_results.csv", index=False)

    total_duration = time.time() - total_start_time
    print("\n" + "!"*20 + " FINAL RESULTS " + "!"*20)
    print(summary_df)
    print("\n" + "="*50)
    print(f"PIPELINE COMPLETE | Total Time: {total_duration/60:.2f} minutes")
    print("="*50)


    return summary_df

if __name__ == "__main__":
    # Define the models list
    mp.set_start_method('spawn', force=True)
    models_to_run = [
#        # Standard Models
        "logistic", 
        "rf", 
        "et", 
        "xgb", 
        "lgbm", 
        "catboost", 
        "knn", 
        "dummy",
        "rusb",
        "mlp", 
        "nca", 
        "tabm", 
        "icl", 
        "tabpfn"
    ]

    run_benchmark(
        n_trials=1, 
        model_names=models_to_run, 
        resampling_methods=[None]
    )
    # Run the benchmark
    #run_benchmark(
    #    n_trials=5, 
    #    model_names=models_to_run, 
    #    resampling_methods=[None, "SMOTE", "ADASYN", "RandomUnderSampler"]
    #)

torch.cuda.empty_cache()