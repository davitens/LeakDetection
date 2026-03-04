from optuna.trial import Trial

def get_optuna_space(trial, name, n_samples=None):
    if name == "catboost":
        return {
            "task_type": "GPU",
            "devices": "0",  # Specify the GPU index
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 5.0, log=True),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 20, log=True),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 8, 100, log=True),
            "model_size_reg": trial.suggest_float("model_size_reg", 0.1, 1.5, log=True),
            "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 2, 5),
        }
    
    elif name == "lgbm":
        return {
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 200, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 64, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 2.0),
        }
    
    elif name == "xgb":
        return {
            "tree_method": "hist",  # Use 'hist' with 'device' for newer XGB versions
            "device": "cuda",               # Use 'device': 'cuda' for XGBoost 2.0+
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 5.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        }
    elif name == "rusb":
        return {
               "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
               "n_estimators": trial.suggest_int("n_estimators", 50,800, log=True)
        }
    elif name == "rf":
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        
        params = {
            "bootstrap": bootstrap,
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 4, log=True),
            "n_estimators": 50,
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 1e-5, 1e-3, log=True),
        }

        # Only suggest max_samples if bootstrap is enabled
        if bootstrap:
            params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
        else:
            params["max_samples"] = None

        return params
    elif name == "logistic":
        return {
            "C": trial.suggest_float("C", 0.1, 1000, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
        }
    elif name == "knn":
        return {
            "n_neighbors": trial.suggest_categorical("n_neighbors", [1, 5, 10, 20, 50, 100]),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }

    elif name == "mlp":
            return {
            "n_hidden_layers": trial.suggest_int("n_hidden_layers", 1, 4),
            "hidden_sizes": "rectangular",
            "hidden_width": trial.suggest_int("hidden_width", 128, 512, step=128),
            "p_drop": trial.suggest_float("p_drop", 0.05, 0.5), 
            "act": "mish",
            # Range reduced from 100 to 2.0 for numerical stability with high LR
            "plr_sigma": trial.suggest_float("plr_sigma", 1e-3, 2.0, log=True), 
            "plr_hidden_1": trial.suggest_int("plr_hidden_1", 16, 32, step=16),
            "plr_hidden_2": trial.suggest_int("plr_hidden_2", 16, 64, step=16),
            "lr": trial.suggest_float("lr", 1e-4 * 32, 1e-1 * 32, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 0.5, log=True), 
            # Capping sq_mom at 0.999 to prevent division by zero/NaN in optimizer
            "sq_mom": 1.0 - trial.suggest_float("sq_mom", 1e-3, 1e-1, log=True),
            "plr_lr_factor": trial.suggest_float("plr_lr_factor", 0.01, 1.0, log=True),
            # Reduced max from 10.0 to 2.0 to prevent effective LR from hitting 30+
            "scale_lr_factor": trial.suggest_float("scale_lr_factor", 1.0, 2.0, log=True),
            "first_layer_lr_factor": trial.suggest_float("first_layer_lr_factor", 0.1, 2.0, log=True),
            "use_ls": True, 
            "ls_eps": trial.suggest_float("ls_eps", 0.001, 0.2, log=True),
            "ls_eps_sched": "coslog4",
            "batch_size": 16384,
            "n_epochs": 128, 
            "use_early_stopping": True
        }
    elif name == "nca":
            # Conditional logic for n_blocks (Choice: 0 OR [0, 2])
            # Since 0 is included in the range [0, 2], we can simplify or use a categorical
            n_blocks_choice = trial.suggest_categorical("n_blocks_choice", [0, "range"])
            n_blocks = 0 if n_blocks_choice == 0 else trial.suggest_int("n_blocks", 0, 2)

            # Conditional logic for weight_decay (Choice: 0.0 OR LogUniform)
            wd_choice = trial.suggest_categorical("wd_choice", [0.0, "loguniform"])
            weight_decay = 0.0 if wd_choice == 0.0 else trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            return {
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "d_block": trial.suggest_int("d_block", 64, 1024),
                "n_blocks": n_blocks,
                "dim": trial.suggest_int("dim", 64, 1024),
                "num_emb_n_frequencies": trial.suggest_int("num_emb_n_frequencies", 16, 96),
                "num_emb_frequency_scale": trial.suggest_float("num_emb_frequency_scale", 0.005, 10.0, log=True),
                "num_emb_d_embedding": trial.suggest_int("num_emb_d_embedding", 16, 64),
                "sample_rate": trial.suggest_float("sample_rate", 0.05, 0.6),
                "lr": trial.suggest_float("lr", 1e-4 * 32, 1e-1 * 32, log=True),
                "weight_decay": weight_decay,
                "temperature": 1.0,
                "num_emb_type": "plr",
                "num_emb_lite": True,
                "batch_size": 4096
            }
    elif name == "tabm":
        wd_choice = trial.suggest_categorical("wd_choice", [0.0, "loguniform"])
        weight_decay = 0.0 if wd_choice == 0.0 else trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        do_choice = trial.suggest_categorical("do_choice", [0.0, "uniform"])
        dropout = 0.0 if do_choice == 0.0 else trial.suggest_float("dropout", 0.0, 0.3)

        return {
            "batch_size": 2048,
            "patience": 16,
            "amp": True,
            "arch_type": "tabm-mini",
            "tabm_k": 32,
            "gradient_clipping_norm": 1.0,
            "share_training_batches": True,
            "lr": trial.suggest_float("lr", 1e-4 * 32, 1e-1 * 32, log=True),
            "weight_decay": weight_decay,
            "n_blocks": trial.suggest_int("n_blocks", 2, 5),
            "d_block": trial.suggest_int("d_block", 128, 1024, step=16),
            "dropout": dropout,
            "num_emb_type": "pwl",
            "d_embedding": trial.suggest_int("d_embedding", 8, 32, step=4),
            "num_emb_n_bins": trial.suggest_int("num_emb_n_bins", 2, 128),
        }
    elif name == "icl":
        return None
    elif name == "tabpfn":
        # 1. Mappings using TUPLES (to avoid Optuna persistent storage warnings)
        scaling_options = {
            "none": ("none",),
            "kdi_0.3_uni": ("kdi_alpha_0.3_uni",),  # Added your specific scaling here
            "q_uni": ("quantile_uni_coarse",),
            "q_norm": ("quantile_norm_coarse",),
            "kdi_uni": ("kdi_uni",),
            "kdi_0.3": ("kdi_alpha_0.3",),
            "kdi_3.0": ("kdi_alpha_3.0",),
            "safe_q": ("safepower", "quantile_uni"),
            "none_q": ("none", "quantile_uni_coarse"),
            "squash_q": ("squashing_scaler_default", "quantile_uni_coarse"),
            "squash": ("squashing_scaler_default",),
        }

        reg_y_options = {
            "none": (None,),
            "none_safe": (None, "safepower"),
            "safe": ("safepower",),
            "kdi_0.3": ("kdi_alpha_0.3",),
            "kdi_1.0": ("kdi_alpha_1.0",),
            "kdi_3.0": ("kdi_alpha_3.0",),
            "q_uni": ("quantile_uni",),
        }

        # 2. Suggest keys (Strings)
        scaling_key = trial.suggest_categorical("scaling_choice", list(scaling_options.keys()))
        reg_y_key = trial.suggest_categorical("reg_y_choice", list(reg_y_options.keys()))

        # 3. Construct the configuration dictionary
        return {
            "softmax_temperature": trial.suggest_categorical("softmax_temperature", [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5]),
            "balance_probabilities": trial.suggest_categorical("balance_probabilities", [True, False]),            
            "inference_config/OUTLIER_REMOVAL_STD": trial.suggest_categorical("inference_config/OUTLIER_REMOVAL_STD", [3, 6, 12]),
            "inference_config/POLYNOMIAL_FEATURES": trial.suggest_categorical("inference_config/POLYNOMIAL_FEATURES", ["no", 25]),
            
            # Map the suggested keys to their tuple values
            "inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS": reg_y_options[reg_y_key],
            "preprocessing/scaling": scaling_options[scaling_key],
            
            "preprocessing/categoricals": trial.suggest_categorical("preprocessing/categoricals", ["numeric", "onehot", "none"]),
            "preprocessing/append_original": trial.suggest_categorical("preprocessing/append_original", [True, False]),
            "preprocessing/global": trial.suggest_categorical("preprocessing/global", [None, "svd", "svd_quarter_components"]),
            "memory_saving_mode":True
        }
    elif name == "et":
        return {
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.75, 1.0]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 32, log=True),
            "bootstrap": False,
            "n_estimators": 50,
            "min_impurity_decrease": trial.suggest_categorical("min_impurity_decrease", [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),    
        }