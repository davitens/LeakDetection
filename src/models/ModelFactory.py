"""Model construction and Optuna search spaces.

The scikit-learn, XGBoost, LightGBM, CatBoost and imbalanced-learn learners are
always available. The tabarena tabular deep-learning models pull in several
extra stacks (and need the git checkout described in the README), so they are
imported only when one of them is actually requested.
"""

import importlib

import torch
from imblearn.ensemble import RUSBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.models.spaces import get_optuna_space

#: tabarena model name -> (module, class) for the optional tabular DL learners.
TABARENA_MODELS = {
    "mlp": ("tabarena.benchmark.models.ag.realmlp.realmlp_model", "RealMLPModel"),
    "nca": ("tabarena.benchmark.models.ag.modernnca.modernnca_model", "ModernNCAModel"),
    "tabm": ("tabarena.benchmark.models.ag.tabm.tabm_model", "TabMModel"),
    "icl": ("tabarena.benchmark.models.ag.tabicl.tabicl_model", "TabICLModel"),
    "tabpfn": ("tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model", "RealTabPFNv25Model"),
}

#: Models whose fit signature takes the tabarena device arguments.
TABARENA_FIT_KWARGS = tuple(TABARENA_MODELS)


def tabarena_fit_kwargs(num_cpus=6):
    """Extra ``fit()`` arguments the tabarena models expect.

    They must never ask for a GPU that does not exist, and ``num_cpus`` is the
    per-model budget, not a share of the machine: keep it small when several
    trials run in parallel.
    """
    return {"num_gpus": 1 if torch.cuda.is_available() else 0, "num_cpus": num_cpus}


def _build_tabarena_model(name, params):
    """Import the tabarena model on demand and wrap it in a fresh instance."""
    module_name, class_name = TABARENA_MODELS[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"the '{name}' model needs the optional 'tabarena' dependency; "
            "see the README for the install step"
        ) from error
    return getattr(module, class_name)(hyperparameters=params)


class ModelFactory:
    @staticmethod
    def get_model(name, params=None):
        params = params or {}

        if name == "logistic":
            if params.get("penalty") == "l1":
                params["solver"] = "liblinear"
            return LogisticRegression(**params)

        elif name == "rf":
            return RandomForestClassifier(**params)

        elif name == "et":
            return ExtraTreesClassifier(**params)

        elif name == "xgb":
            return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)

        elif name == "lgbm":
            if "bagging_fraction" in params and "bagging_freq" not in params:
                params["bagging_freq"] = 1
            return LGBMClassifier(verbose=-1, objective="binary", metric="binary_logloss", **params)

        elif name == "catboost":
            return CatBoostClassifier(verbose=0, boosting_type="Plain", **params)

        elif name == "knn":
            return KNeighborsClassifier(**params)

        elif name == "dummy":
            return DummyClassifier(strategy="most_frequent")

        elif name == "rusb":
            return RUSBoostClassifier(**params)

        elif name in TABARENA_MODELS:
            model = _build_tabarena_model(name, params)
            if name == "tabpfn" and params.get("checkpoint"):
                model.default_classification_model = params["checkpoint"]
            return model

        else:
            raise ValueError(f"Model {name} not found in factory.")

    @staticmethod
    def get_space(trial, name):
        """Helper to access the spaces from the factory"""
        return get_optuna_space(trial, name)
