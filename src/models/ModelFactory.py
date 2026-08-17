# Standard Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Specialized Tabular Models
from tabarena.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel
from tabarena.benchmark.models.ag.tabm.tabm_model import TabMModel
from tabarena.benchmark.models.ag.tabicl.tabicl_model import TabICLModel
from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model


from tabarena.benchmark.models.ag.modernnca.modernnca_model import ModernNCAModel

# Import the spaces you just moved
from src.models.spaces import get_optuna_space
from imblearn.ensemble import RUSBoostClassifier

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
            return XGBClassifier(objective='binary:logistic', eval_metric='logloss', **params)

        elif name == "lgbm":
            if "bagging_fraction" in params and "bagging_freq" not in params:
                params["bagging_freq"] = 1
            return LGBMClassifier(verbose=-1, objective="binary", metric='binary_logloss', **params)

        elif name == "catboost":
            return CatBoostClassifier(verbose=0, boosting_type='Plain', **params)

        elif name == "knn":
            return KNeighborsClassifier(**params)

        elif name == "dummy":
            return DummyClassifier(strategy="most_frequent")

        elif name == "mlp":
            return RealMLPModel(hyperparameters = params)

        elif name == "nca":
            return ModernNCAModel(hyperparameters = params)

        elif name == "tabm":
            return TabMModel(hyperparameters = params)

        elif name == "icl":
            return TabICLModel(hyperparameters = params)

        elif name == "tabpfn":
            model = RealTabPFNv25Model(hyperparameters = params)
            if params and "checkpoint" in params:
                model.default_classification_model = params["checkpoint"]
            return model
        elif name == "rusb":
            return RUSBoostClassifier(**params)
            
        else:
            raise ValueError(f"Model {name} not found in factory.")

    @staticmethod
    def get_space(trial, name):
        """Helper to access the spaces from the factory"""
        return get_optuna_space(trial, name)
