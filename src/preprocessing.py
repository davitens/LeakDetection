"""Fold-local preprocessing: impute, encode nominal codes, scale numerics.

Fit the returned transformer inside every cross-validation fold. Imputation
statistics and encoder categories must never see the validation partition.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

#: Sentinel for a category that was never observed. No code in
#: ``src.data_loader.CAT_FEATURES`` uses -1, so it cannot collide with a real
#: category.
MISSING_CATEGORY = -1


def make_preprocessor(cat_features, num_features):
    """Return an unfitted ``ColumnTransformer``.

    Nominal code columns are imputed with the missing-category sentinel and
    ordinally encoded; numeric columns are median-imputed and standardised.
    Empty features are kept so that the output width cannot change with the
    fold.
    """
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value=MISSING_CATEGORY,
            keep_empty_features=True,
        )),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=MISSING_CATEGORY,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    return preprocessor
