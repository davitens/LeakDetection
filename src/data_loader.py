"""Load and clean the Ljubljana pipe-failure table.

Column meanings are documented in ``data/attributes.txt``. ``DW_Leak_No`` holds
the number of leaks recorded for a pipe (empty when the pipe never leaked) and
the ``*_atr*`` columns are nominal integer codes.
"""

import numpy as np
import pandas as pd

TARGET_COL = "DW_Leak_No"

# Dropped before modelling:
#   DW_Leak_No         -> the label itself
#   DW_Leak_After_Inst -> read off the leak record (label leakage)
#   DE_Pipe_Age        -> current year - DW_leto_izgra, and the year is kept
LEAK_DERIVED_COLS = [TARGET_COL, "DW_Leak_After_Inst", "DE_Pipe_Age"]

# Nominal integer codes. Every value is a category, not a magnitude, so these
# must be imputed and encoded as categories rather than scaled as numbers.
CAT_FEATURES = [
    "DW_atr1",  # pipe material
    "DW_atr4",  # distribution network type
    "RO_atr1",  # road category
    "RR_atr2",  # rail category
    "EE_atr2",  # cable voltage
    "EK_atr1",  # underground cable installation
    "ZP_atr4",  # gas pipe dimension range
    "DH_atr2",  # district heat dimension range
    "WW_atr1",  # wastewater network type
]

#: Dataset used by the benchmark. The manuscript analysis scripts read
#: ``data/modified_real.csv`` instead; see the README.
DEFAULT_DATA_PATH = "data/modified.csv"

#: Pipe material codes. ``data/attributes.txt`` names the column but not its
#: codes, so this is the only copy of the legend in the repository. The model
#: consumes the codes themselves; reporting code uses this to label them.
MATERIAL_LABELS = {
    237: "Ceramics",
    238: "Fiberglass",
    241: "Cast iron",
    243: "Lead",
    244: "Concrete",
    245: "Steel",
    246: "Asbestos cement",
    248: "Other",
    249: "PVC",
    250: "Polyethilen",
    251: "Unknown",
    252: "Polypropilen",
    253: "Ductile iron",
}


def _to_float_or_nan(value):
    """Numeric cell -> float, anything else -> NaN."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def leak_presence(leak_count):
    """1 where a pipe has at least one recorded leak.

    A cell that is not a number carries no leak record whatever token the export
    used, so it means "no leak". A recorded count of 0 keeps the original
    reading: a record exists, so the pipe is labelled as leaking.
    """
    return pd.to_numeric(leak_count, errors="coerce").notna().astype(int)


def load_and_clean_data(file_path=DEFAULT_DATA_PATH):
    """Return ``(X, y, cat_features, num_features)`` for the leak benchmark."""
    df = pd.read_csv(file_path, sep=";", index_col=0)

    y = leak_presence(df[TARGET_COL])
    df = df.drop(columns=[column for column in LEAK_DERIVED_COLS if column in df.columns])
    df = df.map(_to_float_or_nan)

    cat_features = [column for column in CAT_FEATURES if column in df.columns]
    num_features = [column for column in df.columns if column not in cat_features]

    return df, y, cat_features, num_features
