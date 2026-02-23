import pandas as pd
import numpy as np

def load_and_clean_data(file_path="data/modified.csv"):
    df = pd.read_csv(file_path, sep=";", index_col=0)
    
    target_col = "DW_Leak_No"
    df[target_col] = df[target_col].replace("NULL", 0)
    df["LeakPresence"] = df[target_col].notna().astype(int)
    
    df = df.replace("NULL", np.nan)
    
    def to_float_or_nan(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    df = df.map(to_float_or_nan)
    
    code_to_english = {
        237.0: "Ceramics", 238.0: "Fiberglass", 241.0: "Cast iron",
        243.0: "Lead", 244.0: "Concrete", 245.0: "Steel",
        246.0: "Asbestos cement", 248.0: "Other", 249.0: "PVC",
        250.0: "Polyethilen", 251.0: "Unknown", 252.0: "Polypropilen",
        253.0: "Ductile iron"
    }
    
    df["DW_atr1"] = df["DW_atr1"].map(code_to_english)
    
    remove_cols = ["DW_Leak_No", "LeakPresence", "DW_Leak_After_Inst", "DE_Pipe_Age"]
    
    X = df.drop(columns=remove_cols)
    y = df["LeakPresence"]
    
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=['number']).columns.tolist()
    
    return X, y, cat_features, num_features