from functools import lru_cache          # FIX: load once, not on every prediction
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

@lru_cache(maxsize=1)
def load_scaler(scaler_path: Path = SCALER_PATH):
    """
    Load the StandardScaler used during training.
    """
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ------------
    input_data : dict 

    Returns
    -------
    pd.DataFrame with predicted flag
    """

    model = load_model()
    scaler = load_scaler()
    input_df = pd.DataFrame(input_data)

    # FIX: check that all required columns are present
    missing = [c for c in FEATURES if c not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # FIX: scale first, exactly like training
    X = scaler.transform(input_df[FEATURES])

    input_df['Predicted_Flag'] = model.predict(X).astype(int)
    input_df['Risk_Probability'] = model.predict_proba(X)[:, 1].round(3)
    return input_df

if __name__ == "__main__":

    # Local testing
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [50],
        "total_item_dollars": [350.0],
    }
    print(predict_invoice_flag(sample_data))