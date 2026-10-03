from functools import lru_cache      # FIX: load once, not on every prediction
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = Path("../models/predict_freight_model.pkl")

FEATURES = ["Dollars"]

@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ------------
    input_data : dict 

    Returns
    -------
    pd.DataFrame with predicted freight cost
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)

    # FIX: check required column
    missing = [c for c in FEATURES if c not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    preds = np.asarray(model.predict(input_df[FEATURES])).ravel()
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":

    #Example inference run (local testing)
    sample_data = {
        "Dollars" : [18500, 9000, 3000, 200] 
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)