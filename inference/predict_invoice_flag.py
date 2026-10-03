from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("../models/predict_flag_invoice.pkl")
# SCALER_PATH = Path("../models/scaler.pkl")

def load_model(model_path: Path = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

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
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df).round()
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