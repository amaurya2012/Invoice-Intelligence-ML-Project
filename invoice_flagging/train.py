from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from modeling_evaluation import train_random_forest, evaluate_classifier
import joblib

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"

def main():

    print("1. Loading data...")
    df = load_invoice_data()
    df = apply_labels(df)

    print("2. Splitting and scaling data...")
    X_train, X_test, Y_train, Y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, '../models/scaler.pkl'
    )

    print("3. Starting GridSearchCV for Random Forest Classifier...")
    grid_search = train_random_forest(X_train_scaled, Y_train)

    print("4. Evaluating the model...")
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        Y_test,
        "Random Forest Classifier"
    )

    print("5. Saving the model...")
    joblib.dump(grid_search.best_estimator_, '../models/predict_flag_invoice.pkl')
    print("Done! Everything ran successfully.")

if __name__ == "__main__":
    main()