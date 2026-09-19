"""Predict the price of a house using the trained model.

    python src/predict.py                   # shows 5 example neighbourhoods
    python src/predict.py my_houses.csv     # predicts prices for your own file
"""

import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore", message=".*encountered in matmul")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
DATA_PATH = ROOT / "data" / "housing.csv"


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("No model yet. Run this first: python src/train.py")

    # Load the model we trained earlier. No re-learning needed.
    model = joblib.load(MODEL_PATH)

    # Did the user give us a file? If not, use 5 rows from the dataset.
    if len(sys.argv) > 1:
        houses = pd.read_csv(sys.argv[1])
    else:
        houses = pd.read_csv(DATA_PATH).sample(5, random_state=1)
        print("Showing 5 example neighbourhoods.\n")

    # Remove the price column if it is there. That is what we are predicting,
    # so the model must not be allowed to see it.
    clues = houses.drop(columns=["median_house_value"], errors="ignore")

    predicted_prices = model.predict(clues)

    table = pd.DataFrame({
        "income (10k$)": houses["median_income"].round(1).values,
        "rooms": houses["total_rooms"].astype(int).values,
        "location": houses["ocean_proximity"].values,
        "predicted price": [f"${p:,.0f}" for p in predicted_prices],
    })

    # If the real price is in the file, show it so we can compare.
    if "median_house_value" in houses.columns:
        table["real price"] = [f"${p:,.0f}" for p in houses["median_house_value"]]

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
