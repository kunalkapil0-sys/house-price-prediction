"""Train a model that predicts house prices in California.

Steps: load the data, prepare it, train two models, keep the better one.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # Save charts as files instead of opening windows.

# Macs with Apple chips print a scary "overflow" warning that does not affect
# the answer at all. This hides that one message only.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"


def main():
    # ---------------------------------------------------------------
    # 1. Load the data
    # ---------------------------------------------------------------
    if not DATA_PATH.exists():
        raise SystemExit("No data found. Run this first: python src/download_data.py")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} neighbourhoods.")
    print(f"Average house price: ${df['median_house_value'].mean():,.0f}\n")

    # X = the clues we get. y = the answer we want to predict.
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]

    # ---------------------------------------------------------------
    # 2. Prepare the columns
    # ---------------------------------------------------------------
    # Columns full of numbers, and the one column full of words.
    number_columns = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
    ]
    word_columns = ["ocean_proximity"]

    # Numbers: fill the 207 empty total_bedrooms cells with the middle value,
    # then put every column on the same scale.
    number_steps = Pipeline([
        ("fill_blanks", SimpleImputer(strategy="median")),
        ("rescale", StandardScaler()),
    ])

    # Words: "NEAR BAY" becomes a column of 1s and 0s, because a model can
    # only do maths on numbers.
    word_steps = OneHotEncoder(handle_unknown="ignore")

    preparation = ColumnTransformer([
        ("numbers", number_steps, number_columns),
        ("words", word_steps, word_columns),
    ])

    # ---------------------------------------------------------------
    # 3. Split into a study set and a test set
    # ---------------------------------------------------------------
    # The model learns from 80% and is tested on the 20% it has never seen,
    # so the score is honest.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Learning from {len(X_train)} neighbourhoods, "
          f"testing on {len(X_test)}.\n")

    # ---------------------------------------------------------------
    # 4. Train two models and compare them
    # ---------------------------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        # min_samples_leaf=2 stops the trees splitting down to single houses.
        # It is slightly more accurate AND makes the saved file 8x smaller.
        "Random Forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    trained = {}

    for name, algorithm in models.items():
        model = Pipeline([("prep", preparation), ("model", algorithm)])
        model.fit(X_train, y_train)               # <- this is the learning

        guesses = model.predict(X_test)

        # How many dollars off is a typical guess?
        average_error = mean_absolute_error(y_test, guesses)

        # Accuracy as a percentage. First work out how far off each guess is
        # as a share of the real price, then take 100% minus the average.
        # So 82% accurate means a typical guess is 18% away from the truth.
        accuracy = 100 - (mean_absolute_percentage_error(y_test, guesses) * 100)

        # How often does the guess land close to the real price?
        how_far_off = np.abs(guesses - y_test) / y_test
        within_10 = (how_far_off <= 0.10).mean() * 100
        within_20 = (how_far_off <= 0.20).mean() * 100

        # R2: 1.0 would be perfect, 0.0 would be no better than always
        # guessing the average price.
        score = r2_score(y_test, guesses)

        results[name] = {
            "average_error": average_error,
            "accuracy": accuracy,
            "within_10": within_10,
            "within_20": within_20,
            "r2": score,
        }
        trained[name] = model
        print(f"{name:<18} accuracy {accuracy:.1f}%   "
              f"average error ${average_error:,.0f}   R2 {score:.3f}")

    # ---------------------------------------------------------------
    # 5. Keep the better model
    # ---------------------------------------------------------------
    best_name = "Random Forest"        # it wins clearly, see the numbers above
    best_model = trained[best_name]
    best = results[best_name]
    print(f"\nKeeping the {best_name}.")
    print(f"  Accuracy:            {best['accuracy']:.1f}%")
    print(f"  Typical miss:        ${best['average_error']:,.0f} "
          f"(on homes averaging ${y.mean():,.0f})")
    print(f"  Within 10% of price: {best['within_10']:.0f} out of every 100 guesses")
    print(f"  Within 20% of price: {best['within_20']:.0f} out of every 100 guesses")

    # ---------------------------------------------------------------
    # 6. Save the model and a picture of how it did
    # ---------------------------------------------------------------
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    # compress=3 shrinks the saved file so it fits comfortably on GitHub.
    joblib.dump(best_model, ROOT / "models" / "house_price_model.joblib", compress=3)

    # Each dot is one neighbourhood. The closer the dots sit to the red line,
    # the better the guess was.
    guesses = best_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, guesses, alpha=0.2, s=8, color="#2b6cb0")
    ax.plot([0, 500000], [0, 500000], "r--", linewidth=1.5, label="perfect guess")
    ax.set_xlabel("Real price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title(f"{best_name}: predicted vs real price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "predictions.png", dpi=150)
    plt.close(fig)

    (ROOT / "reports" / "metrics.json").write_text(json.dumps({
        "dataset": "California Housing (camnugent/california-housing-prices)",
        "rows": len(df),
        "model_used": best_name,
        "accuracy_percent": round(best["accuracy"], 1),
        "average_error_dollars": round(best["average_error"]),
        "within_10_percent": round(best["within_10"], 1),
        "within_20_percent": round(best["within_20"], 1),
        "r2_score": round(best["r2"], 3),
        "all_models": {k: {"accuracy_percent": round(v["accuracy"], 1),
                           "average_error_dollars": round(v["average_error"]),
                           "r2_score": round(v["r2"], 3)}
                       for k, v in results.items()},
    }, indent=2) + "\n")

    print("\nSaved: models/house_price_model.joblib")
    print("Saved: reports/predictions.png and reports/metrics.json")


if __name__ == "__main__":
    main()
