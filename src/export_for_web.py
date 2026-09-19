"""Turn the trained model into a JSON file a web browser can use.

The website has no server. It downloads this JSON file and does the maths in
JavaScript, which is why the site is instant and free to host.

A Random Forest is just a pile of yes/no question trees, so all we need to save
is: the questions, the answers at the end of each branch, and the numbers needed
to prepare the input the same way training did.
"""

import json
import warnings
from pathlib import Path

import joblib

warnings.filterwarnings("ignore", message=".*encountered in matmul")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
OUTPUT_PATH = ROOT / "web" / "model.json"


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("No model yet. Run this first: python src/train.py")

    model = joblib.load(MODEL_PATH)
    preparation = model.named_steps["prep"]
    forest = model.named_steps["model"]

    # The three preparation steps, saved as plain numbers.
    number_part = preparation.named_transformers_["numbers"]
    medians = number_part.named_steps["fill_blanks"].statistics_
    scaler = number_part.named_steps["rescale"]
    categories = preparation.named_transformers_["words"].categories_[0]

    # Every tree, flattened into simple lists.
    trees = []
    for tree in forest.estimators_:
        t = tree.tree_
        trees.append({
            # -1 means "this is the end of the branch, stop here"
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            # which of the 13 inputs this question is about
            "feature": t.feature.tolist(),
            # The question is: "is that input below this number?"
            # These are NOT rounded. Rounding even to 6 decimals pushes values
            # that sit exactly on a boundary down the wrong branch, which can
            # change a prediction by thousands of dollars.
            "threshold": [float(v) for v in t.threshold],
            # the price this branch predicts
            "value": [round(float(v[0][0]), 1) for v in t.value],
        })

    bundle = {
        "numeric_columns": [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
        ],
        # Full precision here too, for the same reason as the thresholds.
        "medians": [float(v) for v in medians],
        "means": [float(v) for v in scaler.mean_],
        "scales": [float(v) for v in scaler.scale_],
        "categories": list(categories),
        "trees": trees,
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(bundle, separators=(",", ":")))

    size_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    total_nodes = sum(len(t["left"]) for t in trees)
    print(f"Exported {len(trees)} trees, {total_nodes:,} questions in total.")
    print(f"Saved {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
