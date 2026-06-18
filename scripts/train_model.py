"""
train_model.py
================
Trains a model to predict `obj_purchasePrice` (total property purchase
price) for the Niedersachsen real-estate listings in
`data/price_clean.csv`.

What this script does
----------------------
1. Loads the cleaned CSV.
2. Drops columns that are useless or that leak the target.
3. Engineers a handful of new features from the raw columns.
4. Builds a preprocessing pipeline (imputation, scaling, encoding).
5. Trains several candidate models with cross-validation, picks the
   best one, then evaluates it on a held-out test set.
6. Saves the winning pipeline (preprocessing + model bundled together)
   to disk so it can be reused for predictions later.

Run it from the repo root with:
    python train_model.py
or point it at a different file with:
    python train_model.py --data path/to/other_file.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

TARGET = "obj_purchasePrice"
# The listings were scraped in April 2020 -- ages are computed relative
# to that, not to today, since that's what the price reflects.
REFERENCE_YEAR = 2020
RANDOM_STATE = 42


class Winsorizer(BaseEstimator, TransformerMixin):
    """Caps each numeric column to its [lower_q, upper_q] percentile range,
    learned from the training fold only.

    Why this is needed: a couple of listings in this dataset have data-entry
    quirks (e.g. one row lists 564 sqm of living space against a single
    room, giving a "living space per room" of 564 vs. a typical value
    around 25-30). Linear models are very sensitive to such high-leverage
    points -- combined with the log-target transform below, a single
    outlier like that can blow up a prediction from a few hundred thousand
    euros to tens of millions, wrecking the error metrics for that fold.
    Capping extreme values keeps every model honest without throwing the
    rows away entirely.
    """

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {path}")
    return df


# ---------------------------------------------------------------------
# 2 & 3. Cleaning + feature engineering
# ---------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Drop columns that are useless or leak the target ---------
    # obj_regio1 is constant ("Niedersachsen") for every row -> no signal.
    # obj_houseNumber / obj_street are ~91% missing and too granular to
    # use without a geocoding step -> drop.
    # obj_purchasePrice_per_qm is literally obj_purchasePrice /
    # obj_livingSpace -> training on it alongside obj_livingSpace would
    # let the model trivially reconstruct the target (leakage) -> drop.
    # obj_livingSpaceRange is just obj_livingSpace pre-binned by the
    # source site -> redundant with obj_livingSpace -> drop.
    df = df.drop(
        columns=[
            "obj_regio1",
            "obj_houseNumber",
            "obj_street",
            "obj_purchasePrice_per_qm",
            "obj_livingSpaceRange",
        ],
        errors="ignore",
    )

    # --- Missing-value indicator flags ------------------------------
    # Before imputing, record *whether* a value was missing. For listing
    # data, "missing" is often informative on its own (e.g. agents who
    # skip the condition field may be hiding something), so this lets
    # the model use that signal even after the value itself is imputed.
    for col in [
        "obj_firingTypes",
        "obj_condition",
        "obj_telekomInternetProductAvailable",
        "obj_telekomUploadSpeed",
        "obj_telekomDownloadSpeed",
    ]:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # --- Encode y/n and boolean columns as 0/1 -----------------------
    for col in ["obj_newlyConst", "obj_cellar", "obj_barrierFree"]:
        df[col] = df[col].map({"y": 1, "n": 0})

    # obj_telekomInternetProductAvailable is True/False/NaN; missing
    # already captured above, so treat NaN as "not available" (0).
    df["obj_telekomInternetProductAvailable"] = (
        df["obj_telekomInternetProductAvailable"].fillna(False).astype(int)
    )

    # --- Building age instead of raw construction year ---------------
    # A year like 1965 has no meaningful scale to a linear model; age
    # (in years, relative to the 2020 listing snapshot) is monotonic
    # with "how old/new" and is what actually drives price.
    df["building_age"] = REFERENCE_YEAR - df["obj_yearConstructed"]
    df = df.drop(columns=["obj_yearConstructed"])

    # --- Postal code -> coarser region bucket -------------------------
    # geo_plz is a 4-5 digit code; the raw number isn't meaningfully
    # ordinal for price. Its first two digits correspond to a broad
    # postal region in Germany, giving a lower-cardinality location
    # signal that complements geo_krs/obj_regio3.
    df["plz_prefix"] = df["geo_plz"].astype(int).astype(str).str.zfill(5).str[:2]
    df = df.drop(columns=["geo_plz"])

    # --- Living space per room -----------------------------------------
    # Two properties with the same living space but different room
    # counts (open-plan loft vs. many small rooms) tend to be priced
    # differently; this ratio captures that directly.
    df["living_space_per_room"] = df["obj_livingSpace"] / df["obj_noRooms"].replace(0, np.nan)

    return df


def remove_price_outliers(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """Drop extreme price outliers (likely data-entry errors) using the
    IQR rule with a wide multiplier so only genuinely extreme values
    are removed, not just the normal tail of the price distribution."""
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    keep = df[TARGET].between(lower, upper)
    removed = len(df) - keep.sum()
    if removed:
        print(f"Removed {removed} extreme price outlier(s) outside [{lower:,.0f}, {upper:,.0f}]")
    return df[keep]


# ---------------------------------------------------------------------
# 4. Preprocessing pipeline
# ---------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "obj_livingSpace",
        "obj_noRooms",
        "building_age",
        "living_space_per_room",
        "obj_telekomUploadSpeed",
        "obj_telekomDownloadSpeed",
    ]
    binary_features = [
        "obj_newlyConst",
        "obj_cellar",
        "obj_barrierFree",
        "obj_telekomInternetProductAvailable",
        "obj_firingTypes_missing",
        "obj_condition_missing",
        "obj_telekomInternetProductAvailable_missing",
        "obj_telekomUploadSpeed_missing",
        "obj_telekomDownloadSpeed_missing",
    ]
    categorical_features = [
        "obj_firingTypes",
        "obj_condition",
        "geo_krs",
        "obj_regio3",
        "plz_prefix",
    ]

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("winsorize", Winsorizer()),
            ("scale", StandardScaler()),
        ]
    )
    binary_pipe = SimpleImputer(strategy="constant", fill_value=0)
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            # min_frequency buckets rare categories (e.g. small towns
            # with only a couple of listings) into a single "infrequent"
            # group instead of giving each its own sparse column.
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("binary", binary_pipe, binary_features),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )
    return preprocessor


# ---------------------------------------------------------------------
# 5. Model comparison
# ---------------------------------------------------------------------
def candidate_models() -> dict:
    """Each model is wrapped in a TransformedTargetRegressor so it's
    trained on log1p(price). Prices are right-skewed (lots of cheaper
    homes, a long tail of expensive ones); fitting in log-space keeps
    that tail from dominating the loss and tends to improve both
    accuracy and the validity of the error metrics. Predictions are
    automatically converted back to euros via expm1."""

    def wrap(estimator):
        return TransformedTargetRegressor(
            regressor=estimator, func=np.log1p, inverse_func=np.expm1
        )

    return {
        "Ridge Regression": wrap(RidgeCV(alphas=np.logspace(-1, 3, 20))),
        "Random Forest": wrap(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "Hist Gradient Boosting": wrap(
            HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.05,
                max_iter=400,
                random_state=RANDOM_STATE,
            )
        ),
    }


def compare_models(X_train, y_train, preprocessor) -> tuple[str, Pipeline, pd.DataFrame]:
    """Cross-validate each candidate model and return the best one."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted_pipelines = {}

    for name, model in candidate_models().items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        mae_scores = -cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        r2_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        results.append(
            {
                "model": name,
                "cv_mae_mean": mae_scores.mean(),
                "cv_mae_std": mae_scores.std(),
                "cv_r2_mean": r2_scores.mean(),
                "cv_r2_std": r2_scores.std(),
            }
        )
        fitted_pipelines[name] = pipe
        print(
            f"  {name:<25} CV MAE: {mae_scores.mean():>10,.0f} EUR  |  "
            f"CV R2: {r2_scores.mean():.3f}"
        )

    results_df = pd.DataFrame(results).sort_values("cv_r2_mean", ascending=False)
    best_name = results_df.iloc[0]["model"]
    return best_name, fitted_pipelines[best_name], results_df


def print_top_features(pipeline: Pipeline, X_test, y_test, top_n: int = 15) -> None:
    """Show which features matter most to the winning model, using
    permutation importance on the held-out test set: shuffle one feature
    at a time and see how much worse the model gets. This works for any
    model type (unlike `.feature_importances_`, which HistGradientBoosting
    doesn't expose, or `.coef_`, which only linear models have)."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    # permutation_importance is run on the raw input columns (it permutes
    # before the pipeline's preprocessing step), so use X_test's columns.
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    print(f"\nTop {top_n} features (by permutation importance):")
    for idx in order:
        print(f"  {X_test.columns[idx]:<35} {result.importances_mean[idx]:.4f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(data_path: str, model_out_path: str):
    df = load_data(data_path)
    df = engineer_features(df)
    df = df.dropna(subset=[TARGET])
    df = remove_price_outliers(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain rows: {len(X_train):,}   Test rows: {len(X_test):,}\n")

    preprocessor = build_preprocessor()

    print("Cross-validating candidate models (5-fold)...")
    best_name, best_pipeline, results_df = compare_models(X_train, y_train, preprocessor)
    print(f"\nBest model by CV R2: {best_name}\n")

    # Refit the best pipeline on the full training set, then check it
    # against the untouched test set for a final, unbiased read on
    # performance.
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("Held-out test set performance:")
    print(f"  MAE:  {mae:,.0f} EUR")
    print(f"  MAPE: {mape:.1%}")
    print(f"  R2:   {r2:.3f}")

    print_top_features(best_pipeline, X_test, y_test, top_n=15)

    import joblib

    joblib.dump(best_pipeline, model_out_path)
    print(f"\nSaved trained pipeline to: {model_out_path}")

    return best_pipeline, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a price prediction model.")
    parser.add_argument(
        "--data", default="data/price_clean.csv", help="Path to the cleaned CSV file."
    )
    parser.add_argument(
        "--out", default="price_model.joblib", help="Where to save the trained pipeline."
    )
    args = parser.parse_args()
    main(args.data, args.out)
