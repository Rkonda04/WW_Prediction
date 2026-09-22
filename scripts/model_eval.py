"""
RRWWTF Flow Prediction -- Shared Split Definitions and Scoring Function

Import this module from every model script that follows the naive-baseline
validation framework (scripts/validation_framework.py). Every model in
this project MUST use the split constants and score() function below
unchanged, so every model is evaluated on an identical protocol -- no
model script should redefine its own train/test split or its own metric
formulas.

SPLIT (fixed, chronological -- never randomized):
    TRAIN: 2018-01-01 through 2022-12-31
    TEST:  all available days in 2024
    2023 (June-Sept only, ~4 months) is excluded from both train and
    test -- too fragmentary a partial year to serve as a clean,
    representative test period.

Expanding-window CV folds (CV_FOLDS below) are provided for future
hyperparameter search WITHIN the training period only -- they never touch
the 2024 test set.

This module defines splits and scoring only; it does not load data. Every
script that uses it loads its own feature frame (following
validation_framework.py's pattern of reading the corrected master dataset)
and passes date/actual/predicted arrays into score().
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Split definition
# ---------------------------------------------------------------------------

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-12-31")  # actual test days are whatever's available within this window
EXCLUDED_YEAR = 2023
EXCLUDED_YEAR_REASON = (
    "2023 plant logs cover only June-September (a partial year) -- too fragmentary to serve as a "
    "clean, representative train or test period, so it is excluded from both entirely."
)

CV_FOLDS = [
    {"Name": "Fold 1", "Train_Start": pd.Timestamp("2018-01-01"), "Train_End": pd.Timestamp("2019-12-31"),
     "Val_Start": pd.Timestamp("2020-01-01"), "Val_End": pd.Timestamp("2020-12-31")},
    {"Name": "Fold 2", "Train_Start": pd.Timestamp("2018-01-01"), "Train_End": pd.Timestamp("2020-12-31"),
     "Val_Start": pd.Timestamp("2021-01-01"), "Val_End": pd.Timestamp("2021-12-31")},
    {"Name": "Fold 3", "Train_Start": pd.Timestamp("2018-01-01"), "Train_End": pd.Timestamp("2021-12-31"),
     "Val_Start": pd.Timestamp("2022-01-01"), "Val_End": pd.Timestamp("2022-12-31")},
]


def split_mask(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    return (dates >= start) & (dates <= end)


def get_train_test_masks(dates: pd.Series):
    """Boolean masks (aligned to `dates`, positional) for TRAIN and TEST.
    Neither mask includes any 2023 date -- 2023 falls into neither window
    by construction (TRAIN_END is 2022-12-31, TEST_START is 2024-01-01)."""
    train_mask = split_mask(dates, TRAIN_START, TRAIN_END)
    test_mask = split_mask(dates, TEST_START, TEST_END)
    return train_mask, test_mask


def get_cv_fold_masks(dates: pd.Series):
    """Returns a list of (fold_name, train_mask, val_mask) tuples for the
    three expanding-window CV folds, all strictly within the training
    period -- never touching the 2024 test set."""
    out = []
    for fold in CV_FOLDS:
        train_mask = split_mask(dates, fold["Train_Start"], fold["Train_End"])
        val_mask = split_mask(dates, fold["Val_Start"], fold["Val_End"])
        out.append((fold["Name"], train_mask, val_mask))
    return out


# ---------------------------------------------------------------------------
# Scoring -- the single source of truth for every model's metrics
# ---------------------------------------------------------------------------

def score(dates, y_true, y_pred, high_flow_threshold: float = None) -> dict:
    """
    dates, y_true, y_pred: equal-length array-likes, same order.

    Rows with NaN in y_true or y_pred are excluded from every metric --
    never filled or interpolated -- and counted in N_Excluded_NaN. This is
    how an undefined prediction (e.g. persistence on the day after a gap)
    is handled: it is dropped and the drop is reported, not silently
    scored against a filled value.

    If high_flow_threshold is given, also scores the subset of valid rows
    where y_true > high_flow_threshold (High_Flow_N/RMSE/Mean_Error).

    Returns a flat dict -- every model script in this project must call
    this function (not hand-roll its own metric formulas) so scores are
    directly comparable across models.
    """
    df = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(dates).reset_index(drop=True)),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    n_input = len(df)
    valid = df.dropna(subset=["y_true", "y_pred"])
    n_excluded = n_input - len(valid)

    result = {
        "N": len(valid),
        "N_Excluded_NaN": n_excluded,
        "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan,
        "Mean_Error": np.nan, "Pct_Days_Underpredicted": np.nan,
    }

    if len(valid) < 2:
        if high_flow_threshold is not None:
            result.update({"High_Flow_N": 0, "High_Flow_RMSE": np.nan, "High_Flow_Mean_Error": np.nan})
        return result

    err = valid["y_pred"] - valid["y_true"]
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((valid["y_true"] - valid["y_true"].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    nonzero = valid["y_true"] != 0
    mape = float(np.mean(np.abs(err[nonzero] / valid.loc[nonzero, "y_true"])) * 100) if nonzero.any() else np.nan

    mean_error = float(err.mean())
    pct_under = float((valid["y_pred"] < valid["y_true"]).mean() * 100)

    result.update({
        "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape,
        "Mean_Error": mean_error, "Pct_Days_Underpredicted": pct_under,
    })

    if high_flow_threshold is not None:
        hf = valid[valid["y_true"] > high_flow_threshold]
        if len(hf) >= 1:
            hf_err = hf["y_pred"] - hf["y_true"]
            result["High_Flow_N"] = len(hf)
            result["High_Flow_RMSE"] = float(np.sqrt(np.mean(hf_err ** 2)))
            result["High_Flow_Mean_Error"] = float(hf_err.mean())
        else:
            result["High_Flow_N"] = 0
            result["High_Flow_RMSE"] = np.nan
            result["High_Flow_Mean_Error"] = np.nan

    return result
