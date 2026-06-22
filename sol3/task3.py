"""
Task 3: Adversity Prediction Model
Nomura Quant Challenge 5
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    log_loss
)

TAUS = [5, 10, 15, 20, 25, 30]

MI_COLS = {
    5: "M5",
    10: "M10",
    15: "M15",
    20: "M20",
    25: "M25",
    30: "M30"
}

FEATURE_COLS = [
    "Side",
    "log_volume",
    "Spread",
    "tp_vs_m0",
    "rel_spread",
    "eta",
    "momentum",
    "realized_vol",
    "client_A",
    "client_B",
    "client_C",
    "client_D",
    "client_E",
    "client_F"
]

_models = {}
_scalers = {}


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------

def load_clean_data(path="trade_data.csv"):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # Convert Side to +/-1 if needed
    if df["Side"].dtype == object:
        mapping = {
            "Buy": 1,
            "Sell": -1,
            "BUY": 1,
            "SELL": -1,
            "B": 1,
            "S": -1,
            "buy": 1,
            "sell": -1
        }

        df["Side"] = df["Side"].map(mapping)

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["time"].astype(str)
    )

    df = (
        df.sort_values("datetime")
          .reset_index(drop=True)
    )

    return df


# ------------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------------

def prepare_features(df):
    df = df.copy()

    df["log_volume"] = np.log1p(df["Volume"])

    df["tp_vs_m0"] = (
        df["Trade Price"] - df["M0"]
    )

    df["rel_spread"] = np.where(
        df["M0"] != 0,
        df["Spread"] / df["M0"],
        0.0
    )

    t_secs = (
        pd.to_timedelta(df["time"])
        .dt.total_seconds()
    )

    df["eta"] = (
        (t_secs - 9.5 * 3600)
        / (6.5 * 3600)
    ).clip(0.0, 1.0)

    m0_ret = (
        df["M0"]
        .pct_change()
        .fillna(0)
    )

    df["momentum"] = (
        df["M0"]
        .pct_change(5)
        .fillna(0)
    )

    df["realized_vol"] = (
        m0_ret
        .rolling(20, min_periods=1)
        .std()
        .fillna(0.0003)
    )

    for c in ["A", "B", "C", "D", "E", "F"]:
        df[f"client_{c}"] = (
            df["Name"] == c
        ).astype(float)

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.fillna(0, inplace=True)

    return df


# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------

def train_all_models(df):

    dates = sorted(df["Date"].unique())

    n_days = len(dates)

    train_dates = set(
        dates[:int(0.6 * n_days)]
    )

    val_dates = set(
        dates[int(0.6 * n_days):int(0.8 * n_days)]
    )

    test_dates = set(
        dates[int(0.8 * n_days):]
    )

    df_feat = prepare_features(df)

    train_mask = (
        df_feat["Date"]
        .isin(train_dates)
    )

    val_mask = (
        df_feat["Date"]
        .isin(val_dates)
    )

    test_mask = (
        df_feat["Date"]
        .isin(test_dates)
    )

    summary_metrics = {
        "train": [],
        "validation": [],
        "test": []
    }

    for tau in TAUS:

        # Adverse trade label
        df_feat["label"] = (
            (
                df_feat["Side"]
                * (
                    df_feat[MI_COLS[tau]]
                    - df_feat["Trade Price"]
                )
            ) < 0
        ).astype(int)

        X_tr = (
            df_feat.loc[
                train_mask,
                FEATURE_COLS
            ].values
        )

        y_tr = (
            df_feat.loc[
                train_mask,
                "label"
            ].values
        )

        X_va = (
            df_feat.loc[
                val_mask,
                FEATURE_COLS
            ].values
        )

        y_va = (
            df_feat.loc[
                val_mask,
                "label"
            ].values
        )

        X_te = (
            df_feat.loc[
                test_mask,
                FEATURE_COLS
            ].values
        )

        y_te = (
            df_feat.loc[
                test_mask,
                "label"
            ].values
        )

        scaler = StandardScaler()

        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        clf = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42
        )

        clf.fit(X_tr_s, y_tr)

        _models[tau] = clf
        _scalers[tau] = scaler

        for split_name, X_s, y in [
            ("train", X_tr_s, y_tr),
            ("validation", X_va_s, y_va),
            ("test", X_te_s, y_te)
        ]:

            preds = clf.predict(X_s)
            probs = clf.predict_proba(X_s)[:, 1]

            summary_metrics[split_name].append({
                "acc": accuracy_score(y, preds),
                "prec": precision_score(
                    y,
                    preds,
                    zero_division=0
                ),
                "rec": recall_score(
                    y,
                    preds,
                    zero_division=0
                ),
                "ll": log_loss(y, probs)
            })

    return summary_metrics


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":

    df = load_clean_data("trade_data.csv")

    metrics = train_all_models(df)

    rows = []

    print("\n=== Task 3 Averaged Metrics ===\n")

    for split in [
        "train",
        "validation",
        "test"
    ]:

        ms = metrics[split]

        row = {
            "split": split,
            "accuracy": round(
                np.mean([m["acc"] for m in ms]),
                4
            ),
            "precision": round(
                np.mean([m["prec"] for m in ms]),
                4
            ),
            "recall": round(
                np.mean([m["rec"] for m in ms]),
                4
            ),
            "log_loss": round(
                np.mean([m["ll"] for m in ms]),
                4
            )
        }

        print(
            f"{split:<10}"
            f" Acc={row['accuracy']:.4f}"
            f" Prec={row['precision']:.4f}"
            f" Recall={row['recall']:.4f}"
            f" LogLoss={row['log_loss']:.4f}"
        )

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        "task3_results.csv",
        index=False
    )

    print(
        "\nSaved results to task3_results.csv"
    )
