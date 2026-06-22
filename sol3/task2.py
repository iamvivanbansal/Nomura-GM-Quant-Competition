"""
Tasks 1 & 2: Adversity Profile and Client Profitability
Market Making – Nomura Quant Challenge 5
"""

import pandas as pd
import numpy as np
from typing import List

# ── Data loading ─────────────────────────────────────────────────────────────
_df = None

def _load():
    global _df
    if _df is None:
        _df = pd.read_csv("trade_data.csv")
        _df.columns = _df.columns.str.strip()
    return _df

MI_COLS = {5: "M5", 10: "M10", 15: "M15", 20: "M20", 25: "M25", 30: "M30"}
TAUS    = [5, 10, 15, 20, 25, 30]


# ── Task 1 ───────────────────────────────────────────────────────────────────

def adversity_profile(client: str, tau: List[int]) -> List[float]:
    """
    Parameters:
        client : Client identifier (single character)
        tau    : List of horizons e.g. [5, 10, 15, 20, 25, 30]

    Returns:
        List of floats representing adversity percentage (0-100) at each horizon.

    A trade is adverse at horizon τ iff:
        PnL(at t=τ) = side * V * (M_τ - TP) < 0
    i.e.  side * (M_τ - TP) < 0
    Adversity % = 100 * (# adverse trades) / (# total trades)
    """
    df = _load()
    cdf = df[df["Name"] == client]
    result = []
    for t in tau:
        adverse = cdf["Side"] * (cdf[MI_COLS[t]] - cdf["Trade Price"]) < 0
        result.append(float(adverse.mean() * 100.0))
    return result


# ── Task 2 ───────────────────────────────────────────────────────────────────

def expected_pnl(client: str, tau: List[int]) -> dict:
    """
    Parameters:
        client : Client identifier
        tau    : List of horizons e.g. [5, 10, 15, 20, 25, 30]

    Returns:
        Dictionary with keys:
            'per_horizon': List[float]
                Expected PnL per trade at each tau (Corollary 1, eq. 5)
            'aggregate': float
                Expected Aggregate PnL per trade (Corollary 2, eq. 6)
                Uses uniform closing weights w_i = 1/6 for i in {1..6}.
    """
    df = _load()
    cdf = df[df["Name"] == client]

    per_horizon = []
    for t in tau:
        pnl = cdf["Side"] * cdf["Volume"] * (cdf[MI_COLS[t]] - cdf["Trade Price"])
        per_horizon.append(float(pnl.mean()))

    # Aggregate PnL: uniform weights across all 6 horizons
    agg_series = cdf.apply(
        lambda r: r["Side"] * r["Volume"] * np.mean(
            [r[MI_COLS[t]] - r["Trade Price"] for t in TAUS]
        ),
        axis=1,
    )
    return {"per_horizon": per_horizon, "aggregate": float(agg_series.mean())}


def classify_client(client: str) -> str:
    """
    Parameters:
        client : Client identifier

    Returns:
        'profitable' or 'costly'
        Based on the sign of the expected aggregate PnL (eq. 6).
        Positive aggregate PnL → LP earns on average → 'profitable'.
    """
    res = expected_pnl(client, TAUS)
    return "profitable" if res["aggregate"] >= 0.0 else "costly"


def min_half_spread(client: str) -> float:
    """
    Parameters:
        client : Client identifier

    Returns:
        Minimum half-spread δ* (in data price units) such that if the LP
        quotes at M0 ± δ* for all trades with this client, the expected
        aggregate PnL per trade (eq. 6) would be non-negative.

    Derivation:
        If quoted half-spread is δ, LP's effective TP is:
            TP_new = M0 - side * δ    (buy below mid, sell above mid)

        Aggregate PnL per trade with δ:
            E[side * V * avg_i(M_ti - TP_new)]
          = E[side * V * avg_i(M_ti - M0 + side*δ)]
          = E[side * V * avg_i(M_ti - M0)] + δ * E[V]

        Setting ≥ 0:
            δ* = max(0,  -E[side * V * avg_i(M_ti - M0)] / E[V])
    """
    df = _load()
    cdf = df[df["Name"] == client]

    pnl_at_zero_spread = cdf.apply(
        lambda r: r["Side"] * r["Volume"] * np.mean(
            [r[MI_COLS[t]] - r["M0"] for t in TAUS]
        ),
        axis=1,
    )
    mean_vol = float(cdf["Volume"].mean())
    delta_star = max(0.0, -float(pnl_at_zero_spread.mean()) / mean_vol)
    return delta_star


# ── Runner (generates CSVs) ───────────────────────────────────────────────────

if __name__ == "__main__":
    CLIENTS = ["A", "B", "C", "D", "E", "F"]

    # Task 1 CSV
    rows1 = []
    for c in CLIENTS:
        vals = adversity_profile(c, TAUS)
        row = {"client": c}
        for t, v in zip(TAUS, vals):
            row[f"tau={t}"] = round(v, 4)
        rows1.append(row)
    pd.DataFrame(rows1).to_csv("task1_results.csv", index=False)
    print("task1_results.csv written")

    # Task 2 CSV
    rows2 = []
    for c in CLIENTS:
        res = expected_pnl(c, TAUS)
        row = {"client": c}
        for t, v in zip(TAUS, res["per_horizon"]):
            row[f"tau={t}"] = round(v, 6)
        row["agg_pnl"]  = round(res["aggregate"], 6)
        row["delta_star"] = round(min_half_spread(c), 6)
        rows2.append(row)
    pd.DataFrame(rows2).to_csv("task2_results.csv", index=False)
    print("task2_results.csv written")

    # Print summary
    print("\n=== Task 1: Adversity Profiles ===")
    print(pd.DataFrame(rows1).to_string(index=False))
    print("\n=== Task 2: Expected PnL and Spread ===")
    print(pd.DataFrame(rows2).to_string(index=False))
    for c in CLIENTS:
        print(f"  Client {c}: {classify_client(c)}")
