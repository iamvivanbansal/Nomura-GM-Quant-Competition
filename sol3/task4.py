"""
Task 4: Optimal Externalization Threshold — Improved Pipeline
Nomura Quant Challenge 5

Improvements over baseline:
  1. Two-pass theta search: coarse (step=0.05) then fine (step=0.005, ±0.1 window)
  2. Sharpe-weighted theta selection: max PnL / std(daily PnL) rather than raw total
  3. Tau injected as explicit feature + horizon-scaled realized vol window
  4. Degenerate theta guard: clip to [0.10, 0.90] and flag edge solutions
  5. Required plot_pnl_vs_theta() function with per-client subplots
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MI_COLS   = {5: 'M5', 10: 'M10', 15: 'M15', 20: 'M20', 25: 'M25', 30: 'M30'}
TAUS      = [5, 10, 15, 20, 25, 30]
CLIENTS   = ['A', 'B', 'C', 'D', 'E', 'F']
THETA_MIN = 0.10   # Guard against total-externalize (θ*=0) degenerate solutions
THETA_MAX = 0.90   # Guard against total-internalize (θ*=1) degenerate solutions


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & global feature construction (run once)
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str = 'trade_data.csv') -> tuple:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'])
    df = df.sort_values('datetime').reset_index(drop=True)

    df['log_volume'] = np.log1p(df['Volume'])
    df['tp_vs_m0']   = df['Trade Price'] - df['M0']
    df['rel_spread'] = df['Spread'] / df['M0']
    df['tp_vs_halfspread'] = (df['Side'] * (df['M0'] - df['Trade Price'])) / (df['Spread'] / 2 + 1e-10)

    t_secs  = pd.to_timedelta(df['time']).dt.total_seconds()
    df['eta'] = ((t_secs - 9.5 * 3600) / (6.5 * 3600)).clip(0.0, 1.0)

    df['m0_ret']       = df['M0'].pct_change().fillna(0)
    df['realized_vol'] = df['m0_ret'].rolling(20, min_periods=1).std().fillna(0.0003)

    dates       = sorted(df['Date'].unique())
    n_days      = len(dates)
    train_dates = set(dates[:int(0.6 * n_days)])
    val_dates   = set(dates[int(0.6 * n_days):int(0.8 * n_days)])
    test_dates  = set(dates[int(0.8 * n_days):])

    print(f"Data loaded: {len(train_dates)} train | {len(val_dates)} val | {len(test_dates)} test days")
    return df, train_dates, val_dates, test_dates


# ─────────────────────────────────────────────────────────────────────────────
# Client-level feature engine
# Improvement 3: tau injected + tau-scaled vol window
# ─────────────────────────────────────────────────────────────────────────────
def build_client_features(cdf: pd.DataFrame, tau: int) -> pd.DataFrame:
    """Adds all client-specific + tau-aware features in-place."""
    cdf = cdf.copy()

    cdf['signed_mom_3']  = cdf['Side'] * (cdf['M0'].diff(3).fillna(0)  / cdf['M0'].shift(3).fillna(1))
    cdf['signed_mom_10'] = cdf['Side'] * (cdf['M0'].diff(10).fillna(0) / cdf['M0'].shift(10).fillna(1))
    cdf['sweep_intensity'] = (cdf['Side'] * (cdf['M0'] - cdf['Trade Price'])) / (cdf['Spread'] / 2 + 1e-10)

    c_dt = cdf['datetime'].diff().dt.total_seconds().fillna(60.0).clip(0.1, 600.0)
    cdf['urgency_score'] = np.exp(-c_dt / 10.0)

    cdf['c_ofi_3'] = cdf['Side'].rolling(3, min_periods=1).sum()
    cdf['c_ofi_8'] = cdf['Side'].rolling(8, min_periods=1).sum()
    cdf['size_shock'] = cdf['Volume'] / (cdf['Volume'].rolling(20, min_periods=1).median() + 1e-8)

    # Improvement 3a: tau as explicit numeric feature (normalized)
    cdf['tau_norm'] = tau / 30.0

    # Improvement 3b: horizon-scaled volatility window (tau seconds → ~tau/5 trades apart)
    vol_window = max(5, tau // 3)
    cdf['vol_tau'] = cdf['m0_ret'].rolling(vol_window, min_periods=1).std().fillna(0.0003)

    return cdf


BASE_FEATURES = [
    'Side', 'Spread', 'realized_vol', 'eta',
    'signed_mom_3', 'signed_mom_10',
    'sweep_intensity', 'urgency_score',
    'c_ofi_3', 'c_ofi_8', 'size_shock',
    'tau_norm', 'vol_tau'          # new in improved version
]


# ─────────────────────────────────────────────────────────────────────────────
# Two-pass theta search
# Improvement 1: coarse pass then fine zoom
# Improvement 2: Sharpe-weighted objective (PnL / daily std)
# ─────────────────────────────────────────────────────────────────────────────
def _daily_pnl_series(probs: np.ndarray, pnl_raw: np.ndarray,
                      dates_arr: np.ndarray, theta: float) -> np.ndarray:
    """Returns array of per-day PnL under a given theta."""
    mask = (probs <= theta)
    unique_dates = np.unique(dates_arr)
    daily = np.array([
        np.sum(pnl_raw[dates_arr == d] * mask[dates_arr == d])
        for d in unique_dates
    ])
    return daily


def find_optimal_theta(
    val_probs: np.ndarray,
    val_pnl_raw: np.ndarray,
    val_dates: np.ndarray,
    sharpe_weight: float = 0.3,
) -> tuple:
    """
    Two-pass search for theta* that maximises a Sharpe-blended objective:
        score(theta) = mean_daily_pnl(theta) - sharpe_weight * std_daily_pnl(theta)

    Returns (theta_star, best_score, coarse_curve)
    where coarse_curve is a (theta_values, scores) tuple for plotting.
    """
    # ── Pass 1: coarse grid ──────────────────────────────────────────────────
    coarse_thetas = np.round(np.arange(0.0, 1.01, 0.05), 3)
    coarse_scores = []
    for th in coarse_thetas:
        daily = _daily_pnl_series(val_probs, val_pnl_raw, val_dates, th)
        score = daily.sum() - sharpe_weight * (daily.std() + 1e-8)
        coarse_scores.append(score)

    coarse_best_idx = int(np.argmax(coarse_scores))
    coarse_best_th  = coarse_thetas[coarse_best_idx]

    # ── Pass 2: fine zoom ±0.10 around coarse best ───────────────────────────
    fine_lo = max(0.0, coarse_best_th - 0.10)
    fine_hi = min(1.0, coarse_best_th + 0.10)
    fine_thetas = np.round(np.arange(fine_lo, fine_hi + 0.001, 0.005), 4)
    fine_scores = []
    for th in fine_thetas:
        daily = _daily_pnl_series(val_probs, val_pnl_raw, val_dates, th)
        score = daily.sum() - sharpe_weight * (daily.std() + 1e-8)
        fine_scores.append(score)

    fine_best_idx = int(np.argmax(fine_scores))
    theta_star    = float(fine_thetas[fine_best_idx])
    best_score    = fine_scores[fine_best_idx]

    # ── Improvement 4: degenerate guard ─────────────────────────────────────
    flagged = False
    if theta_star < THETA_MIN or theta_star > THETA_MAX:
        flagged = True
        theta_star = float(np.clip(theta_star, THETA_MIN, THETA_MAX))

    return theta_star, best_score, flagged, (coarse_thetas, np.array(coarse_scores))


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline — returns results dataframe and curves for plotting
# ─────────────────────────────────────────────────────────────────────────────
def run_task4(data_path: str = 'trade_data.csv') -> pd.DataFrame:
    df, train_dates, val_dates, test_dates = load_data(data_path)

    rows   = []
    curves = {}   # {(client, tau): (coarse_thetas, coarse_scores)}

    for client in CLIENTS:
        cdf_base = df[df['Name'] == client].copy().reset_index(drop=True)

        for tau in TAUS:
            cdf = build_client_features(cdf_base, tau)

            raw_pnl = cdf['Side'] * cdf['Volume'] * (cdf[MI_COLS[tau]] - cdf['Trade Price'])
            target  = (raw_pnl < 0).astype(int)

            c_tr = cdf['Date'].isin(train_dates)
            c_va = cdf['Date'].isin(val_dates)
            c_te = cdf['Date'].isin(test_dates)

            Xtr, ytr = cdf[c_tr][BASE_FEATURES].values, target[c_tr].values
            Xva, yva = cdf[c_va][BASE_FEATURES].values, target[c_va].values
            Xte, yte = cdf[c_te][BASE_FEATURES].values, target[c_te].values

            if len(ytr) == 0 or len(yva) == 0 or len(yte) == 0:
                continue

            model = HistGradientBoostingClassifier(
                max_iter=150, max_depth=4, learning_rate=0.04,
                min_samples_leaf=20, l2_regularization=1.0,
                random_state=42
            )
            model.fit(Xtr, ytr)

            val_probs  = model.predict_proba(Xva)[:, 1]
            test_probs = model.predict_proba(Xte)[:, 1]

            val_pnl_raw  = raw_pnl[c_va].values
            test_pnl_raw = raw_pnl[c_te].values
            val_dates_arr  = cdf[c_va]['Date'].values

            theta_star, _, flagged, curve = find_optimal_theta(
                val_probs, val_pnl_raw, val_dates_arr
            )
            curves[(client, tau)] = curve

            test_mask    = (test_probs <= theta_star)
            final_pnl    = float(np.sum(test_pnl_raw * test_mask))
            val_pnl_at_star = float(np.sum(val_pnl_raw * (val_probs <= theta_star)))

            rows.append({
                'client':       client,
                'tau':          tau,
                'theta_star':   round(theta_star, 4),
                'val_pnl':      round(val_pnl_at_star, 2),
                'final_pnl':    round(final_pnl, 2),
                'flagged':      flagged,
            })

        print(f"  Client {client} done.")

    results = pd.DataFrame(rows).sort_values(['client', 'tau']).reset_index(drop=True)

    # Export CSV in required format
    out = results[['client', 'tau', 'theta_star', 'final_pnl']].copy()
    out.to_csv('task4_results.csv', index=False)
    print("\nSaved: task4_results.csv")

    return results, curves


# ─────────────────────────────────────────────────────────────────────────────
# Required submission functions
# ─────────────────────────────────────────────────────────────────────────────
def optimal_threshold(client: str, tau: int, results_df: pd.DataFrame = None) -> dict:
    """
    Returns optimal threshold and PnL summary for a given client / tau pair.

    Parameters
    ----------
    client      : one of 'A'..'F'
    tau         : one of 5, 10, 15, 20, 25, 30
    results_df  : pre-computed results DataFrame from run_task4()
                  (pass None to recompute — slow)
    """
    if results_df is None:
        results_df, _ = run_task4()

    row = results_df[(results_df['client'] == client) & (results_df['tau'] == tau)]
    if row.empty:
        raise ValueError(f"No result found for client={client}, tau={tau}")
    row = row.iloc[0]

    return {
        'theta':          row['theta_star'],
        'validation_pnl': row['val_pnl'],
        'test_pnl':       row['final_pnl'],
        'flagged':        row['flagged'],
    }


def plot_pnl_vs_theta(results_df: pd.DataFrame, curves: dict,
                      save_path: str = 'pnl_vs_theta.png') -> None:
    """
    Plots PnL_validation(theta) for each client × tau combination.
    Saves to `save_path`.

    Layout: 6 rows (clients) × 6 cols (taus), theta* marked with a dashed line.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Task 4 — PnL vs θ on validation set', fontsize=14, y=0.99)
    gs  = gridspec.GridSpec(6, 6, figure=fig, hspace=0.55, wspace=0.35)

    for i, client in enumerate(CLIENTS):
        for j, tau in enumerate(TAUS):
            ax = fig.add_subplot(gs[i, j])

            key = (client, tau)
            if key not in curves:
                ax.axis('off')
                continue

            thetas, scores = curves[key]
            ax.plot(thetas, scores, color='steelblue', linewidth=1.0)

            # Mark theta*
            row = results_df[
                (results_df['client'] == client) & (results_df['tau'] == tau)
            ]
            if not row.empty:
                t_star = row.iloc[0]['theta_star']
                ax.axvline(t_star, color='crimson', linewidth=1.0, linestyle='--')
                ax.set_title(f'{client} τ={tau}\nθ*={t_star:.3f}', fontsize=7, pad=2)
            else:
                ax.set_title(f'{client} τ={tau}', fontsize=7, pad=2)

            ax.set_xlabel('θ', fontsize=6)
            ax.set_ylabel('score', fontsize=6)
            ax.tick_params(labelsize=5)
            ax.spines[['top', 'right']].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Task 4: Optimal Externalization Threshold (Improved)")
    print("=" * 60)

    results_df, curves = run_task4('trade_data.csv')

    print("\n=== Full Results ===")
    print(results_df.to_string(index=False))

    flagged = results_df[results_df['flagged']]
    if not flagged.empty:
        print(f"\n⚠  Degenerate θ* clipped for {len(flagged)} client-tau pair(s):")
        print(flagged[['client', 'tau', 'theta_star']].to_string(index=False))

    plot_pnl_vs_theta(results_df, curves)

    print("\nAll outputs saved.")
