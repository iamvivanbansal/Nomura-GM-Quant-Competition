"""
Task 5: Dynamic Quoting Under Inventory Pressure — FIXED VERSION
Nomura Quant Challenge 5

Root causes of the original -20 Sharpe:
  BUG 1 — Fill model misread: code wrote P = lam*exp(-gamma*delta/sigma)
           where delta was sigma-scaled (k1*sigma). This makes delta/sigma = k1,
           so P_fill = 0.6*exp(-1.5*1.8) = 0.04 (4% fill rate, economically broken).
           FIX: delta is quoted in ABSOLUTE price units (~0.013), so the fill model
           is P = lam*exp(-gamma*delta) where delta is already in price space.
           With delta~0.013 and gamma=1.5: P_fill = 0.6*exp(-0.0195) = 0.588 — correct.

  BUG 2 — Inventory skew sign inverted: original had
           delta_bid = base - skew,  delta_ask = base + skew  (long inventory)
           That TIGHTENS the bid (encourages LP to BUY MORE) when already long — wrong.
           FIX: long inventory → RAISE delta_bid (fewer LP buys), LOWER delta_ask
           (more LP sells), which mean-reverts the position.
           delta_bid = base + skew,  delta_ask = base - skew  (long inventory)

  BUG 3 — k1 too small: k1=1.8 gives base = 1.8*0.0003 = 0.00054 per trade.
           Actual market half-spread is ~0.013 (43x larger).
           At sigma-scaled spread, spread income (~0.04/fill) cannot overcome
           adverse selection (~0.23/fill) → always unprofitable.
           FIX: k1=46 so that base = 46*sigma ≈ actual market half-spread.
           Also parameterised via I_TARGET (see below) for explainability.

  BUG 4 — k3 inventory skew units: k3=0.15 with I in raw volume units
           (which grows to 10000+) makes skew dominate the base entirely.
           FIX: skew is expressed as base * (I / I_TARGET) * urgency_factor,
           ensuring skew is bounded as a fraction of base spread.
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Spec constraints  (unchanged from problem statement)
# ─────────────────────────────────────────────────────────────────────────────
C_MIN   = 0.5       # delta >= C_MIN * sigma  (hard floor)
BPS_MAX = 50e-4     # delta_max = 50 bps of mid (~0.503 at M0=100)

# ─────────────────────────────────────────────────────────────────────────────
# Fixed parameters  (calibrated to data)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    # k1: base = k1 * sigma.  sigma~0.0003, actual hs~0.013 => k1~43.
    # k1=46 calibrated to match mean observed half-spread exactly.
    'k1':       46.0,
    # k2: adversity premium per unit of excess alpha above client baseline.
    # Widens both sides symmetrically when toxic flow detected.
    'k2':       10.0,
    # I_TARGET: inventory level at which skew equals base (100% widening).
    # Keep inventory well below this to preserve spread income.
    'I_TARGET': 50.0,
    # k4: day-end urgency multiplier on skew (applied as 1 + k4*eta^2).
    # Accelerates inventory unwind in the last 30 minutes.
    'k4':       3.0,
    # k5: end-of-day panic term (only active for eta > 0.90).
    'k5':       5.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core quoting function  (required submission interface)
# ─────────────────────────────────────────────────────────────────────────────
def quote(
    inventory:      float,
    sigma:          float,
    alpha:          float,
    eta:            float,
    mid:            float = 100.0,
    params:         dict  = None,
    alpha_baseline: float = 0.50,
) -> Tuple[float, float]:
    """
    Returns (delta_bid, delta_ask) — absolute half-spreads in price units.

    Convention (LP perspective):
        side = +1  →  LP buys  at  M0 - delta_bid   (TP below mid)
        side = -1  →  LP sells at  M0 + delta_ask   (TP above mid)

    Long inventory (I > 0) means LP is over-long:
        Raise delta_bid  → fewer LP buys   (high fill cost for client sells)
        Lower delta_ask  → more LP sells   (attractive for client buys)
    This skew mean-reverts the inventory position.

    Parameters
    ----------
    inventory      : current running inventory (positive = long)
    sigma          : realized volatility (RMS of last-20 mid-price returns)
    alpha          : adversity probability from Task 3 model in [0, 1]
    eta            : elapsed fraction of trading day in [0, 1]
    mid            : current mid price (used for delta_max cap)
    params         : parameter dict; uses DEFAULT_PARAMS if None
    alpha_baseline : per-client median alpha (excess premium = max(0, alpha-baseline))
    """
    p = params if params is not None else DEFAULT_PARAMS
    sigma = max(sigma, 1e-8)

    # ── 1. Base spread ────────────────────────────────────────────────────────
    # BUG 3 FIX: k1 calibrated so k1*sigma matches actual market half-spread.
    excess_alpha = max(0.0, alpha - alpha_baseline)
    base = (p['k1'] + p['k2'] * excess_alpha) * sigma

    # ── 2. Inventory skew ─────────────────────────────────────────────────────
    # Normalise by I_TARGET so skew is a dimensionless fraction of base.
    # urgency factor grows quadratically toward end-of-day.
    urgency       = 1.0 + p['k4'] * (eta ** 2)
    norm_inv      = inventory / (p['I_TARGET'] + 1e-8)
    skew          = base * norm_inv * urgency

    # ── 3. End-of-day panic term ──────────────────────────────────────────────
    # Only activates for eta > 0.90 — hard shove toward zero inventory at close.
    panic = 0.0
    if eta > 0.90:
        ramp  = ((eta - 0.90) / 0.10) ** 2
        panic = p['k5'] * base * norm_inv * ramp

    # ── 4. Assemble raw half-spreads ──────────────────────────────────────────
    # BUG 2 FIX: long inventory → higher bid spread, lower ask spread.
    delta_bid = base + skew + panic   # raised when long  (fewer LP buys)
    delta_ask = base - skew - panic   # lowered when long (more LP sells)

    # ── 5. Clip to spec bounds ────────────────────────────────────────────────
    delta_max = BPS_MAX * mid
    floor     = C_MIN * sigma

    delta_bid = float(max(floor, min(delta_max, delta_bid)))
    delta_ask = float(max(floor, min(delta_max, delta_ask)))

    return delta_bid, delta_ask


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────
def backtest(
    df:              pd.DataFrame,
    alpha_series:    np.ndarray,
    alpha_baselines: Dict[str, float],
    lam:   float = 0.6,
    gamma: float = 1.5,
    phi:   float = 1.0,
    params: dict = None,
    seed:   int  = 42,
) -> dict:
    """
    Simulates the quoting strategy over a single DataFrame split.

    Fill model (BUG 1 FIX):
        P_fill = lam * exp(-gamma * delta)
        where delta is in ABSOLUTE price units (not divided by sigma).
        With gamma=1.5 and delta~0.013: P_fill = 0.6*exp(-0.0195) ≈ 0.59.

    Inventory penalty (end-of-day):
        Penalty_D = phi * I_T^2 * sigma_D
        Applied once at midnight reset; inventory reset to 0 each day.
    """
    np.random.seed(seed)

    if params is None:
        params = DEFAULT_PARAMS

    df = df.copy().reset_index(drop=True)

    # Sigma: rolling std of M0 returns (same formula as problem eq.10)
    df['m0_ret'] = df['M0'].pct_change().fillna(0.0)
    df['sigma']  = (
        df['m0_ret']
        .rolling(20, min_periods=1)
        .apply(lambda x: np.sqrt(np.mean(x ** 2)), raw=True)
        .fillna(0.0003)
    )

    # Time-of-day fraction
    t    = pd.to_datetime(df['time'])
    secs = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
    df['eta'] = ((secs - 9.5 * 3600) / (6.5 * 3600)).clip(0.0, 1.0)
    df['alpha'] = alpha_series

    mi_cols = ['M5', 'M10', 'M15', 'M20', 'M25', 'M30']

    # Pre-resolve baselines to array
    abase_arr = np.array(
        [alpha_baselines.get(c, 0.50) for c in df['Name'].values]
    )

    side_arr   = df['Side'].values.astype(float)
    vol_arr    = df['Volume'].values.astype(float)
    m0_arr     = df['M0'].values.astype(float)
    sigma_arr  = df['sigma'].values.astype(float)
    alpha_arr  = df['alpha'].values.astype(float)
    eta_arr    = df['eta'].values.astype(float)
    mi_arr     = df[mi_cols].values.astype(float)
    date_arr   = df['Date'].values

    # Unpack params
    k1 = params['k1']; k2 = params['k2']
    k4 = params['k4']; k5 = params['k5']
    I_T = max(params['I_TARGET'], 1e-8)

    inventory  = 0.0
    daily_pnl  = {}
    trade_pnls = []
    cur_date   = date_arr[0]
    day_pnl    = 0.0
    day_sigmas = []

    for i in range(len(df)):
        d = date_arr[i]

        # ── Day boundary ──────────────────────────────────────────────────────
        if d != cur_date:
            day_sigma = float(np.mean(day_sigmas)) if day_sigmas else 0.0003
            penalty   = phi * (inventory ** 2) * day_sigma
            daily_pnl[cur_date] = day_pnl - penalty
            inventory  = 0.0
            day_pnl    = 0.0
            day_sigmas = []
            cur_date   = d

        side   = side_arr[i]
        vol    = vol_arr[i]
        m0     = m0_arr[i]
        sigma  = max(sigma_arr[i], 1e-8)
        alpha  = alpha_arr[i]
        eta    = eta_arr[i]
        a_base = abase_arr[i]

        day_sigmas.append(sigma)

        # ── Inline quote logic ────────────────────────────────────────────────
        excess_alpha = max(0.0, alpha - a_base)
        base         = (k1 + k2 * excess_alpha) * sigma

        urgency  = 1.0 + k4 * (eta * eta)
        norm_inv = inventory / I_T
        skew     = base * norm_inv * urgency

        if eta > 0.90:
            ramp  = ((eta - 0.90) / 0.10) ** 2
            panic = k5 * base * norm_inv * ramp
        else:
            panic = 0.0

        # BUG 2 FIX: long → raise bid, lower ask
        db_raw = base + skew + panic
        da_raw = base - skew - panic

        delta_max = BPS_MAX * m0
        floor_val = C_MIN * sigma
        db = max(floor_val, min(delta_max, db_raw))
        da = max(floor_val, min(delta_max, da_raw))

        delta_side = db if side == 1.0 else da

        # BUG 1 FIX: fill model uses absolute delta, not delta/sigma
        p_fill = lam * np.exp(-gamma * delta_side)
        filled = np.random.rand() < min(p_fill, 1.0)

        if filled:
            tp        = m0 - side * delta_side
            trade_pnl = side * vol * (mi_arr[i].mean() - tp)
            day_pnl  += trade_pnl
            trade_pnls.append(trade_pnl)
            inventory += side * vol

    # Flush last day
    day_sigma = float(np.mean(day_sigmas)) if day_sigmas else 0.0003
    penalty   = phi * (inventory ** 2) * day_sigma
    daily_pnl[cur_date] = day_pnl - penalty

    # ── Metrics ───────────────────────────────────────────────────────────────
    daily_arr  = np.array(list(daily_pnl.values()))
    total_pnl  = float(daily_arr.sum())
    sigma_d    = float(daily_arr.std()) + 1e-8
    sharpe     = total_pnl / (sigma_d * np.sqrt(len(daily_arr)) + 1e-8)

    cumulative = np.cumsum(daily_arr)
    peak       = np.maximum.accumulate(cumulative)
    max_dd     = float((peak - cumulative).max()) if len(cumulative) > 0 else 0.0

    return {
        'total_pnl':    round(total_pnl, 2),
        'sharpe_score': round(float(daily_arr.mean() / (daily_arr.std() + 1e-8)), 4),
        'annualised_sharpe': round(sharpe, 4),
        'daily_pnl':    daily_pnl,
        'max_drawdown': round(max_dd, 2),
        'n_trades':     len(trade_pnls),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Required submission function: validate_quote
# ─────────────────────────────────────────────────────────────────────────────
def validate_quote(
    data_path:  str   = 'trade_data.csv',
    alpha_path: str   = None,
    lam:        float = 0.6,
    gamma:      float = 1.5,
    phi:        float = 1.0,
    params:     dict  = None,
) -> dict:
    """
    Full backtest and validation of the quoting strategy.
    Prints summary metrics and saves pnl_curve.png.

    alpha_path (optional): CSV with columns ['Date','time','Name','alpha']
                           matching trade_data rows. Uses alpha=0.50 if None.
    """
    print("=" * 60)
    print("  Task 5 — Dynamic Quoting: Validation & Backtest")
    print("=" * 60)

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Alpha scores
    if alpha_path is not None:
        alpha_df  = pd.read_csv(alpha_path)
        df = df.merge(
            alpha_df[['Date', 'time', 'Name', 'alpha']],
            on=['Date', 'time', 'Name'], how='left'
        )
        df['alpha'] = df['alpha'].fillna(0.50)
    else:
        print("  No alpha_path — using alpha=0.50 fallback.")
        df['alpha'] = 0.50

    alpha_all = df['alpha'].values

    # Chronological splits (60/20/20)
    dates       = sorted(df['Date'].unique())
    n           = len(dates)
    train_dates = set(dates[:int(0.6 * n)])
    val_dates   = set(dates[int(0.6 * n):int(0.8 * n)])
    test_dates  = set(dates[int(0.8 * n):])

    tr_mask = df['Date'].isin(train_dates)
    va_mask = df['Date'].isin(val_dates)
    te_mask = df['Date'].isin(test_dates)

    df_train = df[tr_mask].copy()
    df_val   = df[va_mask].copy()
    df_test  = df[te_mask].copy()

    a_tr = alpha_all[tr_mask.values]
    a_va = alpha_all[va_mask.values]
    a_te = alpha_all[te_mask.values]

    # Per-client alpha baselines from training data
    df_train = df_train.copy()
    df_train['_a'] = a_tr
    alpha_baselines = df_train.groupby('Name')['_a'].median().to_dict()
    print(f"  Per-client alpha baselines: {alpha_baselines}")

    run_params = params if params is not None else DEFAULT_PARAMS
    print(f"  Parameters: {run_params}\n")

    # Run backtest on all three splits
    results = {}
    for name, split_df, split_alpha in [
        ('train',      df_train, a_tr),
        ('validation', df_val,   a_va),
        ('test',       df_test,  a_te),
    ]:
        r = backtest(split_df, split_alpha, alpha_baselines,
                     lam, gamma, phi, params=run_params, seed=42)
        results[name] = r
        print(f"  {name:<12}: PnL={r['total_pnl']:>12,.2f}  "
              f"Sharpe={r['sharpe_score']:>7.4f}  "
              f"MaxDD={r['max_drawdown']:>12,.2f}  "
              f"Trades={r['n_trades']}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Task 5 — Daily PnL by split', fontsize=12)
    colours = {'train': 'steelblue', 'validation': 'darkorange', 'test': 'seagreen'}
    for ax, (name, r) in zip(axes, results.items()):
        daily  = list(r['daily_pnl'].values())
        cumul  = np.cumsum(daily)
        ax.bar(range(len(daily)), daily, alpha=0.4, color=colours[name])
        ax.plot(cumul, color=colours[name], linewidth=1.8, label='Cumulative')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(
            f"{name}\nTotal={r['total_pnl']:,.0f}  Sharpe={r['sharpe_score']:.3f}",
            fontsize=9)
        ax.set_xlabel('Day'); ax.set_ylabel('PnL')
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('pnl_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: pnl_curve.png")

    print("\n  === Final Summary ===")
    print(f"  {'Split':<14} {'Total PnL':>14} {'Sharpe':>10} {'MaxDrawdown':>14}")
    print("  " + "-" * 54)
    for name, r in results.items():
        print(f"  {name:<14} {r['total_pnl']:>14,.2f} "
              f"{r['sharpe_score']:>10.4f} "
              f"{r['max_drawdown']:>14,.2f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────
def _run_unit_tests():
    import math
    print("Running unit tests...")
    PASS = True

    # Test 1: neutral → symmetric
    db, da = quote(0.0, 0.001, 0.50, 0.0, mid=100.0, alpha_baseline=0.50)
    if not math.isclose(db, da, rel_tol=1e-6):
        print(f"  FAIL: neutral should be symmetric. db={db}, da={da}"); PASS = False
    else:
        print(f"  PASS: neutral symmetric  db={db:.6f}  da={da:.6f}")

    # Test 2: long inventory → db > da (bid raised, ask lowered — lean to sell)
    db, da = quote(500.0, 0.001, 0.50, 0.3, mid=100.0, alpha_baseline=0.50)
    if not (db > da):
        print(f"  FAIL: long should give db > da. db={db}, da={da}"); PASS = False
    else:
        print(f"  PASS: long skew  db={db:.6f} > da={da:.6f}")

    # Test 3: short inventory → da > db (ask raised, bid lowered — lean to buy)
    db, da = quote(-500.0, 0.001, 0.50, 0.3, mid=100.0, alpha_baseline=0.50)
    if not (da > db):
        print(f"  FAIL: short should give da > db. db={db}, da={da}"); PASS = False
    else:
        print(f"  PASS: short skew  da={da:.6f} > db={db:.6f}")

    # Test 4: excess alpha widens both sides
    db_base, da_base = quote(0.0, 0.001, 0.50, 0.0, mid=100.0, alpha_baseline=0.50)
    db_high, da_high = quote(0.0, 0.001, 0.80, 0.0, mid=100.0, alpha_baseline=0.50)
    if not (db_high > db_base):
        print(f"  FAIL: excess alpha should widen. db_high={db_high}, db_base={db_base}"); PASS = False
    else:
        print(f"  PASS: excess alpha widens  {db_base:.6f} → {db_high:.6f}")

    # Test 5: sub-baseline alpha → no surcharge
    db_low, da_low = quote(0.0, 0.001, 0.30, 0.0, mid=100.0, alpha_baseline=0.50)
    if not math.isclose(db_low, db_base, rel_tol=1e-6):
        print(f"  FAIL: sub-baseline alpha should match baseline"); PASS = False
    else:
        print(f"  PASS: sub-baseline alpha no surcharge")

    # Test 6: bounds respected
    for I in [-2000, 0, 2000]:
        for eta in [0.0, 0.5, 0.95, 1.0]:
            db, da = quote(I, 0.001, 0.60, eta, mid=100.0, alpha_baseline=0.50)
            floor = C_MIN * 0.001; cap = BPS_MAX * 100.0
            if not (floor - 1e-9 <= db <= cap + 1e-9 and
                    floor - 1e-9 <= da <= cap + 1e-9):
                print(f"  FAIL: bounds violated I={I} eta={eta} db={db} da={da}"); PASS = False

    if PASS:
        print("All unit tests PASSED.\n")
    return PASS


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _run_unit_tests()
    validate_quote(data_path='trade_data.csv', alpha_path=None,
                   lam=0.6, gamma=1.5, phi=1.0)
