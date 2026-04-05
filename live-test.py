"""Backtest the HMM model against live Kalshi market data.

Reads scraped match data from scraper/data/, replays the tennis points
through the HMM, and compares the model's live win probability to the
market's implied probability at each point.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import (
    encode_observations,
    invert_match_odds,
    live_win_probability,
    predict_proba,
)

PROJECT_DIR = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "hmm_model.pkl")


def load_scraped_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and clean scraped CSV into separate Kalshi and tennis DataFrames.

    Returns
    -------
    kalshi : DataFrame with timestamp, yes_mid (midpoint), last_price, volume
    tennis_points : DataFrame with point-level tennis data
    match_info : dict with player names and initial state
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Extract match info from snapshot ---
    snap = df[df["event_type"] == "snapshot"]
    match_info = {}
    if len(snap) > 0:
        raw = json.loads(snap.iloc[0]["raw"])
        match_info["first_player"] = raw.get("event_first_player", "Player 1")
        match_info["second_player"] = raw.get("event_second_player", "Player 2")

    # --- Clean Kalshi data ---
    kalshi = df[df["source"] == "kalshi"].copy()
    kalshi["yes_bid"] = pd.to_numeric(kalshi["yes_bid"], errors="coerce")
    kalshi["yes_ask"] = pd.to_numeric(kalshi["yes_ask"], errors="coerce")
    kalshi["last_price"] = pd.to_numeric(kalshi["last_price"], errors="coerce")
    kalshi["volume"] = pd.to_numeric(kalshi["volume"], errors="coerce")
    # Midpoint as best estimate of market probability
    kalshi["yes_mid"] = (kalshi["yes_bid"] + kalshi["yes_ask"]) / 2
    # Drop rows where market is empty (both bid and ask are 0)
    kalshi = kalshi[(kalshi["yes_bid"] > 0) | (kalshi["yes_ask"] > 0)].copy()
    kalshi = kalshi.reset_index(drop=True)

    # --- Clean tennis point data ---
    tennis_points = df[df["event_type"] == "point"].copy()
    tennis_points = tennis_points.reset_index(drop=True)

    return kalshi, tennis_points, match_info


def build_point_observations(tennis_points: pd.DataFrame) -> pd.DataFrame:
    """Convert scraped tennis points into the format encode_observations expects.

    The scraped data has server/winner as "First Player"/"Second Player".
    We convert to the numeric format the model expects.
    """
    rows = []
    for _, pt in tennis_points.iterrows():
        server = pt["server"]
        winner = pt["point_winner"]

        # Map to numeric: First Player = 1, Second Player = 2
        point_server = 1 if "First" in str(server) else 2
        point_winner = 1 if "First" in str(winner) else 2

        # Determine ace/double fault from context
        # We don't have ace/df flags in scraped data, so default to 0
        rows.append({
            "PointServer": point_server,
            "PointWinner": point_winner,
            "P1Ace": 0,
            "P2Ace": 0,
            "P1DoubleFault": 0,
            "P2DoubleFault": 0,
            "timestamp": pt["timestamp"],
            "set_score": pt["set_score"],
            "game_score": pt["game_score"],
            "point_score": pt["point_score"],
        })

    return pd.DataFrame(rows)


def run_model_on_points(model, points_df: pd.DataFrame, pre_match_prob: float):
    """Run HMM on each point cumulatively and return model win probabilities.

    At each point i, we feed points 0..i through the HMM to get posteriors,
    then compute P(second player wins) to compare with Kalshi (which prices
    the second player winning).
    """
    # Invert pre-match odds to get baseline serve/return for player 2
    baseline_s2, baseline_r2 = invert_match_odds(pre_match_prob, best_of=3)

    encoded = encode_observations(points_df)
    observations = encoded["observation"].values

    model_probs = []
    for i in range(1, len(observations) + 1):
        X = observations[:i].reshape(-1, 1)
        posteriors = predict_proba(model, X, np.array([i]))

        # Get P(server wins) at the last point
        is_p2_serving = points_df.iloc[i - 1]["PointServer"] == 2
        score_dict = {"server_serving": not is_p2_serving}  # from P2's perspective

        p2_win = live_win_probability(
            model, posteriors, i - 1,
            baseline_s2, baseline_r2,
            score=score_dict,
            best_of=3,
        )
        model_probs.append(p2_win)

    return model_probs


def find_market_price_at_point(kalshi: pd.DataFrame, point_time: pd.Timestamp) -> float:
    """Find the closest Kalshi market midpoint price at a given time."""
    # Find the last kalshi row before or at this time
    before = kalshi[kalshi["timestamp"] <= point_time]
    if len(before) > 0:
        return before.iloc[-1]["yes_mid"]
    # Fallback: closest row
    idx = (kalshi["timestamp"] - point_time).abs().idxmin()
    return kalshi.loc[idx, "yes_mid"]


def plot_model_vs_market(
    kalshi: pd.DataFrame,
    tennis_points: pd.DataFrame,
    model_probs: list[float],
    match_info: dict,
    pre_match_prob: float,
    output_path: str,
):
    """Plot market probability and model probability over time."""
    p1_name = match_info.get("first_player", "Player 1")
    p2_name = match_info.get("second_player", "Player 2")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # --- Top: probability over time ---
    ax = axes[0]

    # Market probability (continuous)
    market_times = kalshi["timestamp"]
    market_probs = kalshi["yes_mid"]
    ax.plot(market_times, market_probs, linewidth=0.8, color="#f59e0b",
            alpha=0.7, label="Kalshi Market")

    # Model probability (at each point)
    point_times = pd.to_datetime(tennis_points["timestamp"])
    ax.plot(point_times, model_probs, "o-", color="#2563eb", linewidth=1.5,
            markersize=5, label="HMM Model", zorder=5)

    # Pre-match line
    ax.axhline(pre_match_prob, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5, label=f"Pre-match ({pre_match_prob:.0%})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

    ax.set_ylabel(f"P({p2_name} wins)")
    ax.set_title(
        f"Live Model vs Market: {p1_name} vs {p2_name}\n"
        f"Kalshi market = P({p2_name} wins)"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    # --- Bottom: model - market spread ---
    ax = axes[1]
    market_at_points = [
        find_market_price_at_point(kalshi, t) for t in point_times
    ]
    spread = [m - k for m, k in zip(model_probs, market_at_points)]

    colors = ["#22c55e" if s > 0 else "#ef4444" for s in spread]
    ax.bar(range(len(spread)), spread, color=colors, width=0.6, edgecolor="white")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Point Number")
    ax.set_ylabel("Model - Market")
    ax.set_title("Model vs Market Spread (positive = model more bullish on P2)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_accuracy_summary(
    model_probs: list[float],
    market_at_points: list[float],
    output_path: str,
):
    """Scatter plot of model prob vs market prob at each point."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(market_at_points, model_probs, s=50, color="#2563eb",
               edgecolors="white", linewidth=0.5, zorder=5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect agreement")

    ax.set_xlabel("Market P(P2 wins)")
    ax.set_ylabel("Model P(P2 wins)")
    ax.set_title("Model vs Market at Each Point")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    # Annotate stats
    errors = [abs(m - k) for m, k in zip(model_probs, market_at_points)]
    mae = np.mean(errors)
    ax.text(0.05, 0.92, f"MAE: {mae:.3f}\nn={len(model_probs)} points",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    csv_path = os.path.join(
        PROJECT_DIR, "scraper", "data",
        "KXATPCHALLENGERMATCH-26APR04DUCPAC-PAC.csv",
    )

    print("Loading scraped data...")
    kalshi, tennis_points, match_info = load_scraped_data(csv_path)
    print(f"  Kalshi rows: {len(kalshi)}")
    print(f"  Tennis points: {len(tennis_points)}")
    print(f"  {match_info.get('first_player')} vs {match_info.get('second_player')}")

    if len(tennis_points) == 0:
        print("No tennis points to analyze.")
        return

    # Pre-match probability from first Kalshi price
    pre_match_prob = float(kalshi.iloc[0]["yes_mid"])
    print(f"  Pre-match P({match_info.get('second_player')} wins): {pre_match_prob:.2f}")

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Building point observations...")
    points_df = build_point_observations(tennis_points)

    print("Running model on points...")
    model_probs = run_model_on_points(model, points_df, pre_match_prob)

    print(f"\nPoint-by-point comparison:")
    print(f"{'#':>3}  {'Time':>8}  {'Score':<12} {'Market':>7} {'Model':>7} {'Spread':>7}")
    print("-" * 55)
    for i, (_, pt) in enumerate(tennis_points.iterrows()):
        t = pd.to_datetime(pt["timestamp"])
        mkt = find_market_price_at_point(kalshi, t)
        mdl = model_probs[i]
        score = f"{pt['set_score']} {pt['game_score']}"
        print(f"{i+1:>3}  {t.strftime('%H:%M'):>8}  {score:<12} {mkt:>7.2f} {mdl:>7.2f} {mdl-mkt:>+7.2f}")

    # Generate figures
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_model_vs_market(
        kalshi, tennis_points, model_probs, match_info, pre_match_prob,
        os.path.join(FIGURES_DIR, "live_model_vs_market.png"),
    )

    market_at_points = [
        find_market_price_at_point(kalshi, pd.to_datetime(t))
        for t in tennis_points["timestamp"]
    ]
    plot_accuracy_summary(
        model_probs, market_at_points,
        os.path.join(FIGURES_DIR, "live_model_accuracy.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
