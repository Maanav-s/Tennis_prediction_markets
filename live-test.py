"""Backtest the HMM model against live Kalshi market data.

Reads scraped match data from scraper/data/, replays the tennis points
through the HMM, and compares the model's live win probability to the
market's implied probability at each score change.
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
    match_win_probability,
    predict_proba,
)

PROJECT_DIR = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "hmm_model.pkl")


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def _parse_score(score_str: str) -> tuple[int, int]:
    """Parse 'X - Y' into (x, y) ints."""
    parts = str(score_str).split("-")
    if len(parts) == 2:
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            pass
    return 0, 0


def load_scraped_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and clean scraped CSV.

    Returns
    -------
    kalshi : DataFrame with market price data
    score_events : DataFrame with all score-change events (point/game/set)
    match_info : dict with player names
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Match info from snapshot + Kalshi market ---
    match_info = {}
    snap = df[df["event_type"] == "snapshot"]
    if len(snap) > 0:
        try:
            raw = json.loads(snap.iloc[0]["raw"])
            match_info["first_player"] = raw.get("event_first_player", "Player 1")
            match_info["second_player"] = raw.get("event_second_player", "Player 2")
        except (json.JSONDecodeError, TypeError):
            match_info["first_player"] = "Player 1"
            match_info["second_player"] = "Player 2"

    # Determine which player Kalshi "yes" refers to
    kalshi_rows = df[df["source"] == "kalshi"]
    if len(kalshi_rows) > 0:
        try:
            kraw = json.loads(kalshi_rows.iloc[0]["raw"])
            yes_player = kraw.get("yes_sub_title", "")
            match_info["kalshi_yes_player"] = yes_player
            # Does "yes" refer to first or second player?
            if match_info.get("first_player", "") and match_info["first_player"].split()[-1] in yes_player:
                match_info["yes_is_p1"] = True
            else:
                match_info["yes_is_p1"] = False
        except (json.JSONDecodeError, TypeError):
            match_info["yes_is_p1"] = False

    # --- Kalshi data ---
    kalshi = df[df["source"] == "kalshi"].copy()
    kalshi["yes_bid"] = pd.to_numeric(kalshi["yes_bid"], errors="coerce")
    kalshi["yes_ask"] = pd.to_numeric(kalshi["yes_ask"], errors="coerce")
    kalshi["last_price"] = pd.to_numeric(kalshi["last_price"], errors="coerce")
    kalshi["volume"] = pd.to_numeric(kalshi["volume"], errors="coerce")
    kalshi["yes_mid"] = (kalshi["yes_bid"] + kalshi["yes_ask"]) / 2
    kalshi = kalshi[(kalshi["yes_bid"] > 0) | (kalshi["yes_ask"] > 0)].copy()
    kalshi = kalshi.reset_index(drop=True)

    # --- Score events (point, game, set changes) ---
    score_events = df[
        df["event_type"].isin(["point", "game", "set"])
    ].copy().reset_index(drop=True)

    return kalshi, score_events, match_info


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def infer_point_winner(prev_score: str, cur_score: str, server: str) -> str:
    """Infer who won a point by comparing consecutive point scores.

    Returns 'server' or 'returner' based on which side's score increased.
    Falls back to 'server' if ambiguous.
    """
    if not prev_score or not cur_score:
        return "server"

    score_order = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4}

    prev_parts = str(prev_score).split("-")
    cur_parts = str(cur_score).split("-")

    if len(prev_parts) != 2 or len(cur_parts) != 2:
        return "server"

    prev_s = score_order.get(prev_parts[0].strip(), 0)
    prev_r = score_order.get(prev_parts[1].strip(), 0)
    cur_s = score_order.get(cur_parts[0].strip(), 0)
    cur_r = score_order.get(cur_parts[1].strip(), 0)

    # Server's score went up, or returner's went down (lost advantage)
    if cur_s > prev_s or cur_r < prev_r:
        return "server"
    return "returner"


def build_observations_from_events(
    score_events: pd.DataFrame,
) -> pd.DataFrame:
    """Build observation DataFrame for the HMM from score-change events.

    Each event becomes a point observation. We infer who won each point
    from the score progression.
    """
    rows = []
    prev_point_score = None
    prev_sets = (0, 0)
    prev_games = (0, 0)

    for _, ev in score_events.iterrows():
        server_str = str(ev.get("server", ""))
        point_server = 1 if "First" in server_str else 2

        point_score = str(ev.get("point_score", ""))
        event_type = ev["event_type"]

        # Infer winner
        if event_type in ("game", "set"):
            point_winner = point_server  # default: assume server held
        else:
            winner_side = infer_point_winner(prev_point_score, point_score, server_str)
            point_winner = point_server if winner_side == "server" else (3 - point_server)

        prev_point_score = point_score if event_type == "point" else None

        # Parse scores
        p1_sets, p2_sets = _parse_score(ev.get("set_score", "0 - 0"))
        p1_games, p2_games = _parse_score(ev.get("game_score", "0 - 0"))

        # Fix stale game scores from the API.
        # The API sometimes reports the previous set's final game score
        # (e.g. "1 - 6") for several polls after a new set starts.
        # Detect: if one player has >= 6 games but the game total is
        # consistent with a completed set, it's stale data.  In an active
        # set, both can only reach 6+ during a tiebreak (6-6 or 7-6).
        completed_set = (
            (p1_games >= 6 and p1_games - p2_games >= 2) or
            (p2_games >= 6 and p2_games - p1_games >= 2)
        )
        if completed_set:
            # This looks like a final set score, not an in-progress set
            p1_games, p2_games = prev_games if (p1_sets, p2_sets) == prev_sets else (0, 0)

        # Clamp impossible game counts
        p1_games = min(p1_games, 7)
        p2_games = min(p2_games, 7)

        # If game score went backwards significantly within same set,
        # keep previous (API glitch)
        if (p1_sets, p2_sets) == prev_sets:
            if (p1_games + p2_games) < (prev_games[0] + prev_games[1]) - 1:
                p1_games, p2_games = prev_games

        prev_sets = (p1_sets, p2_sets)
        prev_games = (p1_games, p2_games)

        rows.append({
            "PointServer": point_server,
            "PointWinner": point_winner,
            "P1Ace": 0,
            "P2Ace": 0,
            "P1DoubleFault": 0,
            "P2DoubleFault": 0,
            "timestamp": ev["timestamp"],
            "event_type": event_type,
            "p1_sets": p1_sets,
            "p2_sets": p2_sets,
            "p1_games": p1_games,
            "p2_games": p2_games,
            "point_score": point_score,
        })

    return pd.DataFrame(rows)


def run_model(
    model, obs_df: pd.DataFrame, kalshi: pd.DataFrame,
    pre_match_prob: float, yes_is_p1: bool,
    recalibrate: bool = True,
):
    """Run HMM on each score event cumulatively and return model win probs.

    Parameters
    ----------
    recalibrate : bool
        If True, re-anchor baseline to current market price at each point.
        If False, use only the pre-match odds as baseline throughout.
    """
    baseline_s, baseline_r = invert_match_odds(pre_match_prob, best_of=3)

    encoded = encode_observations(obs_df)
    observations = encoded["observation"].values

    model_probs = []
    for i in range(1, len(observations) + 1):
        X = observations[:i].reshape(-1, 1)
        posteriors = predict_proba(model, X, np.array([i]))

        row = obs_df.iloc[i - 1]
        point_time = pd.to_datetime(row["timestamp"])

        mkt_price = find_market_price_at(kalshi, point_time) if recalibrate else None

        if yes_is_p1:
            is_yes_serving = row["PointServer"] == 1
            score_dict = {
                "s_sets": row["p1_sets"],
                "r_sets": row["p2_sets"],
                "s_games": row["p1_games"],
                "r_games": row["p2_games"],
                "server_serving": is_yes_serving,
            }
        else:
            is_yes_serving = row["PointServer"] == 2
            score_dict = {
                "s_sets": row["p2_sets"],
                "r_sets": row["p1_sets"],
                "s_games": row["p2_games"],
                "r_games": row["p1_games"],
                "server_serving": is_yes_serving,
            }

        yes_win = live_win_probability(
            model, posteriors, i - 1,
            baseline_s, baseline_r,
            score=score_dict,
            best_of=3,
            market_prob=mkt_price,
        )
        model_probs.append(yes_win)

    return model_probs


def find_market_price_at(kalshi: pd.DataFrame, t: pd.Timestamp) -> float:
    """Find the closest Kalshi midpoint price at a given time."""
    before = kalshi[kalshi["timestamp"] <= t]
    if len(before) > 0:
        return before.iloc[-1]["yes_mid"]
    idx = (kalshi["timestamp"] - t).abs().idxmin()
    return kalshi.loc[idx, "yes_mid"]


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_model_vs_market(
    kalshi, score_events, model_probs, match_info, pre_match_prob, output_path,
):
    yes_player = match_info.get("kalshi_yes_player", "Yes Player")
    yes_is_p1 = match_info.get("yes_is_p1", False)
    p1 = match_info.get("first_player", "Player 1")
    p2 = match_info.get("second_player", "Player 2")

    point_times = pd.to_datetime(score_events["timestamp"])
    market_at_events = [find_market_price_at(kalshi, t) for t in point_times]
    spread = [m - k for m, k in zip(model_probs, market_at_events)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    # --- Top: probabilities over time ---
    ax = axes[0]
    ax.plot(kalshi["timestamp"], kalshi["yes_mid"], linewidth=0.8,
            color="#f59e0b", alpha=0.7, label="Kalshi Market")
    ax.plot(point_times, model_probs, "o-", color="#2563eb", linewidth=1.5,
            markersize=4, label="HMM Model", zorder=5)

    # Odds-only baseline (no HMM momentum, just score + pre-match odds)
    obs_df_temp = build_observations_from_events(score_events)
    baseline_s, baseline_r = invert_match_odds(pre_match_prob, best_of=3)
    odds_only = []
    for _, row in obs_df_temp.iterrows():
        if yes_is_p1:
            is_yes_serving = row["PointServer"] == 1
            score_dict = {
                "s_sets": row["p1_sets"], "r_sets": row["p2_sets"],
                "s_games": row["p1_games"], "r_games": row["p2_games"],
                "server_serving": is_yes_serving,
            }
        else:
            is_yes_serving = row["PointServer"] == 2
            score_dict = {
                "s_sets": row["p2_sets"], "r_sets": row["p1_sets"],
                "s_games": row["p2_games"], "r_games": row["p1_games"],
                "server_serving": is_yes_serving,
            }
        odds_only.append(match_win_probability(
            baseline_s, baseline_r, score=score_dict, best_of=3,
        ))
    ax.plot(point_times, odds_only, "s--", color="#94a3b8", linewidth=1.0,
            markersize=3, alpha=0.7, label="Odds + Score (no HMM)")

    ax.axhline(pre_match_prob, color="gray", linestyle="-.", linewidth=0.8,
               alpha=0.5, label=f"Pre-match ({pre_match_prob:.0%})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

    ax.set_ylabel(f"P({yes_player} wins)")
    ax.set_title(f"Live Model vs Market: {p1} vs {p2}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)

    # --- Bottom: spread over time ---
    ax = axes[1]
    ax.fill_between(point_times, spread, 0,
                    where=[s >= 0 for s in spread],
                    color="#22c55e", alpha=0.5, interpolate=True, label="Model > Market")
    ax.fill_between(point_times, spread, 0,
                    where=[s < 0 for s in spread],
                    color="#ef4444", alpha=0.5, interpolate=True, label="Model < Market")
    ax.plot(point_times, spread, color="#333", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Model - Market")
    ax.set_title(f"Spread (positive = model more bullish on {yes_player})")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_accuracy_scatter(model_probs, market_at_events, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(market_at_events, model_probs, s=50, color="#2563eb",
               edgecolors="white", linewidth=0.5, zorder=5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect agreement")

    ax.set_xlabel("Market P(P2 wins)")
    ax.set_ylabel("Model P(P2 wins)")
    ax.set_title("Model vs Market at Each Score Event")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    errors = [abs(m - k) for m, k in zip(model_probs, market_at_events)]
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    ax.text(0.05, 0.92, f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nn={len(model_probs)} events",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_market_around_points(kalshi, score_events, match_info, output_path):
    """Show how the market reacts around each score event."""
    p2 = match_info.get("second_player", "Player 2")
    point_times = pd.to_datetime(score_events["timestamp"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(kalshi["timestamp"], kalshi["yes_mid"], linewidth=0.8,
            color="#f59e0b", alpha=0.8, label="Kalshi mid")

    # Mark points, games, sets with different markers
    for etype, marker, color, size in [
        ("point", "o", "#2563eb", 30),
        ("game", "D", "#22c55e", 50),
        ("set", "*", "#ef4444", 100),
    ]:
        mask = score_events["event_type"] == etype
        if mask.sum() > 0:
            times = point_times[mask]
            prices = [find_market_price_at(kalshi, t) for t in times]
            ax.scatter(times, prices, s=size, marker=marker, color=color,
                       zorder=5, label=f"{etype.title()} scored", edgecolors="white")

    ax.set_ylabel(f"P({p2} wins)")
    ax.set_title(f"Market Price with Score Events")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_odds_score_vs_market(
    kalshi, score_events, match_info, pre_match_prob, output_path,
    window: int = 5,
):
    """Plot market vs odds+score model with a moving average smoothing."""
    yes_player = match_info.get("kalshi_yes_player", "Yes Player")
    yes_is_p1 = match_info.get("yes_is_p1", False)
    p1 = match_info.get("first_player", "Player 1")
    p2 = match_info.get("second_player", "Player 2")

    point_times = pd.to_datetime(score_events["timestamp"])

    # Compute raw odds+score at each event
    obs_df = build_observations_from_events(score_events)
    baseline_s, baseline_r = invert_match_odds(pre_match_prob, best_of=3)
    raw_probs = []
    for _, row in obs_df.iterrows():
        if yes_is_p1:
            is_yes_serving = row["PointServer"] == 1
            score_dict = {
                "s_sets": row["p1_sets"], "r_sets": row["p2_sets"],
                "s_games": row["p1_games"], "r_games": row["p2_games"],
                "server_serving": is_yes_serving,
            }
        else:
            is_yes_serving = row["PointServer"] == 2
            score_dict = {
                "s_sets": row["p2_sets"], "r_sets": row["p1_sets"],
                "s_games": row["p2_games"], "r_games": row["p1_games"],
                "server_serving": is_yes_serving,
            }
        raw_probs.append(match_win_probability(
            baseline_s, baseline_r, score=score_dict, best_of=3,
        ))

    raw_probs = np.array(raw_probs)
    smoothed = pd.Series(raw_probs).rolling(window=window, min_periods=1, center=True).mean().values

    market_at_events = [find_market_price_at(kalshi, t) for t in point_times]
    spread = smoothed - np.array(market_at_events)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    # --- Top: market vs smoothed odds+score ---
    ax = axes[0]
    ax.plot(kalshi["timestamp"], kalshi["yes_mid"], linewidth=0.8,
            color="#f59e0b", alpha=0.7, label="Kalshi Market")
    ax.plot(point_times, raw_probs, ".", color="#94a3b8", markersize=3,
            alpha=0.4, label="Odds + Score (raw)")
    ax.plot(point_times, smoothed, "-", color="#2563eb", linewidth=1.8,
            label=f"Odds + Score ({window}-pt avg)", zorder=5)
    ax.axhline(pre_match_prob, color="gray", linestyle="-.", linewidth=0.8,
               alpha=0.5, label=f"Pre-match ({pre_match_prob:.0%})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

    ax.set_ylabel(f"P({yes_player} wins)")
    ax.set_title(f"Market vs Odds + Score (no HMM): {p1} vs {p2}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)

    # --- Bottom: spread ---
    ax = axes[1]
    ax.fill_between(point_times, spread, 0,
                    where=spread >= 0,
                    color="#22c55e", alpha=0.5, interpolate=True, label="Model > Market")
    ax.fill_between(point_times, spread, 0,
                    where=spread < 0,
                    color="#ef4444", alpha=0.5, interpolate=True, label="Model < Market")
    ax.plot(point_times, spread, color="#333", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Smoothed Model - Market")
    ax.set_title(f"Spread ({window}-pt moving average)")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_recalibration_comparison(
    kalshi, score_events, model_recal, model_no_recal,
    match_info, pre_match_prob, output_path,
):
    """Compare model output with and without market recalibration."""
    yes_player = match_info.get("kalshi_yes_player", "Yes Player")
    p1 = match_info.get("first_player", "Player 1")
    p2 = match_info.get("second_player", "Player 2")
    point_times = pd.to_datetime(score_events["timestamp"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    # --- Top: all three lines ---
    ax = axes[0]
    ax.plot(kalshi["timestamp"], kalshi["yes_mid"], linewidth=0.8,
            color="#f59e0b", alpha=0.7, label="Kalshi Market")
    ax.plot(point_times, model_recal, "o-", color="#2563eb", linewidth=1.5,
            markersize=4, label="HMM + Recalibration", zorder=5)
    ax.plot(point_times, model_no_recal, "s--", color="#ef4444", linewidth=1.2,
            markersize=3, alpha=0.8, label="HMM (pre-match odds only)")
    ax.axhline(pre_match_prob, color="gray", linestyle="-.", linewidth=0.8,
               alpha=0.5, label=f"Pre-match ({pre_match_prob:.0%})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

    ax.set_ylabel(f"P({yes_player} wins)")
    ax.set_title(f"Recalibration Comparison: {p1} vs {p2}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)

    # --- Bottom: both spreads vs market ---
    ax = axes[1]
    market_at_events = [find_market_price_at(kalshi, t) for t in point_times]
    spread_recal = [m - k for m, k in zip(model_recal, market_at_events)]
    spread_no_recal = [m - k for m, k in zip(model_no_recal, market_at_events)]

    ax.plot(point_times, spread_recal, color="#2563eb", linewidth=1.2,
            label="With recalibration")
    ax.plot(point_times, spread_no_recal, color="#ef4444", linewidth=1.2,
            alpha=0.8, linestyle="--", label="Without recalibration")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Model - Market")
    ax.set_title("Spread vs Market")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(PROJECT_DIR, csv_path)
    else:
        csv_path = os.path.join(
            PROJECT_DIR, "scraper", "data",
            "KXATPCHALLENGERMATCH-26APR04BROWEN-BRO.csv",
        )

    print("Loading scraped data...")
    kalshi, score_events, match_info = load_scraped_data(csv_path)
    p1 = match_info.get("first_player", "Player 1")
    p2 = match_info.get("second_player", "Player 2")
    print(f"  {p1} vs {p2}")
    print(f"  Kalshi rows: {len(kalshi)}")
    print(f"  Score events: {len(score_events)} (point/game/set)")

    if len(score_events) == 0:
        print("No score events to analyze.")
        return

    yes_is_p1 = match_info.get("yes_is_p1", False)
    yes_player = match_info.get("kalshi_yes_player", p2)
    pre_match_prob = float(kalshi.iloc[0]["yes_mid"])
    print(f"  Kalshi yes = {yes_player} ({'P1' if yes_is_p1 else 'P2'})")
    print(f"  Initial market P({yes_player} wins): {pre_match_prob:.2f}")

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Building observations...")
    obs_df = build_observations_from_events(score_events)

    print("Running model (with market recalibration)...")
    model_probs = run_model(model, obs_df, kalshi, pre_match_prob, yes_is_p1,
                            recalibrate=True)

    print("Running model (without recalibration)...")
    model_probs_no_recal = run_model(model, obs_df, kalshi, pre_match_prob, yes_is_p1,
                                     recalibrate=False)

    # Print table
    point_times = pd.to_datetime(score_events["timestamp"])
    market_at_events = [find_market_price_at(kalshi, t) for t in point_times]

    print(f"\n{'#':>3}  {'Time':>8}  {'Type':<5} {'Sets':<6} {'Games':<6} {'Pts':<8} "
          f"{'Mkt':>6} {'Recal':>6} {'NoRecal':>7}")
    print("-" * 72)
    for i, (_, ev) in enumerate(score_events.iterrows()):
        t = pd.to_datetime(ev["timestamp"])
        mkt = market_at_events[i]
        mdl = model_probs[i]
        mdl_nr = model_probs_no_recal[i]
        etype = ev["event_type"][0].upper()
        print(f"{i+1:>3}  {t.strftime('%H:%M'):>8}  {etype:<5} {ev['set_score']:<6} "
              f"{ev['game_score']:<6} {str(ev['point_score']):<8} "
              f"{mkt:>6.2f} {mdl:>6.2f} {mdl_nr:>7.2f}")

    # Generate figures
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_model_vs_market(
        kalshi, score_events, model_probs, match_info, pre_match_prob,
        os.path.join(FIGURES_DIR, "live_model_vs_market.png"),
    )
    plot_accuracy_scatter(
        model_probs, market_at_events,
        os.path.join(FIGURES_DIR, "live_model_accuracy.png"),
    )
    plot_market_around_points(
        kalshi, score_events, match_info,
        os.path.join(FIGURES_DIR, "live_market_events.png"),
    )
    plot_odds_score_vs_market(
        kalshi, score_events, match_info, pre_match_prob,
        os.path.join(FIGURES_DIR, "live_odds_score_vs_market.png"),
    )
    plot_recalibration_comparison(
        kalshi, score_events, model_probs, model_probs_no_recal,
        match_info, pre_match_prob,
        os.path.join(FIGURES_DIR, "live_recalibration_comparison.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
