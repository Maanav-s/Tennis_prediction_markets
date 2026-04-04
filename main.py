"""
Markov chain model for predicting tennis match outcomes point-by-point.

State = (server_sets, returner_sets, server_games, returner_games,
         server_points, returner_points, is_tiebreak)

Trained on Grand Slam men's singles point-by-point data.
"""

import glob
import os
import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42
N_SIMULATIONS = 1000
TRAIN_FRAC = 0.8
SETS_TO_WIN = 3  # best-of-5

# Absorbing / terminal labels
SERVER_WINS = "SERVER_WINS"
RETURNER_WINS = "RETURNER_WINS"

# Tiebreak point cap (scores above this are grouped together)
TB_POINT_CAP = 7

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_points() -> pd.DataFrame:
    """Load all singles point-by-point CSVs into one DataFrame."""
    pattern = os.path.join(DATA_DIR, "*-points.csv")
    files = [f for f in glob.glob(pattern)
             if "doubles" not in f and "mixed" not in f]
    frames = []
    for f in sorted(files):
        df = pd.read_csv(f, dtype=str)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Filter to men's singles.
    # Newer files use "MS###" suffix for men's singles, "WS###" for women's.
    # Older files use numeric suffixes: < 2000 = men's, >= 2000 = women's.
    suffix = df["match_id"].str.split("-").str[-1]
    is_ms_prefix = suffix.str.startswith("MS")
    numeric_val = pd.to_numeric(suffix, errors="coerce")
    is_numeric_mens = numeric_val.notna() & (numeric_val < 2000)
    df = df[is_ms_prefix | is_numeric_mens].copy()

    # Drop sentinel rows (0X, 0Y)
    df = df[df["PointNumber"].str.isnumeric()].copy()

    # Convert numeric columns, dropping rows where these are missing
    int_cols = ["SetNo", "P1GamesWon", "P2GamesWon", "SetWinner",
                "GameWinner", "PointNumber", "PointWinner", "PointServer"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=int_cols).copy()
    for col in int_cols:
        df[col] = df[col].astype(int)

    df = df.sort_values(["match_id", "PointNumber"]).reset_index(drop=True)
    return df


def derive_match_winners(df: pd.DataFrame) -> dict[str, int]:
    """Return {match_id: winner (1 or 2)} derived from set wins in the points."""
    set_wins = df[df["SetWinner"].isin([1, 2])].copy()
    counts = set_wins.groupby(["match_id", "SetWinner"]).size().unstack(fill_value=0)
    winners = {}
    for mid, row in counts.iterrows():
        p1_sets = row.get(1, 0)
        p2_sets = row.get(2, 0)
        if p1_sets >= SETS_TO_WIN:
            winners[mid] = 1
        elif p2_sets >= SETS_TO_WIN:
            winners[mid] = 2
    return winners


# ---------------------------------------------------------------------------
# State computation
# ---------------------------------------------------------------------------

def _normalise_point_score(score_str: str, is_tiebreak: bool) -> str:
    """Normalise a point score string."""
    if is_tiebreak:
        try:
            v = int(score_str)
        except (ValueError, TypeError):
            return score_str
        return str(min(v, TB_POINT_CAP))
    return score_str  # "0", "15", "30", "40", "AD"


def compute_states_for_match(match_df: pd.DataFrame, match_winner: int):
    """
    Yield (state_before_point, state_after_point) for every point in a match.

    The last point's state_after is the appropriate absorbing state.
    States are from the server's perspective at each point.
    """
    rows = match_df.to_dict("records")

    # Pre-compute cumulative sets won at each point.
    # SetWinner is nonzero on the row where a set ends. The games shown on
    # that row already reflect the completed game, and P1Score/P2Score reset
    # to 0. We count set wins *before* the current row to get the score at
    # the start of each point.
    p1_sets_cum = []
    p2_sets_cum = []
    p1s = 0
    p2s = 0
    for r in rows:
        p1_sets_cum.append(p1s)
        p2_sets_cum.append(p2s)
        if r["SetWinner"] == 1:
            p1s += 1
        elif r["SetWinner"] == 2:
            p2s += 1

    def _make_state(idx):
        r = rows[idx]
        server = r["PointServer"]  # 1 or 2
        is_tb = (r["P1GamesWon"] == 6 and r["P2GamesWon"] == 6)

        if server == 1:
            s_sets, r_sets = p1_sets_cum[idx], p2_sets_cum[idx]
            s_games, r_games = r["P1GamesWon"], r["P2GamesWon"]
            s_pts = _normalise_point_score(r["P1Score"], is_tb)
            r_pts = _normalise_point_score(r["P2Score"], is_tb)
        else:
            s_sets, r_sets = p2_sets_cum[idx], p1_sets_cum[idx]
            s_games, r_games = r["P2GamesWon"], r["P1GamesWon"]
            s_pts = _normalise_point_score(r["P2Score"], is_tb)
            r_pts = _normalise_point_score(r["P1Score"], is_tb)

        return (s_sets, r_sets, s_games, r_games, s_pts, r_pts, is_tb)

    for i in range(len(rows)):
        state_before = _make_state(i)

        if i + 1 < len(rows):
            state_after = _make_state(i + 1)
        else:
            # Last point — determine absorbing state
            server = rows[i]["PointServer"]
            if match_winner == server:
                state_after = SERVER_WINS
            else:
                state_after = RETURNER_WINS
            # But we also need to know who was serving for mapping back,
            # so yield the server identity alongside.

        yield state_before, state_after, rows[i]["PointServer"]


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def build_transitions(df: pd.DataFrame, match_ids: set, winners: dict):
    """Build transition counts from training matches."""
    transitions = defaultdict(Counter)
    serve_points_won = 0
    serve_points_total = 0

    for mid in sorted(match_ids):
        if mid not in winners:
            continue
        mdf = df[df["match_id"] == mid]
        for s_before, s_after, server in compute_states_for_match(mdf, winners[mid]):
            transitions[s_before][s_after] += 1

        # Tally serve point win rate
        mrows = mdf[mdf["PointWinner"].isin([1, 2]) & mdf["PointServer"].isin([1, 2])]
        serve_points_won += (mrows["PointWinner"] == mrows["PointServer"]).sum()
        serve_points_total += len(mrows)

    # Convert to probabilities
    trans_prob = {}
    for s_from, counter in transitions.items():
        total = sum(counter.values())
        trans_prob[s_from] = {s_to: c / total for s_to, c in counter.items()}

    serve_win_rate = serve_points_won / serve_points_total if serve_points_total else 0.6
    return trans_prob, serve_win_rate


# ---------------------------------------------------------------------------
# Tennis scoring rules engine (fallback for unseen states)
# ---------------------------------------------------------------------------

REGULAR_SCORES = ["0", "15", "30", "40"]

def next_state_by_rules(state, server_wins_point: bool):
    """
    Given a state tuple and who won the point, return the next state
    deterministically using tennis rules. Returns a state tuple or an
    absorbing state string.
    """
    s_sets, r_sets, s_games, r_games, s_pts, r_pts, is_tb = state

    if is_tb:
        return _next_tiebreak(s_sets, r_sets, s_games, r_games,
                              s_pts, r_pts, server_wins_point)
    else:
        return _next_regular(s_sets, r_sets, s_games, r_games,
                             s_pts, r_pts, server_wins_point)


def _advance_set(s_sets, r_sets, server_won_set: bool):
    """After a set is won, return new state or absorbing."""
    if server_won_set:
        s_sets += 1
        if s_sets >= SETS_TO_WIN:
            return SERVER_WINS
    else:
        r_sets += 1
        if r_sets >= SETS_TO_WIN:
            return RETURNER_WINS
    # New set starts: returner now serves (service alternates from last game)
    # From the new server's perspective, they are the "server" with 0-0, 0-0.
    # But since service switches, the current returner becomes the server.
    # In our state representation, we always orient around the current server,
    # so the new state is simply (r_sets_new, s_sets_new, 0, 0, "0", "0", False)
    # because the roles swap.
    if server_won_set:
        return (r_sets, s_sets, 0, 0, "0", "0", False)
    else:
        return (r_sets, s_sets, 0, 0, "0", "0", False)


def _advance_game(s_sets, r_sets, s_games, r_games, server_won_game: bool):
    """After a game is won (non-tiebreak), return new state."""
    if server_won_game:
        s_games += 1
    else:
        r_games += 1

    # Check if set is won
    if s_games >= 6 and s_games - r_games >= 2:
        return _advance_set(s_sets, r_sets, True)
    if r_games >= 6 and r_games - s_games >= 2:
        return _advance_set(s_sets, r_sets, False)

    # Check for tiebreak
    if s_games == 6 and r_games == 6:
        # Tiebreak starts; the same server continues for the first point
        return (s_sets, r_sets, 6, 6, "0", "0", True)

    # Next game: service switches, so swap perspective
    return (r_sets, s_sets, r_games, s_games, "0", "0", False)


def _next_regular(s_sets, r_sets, s_games, r_games, s_pts, r_pts,
                  server_wins_point: bool):
    """Next state in a regular (non-tiebreak) game."""
    if server_wins_point:
        if s_pts == "0":
            new_s, new_r = "15", r_pts
        elif s_pts == "15":
            new_s, new_r = "30", r_pts
        elif s_pts == "30":
            new_s, new_r = "40", r_pts
        elif s_pts == "40" and r_pts != "40" and r_pts != "AD":
            # Server wins game
            return _advance_game(s_sets, r_sets, s_games, r_games, True)
        elif s_pts == "40" and r_pts == "40":
            new_s, new_r = "AD", "40"
        elif s_pts == "40" and r_pts == "AD":
            new_s, new_r = "40", "40"  # back to deuce
        elif s_pts == "AD":
            # Server wins game
            return _advance_game(s_sets, r_sets, s_games, r_games, True)
        else:
            new_s, new_r = s_pts, r_pts
    else:
        if r_pts == "0":
            new_s, new_r = s_pts, "15"
        elif r_pts == "15":
            new_s, new_r = s_pts, "30"
        elif r_pts == "30":
            new_s, new_r = s_pts, "40"
        elif r_pts == "40" and s_pts != "40" and s_pts != "AD":
            # Returner wins game (break)
            return _advance_game(s_sets, r_sets, s_games, r_games, False)
        elif r_pts == "40" and s_pts == "40":
            new_s, new_r = "40", "AD"
        elif r_pts == "40" and s_pts == "AD":
            new_s, new_r = "40", "40"
        elif r_pts == "AD":
            return _advance_game(s_sets, r_sets, s_games, r_games, False)
        else:
            new_s, new_r = s_pts, r_pts

    return (s_sets, r_sets, s_games, r_games, new_s, new_r, False)


def _next_tiebreak(s_sets, r_sets, s_games, r_games, s_pts, r_pts,
                   server_wins_point: bool):
    """Next state in a tiebreak game."""
    sp = int(s_pts)
    rp = int(r_pts)

    if server_wins_point:
        sp += 1
    else:
        rp += 1

    # Check tiebreak win: first to 7, lead by 2
    if sp >= 7 and sp - rp >= 2:
        return _advance_set(s_sets, r_sets, True)
    if rp >= 7 and rp - sp >= 2:
        return _advance_set(s_sets, r_sets, False)

    # Service changes after 1st point, then every 2 points
    total_pts = sp + rp
    # In a tiebreak, service switches after 1 point, then every 2.
    # Since we track state from server perspective, we need to swap when
    # service changes. Service changes when total_pts is odd.
    if total_pts % 2 == 1:
        # Swap perspective
        return (r_sets, s_sets, r_games, s_games,
                str(min(rp, TB_POINT_CAP)), str(min(sp, TB_POINT_CAP)), True)
    else:
        return (s_sets, r_sets, s_games, r_games,
                str(min(sp, TB_POINT_CAP)), str(min(rp, TB_POINT_CAP)), True)


# ---------------------------------------------------------------------------
# Monte Carlo prediction
# ---------------------------------------------------------------------------

def simulate_once(state, trans_prob, serve_win_rate, rng, max_steps=500):
    """
    Simulate from a state to an absorbing state.
    Returns True if the initial-point server wins the match.

    Since we always track state from the *current* server's perspective,
    we must also track whether the initial-point server is currently the
    "server" or "returner" in the state representation. Each time service
    swaps (detected by the state transition swapping sets/games perspective),
    we flip our tracking bit.
    """
    initial_server_is_server = True  # at the start, initial server = state's server
    current = state

    for _ in range(max_steps):
        if current == SERVER_WINS:
            return initial_server_is_server
        if current == RETURNER_WINS:
            return not initial_server_is_server

        if current in trans_prob:
            targets = list(trans_prob[current].keys())
            probs = list(trans_prob[current].values())
            idx = rng.choice(len(targets), p=probs)
            next_s = targets[idx]
        else:
            # Fallback: use serve win rate + rules
            server_wins = rng.random() < serve_win_rate
            next_s = next_state_by_rules(current, server_wins)

        # Detect service swap: if next state is a tuple and the sets/games
        # have been swapped (returner's games became server's games), then
        # service changed.
        if isinstance(next_s, tuple) and isinstance(current, tuple):
            # A service swap is indicated when the sets perspective flips.
            # We detect this by checking if (s_sets, r_sets) of next ==
            # (r_sets, s_sets) of current (accounting for no set win).
            c_s_sets, c_r_sets = current[0], current[1]
            n_s_sets, n_r_sets = next_s[0], next_s[1]
            # If sets swapped (and no set was just won/lost):
            if (n_s_sets == c_r_sets and n_r_sets == c_s_sets
                    and n_s_sets + n_r_sets == c_s_sets + c_r_sets):
                initial_server_is_server = not initial_server_is_server

        current = next_s

    # Didn't terminate — use serve win rate as tiebreaker
    return rng.random() < 0.5


def predict_server_win_prob(state, trans_prob, serve_win_rate,
                            n_sims=N_SIMULATIONS, rng=None):
    """P(server at this point wins the match) via Monte Carlo."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    wins = sum(simulate_once(state, trans_prob, serve_win_rate, rng)
               for _ in range(n_sims))
    return wins / n_sims


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_matches(df, test_ids, winners, trans_prob, serve_win_rate):
    """
    For each point in each test match, predict P(P1 wins).
    Returns a DataFrame with columns:
        match_id, point_idx, frac_complete, p1_win_prob, p1_actually_won, server
    """
    rng = np.random.default_rng(SEED)
    # Memoize predictions for each unique state
    memo = {}
    results = []

    test_ids_sorted = sorted(tid for tid in test_ids if tid in winners)
    for i, mid in enumerate(test_ids_sorted):
        if (i + 1) % 50 == 0:
            print(f"  Evaluating match {i+1}/{len(test_ids_sorted)} ...")
        mdf = df[df["match_id"] == mid]
        p1_won = 1 if winners[mid] == 1 else 0
        states = list(compute_states_for_match(mdf, winners[mid]))
        n_points = len(states)

        for j, (s_before, _, server) in enumerate(states):
            if s_before not in memo:
                memo[s_before] = predict_server_win_prob(
                    s_before, trans_prob, serve_win_rate, N_SIMULATIONS, rng
                )
            p_server_wins = memo[s_before]

            # Convert to P(P1 wins)
            if server == 1:
                p1_prob = p_server_wins
            else:
                p1_prob = 1.0 - p_server_wins

            results.append({
                "match_id": mid,
                "point_idx": j,
                "frac_complete": j / max(n_points - 1, 1),
                "p1_win_prob": p1_prob,
                "p1_actually_won": p1_won,
                "server": server,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_calibration(results: pd.DataFrame, save_dir: str):
    """Calibration plot: binned predicted probability vs actual outcome."""
    results = results.copy()
    results["bin"] = pd.cut(results["p1_win_prob"], bins=10, labels=False)
    grouped = results.groupby("bin").agg(
        mean_pred=("p1_win_prob", "mean"),
        mean_actual=("p1_actually_won", "mean"),
        count=("p1_win_prob", "size"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.scatter(grouped["mean_pred"], grouped["mean_actual"],
               s=grouped["count"] / grouped["count"].max() * 300,
               zorder=3, edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Predicted P(P1 wins)")
    ax.set_ylabel("Actual P(P1 wins)")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "calibration.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved calibration.png")


def plot_brier_over_time(results: pd.DataFrame, save_dir: str):
    """Brier score as a function of match progress."""
    results = results.copy()
    results["brier"] = (results["p1_win_prob"] - results["p1_actually_won"]) ** 2
    results["progress_bin"] = pd.cut(results["frac_complete"], bins=20, labels=False)
    grouped = results.groupby("progress_bin").agg(
        mean_progress=("frac_complete", "mean"),
        mean_brier=("brier", "mean"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["mean_progress"], grouped["mean_brier"], "o-", color="steelblue")
    ax.set_xlabel("Match progress (fraction of points played)")
    ax.set_ylabel("Mean Brier score")
    ax.set_title("Prediction Accuracy Over Match Progress")
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Coin-flip baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "brier_over_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved brier_over_time.png")


def plot_match_traces(results: pd.DataFrame, save_dir: str, n_matches=4):
    """Plot P(P1 wins) over time for a few example test matches."""
    match_ids = results["match_id"].unique()
    rng = np.random.default_rng(SEED + 1)
    chosen = rng.choice(match_ids, size=min(n_matches, len(match_ids)), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, mid in zip(axes, chosen):
        mdata = results[results["match_id"] == mid].sort_values("point_idx")
        won = mdata["p1_actually_won"].iloc[0]
        ax.plot(mdata["point_idx"], mdata["p1_win_prob"], color="steelblue",
                linewidth=1)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.axhline(won, color="green" if won == 1 else "red",
                   linestyle=":", alpha=0.6, label=f"P1 {'won' if won else 'lost'}")
        ax.set_xlabel("Point number")
        ax.set_ylabel("P(P1 wins)")
        ax.set_title(mid, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("Match Win Probability Traces", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "match_traces.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved match_traces.png")


def plot_prediction_histogram(results: pd.DataFrame, save_dir: str):
    """Histogram of predicted P(P1 wins) at the first point of each match."""
    first_points = results[results["point_idx"] == 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(first_points["p1_win_prob"], bins=30, edgecolor="black",
            color="steelblue", alpha=0.7)
    ax.set_xlabel("Predicted P(P1 wins) at match start")
    ax.set_ylabel("Count")
    ax.set_title("Pre-Match Win Probability Distribution\n(player-agnostic model)")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "prematch_histogram.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved prematch_histogram.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    df = load_points()
    print(f"  {len(df):,} points across {df['match_id'].nunique()} men's singles matches")

    print("Deriving match winners ...")
    winners = derive_match_winners(df)
    print(f"  {len(winners)} complete matches with a winner")

    # Train/test split
    all_ids = sorted(winners.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_ids)
    split = int(len(all_ids) * TRAIN_FRAC)
    train_ids = set(all_ids[:split])
    test_ids = set(all_ids[split:])
    print(f"  Train: {len(train_ids)}  Test: {len(test_ids)}")

    print("Building transition matrix ...")
    trans_prob, serve_win_rate = build_transitions(df, train_ids, winners)
    print(f"  {len(trans_prob)} unique states observed")
    print(f"  Serve win rate: {serve_win_rate:.3f}")

    print("Evaluating on test set (this may take a few minutes) ...")
    results = evaluate_matches(df, test_ids, winners, trans_prob, serve_win_rate)

    # Overall Brier score
    brier = ((results["p1_win_prob"] - results["p1_actually_won"]) ** 2).mean()
    print(f"  Overall Brier score: {brier:.4f}  (coin-flip baseline: 0.2500)")

    # Match-level accuracy (did the model's pre-match favourite actually win?)
    first_pts = results[results["point_idx"] == 0].copy()
    first_pts["predicted_p1"] = first_pts["p1_win_prob"] >= 0.5
    first_pts["correct"] = first_pts["predicted_p1"] == first_pts["p1_actually_won"].astype(bool)
    prematch_acc = first_pts["correct"].mean()
    print(f"  Pre-match accuracy: {prematch_acc:.3f}")

    # Save plots
    save_dir = os.path.dirname(__file__)
    print("Generating visualisations ...")
    plot_calibration(results, save_dir)
    plot_brier_over_time(results, save_dir)
    plot_match_traces(results, save_dir)
    plot_prediction_histogram(results, save_dir)

    print("Done.")


if __name__ == "__main__":
    main()
