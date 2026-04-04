import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataservice import (
    load_singles_data,
    merge_odds_with_matches,
    train_val_split,
)
from model import (
    build_sequences,
    create_model,
    decode,
    encode_observations,
    get_model_params,
    hmm_momentum_adjustment,
    invert_match_odds,
    live_win_probability,
    match_win_probability,
    predict_proba,
    score,
    train,
    N_CATEGORIES,
)

PROJECT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "hmm_model.pkl")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

OBS_LABELS = ["Server Win", "Ace", "Server Loss", "Double Fault"]


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_cached_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_mens(match_row) -> bool:
    event = str(match_row.get("event_name", ""))
    if "Men" in event or event == "event_MS":
        return True
    # Fallback: match_num < 2000 = men's in Sackmann data
    mid = str(match_row.get("match_id", ""))
    parts = mid.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1]) < 2000
    return True


def _best_of(match_row) -> int:
    return 5 if _is_mens(match_row) else 3


# ---------------------------------------------------------------------------
# Validation — odds-informed
# ---------------------------------------------------------------------------

def evaluate(model, val_matches, val_points):
    """Predict match winners using market odds as baseline + HMM momentum."""
    val_enc = encode_observations(val_points)

    # Only evaluate completed men's matches with known winner AND odds
    valid_matches = val_matches[
        val_matches["winner"].isin([1.0, 2.0])
        & val_matches["implied_p1_prob"].notna()
    ].copy()
    valid_ids = set(valid_matches["match_id"])
    val_enc_valid = val_enc[val_enc["match_id"].isin(valid_ids)]

    results = []
    grouped = val_enc_valid.groupby("match_id", sort=False)
    for match_id, group in grouped:
        match_row = valid_matches[valid_matches["match_id"] == match_id].iloc[0]
        best_of = _best_of(match_row)
        implied_prob = match_row["implied_p1_prob"]
        slam = match_row.get("slam")

        # Odds-only prediction (baseline)
        odds_pred = 1 if implied_prob > 0.5 else 2

        # Invert odds with surface-specific base rate
        baseline_s, baseline_r = invert_match_odds(
            implied_prob, best_of=best_of, slam=slam,
        )

        # HMM-adjusted prediction using full match observations
        X_m = group["observation"].values.reshape(-1, 1)
        posteriors = predict_proba(model, X_m, np.array([len(X_m)]))

        # Use final-point posterior for end-of-match assessment
        last_idx = len(X_m) - 1
        is_p1_serving = group["PointServer"].values[-1] == 1
        hmm_prob = live_win_probability(
            model, posteriors, last_idx,
            baseline_s, baseline_r,
            score={"server_serving": is_p1_serving},
            best_of=best_of,
            slam=slam,
        )

        actual_winner = int(match_row["winner"])

        results.append({
            "match_id": match_id,
            "implied_p1_prob": implied_prob,
            "odds_predicted": odds_pred,
            "hmm_prob": hmm_prob,
            "hmm_predicted": 1 if hmm_prob > 0.5 else 2,
            "actual_winner": actual_winner,
            "best_of": best_of,
        })

    df = pd.DataFrame(results)
    df["odds_correct"] = df["odds_predicted"] == df["actual_winner"]
    df["hmm_correct"] = df["hmm_predicted"] == df["actual_winner"]
    return df


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_win_prob_evolution(model, val_enc, val_matches):
    """Plot live P1 win probability evolving point-by-point for a sample match,
    using market odds as baseline and HMM momentum adjustments."""
    valid = val_matches[
        val_matches["winner"].isin([1.0, 2.0])
        & val_matches["implied_p1_prob"].notna()
    ]
    match_lengths = val_enc.groupby("match_id").size()
    valid_lengths = match_lengths[match_lengths.index.isin(valid["match_id"])]
    # Pick a long match for an interesting plot
    target_len = valid_lengths.quantile(0.9)
    sample_id = (valid_lengths - target_len).abs().idxmin()
    sample_pts = val_enc[val_enc["match_id"] == sample_id]
    match_row = valid[valid["match_id"] == sample_id].iloc[0]

    best_of = _best_of(match_row)
    implied_prob = match_row["implied_p1_prob"]
    slam = match_row.get("slam")
    baseline_s, baseline_r = invert_match_odds(
        implied_prob, best_of=best_of, slam=slam,
    )

    X_m = sample_pts["observation"].values.reshape(-1, 1)
    posteriors = predict_proba(model, X_m, np.array([len(X_m)]))
    is_p1_serving = sample_pts["PointServer"].values == 1

    # Compute live win prob at each point (HMM-adjusted)
    hmm_probs = []
    # Also compute odds-only baseline (no momentum) at each point
    odds_probs = []
    for i in range(len(sample_pts)):
        score_dict = {"server_serving": bool(is_p1_serving[i])}
        hmm_wp = live_win_probability(
            model, posteriors, i, baseline_s, baseline_r,
            score=score_dict, best_of=best_of, slam=slam,
        )
        odds_wp = match_win_probability(
            baseline_s, baseline_r, score=score_dict, best_of=best_of,
        )
        hmm_probs.append(hmm_wp)
        odds_probs.append(odds_wp)

    hmm_probs = np.array(hmm_probs)
    odds_probs = np.array(odds_probs)

    # Set boundaries
    set_boundaries = []
    if "SetNo" in sample_pts.columns:
        sets = sample_pts["SetNo"].values
        for i in range(1, len(sets)):
            if sets[i] != sets[i - 1]:
                set_boundaries.append(i)

    p1_name = match_row.get("player1", "Player 1")
    p2_name = match_row.get("player2", "Player 2")
    winner_name = p1_name if int(match_row["winner"]) == 1 else p2_name

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hmm_probs, linewidth=1.2, color="#2563eb", label="HMM + Odds (live)")
    ax.plot(odds_probs, linewidth=1.0, color="#94a3b8", alpha=0.7,
            linestyle="--", label="Odds only (static baseline)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(implied_prob, color="#f59e0b", linestyle="-.", linewidth=0.8,
               alpha=0.7, label=f"Pre-match odds ({implied_prob:.0%})")
    for b in set_boundaries:
        ax.axvline(b, color="#ef4444", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.fill_between(range(len(hmm_probs)), hmm_probs, 0.5, alpha=0.1, color="#2563eb")
    ax.set_xlabel("Point Number")
    ax.set_ylabel(f"P({p1_name} wins match)")
    ax.set_title(
        f"Live Win Probability: {p1_name} vs {p2_name}\n"
        f"Winner: {winner_name} | {match_row.get('slam', '')} "
        f"{match_row.get('year', '')} | Pre-match: {p1_name} {implied_prob:.0%}"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "win_prob_evolution.png"), dpi=150)
    plt.close(fig)
    print(f"Saved win_prob_evolution.png ({sample_id})")


def plot_hmm_vs_odds(eval_df):
    """Compare HMM+odds accuracy vs odds-only, with calibration."""
    df = eval_df.copy()
    df["p1_won"] = (df["actual_winner"] == 1).astype(int)

    hmm_acc = df["hmm_correct"].mean()
    odds_acc = df["odds_correct"].mean()
    baseline_acc = max(df["p1_won"].mean(), 1 - df["p1_won"].mean())

    # Calibration: bin by implied prob
    n_bins = 10
    df["prob_bin"] = pd.cut(df["implied_p1_prob"], bins=n_bins)
    cal = df.groupby("prob_bin", observed=True).agg(
        mean_predicted=("implied_p1_prob", "mean"),
        mean_actual=("p1_won", "mean"),
        count=("p1_won", "count"),
    ).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: calibration of market odds
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.scatter(cal["mean_predicted"], cal["mean_actual"], s=cal["count"] * 3,
               color="#f59e0b", zorder=5, edgecolors="white", linewidth=0.5)
    ax.plot(cal["mean_predicted"], cal["mean_actual"], color="#f59e0b",
            linewidth=1.2, label="Market odds calibration")
    ax.set_xlabel("Market Implied P(Player 1 Wins)")
    ax.set_ylabel("Actual P(Player 1 Wins)")
    ax.set_title("Market Odds Calibration")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Right: accuracy comparison
    ax = axes[1]
    labels = ["50/50\n(majority)", "Market\nOdds Only", "HMM +\nOdds"]
    values = [baseline_acc * 100, odds_acc * 100, hmm_acc * 100]
    colors = ["#94a3b8", "#f59e0b", "#2563eb"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=11, padding=3)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Match Winner Prediction Accuracy")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "hmm_vs_odds.png"), dpi=150)
    plt.close(fig)
    print(f"Saved hmm_vs_odds.png (odds={odds_acc:.1%}, HMM+odds={hmm_acc:.1%})")


def plot_model_states(model):
    """Visualize the HMM's learned parameters: emissions, transitions, start probs."""
    params = get_model_params(model)
    n_states = params["emissionprob"].shape[0]
    state_labels = [f"State {i}" for i in range(n_states)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) Emission probabilities
    ax = axes[0]
    x = np.arange(N_CATEGORIES)
    width = 0.8 / n_states
    colors = plt.cm.Set2(np.linspace(0, 0.6, n_states))
    for s in range(n_states):
        ax.bar(x + s * width, params["emissionprob"][s], width,
               label=state_labels[s], color=colors[s], edgecolor="white")
    ax.set_xticks(x + width * (n_states - 1) / 2)
    ax.set_xticklabels(OBS_LABELS, fontsize=9)
    ax.set_ylabel("Probability")
    ax.set_title("Emission Probabilities")
    ax.legend(fontsize=8)

    # 2) Transition matrix
    ax = axes[1]
    im = ax.imshow(params["transmat"], cmap="Blues", vmin=0, vmax=1)
    for i in range(n_states):
        for j in range(n_states):
            v = params["transmat"][i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color=color)
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(state_labels, fontsize=9)
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title("Transition Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3) Start probabilities
    ax = axes[2]
    bars = ax.bar(state_labels, params["startprob"], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", fontsize=10, padding=3)
    ax.set_ylabel("Probability")
    ax.set_title("Initial State Distribution")
    ax.set_ylim(0, 1.1)

    fig.suptitle("HMM Learned Parameters (3 Hidden States)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "model_states.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved model_states.png")


def plot_state_timeline(model, val_enc, val_matches):
    """Show hidden state assignments and momentum over a match."""
    valid = val_matches[
        val_matches["winner"].isin([1.0, 2.0])
        & val_matches["implied_p1_prob"].notna()
    ]
    match_lengths = val_enc.groupby("match_id").size()
    valid_lengths = match_lengths[match_lengths.index.isin(valid["match_id"])]
    target_len = valid_lengths.quantile(0.75)
    sample_id = (valid_lengths - target_len).abs().idxmin()
    sample_pts = val_enc[val_enc["match_id"] == sample_id]
    match_row = valid[valid["match_id"] == sample_id].iloc[0]

    slam = match_row.get("slam")

    X_m = sample_pts["observation"].values.reshape(-1, 1)
    _, states = decode(model, X_m, np.array([len(X_m)]))
    posteriors = predict_proba(model, X_m, np.array([len(X_m)]))

    # Compute momentum adjustment at each point (surface-aware)
    momentum = np.array([
        hmm_momentum_adjustment(model, posteriors, i, slam=slam)
        for i in range(len(X_m))
    ])

    n_states = model.n_components
    colors = plt.cm.Set2(np.linspace(0, 0.6, n_states))

    p1_name = match_row.get("player1", "Player 1")
    p2_name = match_row.get("player2", "Player 2")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2, 2]})

    # Top: Viterbi state
    ax = axes[0]
    for s in range(n_states):
        mask = states == s
        ax.scatter(np.where(mask)[0], np.ones(mask.sum()) * 0.5,
                   c=[colors[s]], s=8, marker="s", label=f"State {s}")
    ax.set_yticks([])
    ax.set_title(f"Hidden State Timeline: {p1_name} vs {p2_name} | "
                 f"{match_row.get('slam', '')} {match_row.get('year', '')}")
    ax.legend(loc="upper right", fontsize=8, ncol=n_states)

    # Middle: posterior probabilities stacked
    ax = axes[1]
    bottom = np.zeros(len(posteriors))
    for s in range(n_states):
        ax.fill_between(range(len(posteriors)), bottom, bottom + posteriors[:, s],
                        color=colors[s], alpha=0.7, label=f"State {s}")
        bottom += posteriors[:, s]
    ax.set_ylabel("P(State)")
    ax.set_title("Posterior State Probabilities")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=n_states)

    # Bottom: momentum adjustment
    ax = axes[2]
    ax.fill_between(range(len(momentum)), momentum, 0,
                    where=momentum >= 0, color="#22c55e", alpha=0.5, label="Hot (server)")
    ax.fill_between(range(len(momentum)), momentum, 0,
                    where=momentum < 0, color="#ef4444", alpha=0.5, label="Cold (server)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Point Number")
    ax.set_ylabel("Momentum \u0394")
    ax.set_title("HMM Momentum Adjustment (vs Global Average)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "state_timeline.png"), dpi=150)
    plt.close(fig)
    print(f"Saved state_timeline.png ({sample_id})")


def plot_odds_distribution(eval_df):
    """Histogram of market-implied probs and HMM-adjusted probs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(eval_df["implied_p1_prob"], bins=30, color="#f59e0b", alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="#ef4444", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Market Implied P(P1 Wins)")
    ax.set_ylabel("Number of Matches")
    ax.set_title("Pre-Match Odds Distribution")

    ax = axes[1]
    ax.hist(eval_df["hmm_prob"], bins=30, color="#2563eb", alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="#ef4444", linestyle="--", linewidth=1.2)
    ax.set_xlabel("HMM + Odds P(P1 Wins)")
    ax.set_ylabel("Number of Matches")
    ax.set_title("HMM-Adjusted Probability Distribution")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "probability_distributions.png"), dpi=150)
    plt.close(fig)
    print("Saved probability_distributions.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tennis HMM Prediction Model")
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retraining the model even if a cached version exists",
    )
    args = parser.parse_args()

    # --- Load data ---
    print("Loading match & point data...")
    matches, points = load_singles_data()

    print("Loading odds & merging...")
    matches = merge_odds_with_matches(matches)
    has_odds = matches["implied_p1_prob"].notna().sum()
    print(f"  Odds joined: {has_odds}/{len(matches)} matches")

    # Filter to men's only (ATP odds data)
    matches["_num"] = matches["match_id"].str.extract(r"-(\d+)$").astype(float)
    matches = matches[matches["_num"] < 2000].drop(columns=["_num"]).reset_index(drop=True)
    points_ids = set(matches["match_id"])
    points = points[points["match_id"].isin(points_ids)].reset_index(drop=True)
    print(f"  Men's singles: {len(matches)} matches")

    train_matches, val_matches, train_points, val_points = train_val_split(matches, points)
    print(f"  Train: {len(train_matches)} | Val: {len(val_matches)}")

    # --- Encode ---
    print("Encoding observations...")
    train_enc = encode_observations(train_points)
    val_enc = encode_observations(val_points)
    X_train, len_train = build_sequences(train_enc)
    print(f"  Train points: {X_train.shape[0]} | Val points: {len(val_enc)}")

    # --- Model ---
    hmm = None
    if not args.retrain:
        hmm = load_cached_model()

    if hmm is None:
        print("Training HMM (3 hidden states)...")
        hmm = create_model()
        train(hmm, X_train, len_train)
        save_model(hmm)

    # --- Log-likelihood ---
    X_val, len_val = build_sequences(val_enc)
    train_ll = score(hmm, X_train, len_train)
    val_ll = score(hmm, X_val, len_val)
    print(f"  Train LL: {train_ll:.1f} ({train_ll / X_train.shape[0]:.4f}/pt)")
    print(f"  Val   LL: {val_ll:.1f} ({val_ll / X_val.shape[0]:.4f}/pt)")

    # --- Validation accuracy ---
    print("Evaluating on validation set...")
    eval_df = evaluate(hmm, val_matches, val_points)
    odds_acc = eval_df["odds_correct"].mean()
    hmm_acc = eval_df["hmm_correct"].mean()
    print(f"  Matches evaluated: {len(eval_df)}")
    print(f"  Odds-only accuracy:  {odds_acc:.1%}")
    print(f"  HMM+Odds accuracy:  {hmm_acc:.1%}")

    # --- Figures ---
    print("Generating figures...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_model_states(hmm)
    plot_hmm_vs_odds(eval_df)
    plot_win_prob_evolution(hmm, val_enc, val_matches)
    plot_state_timeline(hmm, val_enc, val_matches)
    plot_odds_distribution(eval_df)

    print("Done.")


if __name__ == "__main__":
    main()
