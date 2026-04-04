import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataservice import load_singles_data, train_val_split
from model import (
    build_sequences,
    create_model,
    decode,
    encode_observations,
    estimate_serve_return_probs,
    get_model_params,
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


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model, val_matches, val_points):
    """Predict match winners on validation set and return accuracy metrics."""
    val_enc = encode_observations(val_points)

    # Only evaluate completed matches with a known winner
    valid_matches = val_matches[
        val_matches["winner"].isin([1.0, 2.0])
    ].copy()
    valid_ids = set(valid_matches["match_id"])
    val_enc_valid = val_enc[val_enc["match_id"].isin(valid_ids)]

    results = []
    grouped = val_enc_valid.groupby("match_id", sort=False)
    for match_id, group in grouped:
        X_m = group["observation"].values.reshape(-1, 1)
        posteriors = predict_proba(model, X_m, np.array([len(X_m)]))

        p_serve_p1, p_return_p1 = estimate_serve_return_probs(
            model, group, posteriors, player=1
        )
        # Determine best_of from event name
        match_row = valid_matches[valid_matches["match_id"] == match_id].iloc[0]
        event = str(match_row.get("event_name", ""))
        is_mens = "Men" in event or event == "event_MS"
        best_of = 5 if is_mens else 3

        p1_win = match_win_probability(p_serve_p1, p_return_p1, best_of=best_of)
        actual_winner = int(match_row["winner"])

        results.append({
            "match_id": match_id,
            "p1_win_prob": p1_win,
            "predicted_winner": 1 if p1_win > 0.5 else 2,
            "actual_winner": actual_winner,
            "best_of": best_of,
        })

    df = pd.DataFrame(results)
    df["correct"] = df["predicted_winner"] == df["actual_winner"]
    return df


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_win_prob_evolution(model, val_enc, val_matches):
    """Plot P1 win probability evolving point-by-point for a sample match."""
    # Pick a long, completed match for an interesting plot
    valid = val_matches[val_matches["winner"].isin([1.0, 2.0])]
    match_lengths = val_enc.groupby("match_id").size()
    valid_lengths = match_lengths[match_lengths.index.isin(valid["match_id"])]
    # Pick the match closest to the 90th percentile in length
    target_len = valid_lengths.quantile(0.9)
    sample_id = (valid_lengths - target_len).abs().idxmin()
    sample_pts = val_enc[val_enc["match_id"] == sample_id]
    match_row = valid[valid["match_id"] == sample_id].iloc[0]

    event = str(match_row.get("event_name", ""))
    is_mens = "Men" in event or event == "event_MS"
    best_of = 5 if is_mens else 3

    X_m = sample_pts["observation"].values.reshape(-1, 1)
    posteriors = predict_proba(model, X_m, np.array([len(X_m)]))

    # Compute running win probability from P1's perspective at each point
    emission = model.emissionprob_
    p_srv_win = posteriors @ (emission[:, 0] + emission[:, 1])
    is_p1_serving = sample_pts["PointServer"].values == 1

    win_probs = []
    for i in range(len(sample_pts)):
        # Use expanding average up to this point for serve/return estimates
        mask_serve = is_p1_serving[:i + 1]
        mask_return = np.logical_not(mask_serve)
        p_s = float(p_srv_win[:i + 1][mask_serve].mean()) if mask_serve.sum() > 0 else 0.6
        p_r = float((1 - p_srv_win[:i + 1][mask_return]).mean()) if mask_return.sum() > 0 else 0.4
        wp = match_win_probability(p_s, p_r, best_of=best_of)
        win_probs.append(wp)

    win_probs = np.array(win_probs)

    # Identify set boundaries
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
    ax.plot(win_probs, linewidth=1.0, color="#2563eb", label=f"{p1_name} win prob")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="50/50")
    for b in set_boundaries:
        ax.axvline(b, color="#ef4444", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.fill_between(range(len(win_probs)), win_probs, 0.5, alpha=0.15, color="#2563eb")
    ax.set_xlabel("Point Number")
    ax.set_ylabel(f"P({p1_name} wins match)")
    ax.set_title(f"Win Probability Evolution: {p1_name} vs {p2_name}\n"
                 f"Winner: {winner_name} | {match_row.get('slam', '')} {match_row.get('year', '')}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "win_prob_evolution.png"), dpi=150)
    plt.close(fig)
    print(f"Saved win_prob_evolution.png ({sample_id})")


def plot_hmm_vs_baseline(eval_df):
    """Compare HMM accuracy vs 50/50 baseline across probability bins."""
    # Calibration plot: bin predicted P1 win prob, compare to actual win rate
    df = eval_df.copy()
    df["p1_won"] = (df["actual_winner"] == 1).astype(int)

    n_bins = 10
    df["prob_bin"] = pd.cut(df["p1_win_prob"], bins=n_bins)
    cal = df.groupby("prob_bin", observed=True).agg(
        mean_predicted=("p1_win_prob", "mean"),
        mean_actual=("p1_won", "mean"),
        count=("p1_won", "count"),
    ).dropna()

    hmm_acc = eval_df["correct"].mean()
    baseline_acc = max(
        (eval_df["actual_winner"] == 1).mean(),
        (eval_df["actual_winner"] == 2).mean(),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: calibration curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.scatter(cal["mean_predicted"], cal["mean_actual"], s=cal["count"] * 2,
               color="#2563eb", zorder=5, edgecolors="white", linewidth=0.5)
    ax.plot(cal["mean_predicted"], cal["mean_actual"], color="#2563eb",
            linewidth=1.2, label="HMM calibration")
    ax.set_xlabel("Predicted P(Player 1 Wins)")
    ax.set_ylabel("Actual P(Player 1 Wins)")
    ax.set_title("Calibration Curve")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Right: accuracy bar chart
    ax = axes[1]
    bars = ax.bar(
        ["50/50\n(majority class)", "HMM"],
        [baseline_acc * 100, hmm_acc * 100],
        color=["#94a3b8", "#2563eb"],
        width=0.5,
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.1f%%", fontsize=11, padding=3)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Match Winner Prediction Accuracy")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "hmm_vs_baseline.png"), dpi=150)
    plt.close(fig)
    print(f"Saved hmm_vs_baseline.png (HMM={hmm_acc:.1%}, baseline={baseline_acc:.1%})")


def plot_model_states(model):
    """Visualize the HMM's learned parameters: emissions, transitions, start probs."""
    params = get_model_params(model)
    n_states = params["emissionprob"].shape[0]
    state_labels = [f"State {i}" for i in range(n_states)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) Emission probabilities (grouped bar chart)
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

    # 2) Transition matrix (heatmap)
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

    # 3) Start probabilities (bar chart)
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
    """Show hidden state assignments over a match alongside point outcomes."""
    valid = val_matches[val_matches["winner"].isin([1.0, 2.0])]
    match_lengths = val_enc.groupby("match_id").size()
    valid_lengths = match_lengths[match_lengths.index.isin(valid["match_id"])]
    target_len = valid_lengths.quantile(0.75)
    sample_id = (valid_lengths - target_len).abs().idxmin()
    sample_pts = val_enc[val_enc["match_id"] == sample_id]
    match_row = valid[valid["match_id"] == sample_id].iloc[0]

    X_m = sample_pts["observation"].values.reshape(-1, 1)
    _, states = decode(model, X_m, np.array([len(X_m)]))
    posteriors = predict_proba(model, X_m, np.array([len(X_m)]))

    n_states = model.n_components
    colors = plt.cm.Set2(np.linspace(0, 0.6, n_states))

    p1_name = match_row.get("player1", "Player 1")
    p2_name = match_row.get("player2", "Player 2")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})

    # Top: Viterbi state assignment
    ax = axes[0]
    for s in range(n_states):
        mask = states == s
        ax.scatter(np.where(mask)[0], np.ones(mask.sum()) * 0.5,
                   c=[colors[s]], s=8, marker="s", label=f"State {s}")
    ax.set_yticks([])
    ax.set_title(f"Hidden State Timeline: {p1_name} vs {p2_name} | "
                 f"{match_row.get('slam', '')} {match_row.get('year', '')}")
    ax.legend(loc="upper right", fontsize=8, ncol=n_states)

    # Bottom: posterior probabilities stacked
    ax = axes[1]
    bottom = np.zeros(len(posteriors))
    for s in range(n_states):
        ax.fill_between(range(len(posteriors)), bottom, bottom + posteriors[:, s],
                        color=colors[s], alpha=0.7, label=f"State {s}")
        bottom += posteriors[:, s]
    ax.set_xlabel("Point Number")
    ax.set_ylabel("P(State)")
    ax.set_title("Posterior State Probabilities")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=n_states)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "state_timeline.png"), dpi=150)
    plt.close(fig)
    print(f"Saved state_timeline.png ({sample_id})")


def plot_win_prob_distribution(eval_df):
    """Histogram of estimated serve/return win probs across matches."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(eval_df["p1_win_prob"], bins=40, color="#2563eb", alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="#ef4444", linestyle="--", linewidth=1.2, label="50/50")
    ax.set_xlabel("P(Player 1 Wins Match)")
    ax.set_ylabel("Number of Matches")
    ax.set_title("Distribution of HMM Match Win Probabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "win_prob_distribution.png"), dpi=150)
    plt.close(fig)
    print("Saved win_prob_distribution.png")


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
    print("Loading data...")
    matches, points = load_singles_data()
    train_matches, val_matches, train_points, val_points = train_val_split(matches, points)
    print(f"  Train: {len(train_matches)} matches | Val: {len(val_matches)} matches")

    # --- Encode ---
    print("Encoding observations...")
    train_enc = encode_observations(train_points)
    val_enc = encode_observations(val_points)
    X_train, len_train = build_sequences(train_enc)
    print(f"  Train points: {X_train.shape[0]} | Val points: {len(val_enc)}")

    # --- Model ---
    hmm = None
    if not args.retrain:
        hmm = load_model()

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
    accuracy = eval_df["correct"].mean()
    print(f"  Match winner accuracy: {accuracy:.1%} ({eval_df['correct'].sum()}/{len(eval_df)})")

    # --- Figures ---
    print("Generating figures...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_model_states(hmm)
    plot_hmm_vs_baseline(eval_df)
    plot_win_prob_evolution(hmm, val_enc, val_matches)
    plot_state_timeline(hmm, val_enc, val_matches)
    plot_win_prob_distribution(eval_df)

    print("Done.")


if __name__ == "__main__":
    main()
