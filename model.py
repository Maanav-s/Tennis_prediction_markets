import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------
# Each point is encoded from the *server's* perspective into a feature vector
# for GaussianHMM.  Features (all binary, from server's perspective):
#   0 — server_won      (1 if server won the point)
#   1 — was_ace          (1 if point was an ace)
#   2 — was_double_fault (1 if point was a double fault)
#   3 — is_second_serve  (1 if 2nd serve, 0 if 1st or unknown)
N_FEATURES = 4
FEATURE_NAMES = ["Server Won", "Ace", "Double Fault", "2nd Serve"]


def encode_observations(points: pd.DataFrame) -> pd.DataFrame:
    """Convert a points DataFrame into per-point feature observations.

    Filters out non-point rows (PointWinner == 0 / NaN) and adds feature
    columns from the server's perspective.

    Returns a copy with only actual points and new columns appended.
    """
    df = points[points["PointWinner"].isin([1, 2])].copy()

    is_p1_serving = df["PointServer"] == 1
    server_won = (df["PointWinner"] == df["PointServer"]).astype(int)
    server_ace = np.where(is_p1_serving, df["P1Ace"], df["P2Ace"]).astype(int)
    server_df = np.where(
        is_p1_serving, df["P1DoubleFault"], df["P2DoubleFault"]
    ).astype(int)
    is_second = (
        (df["ServeNumber"].values == 2).astype(int)
        if "ServeNumber" in df.columns
        else np.zeros(len(df), dtype=int)
    )

    df["feat_server_won"] = server_won.values
    df["feat_ace"] = server_ace
    df["feat_double_fault"] = server_df
    df["feat_second_serve"] = is_second

    # Keep observation column for backward compat (used by some viz)
    obs = np.full(len(df), 2, dtype=int)
    obs[server_won.values.astype(bool) & ~server_ace.astype(bool)] = 0
    obs[server_won.values.astype(bool) & server_ace.astype(bool)] = 1
    obs[~server_won.values.astype(bool) & server_df.astype(bool)] = 3
    df["observation"] = obs

    return df


def build_feature_matrix(encoded_points: pd.DataFrame) -> np.ndarray:
    """Extract the feature matrix from encoded points.

    Returns
    -------
    np.ndarray, shape (n_points, N_FEATURES)
    """
    return encoded_points[
        ["feat_server_won", "feat_ace", "feat_double_fault", "feat_second_serve"]
    ].values.astype(np.float64)


def build_sequences(encoded_points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Group encoded points into per-match sequences for hmmlearn.

    Returns
    -------
    X : np.ndarray, shape (total_points, N_FEATURES)
        Concatenated feature vectors.
    lengths : np.ndarray, shape (n_matches,)
        Number of points in each match sequence.
    """
    grouped = encoded_points.groupby("match_id", sort=False)
    sequences = []
    lengths = []
    for _, group in grouped:
        feat = build_feature_matrix(group)
        sequences.append(feat)
        lengths.append(len(feat))

    X = np.vstack(sequences)
    lengths = np.array(lengths, dtype=int)
    return X, lengths


# ---------------------------------------------------------------------------
# Model creation / training / inference
# ---------------------------------------------------------------------------

def create_model(
    n_states: int = 3,
    n_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> GaussianHMM:
    """Instantiate an untrained GaussianHMM with diagonal covariance.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3).
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold for EM.
    random_state : int
        Seed for reproducibility.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )
    return model


def train(
    model: GaussianHMM,
    X: np.ndarray,
    lengths: np.ndarray,
) -> GaussianHMM:
    """Fit the HMM on training sequences via Baum-Welch (EM)."""
    model.fit(X, lengths)
    return model


def decode(
    model: GaussianHMM,
    X: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Viterbi decoding to find the most likely hidden state sequence."""
    log_prob, states = model.decode(X, lengths)
    return log_prob, states


def score(
    model: GaussianHMM,
    X: np.ndarray,
    lengths: np.ndarray,
) -> float:
    """Compute total log-likelihood of observation sequences under the model."""
    return model.score(X, lengths)


def predict_proba(
    model: GaussianHMM,
    X: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Compute posterior state probabilities for each point."""
    return model.predict_proba(X, lengths)


# ---------------------------------------------------------------------------
# Analytical score model
# ---------------------------------------------------------------------------
# Tennis scoring: point → game → set → match.  Each level is a finite Markov
# chain whose transition probabilities depend only on P(server wins point).
# We compute P(server wins game/set/match) recursively with memoisation.
# ---------------------------------------------------------------------------

from functools import lru_cache


def p_server_wins_point(
    model: GaussianHMM,
    posteriors: np.ndarray,
    point_idx: int,
) -> float:
    """Server win probability at a specific point, given HMM posteriors.

    For GaussianHMM, feature 0 is ``server_won``.  The mean of this feature
    per state approximates P(server wins | state).  We weight by the
    posterior state distribution.
    """
    state_probs = posteriors[point_idx]       # shape (n_states,)
    server_win_means = model.means_[:, 0]     # shape (n_states,)
    return float(np.clip(state_probs @ server_win_means, 0.0, 1.0))


def estimate_serve_return_probs(
    model: GaussianHMM,
    encoded_points: pd.DataFrame,
    posteriors: np.ndarray,
    player: int,
) -> tuple[float, float]:
    """Estimate a player's serve and return win probabilities over a match.

    Averages the HMM's point-level P(server wins) separately across points
    where ``player`` is serving vs returning.

    Parameters
    ----------
    model : GaussianHMM
        A fitted model.
    encoded_points : pd.DataFrame
        Encoded points for a single match (from ``encode_observations``).
        Must have a ``PointServer`` column.
    posteriors : np.ndarray, shape (n_points, n_states)
        Posterior state probabilities for this match (from ``predict_proba``).
    player : int
        Which player (1 or 2) to compute probabilities for.

    Returns
    -------
    p_serve : float
        P(player wins a point on their own serve).
    p_return : float
        P(player wins a point when returning).
    """
    server_win_means = model.means_[:, 0]
    # P(server wins point) at each point, weighted by posterior
    p_srv_win = np.clip(posteriors @ server_win_means, 0.0, 1.0)

    is_serving = encoded_points["PointServer"].values == player
    is_returning = np.logical_not(is_serving)

    if is_serving.sum() > 0:
        p_serve = float(p_srv_win[is_serving].mean())
    else:
        p_serve = float(p_srv_win.mean())

    if is_returning.sum() > 0:
        # When player is returning, P(player wins) = 1 - P(server wins)
        p_return = float((1 - p_srv_win[is_returning]).mean())
    else:
        p_return = float((1 - p_srv_win).mean())

    return p_serve, p_return


@lru_cache(maxsize=None)
def p_game(p: float, s_pts: int, r_pts: int) -> float:
    """P(server wins game) from score s_pts-r_pts (standard game, not tiebreak).

    Points are encoded as 0,1,2,3 = 0,15,30,40.
    Deuce is solved analytically: P(win from deuce) = p^2 / (p^2 + (1-p)^2).
    """
    if s_pts >= 4 and s_pts - r_pts >= 2:
        return 1.0
    if r_pts >= 4 and r_pts - s_pts >= 2:
        return 0.0
    if s_pts >= 3 and r_pts >= 3:
        # Closed-form from deuce: P = p^2 / (p^2 + q^2) where q = 1-p
        q = 1 - p
        p_from_deuce = p * p / (p * p + q * q)
        lead = s_pts - r_pts
        if lead == 0:
            return p_from_deuce
        if lead == 1:  # ad server
            return p + q * p_from_deuce
        if lead == -1:  # ad returner
            return p * p_from_deuce
        return 1.0 if lead >= 2 else 0.0
    return p * p_game(p, s_pts + 1, r_pts) + (1 - p) * p_game(p, s_pts, r_pts + 1)


@lru_cache(maxsize=None)
def p_tiebreak(p_s: float, p_r: float, s_pts: int, r_pts: int, server_serving: bool) -> float:
    """P(server-of-first-game wins tiebreak) from s_pts-r_pts.

    Service alternates every 2 points (after the first), so we track who is
    currently serving via ``server_serving``.  ``p_s`` is P(point) when the
    set-server serves, ``p_r`` when the set-returner serves (and the
    set-server is returning).
    """
    if s_pts >= 7 and s_pts - r_pts >= 2:
        return 1.0
    if r_pts >= 7 and r_pts - s_pts >= 2:
        return 0.0
    if s_pts >= 6 and r_pts >= 6:
        lead = s_pts - r_pts
        if lead >= 2:
            return 1.0
        if lead <= -2:
            return 0.0
        # At 6-6+, service alternates every 2 points.  The next two points
        # are served by the same player, then it switches.  Solve the
        # two-point "mini-deuce" analytically to avoid infinite recursion.
        #
        # Determine who serves the next point at this score.
        total = s_pts + r_pts
        if total == 0:
            next_server = server_serving
        else:
            next_server = ((total - 1) // 2) % 2 == 0
            if not server_serving:
                next_server = not next_server
        pa = p_s if next_server else p_r        # P(set-server wins this pt)
        pb = p_s if not next_server else p_r    # P(set-server wins next pt)
        # P(win two-point block) and P(lose two-point block)
        p_ww = pa * pb
        p_ll = (1 - pa) * (1 - pb)
        # P(return to deuce) = 1 - p_ww - p_ll
        # P(win from deuce) = p_ww / (p_ww + p_ll)
        if lead == 0:
            return p_ww / (p_ww + p_ll)
        if lead == 1:  # set-server ahead by 1
            return pa + (1 - pa) * (p_ww / (p_ww + p_ll))
        # lead == -1: set-server behind by 1
        return pa * (p_ww / (p_ww + p_ll))

    total = s_pts + r_pts
    # After 1st point, service switches every 2 points
    if total == 0:
        next_server = server_serving
    else:
        next_server = ((total - 1) // 2) % 2 == 0
        if not server_serving:
            next_server = not next_server

    pw = p_s if next_server else p_r
    return pw * p_tiebreak(p_s, p_r, s_pts + 1, r_pts, server_serving) + \
           (1 - pw) * p_tiebreak(p_s, p_r, s_pts, r_pts + 1, server_serving)


@lru_cache(maxsize=None)
def p_set(p_s: float, p_r: float, s_games: int, r_games: int, server_serving: bool) -> float:
    """P(P1 wins set) from s_games-r_games.

    ``p_s`` = P(P1 wins a point on P1's serve).
    ``p_r`` = P(P1 wins a point when returning (i.e. opponent is serving)).
    ``server_serving`` = True if P1 is currently serving this game.
    """
    if s_games >= 6 and s_games - r_games >= 2:
        return 1.0
    if r_games >= 6 and r_games - s_games >= 2:
        return 0.0
    if s_games == 6 and r_games == 6:
        return p_tiebreak(p_s, p_r, 0, 0, server_serving)
    # Current game win probability
    if server_serving:
        pg = p_game(p_s, 0, 0)  # P1 serving, wins point with p_s
    else:
        # Opponent serving, wins point with (1 - p_r); P1 wins game
        pg = 1 - p_game(1 - p_r, 0, 0)
    # If P1 wins this game
    return pg * p_set(p_s, p_r, s_games + 1, r_games, not server_serving) + \
           (1 - pg) * p_set(p_s, p_r, s_games, r_games + 1, not server_serving)


@lru_cache(maxsize=None)
def p_match(
    p_s: float,
    p_r: float,
    s_sets: int,
    r_sets: int,
    s_games: int,
    r_games: int,
    server_serving: bool,
    best_of: int = 3,
) -> float:
    """P(server-of-first-game wins match) from the current score.

    Parameters
    ----------
    p_s : float
        P(server wins a point on their serve).
    p_r : float
        P(server wins a point when returning) = 1 - P(opponent wins on serve).
    s_sets, r_sets : int
        Sets won by server / returner.
    s_games, r_games : int
        Games in the current set.
    server_serving : bool
        True if the match-server is currently serving.
    best_of : int
        3 (WTA / most ATP) or 5 (Grand Slam men's).
    """
    sets_to_win = (best_of + 1) // 2
    if s_sets >= sets_to_win:
        return 1.0
    if r_sets >= sets_to_win:
        return 0.0
    # Probability server wins the current set from current game score
    ps = p_set(p_s, p_r, s_games, r_games, server_serving)
    return ps * p_match(p_s, p_r, s_sets + 1, r_sets, 0, 0, server_serving, best_of) + \
           (1 - ps) * p_match(p_s, p_r, s_sets, r_sets + 1, 0, 0, server_serving, best_of)


def match_win_probability(
    p_serve: float,
    p_return: float,
    score: dict | None = None,
    best_of: int = 3,
) -> float:
    """Calculate P(player wins match) from current score and point-win probs.

    Parameters
    ----------
    p_serve : float
        P(player wins a point on their own serve).
    p_return : float
        P(player wins a point when returning).
    score : dict, optional
        Current match score with keys:
        - ``s_sets``, ``r_sets``: sets won (default 0-0)
        - ``s_games``, ``r_games``: games in current set (default 0-0)
        - ``server_serving``: bool, True if player is serving (default True)
    best_of : int
        3 or 5.
    """
    if score is None:
        score = {}
    s_sets = score.get("s_sets", 0)
    r_sets = score.get("r_sets", 0)
    s_games = score.get("s_games", 0)
    r_games = score.get("r_games", 0)
    server_serving = score.get("server_serving", True)

    # Round to limit cache explosion (5 decimal places → ~0.001% precision)
    p_s = round(p_serve, 5)
    p_r = round(p_return, 5)

    return p_match(p_s, p_r, s_sets, r_sets, s_games, r_games, server_serving, best_of)


# ---------------------------------------------------------------------------
# Odds-informed inference
# ---------------------------------------------------------------------------
# The HMM is player-agnostic — it learns a global P(server wins point).
# Pre-match odds give us a player-specific baseline.  We invert the
# analytical score model to recover implied serve/return rates, then use
# the HMM's point-level posterior to detect momentum shifts *relative* to
# the global average.
# ---------------------------------------------------------------------------

# Average P(server wins point) — global and per-surface.
# Used as the neutral reference for odds inversion and HMM momentum.
GLOBAL_SERVE_WIN_RATE = 0.6125

SURFACE_SERVE_WIN_RATE = {
    "wimbledon": 0.6598,    # grass — highest serve advantage
    "usopen": 0.6322,       # hard (DecoTurf)
    "ausopen": 0.6393,      # hard (GreenSet/Plexicushion)
    "frenchopen": 0.6193,   # clay — lowest serve advantage
}


def get_serve_win_rate(slam: str | None = None) -> float:
    """Return the surface-specific serve win rate, or the global average."""
    if slam is not None:
        return SURFACE_SERVE_WIN_RATE.get(slam, GLOBAL_SERVE_WIN_RATE)
    return GLOBAL_SERVE_WIN_RATE


def invert_match_odds(
    implied_p1_prob: float,
    best_of: int = 5,
    base_serve: float | None = None,
    slam: str | None = None,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Recover (p_serve, p_return) for P1 from pre-match implied probability.

    Assumes a single-parameter skill model where both players share the
    same serve advantage (``base_serve``), and P1's edge is a symmetric
    shift ``delta``:

        p1_serve  = base_serve + delta
        p1_return = (1 - base_serve) + delta

    Binary-searches for the ``delta`` that makes
    ``match_win_probability(p1_serve, p1_return, best_of)`` equal to
    ``implied_p1_prob``.

    Parameters
    ----------
    implied_p1_prob : float
        Market-implied probability that player 1 wins the match.
    best_of : int
        3 or 5.
    base_serve : float, optional
        Neutral serve-win rate.  If None, uses surface-specific rate.
    slam : str, optional
        Tournament name for surface-specific base rate.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    p_serve : float
        P(P1 wins a point on their serve).
    p_return : float
        P(P1 wins a point when returning).
    """
    if base_serve is None:
        base_serve = get_serve_win_rate(slam)

    # delta range: p_serve and p_return must stay in (0.01, 0.99)
    lo = max(-base_serve + 0.01, -(1 - base_serve) + 0.01)
    hi = min(1 - base_serve - 0.01, 1 - (1 - base_serve) - 0.01)

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p_s = base_serve + mid
        p_r = (1 - base_serve) + mid
        modeled = match_win_probability(p_s, p_r, best_of=best_of)
        if abs(modeled - implied_p1_prob) < tol:
            return p_s, p_r
        if modeled < implied_p1_prob:
            lo = mid
        else:
            hi = mid

    p_s = base_serve + (lo + hi) / 2
    p_r = (1 - base_serve) + (lo + hi) / 2
    return p_s, p_r


def hmm_momentum_adjustment(
    model: GaussianHMM,
    posteriors: np.ndarray,
    point_idx: int,
    slam: str | None = None,
) -> float:
    """Compute the HMM momentum adjustment at a specific point.

    Returns the difference between the HMM's posterior-weighted
    P(server wins) and the surface-specific average.  Positive means the
    server is performing above average ("hot"); negative means below ("cold").

    Parameters
    ----------
    model : GaussianHMM
        A fitted model.
    posteriors : np.ndarray, shape (n_points, n_states)
        Posterior state probabilities for this match.
    point_idx : int
        Which point to evaluate.
    slam : str, optional
        Tournament name (e.g. "wimbledon") for surface-specific reference.

    Returns
    -------
    delta : float
        Adjustment to add to baseline serve/return rates.
    """
    hmm_p = p_server_wins_point(model, posteriors, point_idx)
    return hmm_p - get_serve_win_rate(slam)


def live_win_probability(
    model: GaussianHMM,
    posteriors: np.ndarray,
    point_idx: int,
    baseline_serve: float,
    baseline_return: float,
    score: dict,
    best_of: int = 5,
    slam: str | None = None,
) -> float:
    """Compute live P(P1 wins match) combining odds baseline + HMM momentum.

    At each point the HMM detects whether the server is performing above
    or below the surface average.  That momentum shift is applied
    symmetrically to P1's baseline serve and return rates, then the
    analytical score model converts to a match-win probability at the
    current score.

    Parameters
    ----------
    model : GaussianHMM
        A fitted model.
    posteriors : np.ndarray, shape (n_points, n_states)
        Posterior state probabilities for this match.
    point_idx : int
        Current point index in the posteriors array.
    baseline_serve : float
        P1's odds-implied serve-win rate (from ``invert_match_odds``).
    baseline_return : float
        P1's odds-implied return-win rate (from ``invert_match_odds``).
    score : dict
        Current match score (see ``match_win_probability``).
    best_of : int
        3 or 5.
    slam : str, optional
        Tournament name for surface-specific momentum reference.

    Returns
    -------
    float
        P(P1 wins match) incorporating momentum and current score.
    """
    delta = hmm_momentum_adjustment(model, posteriors, point_idx, slam=slam)

    # Apply momentum: if P1 is serving and "hot", their serve rate goes up;
    # if P1 is returning and server is "hot", P1's return rate goes down.
    # We apply the adjustment from P1's perspective based on who is serving.
    is_p1_serving = score.get("server_serving", True)

    if is_p1_serving:
        # Server is hot/cold → P1's serve ability shifts
        adj_serve = np.clip(baseline_serve + delta, 0.01, 0.99)
        adj_return = baseline_return
    else:
        # Server (P2) is hot/cold → P1's return ability shifts inversely
        adj_serve = baseline_serve
        adj_return = np.clip(baseline_return - delta, 0.01, 0.99)

    return match_win_probability(
        float(adj_serve), float(adj_return), score=score, best_of=best_of
    )


def get_model_params(model: GaussianHMM) -> dict:
    """Extract learned model parameters as a plain dict.

    Keys
    ----
    startprob : np.ndarray, shape (n_states,)
        Initial state distribution.
    transmat : np.ndarray, shape (n_states, n_states)
        State transition probabilities.
    means : np.ndarray, shape (n_states, N_FEATURES)
        Per-state feature means (server_won, ace, double_fault, 2nd_serve).
    covars : np.ndarray, shape (n_states, N_FEATURES)
        Per-state feature variances (diagonal covariance).
    """
    return {
        "startprob": model.startprob_,
        "transmat": model.transmat_,
        "means": model.means_,
        "covars": model.covars_,
    }
