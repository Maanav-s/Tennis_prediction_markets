import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM

# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------
# Each point is encoded from the *server's* perspective into one of four
# discrete categories so the model is player-agnostic:
#   0 — server wins the point (normal rally)
#   1 — server wins via ace
#   2 — server loses the point (normal rally)
#   3 — server loses via double fault
N_CATEGORIES = 4

# Which observation indices represent a server win (used for P(server wins)).
_SERVER_WIN_OBS = [0, 1]


def encode_observations(points: pd.DataFrame) -> pd.DataFrame:
    """Convert a points DataFrame into per-point categorical observations.

    Filters out non-point rows (PointWinner == 0 / NaN) and adds an integer
    ``observation`` column encoded from the server's perspective.

    Returns a copy with only actual points and the new column appended.
    """
    df = points[points["PointWinner"].isin([1, 2])].copy()

    is_p1_serving = df["PointServer"] == 1
    server_won = df["PointWinner"] == df["PointServer"]
    server_ace = np.where(is_p1_serving, df["P1Ace"], df["P2Ace"]).astype(bool)
    server_df = np.where(
        is_p1_serving, df["P1DoubleFault"], df["P2DoubleFault"]
    ).astype(bool)

    obs = np.full(len(df), 2, dtype=int)  # default: server loses (normal)
    obs[server_won & ~server_ace] = 0     # server wins (normal)
    obs[server_won & server_ace] = 1      # server wins via ace
    obs[~server_won & server_df] = 3      # server loses via double fault

    df["observation"] = obs
    return df


def build_sequences(encoded_points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Group encoded points into per-match sequences for hmmlearn.

    Parameters
    ----------
    encoded_points : pd.DataFrame
        Output of ``encode_observations`` (must have ``match_id`` and
        ``observation`` columns, with rows sorted by point order within
        each match).

    Returns
    -------
    X : np.ndarray, shape (total_points, 1)
        Concatenated observation sequences (column vector as required by
        hmmlearn).
    lengths : np.ndarray, shape (n_matches,)
        Number of points in each match sequence.
    """
    grouped = encoded_points.groupby("match_id", sort=False)
    sequences = []
    lengths = []
    for _, group in grouped:
        obs = group["observation"].values
        sequences.append(obs)
        lengths.append(len(obs))

    X = np.concatenate(sequences).reshape(-1, 1)
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
) -> CategoricalHMM:
    """Instantiate an untrained CategoricalHMM.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 2: e.g. "cold" vs "hot").
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold for EM.
    random_state : int
        Seed for reproducibility.
    """
    model = CategoricalHMM(
        n_components=n_states,
        n_features=N_CATEGORIES,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )
    return model


def train(
    model: CategoricalHMM,
    X: np.ndarray,
    lengths: np.ndarray,
) -> CategoricalHMM:
    """Fit the HMM on training sequences via Baum-Welch (EM).

    Parameters
    ----------
    model : CategoricalHMM
        An untrained model from ``create_model``.
    X : np.ndarray
        Concatenated observations from ``build_sequences``.
    lengths : np.ndarray
        Per-match sequence lengths from ``build_sequences``.

    Returns
    -------
    The fitted model (same object, returned for convenience).
    """
    model.fit(X, lengths)
    return model


def decode(
    model: CategoricalHMM,
    X: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Viterbi decoding to find the most likely hidden state sequence.

    Returns
    -------
    log_probs : np.ndarray, shape (n_matches,)
        Log-probability of each match's observation sequence under the model.
    state_sequences : np.ndarray, shape (total_points,)
        Most likely hidden state at each point (concatenated across matches).
    """
    log_prob, states = model.decode(X, lengths)
    return log_prob, states


def score(
    model: CategoricalHMM,
    X: np.ndarray,
    lengths: np.ndarray,
) -> float:
    """Compute total log-likelihood of observation sequences under the model."""
    return model.score(X, lengths)


def predict_proba(
    model: CategoricalHMM,
    X: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Compute posterior state probabilities for each point.

    Returns
    -------
    posteriors : np.ndarray, shape (total_points, n_states)
        posteriors[t, s] = P(hidden_state=s | observations) at point t.
    """
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
    model: CategoricalHMM,
    posteriors: np.ndarray,
    point_idx: int,
) -> float:
    """Server win probability at a specific point, given HMM posteriors.

    Combines emission probabilities (obs 0 = server win, obs 1 = ace) weighted
    by the posterior state distribution at ``point_idx``.
    """
    state_probs = posteriors[point_idx]  # shape (n_states,)
    emission = model.emissionprob_       # shape (n_states, N_CATEGORIES)
    # P(server wins) = Σ_s P(state=s) * Σ_obs_in_server_wins P(obs|s)
    server_win_probs = emission[:, _SERVER_WIN_OBS].sum(axis=1)
    return float(state_probs @ server_win_probs)


def estimate_serve_return_probs(
    model: CategoricalHMM,
    encoded_points: pd.DataFrame,
    posteriors: np.ndarray,
    player: int,
) -> tuple[float, float]:
    """Estimate a player's serve and return win probabilities over a match.

    Averages the HMM's point-level P(server wins) separately across points
    where ``player`` is serving vs returning.

    Parameters
    ----------
    model : CategoricalHMM
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
    emission = model.emissionprob_
    # P(server wins point) at each point, weighted by posterior
    server_win_probs = emission[:, _SERVER_WIN_OBS].sum(axis=1)
    p_srv_win = posteriors @ server_win_probs

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


# 1st vs 2nd serve dynamics (from data analysis).
FIRST_SERVE_WIN_RATE = 0.691
SECOND_SERVE_WIN_RATE = 0.535
FIRST_SERVE_FREQUENCY = 0.64   # fraction of points that are 1st serves


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
    model: CategoricalHMM,
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
    model : CategoricalHMM
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


def adjust_baseline_for_serve_mix(
    baseline_serve: float,
    baseline_return: float,
    first_serve_pct: float,
) -> tuple[float, float]:
    """Adjust baseline serve/return rates for observed 1st-serve frequency.

    If a player is landing more 1st serves than average, their effective
    serve-win rate is higher.  This computes the delta between the expected
    serve-win rate at the average 1st-serve frequency and at the observed
    frequency, and applies it to the baseline serve rate.

    The return rate is not adjusted (the opponent's serve mix is independent).
    """
    expected = (FIRST_SERVE_FREQUENCY * FIRST_SERVE_WIN_RATE
                + (1 - FIRST_SERVE_FREQUENCY) * SECOND_SERVE_WIN_RATE)
    actual = (first_serve_pct * FIRST_SERVE_WIN_RATE
              + (1 - first_serve_pct) * SECOND_SERVE_WIN_RATE)
    delta = actual - expected

    adj_serve = float(np.clip(baseline_serve + delta, 0.01, 0.99))
    return adj_serve, baseline_return


def recalibrate_baseline(
    market_prob: float,
    best_of: int = 3,
    slam: str | None = None,
) -> tuple[float, float]:
    """Re-anchor baseline serve/return rates to the current market price.

    Instead of using pre-match odds for the entire match, periodically
    re-invert the live market probability to get updated baseline rates.
    This lets the HMM focus on detecting *short-term momentum shifts*
    relative to what the market currently believes, rather than drifting
    from stale pre-match odds.

    Parameters
    ----------
    market_prob : float
        Current market-implied P(player wins match).
    best_of : int
        3 or 5.
    slam : str, optional
        Tournament for surface-specific base rate.

    Returns
    -------
    p_serve, p_return : float
        Re-calibrated baseline rates.
    """
    return invert_match_odds(market_prob, best_of=best_of, slam=slam)


def live_win_probability(
    model: CategoricalHMM,
    posteriors: np.ndarray,
    point_idx: int,
    baseline_serve: float,
    baseline_return: float,
    score: dict,
    best_of: int = 5,
    slam: str | None = None,
    first_serve_pct: float | None = None,
    market_prob: float | None = None,
    confidence_halflife: int = 30,
) -> float:
    """Compute live P(P1 wins match) combining odds baseline + HMM momentum.

    At each point the HMM detects whether the server is performing above
    or below the surface average.  That momentum shift is scaled by a
    confidence weight that grows as the HMM sees more points, then applied
    to P1's baseline serve and return rates.

    Parameters
    ----------
    model : CategoricalHMM
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
    first_serve_pct : float, optional
        Observed 1st-serve percentage so far.  When provided, adjusts the
        baseline serve rate for the player's serve mix.
    market_prob : float, optional
        Current market price for this player.  When provided, re-anchors
        the baseline to the market's current view so the HMM only detects
        *deviations* from the market rather than maintaining an independent
        estimate.
    confidence_halflife : int
        Number of points after which the HMM confidence reaches 50%.
        At point 0 the model fully trusts the market; the HMM's influence
        grows as 1 - 0.5^(n_points / halflife).  Default 30 (~1 set).

    Returns
    -------
    float
        P(P1 wins match) incorporating momentum and current score.
    """
    # Re-anchor baseline to current market price if available
    if market_prob is not None:
        market_prob = float(np.clip(market_prob, 0.02, 0.98))
        baseline_serve, baseline_return = recalibrate_baseline(
            market_prob, best_of=best_of, slam=slam,
        )

    # Adjust baseline for serve mix if available
    if first_serve_pct is not None:
        baseline_serve, baseline_return = adjust_baseline_for_serve_mix(
            baseline_serve, baseline_return, first_serve_pct,
        )

    delta = hmm_momentum_adjustment(model, posteriors, point_idx, slam=slam)

    # Apply momentum: if P1 is serving and "hot", their serve rate goes up;
    # if P1 is returning and server is "hot", P1's return rate goes down.
    is_p1_serving = score.get("server_serving", True)

    if is_p1_serving:
        adj_serve = np.clip(baseline_serve + delta, 0.01, 0.99)
        adj_return = baseline_return
    else:
        adj_serve = baseline_serve
        adj_return = np.clip(baseline_return - delta, 0.01, 0.99)

    hmm_result = match_win_probability(
        float(adj_serve), float(adj_return), score=score, best_of=best_of
    )

    # Confidence decay: blend between raw market price and HMM-adjusted
    # result.  Early in the match, trust the market more; as points
    # accumulate, trust the HMM's momentum signal more.
    if market_prob is not None:
        n_points = point_idx + 1
        confidence = 1.0 - 0.5 ** (n_points / confidence_halflife)
        return confidence * hmm_result + (1 - confidence) * market_prob

    return hmm_result


def get_model_params(model: CategoricalHMM) -> dict:
    """Extract learned model parameters as a plain dict.

    Keys
    ----
    startprob : np.ndarray, shape (n_states,)
        Initial state distribution.
    transmat : np.ndarray, shape (n_states, n_states)
        State transition probabilities.
    emissionprob : np.ndarray, shape (n_states, N_CATEGORIES)
        Per-state observation probabilities (columns ordered as
        server-win-normal, ace, server-loss-normal, double-fault).
    """
    return {
        "startprob": model.startprob_,
        "transmat": model.transmat_,
        "emissionprob": model.emissionprob_,
    }
