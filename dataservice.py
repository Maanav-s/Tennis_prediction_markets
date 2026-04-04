import glob
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SACKMANN_DIR = os.path.join(DATA_DIR, "Sackmann")
ODDS_DIR = os.path.join(DATA_DIR, "tennis-data-co-uk")

SLAM_TOURNAMENT_MAP = {
    "ausopen": "Australian Open",
    "frenchopen": "French Open",
    "wimbledon": "Wimbledon",
    "usopen": "US Open",
}


# ---------------------------------------------------------------------------
# Sackmann point-by-point data
# ---------------------------------------------------------------------------

def _singles_file_paths(suffix: str) -> list[str]:
    """Return sorted paths for all singles CSV files matching *-{suffix}.csv,
    excluding doubles and mixed files."""
    pattern = os.path.join(SACKMANN_DIR, f"*-{suffix}.csv")
    paths = sorted(glob.glob(pattern))
    return [p for p in paths if "doubles" not in p and "mixed" not in p]


def load_all_singles_matches() -> pd.DataFrame:
    """Load and concatenate all singles match CSVs into one DataFrame.

    Adds ``year`` and ``slam`` columns parsed from each filename when they are
    missing from the CSV itself (some newer files leave these blank).
    """
    frames = []
    for path in _singles_file_paths("matches"):
        df = pd.read_csv(path)
        # Parse year/slam from filename as fallback
        basename = os.path.basename(path)  # e.g. "2024-usopen-matches.csv"
        parts = basename.split("-")
        file_year = int(parts[0])
        file_slam = parts[1]

        if "year" in df.columns:
            df["year"] = df["year"].fillna(file_year).astype(int)
        else:
            df["year"] = file_year

        if "slam" in df.columns:
            df["slam"] = df["slam"].fillna(file_slam)
        else:
            df["slam"] = file_slam

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_all_singles_points() -> pd.DataFrame:
    """Load and concatenate all singles point-by-point CSVs.

    Every file's columns are kept (outer join), so columns only present in
    newer files will be NaN for older tournaments.
    """
    frames = []
    for path in _singles_file_paths("points"):
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# tennis-data.co.uk betting odds
# ---------------------------------------------------------------------------

def load_odds(years: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Load tennis-data.co.uk odds from .xls/.xlsx files.

    Parameters
    ----------
    years : tuple of int, optional
        Restrict to specific years.  Defaults to all available files.

    Returns
    -------
    DataFrame with columns from the raw files plus a normalised ``year`` column.
    Only Grand Slam rows are returned.
    """
    frames = []
    patterns = [
        os.path.join(ODDS_DIR, "*.xls"),
        os.path.join(ODDS_DIR, "*.xlsx"),
    ]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})

    for path in paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        try:
            file_year = int(basename)
        except ValueError:
            continue
        if years is not None and file_year not in years:
            continue

        df = pd.read_excel(path)
        df["year"] = file_year
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Keep only Grand Slam matches
    slam_names = set(SLAM_TOURNAMENT_MAP.values())
    combined = combined[combined["Tournament"].isin(slam_names)].reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# Name normalisation & join
# ---------------------------------------------------------------------------

# Manual overrides for names where the algorithmic conversion fails.
# Sackmann full name -> tennis-data.co.uk abbreviated name.
_NAME_OVERRIDES: dict[str, str] = {
    "Adrian Menendez Maceiras": "Menendez-Maceiras A.",
    "Albert Ramos Vinolas": "Ramos-Vinolas A.",
    "Andreas Haider Maurer": "Haider-Maurer A.",
    "Cedrik Marcel Stebe": "Stebe C.M.",
    "Chun Hsin Tseng": "Tseng C.H.",
    "Chun hsin Tseng": "Tseng C.H.",
    "Daniel Elahi Galan": "Galan D.",
    "Daniel Gimeno Traver": "Gimeno-Traver D.",
    "Ds.Schwartzman": "Schwartzman D.",
    "Edouard Roger Vasselin": "Roger-Vasselin E.",
    "Felix Auger Aliassime": "Auger-Aliassime F.",
    "Grigor DiMitrov": "Dimitrov G.",
    "Guillermo Garcia Lopez": "Garcia-Lopez G.",
    "Holger Vitus Nodskov Rune": "Rune H.",
    "James McGee": "Mcgee J.",
    "Jan Lennard Struff": "Struff J.L.",
    "Jo Wilfried Tsonga": "Tsonga J.W.",
    "John Patrick Smith": "Smith J.P.",
    "Jonathan Dasnieres De Veigy": "Dasnieres de Veigy J.",
    "Juan Carlos Ferrero": "Ferrero J.C.",
    "Juan Ignacio Chela": "Chela J.I.",
    "Juan Ignacio Londero": "Londero J.I.",
    "Juan Manuel Cerundolo": "Cerundolo J.M.",
    "Juan Martin Del Potro": "Del Potro J.M.",
    "Juan Martin del Potro": "Del Potro J.M.",
    "Juan Pablo Varillas": "Varillas J.P.",
    "Marc Andrea Huesler": "Huesler M.A.",
    "Paul Henri Mathieu": "Mathieu P.H.",
    "Pierre Hugues Herbert": "Herbert P.H.",
    "Ruben Ramirez Hidalgo": "Ramirez-Hidalgo R.",
    "Thai Son Kwiatkowski": "Kwiatkowski T.S.",
    "Tomas Martin Etcheverry": "Etcheverry T.",
    "Victor Estrella": "Estrella Burgos V.",
    "Yen Hsun Lu": "Lu Y.",
    "Yu Hsiou Hsu": "Hsu Y.",
    "D. Mu\udcf1oz de la Nava": "Munoz De La Nava D.",
    "Giovanni Mpetshi Perricard": "Mpetshi G.",
    "Inigo Cervantes Huegun": "Cervantes I.",
    "J. Del Potro": "Del Potro J.M.",
    "Marcelo Tomas Barrios Vera": "Barrios M.",
    "Tomas Barrios Vera": "Barrios M.",
}


def _sackmann_to_td_name(full_name: str) -> str:
    """Convert Sackmann full name to tennis-data.co.uk abbreviated format.

    Examples
    --------
    >>> _sackmann_to_td_name("Novak Djokovic")
    'Djokovic N.'
    >>> _sackmann_to_td_name("Roberto Carballes Baena")
    'Carballes Baena R.'
    >>> _sackmann_to_td_name("Alex de Minaur")
    'De Minaur A.'
    >>> _sackmann_to_td_name("Christopher O'Connell")
    "O Connell C."
    """
    if full_name in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[full_name]

    # tennis-data strips apostrophes
    name = full_name.replace("'", " ")
    parts = name.split()
    if len(parts) < 2:
        return full_name

    first_initial = parts[0][0] + "."
    surname = " ".join(parts[1:])
    # tennis-data capitalises the first letter of each surname part
    surname = " ".join(w.capitalize() if w.islower() else w for w in surname.split())
    return f"{surname} {first_initial}"


def _build_name_lookup(matches: pd.DataFrame, odds: pd.DataFrame) -> dict[str, str]:
    """Build a verified mapping from Sackmann names to tennis-data.co.uk names.

    Uses the conversion rule + manual overrides, then falls back to
    case-insensitive and surname-only matching for remaining names.
    """
    sackmann_names = set(matches["player1"].dropna()) | set(matches["player2"].dropna())
    td_names = set(odds["Winner"].dropna()) | set(odds["Loser"].dropna())
    td_names_lower = {n.lower(): n for n in td_names}

    lookup: dict[str, str] = {}
    unmatched: list[str] = []

    for name in sackmann_names:
        converted = _sackmann_to_td_name(name)
        if converted in td_names:
            lookup[name] = converted
        elif converted.lower() in td_names_lower:
            lookup[name] = td_names_lower[converted.lower()]
        else:
            unmatched.append(name)

    # Fallback: try matching on last surname token + first initial
    td_by_surname: dict[str, list[str]] = {}
    for tn in td_names:
        parts = tn.rsplit(" ", 1)
        if len(parts) == 2:
            surname = parts[0].lower()
            td_by_surname.setdefault(surname, []).append(tn)

    still_unmatched = []
    for name in unmatched:
        parts = name.replace("'", " ").split()
        if len(parts) < 2:
            continue
        # Try last token as surname (handles multi-word first names)
        last_token = parts[-1].lower()
        first_initial = parts[0][0].upper() + "."
        candidates = td_by_surname.get(last_token, [])
        # Filter to same initial
        initial_matches = [c for c in candidates if c.endswith(first_initial)]
        if len(initial_matches) == 1:
            lookup[name] = initial_matches[0]
        elif len(candidates) == 1:
            lookup[name] = candidates[0]
        else:
            still_unmatched.append(name)

    return lookup


def merge_odds_with_matches(
    matches: pd.DataFrame,
    odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join pre-match betting odds onto the Sackmann matches DataFrame.

    Adds columns ``odds_p1`` and ``odds_p2`` (Pinnacle decimal odds, falling
    back to Bet365 then to average) and ``implied_p1_prob`` (implied
    probability for player 1, adjusted for overround).

    Parameters
    ----------
    matches : pd.DataFrame
        Output of ``load_all_singles_matches``.
    odds : pd.DataFrame, optional
        Output of ``load_odds``.  Loaded automatically if not provided.

    Returns
    -------
    A copy of ``matches`` with odds columns appended.  Rows without a
    matching odds entry will have NaN in the odds columns.
    """
    if odds is None:
        odds = load_odds()

    name_lookup = _build_name_lookup(matches, odds)

    # Normalise Sackmann names in a copy for joining
    matches = matches.copy()
    matches["_p1_td"] = matches["player1"].map(name_lookup)
    matches["_p2_td"] = matches["player2"].map(name_lookup)
    matches["_tournament"] = matches["slam"].map(SLAM_TOURNAMENT_MAP)

    # In odds data, Winner/Loser don't correspond to p1/p2 — we need to
    # match the *pair* of players regardless of order.
    odds = odds.copy()
    # Create a frozenset key for each odds row
    odds["_pair"] = odds.apply(
        lambda r: frozenset([r["Winner"], r["Loser"]]) if pd.notna(r["Winner"]) else None,
        axis=1,
    )
    odds["_tournament"] = odds["Tournament"]

    # Build odds lookup: (year, tournament, frozenset(p1, p2)) -> row
    odds_lookup: dict[tuple, pd.Series] = {}
    for _, row in odds.iterrows():
        if row["_pair"] is None:
            continue
        key = (row["year"], row["_tournament"], row["_pair"])
        odds_lookup[key] = row

    odds_p1 = []
    odds_p2 = []
    for _, mrow in matches.iterrows():
        p1_td = mrow["_p1_td"]
        p2_td = mrow["_p2_td"]
        tournament = mrow["_tournament"]
        year = mrow["year"]

        if pd.isna(p1_td) or pd.isna(p2_td) or pd.isna(tournament):
            odds_p1.append(None)
            odds_p2.append(None)
            continue

        key = (year, tournament, frozenset([p1_td, p2_td]))
        orow = odds_lookup.get(key)
        if orow is None:
            odds_p1.append(None)
            odds_p2.append(None)
            continue

        # Determine which odds column maps to p1 vs p2
        # Pick best available odds source: Pinnacle > Bet365 > Average
        for w_col, l_col in [("PSW", "PSL"), ("B365W", "B365L"), ("AvgW", "AvgL")]:
            if w_col in orow and pd.notna(orow.get(w_col)):
                winner_odds = orow[w_col]
                loser_odds = orow[l_col]
                break
        else:
            odds_p1.append(None)
            odds_p2.append(None)
            continue

        if orow["Winner"] == p1_td:
            odds_p1.append(winner_odds)
            odds_p2.append(loser_odds)
        else:
            odds_p1.append(loser_odds)
            odds_p2.append(winner_odds)

    matches["odds_p1"] = pd.array(odds_p1, dtype=pd.Float64Dtype())
    matches["odds_p2"] = pd.array(odds_p2, dtype=pd.Float64Dtype())

    # Implied probability (remove overround via normalisation)
    ip1 = 1.0 / matches["odds_p1"]
    ip2 = 1.0 / matches["odds_p2"]
    total = ip1 + ip2
    matches["implied_p1_prob"] = ip1 / total

    # Clean up temp columns
    matches.drop(columns=["_p1_td", "_p2_td", "_tournament"], inplace=True)

    return matches


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def train_val_split(
    matches: pd.DataFrame,
    points: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split matches and points into training and validation sets.

    Splitting is done at the match level so all points from a given match
    stay together in the same set.

    Parameters
    ----------
    matches : pd.DataFrame
        Combined matches DataFrame (from ``load_all_singles_matches``).
    points : pd.DataFrame
        Combined points DataFrame (from ``load_all_singles_points``).
    val_fraction : float
        Fraction of *matches* to hold out for validation (default 0.2).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_matches, val_matches, train_points, val_points
    """
    match_ids = matches["match_id"].unique()
    rng = pd.Series(match_ids).sample(frac=1, random_state=seed)
    split_idx = int(len(rng) * (1 - val_fraction))
    train_ids = set(rng.iloc[:split_idx])
    val_ids = set(rng.iloc[split_idx:])

    train_matches = matches[matches["match_id"].isin(train_ids)].reset_index(drop=True)
    val_matches = matches[matches["match_id"].isin(val_ids)].reset_index(drop=True)
    train_points = points[points["match_id"].isin(train_ids)].reset_index(drop=True)
    val_points = points[points["match_id"].isin(val_ids)].reset_index(drop=True)

    return train_matches, val_matches, train_points, val_points


def load_singles_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function: load both matches and points DataFrames.

    Returns
    -------
    matches : pd.DataFrame
    points : pd.DataFrame
    """
    return load_all_singles_matches(), load_all_singles_points()
