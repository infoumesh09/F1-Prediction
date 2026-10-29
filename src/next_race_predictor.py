"""
F1 2026 Next Race Predictor
============================
Predicts the finishing order for the next F1 race using:
  - FastF1 session data (FP1, FP2, FP3, Qualifying)
  - Rolling driver/constructor form from last 3 races
  - Championship standings via Ergast API
  - Ridge regression trained on all completed 2026 races

Based on the pipeline from F1-2026-Prediction-main (Races 09-13 mature scripts).

Usage (live - requires Qualifying data to be available):
    predictor = NextRacePredictor()
    results = predictor.predict("Singapore")

Usage (from cached TrainingData.csv only):
    predictor = NextRacePredictor()
    predictor.train_from_existing_data()
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import joblib
import fastf1
from fastf1.ergast import Ergast
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "race_data")
_MODEL_DIR = os.path.join(_ROOT_DIR, "models")
_RESULTS_DIR = os.path.join(_ROOT_DIR, "results")
_CACHE_DIR = os.path.join(_ROOT_DIR, "scripts", ".fastf1_cache")

os.makedirs(_DATA_DIR, exist_ok=True)
os.makedirs(_MODEL_DIR, exist_ok=True)
os.makedirs(_RESULTS_DIR, exist_ok=True)
os.makedirs(_CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2026 Season Mappings  (from F1-2026-Prediction-main ETL scripts)
# ---------------------------------------------------------------------------
DRIVER_TEAM_MAP = {
    "NOR": "McLaren",        "PIA": "McLaren",
    "VER": "Red Bull Racing","HAD": "Red Bull Racing",
    "LEC": "Ferrari",        "HAM": "Ferrari",
    "RUS": "Mercedes",       "ANT": "Mercedes",
    "ALO": "Aston Martin",   "STR": "Aston Martin",
    "ALB": "Williams",       "SAI": "Williams",
    "OCO": "Haas F1 Team",   "BEA": "Haas F1 Team",
    "GAS": "Alpine",         "COL": "Alpine",
    "HUL": "Audi",           "BOR": "Audi",
    "PER": "Cadillac",       "BOT": "Cadillac",
    "LAW": "Racing Bulls",   "LIN": "Racing Bulls",
    "TSU": "Racing Bulls",   # Yuki Tsunoda (mid-season)
}

CONSTRUCTOR_ID_MAP = {
    "mercedes":     "Mercedes",
    "ferrari":      "Ferrari",
    "mclaren":      "McLaren",
    "red_bull":     "Red Bull Racing",
    "alpine":       "Alpine",
    "haas":         "Haas F1 Team",
    "rb":           "Racing Bulls",
    "racing_bulls": "Racing Bulls",
    "williams":     "Williams",
    "audi":         "Audi",
    "cadillac":     "Cadillac",
    "aston_martin": "Aston Martin",
}

# Sprint weekends in 2026 (use SQ instead of FP2, S instead of FP3)
SPRINT_ROUNDS = {2, 4, 5, 9}  # China, Miami, Canada, British

# Round number lookup for all 2026 races
RACE_ROUND_MAP = {
    "Australia": 1,  "China": 2,       "Japan": 3,
    "Miami": 4,      "Canada": 5,      "Monaco": 6,
    "Barcelona": 7,  "Austria": 8,     "British": 9,
    "Belgian": 10,   "Hungarian": 11,  "Netherlands": 12,
    "Italy": 13,     "Spain": 14,      "Spanish": 14,
    "Azerbaijan": 15, "Bahrain": 16,   "Singapore": 17,
    "UnitedStates": 18, "United States": 18, "Mexico": 19,
    "Brazil": 20,    "SaoPaulo": 20,   "S\u00e3o Paulo": 20,
    "LasVegas": 21,  "Las Vegas": 21,  "Qatar": 22,
    "AbuDhabi": 23,  "Abu Dhabi": 23,
}

# The 17 features the model uses
MODEL_FEATURES = [
    "AveragePositionFromLast3Races",
    "AveragePointsFromLast3Races",
    "ConstructorAveragePointFromLast3Races",
    "Qualifying_Time(s)",
    "Starting_Pos",
    "DriverPoints",
    "ConstructorPoints",
    "FP1_Rank",
    "FP2_Rank",
    "FP3_Rank",
    "FP1_DeltaToFastest",
    "FP2_DeltaToFastest",
    "FP3_DeltaToFastest",
    "LapTime_Rank",
    "StartXConst",
    "DriXConst",
    "FP3XStart",
]


# ===========================================================================
# 1. DATA EXTRACTION  (ETL layer)
# ===========================================================================

def _get_fp1_best(fp1_session) -> pd.DataFrame:
    laps = fp1_session.laps.pick_quicklaps()
    best = laps.groupby("Driver")["LapTime"].min().dt.total_seconds().reset_index()
    best.columns = ["Driver", "FP1_BestTime(s)"]
    return best


def _get_fp2_best(fp2_session) -> pd.DataFrame:
    laps = fp2_session.laps.pick_quicklaps()
    best = laps.groupby("Driver")["LapTime"].min().dt.total_seconds().reset_index()
    best.columns = ["Driver", "FP2_BestTime(s)"]
    return best


def _get_fp3_best(fp3_session) -> pd.DataFrame:
    laps = fp3_session.laps.pick_quicklaps()
    best = laps.groupby("Driver")["LapTime"].min().dt.total_seconds().reset_index()
    best.columns = ["Driver", "FP3_BestTime(s)"]
    return best


def _get_sector_times(fp2_session) -> pd.DataFrame:
    laps = fp2_session.laps
    st = (
        laps.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]]
        .mean()
        .apply(lambda x: x.dt.total_seconds())
        .reset_index()
    )
    st.columns = ["Driver", "Sector1Time(s)", "Sector2Time(s)", "Sector3Time(s)"]
    return st


def _get_average_laptime(fp2_session) -> pd.DataFrame:
    laps = fp2_session.laps
    avg = laps.groupby("Driver")["LapTime"].mean().dt.total_seconds().reset_index()
    avg.columns = ["Driver", "Average_Laptime(s)"]
    return avg


def _get_qualifying_data(qualify_session) -> pd.DataFrame:
    results = qualify_session.results.copy()
    qual_time = results[["Q3", "Q2", "Q1"]].bfill(axis=1).iloc[:, 0]
    qual = results[["Abbreviation"]].copy()
    qual["Qualifying_Time(s)"] = qual_time.dt.total_seconds()
    qual.rename(columns={"Abbreviation": "Driver"}, inplace=True)
    return qual


def _get_starting_position(qualify_session) -> pd.DataFrame:
    df = qualify_session.results[["Abbreviation", "Position"]].copy()
    df.rename(columns={"Abbreviation": "Driver"}, inplace=True)
    return df


def _get_race_results(result_session) -> pd.DataFrame:
    df = result_session.results[["Abbreviation", "Position"]].copy()
    df.rename(columns={"Abbreviation": "Driver"}, inplace=True)
    return df


def _get_driver_form(sessions_list) -> pd.DataFrame:
    """Rolling average position and points from last 3 races."""
    if not sessions_list:
        return pd.DataFrame({
            "Driver": list(DRIVER_TEAM_MAP.keys()),
            "AveragePositionFromLast3Races": 22.0,
            "AveragePointsFromLast3Races": 0.0,
        })

    all_data = []
    for session in sessions_list:
        event_name = session.event["EventName"]
        results = session.results.copy()
        d = results[["Abbreviation", "Position", "Points"]].copy()
        d.rename(columns={"Abbreviation": "Driver"}, inplace=True)
        d["Event"] = event_name
        all_data.append(d)

    combined = pd.concat(all_data, ignore_index=True)
    pivot = combined.pivot(index="Driver", columns="Event", values=["Position", "Points"])
    pivot.columns = [f"{c[0]}_{c[1]}" for c in pivot.columns]
    pivot = pivot.reset_index()

    pos_cols = [c for c in pivot.columns if c.startswith("Position_")]
    pts_cols = [c for c in pivot.columns if c.startswith("Points_")]
    pivot[pos_cols] = pivot[pos_cols].fillna(22)
    pivot[pts_cols] = pivot[pts_cols].fillna(0)
    pivot["AveragePositionFromLast3Races"] = pivot[pos_cols].mean(axis=1)
    pivot["AveragePointsFromLast3Races"]   = pivot[pts_cols].mean(axis=1)

    return pivot[["Driver", "AveragePositionFromLast3Races", "AveragePointsFromLast3Races"]]


def _get_constructor_form(sessions_list) -> pd.DataFrame:
    """Rolling average constructor points from last 3 races."""
    if not sessions_list:
        return pd.DataFrame({
            "Constructor": list(set(DRIVER_TEAM_MAP.values())),
            "ConstructorAveragePointFromLast3Races": 0.0,
        })

    all_data = []
    for session in sessions_list:
        event_name = session.event["EventName"]
        results = session.results.copy()
        tp = results.groupby("TeamName")["Points"].sum().reset_index()
        tp.rename(columns={"TeamName": "Constructor"}, inplace=True)
        tp["Event"] = event_name
        all_data.append(tp)

    combined = pd.concat(all_data, ignore_index=True)
    pivot = combined.pivot(index="Constructor", columns="Event", values="Points")
    pivot.columns = [f"Pts_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()
    pts_cols = [c for c in pivot.columns if c.startswith("Pts_")]
    pivot["ConstructorAveragePointFromLast3Races"] = pivot[pts_cols].mean(axis=1)
    return pivot[["Constructor", "ConstructorAveragePointFromLast3Races"]]


def _get_driver_points(season: int, race_name: str) -> pd.DataFrame:
    """Championship standings before this race."""
    event = fastf1.get_event(season, race_name)
    current_round = event.RoundNumber
    ergast = Ergast()

    if current_round == 1:
        driver_points = pd.DataFrame({
            "Driver": list(DRIVER_TEAM_MAP.keys()),
            "DriverPoints": 0.0,
        })
        return driver_points

    resp = ergast.get_driver_standings(season=season, round=current_round - 1)
    standings = resp.content[0] if hasattr(resp, "content") else resp
    df = standings[["driverCode", "points"]].copy()
    df.rename(columns={"driverCode": "Driver", "points": "DriverPoints"}, inplace=True)
    return df


def _get_constructor_points(season: int, race_name: str) -> pd.DataFrame:
    """Constructor championship standings before this race."""
    event = fastf1.get_event(season, race_name)
    current_round = event.RoundNumber
    ergast = Ergast()

    if current_round == 1:
        return pd.DataFrame({
            "Constructor": list(set(DRIVER_TEAM_MAP.values())),
            "ConstructorPoints": 0.0,
        })

    resp = ergast.get_constructor_standings(season=season, round=current_round - 1)
    standings = resp.content[0] if hasattr(resp, "content") else resp
    df = standings[["constructorId", "points"]].copy()
    df.rename(columns={"constructorId": "Constructor", "points": "ConstructorPoints"}, inplace=True)
    df["Constructor"] = df["Constructor"].map(CONSTRUCTOR_ID_MAP).fillna(df["Constructor"])
    return df


# ===========================================================================
# 2. FEATURE ENGINEERING  (EDA layer)
# ===========================================================================

def engineer_features(df: pd.DataFrame, has_result: bool = True) -> pd.DataFrame:
    """
    Apply rank + delta + interaction features to a single-race DataFrame.
    Works for both training data (has_result=True) and prediction data (has_result=False).
    """
    d = df.copy()

    # ---- imputation --------------------------------------------------------
    if has_result:
        d.dropna(subset=["Qualifying_Time(s)", "Race_Result"], inplace=True)
    else:
        d.dropna(subset=["Qualifying_Time(s)"], inplace=True)

    for col in ["FP1_BestTime(s)", "FP2_BestTime(s)", "FP3_BestTime(s)",
                "Sector1Time(s)", "Sector2Time(s)", "Sector3Time(s)"]:
        if col in d.columns:
            d[col].fillna(d[col].median(), inplace=True)

    if "Average_Laptime(s)" in d.columns:
        d["Average_Laptime(s)"].fillna(d["Average_Laptime(s)"].mean(), inplace=True)

    if "AveragePointsFromLast3Races" in d.columns:
        d["AveragePointsFromLast3Races"].fillna(0, inplace=True)

    # Fill points columns so interaction features don't produce NaN
    for col in ["DriverPoints", "ConstructorPoints",
                "AveragePositionFromLast3Races", "ConstructorAveragePointFromLast3Races"]:
        if col in d.columns:
            fill_val = 22.0 if "Position" in col else 0.0
            d[col].fillna(fill_val, inplace=True)

    # ---- rank features -----------------------------------------------------
    for col, rank_col in [
        ("FP1_BestTime(s)", "FP1_Rank"),
        ("FP2_BestTime(s)", "FP2_Rank"),
        ("FP3_BestTime(s)", "FP3_Rank"),
        ("Sector1Time(s)",  "Sector1_Rank"),
        ("Sector2Time(s)",  "Sector2_Rank"),
        ("Sector3Time(s)",  "Sector3_Rank"),
        ("Average_Laptime(s)", "LapTime_Rank"),
    ]:
        if col in d.columns:
            d = d.sort_values(col, ascending=True).reset_index(drop=True)
            d[rank_col] = d.index + 1

    # ---- delta features ----------------------------------------------------
    for col, delta_col in [
        ("FP1_BestTime(s)", "FP1_DeltaToFastest"),
        ("FP2_BestTime(s)", "FP2_DeltaToFastest"),
        ("FP3_BestTime(s)", "FP3_DeltaToFastest"),
        ("Average_Laptime(s)", "DeltaToFastestLap"),
    ]:
        if col in d.columns:
            d[delta_col] = d[col] - d[col].min()

    # ---- combined sector ---------------------------------------------------
    if all(c in d.columns for c in ["Sector1Time(s)", "Sector2Time(s)", "Sector3Time(s)"]):
        d["CombinedSectorTime"]  = d["Sector1Time(s)"] + d["Sector2Time(s)"] + d["Sector3Time(s)"]
        d["CombinedSectorDelta"] = d["CombinedSectorTime"] - d["CombinedSectorTime"].min()

    # ---- interaction features ----------------------------------------------
    d["StartXConst"] = d["Starting_Pos"]   * d["ConstructorPoints"]
    d["DriXConst"]   = d["DriverPoints"]   * d["ConstructorPoints"]
    d["FP3XStart"]   = d["FP3_Rank"]       * d["Starting_Pos"]

    return d.sort_values("Driver").reset_index(drop=True)


def build_training_data(race_csvs: list) -> pd.DataFrame:
    """
    Concatenate per-race CSVs, apply feature engineering, return combined training DataFrame.
    race_csvs: list of file paths to *GP.csv files with Race_Result column.
    """
    frames = []
    for path in race_csvs:
        df = pd.read_csv(path)
        if "Race_Result" not in df.columns:
            logger.warning(f"Skipping {path} — no Race_Result column")
            continue
        df_eng = engineer_features(df, has_result=True)
        frames.append(df_eng)

    if not frames:
        raise ValueError("No valid training race CSVs found.")

    training = pd.concat(frames, axis=0, ignore_index=True)
    logger.info(f"Training data: {len(training)} rows from {len(frames)} races")
    return training


# ===========================================================================
# 3. MODEL TRAINING
# ===========================================================================

def _spearman_scorer_fn(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return spearmanr(y_true, y_pred).statistic


_spearman_scorer = make_scorer(_spearman_scorer_fn, greater_is_better=True)


def train_model(training_df: pd.DataFrame, model_save_path: str = None) -> tuple:
    """
    Train Ridge regression pipeline on training_df.
    Returns (fitted best estimator, spearman_score).
    """
    available_features = [f for f in MODEL_FEATURES if f in training_df.columns]
    missing = set(MODEL_FEATURES) - set(available_features)
    if missing:
        logger.warning(f"Missing features in training data: {missing}")

    X = training_df[available_features]
    y = training_df["Race_Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge()),
    ])

    param_grid = {
        "ridge__alpha":         [0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000],
        "ridge__fit_intercept": [True, False],
        "ridge__solver":        ["auto", "svd", "cholesky", "lsqr"],
        "ridge__positive":      [True, False],
        "ridge__random_state":  [42],
    }

    logger.info("Training Ridge model with GridSearchCV (Spearman scorer)...")
    # The training data is deliberately small (one row per driver per race).
    # A single worker avoids spawning orphaned worker processes on Windows.
    cv = GridSearchCV(pipeline, param_grid, cv=5, scoring=_spearman_scorer, n_jobs=1)
    cv.fit(X_train, y_train)

    y_pred = cv.predict(X_test)
    spearman, _ = spearmanr(y_pred, y_test)
    logger.info(f"Model Spearman Rank (test set): {spearman:.3f}")
    logger.info(f"Best params: {cv.best_params_}")

    # Persist only the fitted estimator.  Saving the GridSearchCV object would
    # also pickle the custom scorer, making models trained from the CLI depend
    # on Python's transient ``__main__`` module and impossible to reload.
    fitted_model = cv.best_estimator_
    if model_save_path:
        joblib.dump(fitted_model, model_save_path)
        logger.info(f"Model saved to {model_save_path}")

    return fitted_model, spearman


# ===========================================================================
# 4. PREDICTION
# ===========================================================================

def predict_from_df(model, prediction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run model on engineered prediction DataFrame.
    Returns DataFrame sorted by predicted finishing position.
    """
    available_features = [f for f in MODEL_FEATURES if f in prediction_df.columns]
    X = prediction_df[available_features]
    prediction_df = prediction_df.copy()
    prediction_df["Predicted_Score"] = model.predict(X)
    prediction_df = prediction_df.sort_values("Predicted_Score", ascending=True).reset_index(drop=True)
    prediction_df["Predicted_Pos"] = prediction_df.index + 1
    return prediction_df


# ===========================================================================
# 5. MAIN CLASS
# ===========================================================================

class NextRacePredictor:
    """
    Unified next race predictor.

    Workflow A — live prediction (requires FP1/FP2/FP3/Qualifying data):
        predictor = NextRacePredictor()
        results_df = predictor.predict("Singapore")

    Workflow B — retrain on cached data, then predict:
        predictor = NextRacePredictor()
        predictor.train_from_existing_data()
        results_df = predictor.predict("Singapore")

    Workflow C — just retrain (no upcoming race yet):
        predictor = NextRacePredictor()
        predictor.train_from_existing_data()
    """

    SEASON = 2026

    def __init__(self, data_dir: str = _DATA_DIR, model_dir: str = _MODEL_DIR,
                 results_dir: str = _RESULTS_DIR, cache_dir: str = _CACHE_DIR):
        self.data_dir    = data_dir
        self.model_dir   = model_dir
        self.results_dir = results_dir
        self.cache_dir   = cache_dir
        self.model       = None
        self.spearman    = None

        fastf1.Cache.enable_cache(self.cache_dir)

    # ------------------------------------------------------------------
    # A) ETL for the upcoming race (live FastF1 fetch)
    # ------------------------------------------------------------------

    def fetch_race_data(self, race_name: str) -> pd.DataFrame:
        """
        Extract FP1/FP2/FP3/Qualifying/form data for the upcoming race.
        Returns raw DataFrame (before feature engineering).
        Does NOT load the Race session (race hasn't happened yet).
        """
        season = self.SEASON
        round_num = RACE_ROUND_MAP.get(race_name)
        is_sprint = round_num in SPRINT_ROUNDS if round_num else False

        logger.info(f"Fetching data for {race_name} (Round {round_num}, sprint={is_sprint})...")

        # Session labels differ for sprint weekends
        fp2_label = "SQ" if is_sprint else "FP2"
        fp3_label = "S"  if is_sprint else "FP3"

        logger.info("Loading FP1...")
        FP1 = fastf1.get_session(season, race_name, "FP1"); FP1.load()

        logger.info(f"Loading {fp2_label}...")
        FP2 = fastf1.get_session(season, race_name, fp2_label); FP2.load()

        logger.info(f"Loading {fp3_label}...")
        FP3 = fastf1.get_session(season, race_name, fp3_label); FP3.load()

        logger.info("Loading Qualifying...")
        Qualify = fastf1.get_session(season, race_name, "Q"); Qualify.load()

        # Rolling form — last 3 completed races
        if round_num:
            start = max(1, round_num - 2)
            form_sessions = []
            for r in range(start, round_num):
                try:
                    s = fastf1.get_session(season, r, "R")
                    s.load()
                    form_sessions.append(s)
                except Exception as e:
                    logger.warning(f"  Could not load round {r} result: {e}")
        else:
            form_sessions = []

        logger.info("Computing rolling driver/constructor form...")
        driver_form      = _get_driver_form(form_sessions)
        constructor_form = _get_constructor_form(form_sessions)

        logger.info("Fetching championship standings...")
        try:
            driver_pts_df      = _get_driver_points(season, race_name)
            constructor_pts_df = _get_constructor_points(season, race_name)
        except Exception as e:
            logger.warning(f"Standings fetch failed ({e}), using zeros.")
            driver_pts_df = pd.DataFrame({
                "Driver": list(DRIVER_TEAM_MAP.keys()), "DriverPoints": 0.0
            })
            constructor_pts_df = pd.DataFrame({
                "Constructor": list(set(DRIVER_TEAM_MAP.values())), "ConstructorPoints": 0.0
            })

        # Extract session data
        fp1_best     = _get_fp1_best(FP1)
        fp2_best     = _get_fp2_best(FP2)
        fp3_best     = _get_fp3_best(FP3)
        sector_times = _get_sector_times(FP2)
        avg_laptime  = _get_average_laptime(FP2)
        qual_times   = _get_qualifying_data(Qualify)
        start_pos    = _get_starting_position(Qualify)

        # Add Constructor column to driver form
        driver_form["Constructor"] = driver_form["Driver"].map(DRIVER_TEAM_MAP)

        # Merge constructor form
        driver_form = driver_form.merge(
            constructor_form[["Constructor", "ConstructorAveragePointFromLast3Races"]],
            on="Constructor", how="left"
        )
        driver_form["ConstructorAveragePointFromLast3Races"] = \
            driver_form["ConstructorAveragePointFromLast3Races"].fillna(0)

        # Merge driver championship points
        driver_pts_df["Constructor"] = driver_pts_df["Driver"].map(DRIVER_TEAM_MAP)
        constructor_pts_df["Constructor"] = constructor_pts_df["Constructor"].map(
            lambda x: CONSTRUCTOR_ID_MAP.get(x, x)
        )
        driver_pts_df = driver_pts_df.merge(
            constructor_pts_df[["Constructor", "ConstructorPoints"]],
            on="Constructor", how="left"
        )
        driver_pts_df["ConstructorPoints"] = driver_pts_df["ConstructorPoints"].fillna(0)

        # Build master DataFrame
        for df_part in [fp1_best, fp2_best, fp3_best, sector_times,
                        avg_laptime, qual_times, start_pos,
                        driver_pts_df[["Driver", "DriverPoints", "ConstructorPoints"]]]:
            driver_form = driver_form.merge(df_part, on="Driver", how="left")

        # Rename DriverPoints/ConstructorPoints if duplicated from standings merge
        cols = list(driver_form.columns)
        if "DriverPoints_x" in cols:
            driver_form.rename(columns={"DriverPoints_x": "DriverPoints"}, inplace=True)
            driver_form.drop(columns=["DriverPoints_y"], errors="ignore", inplace=True)
        if "ConstructorPoints_x" in cols:
            driver_form.rename(columns={"ConstructorPoints_x": "ConstructorPoints"}, inplace=True)
            driver_form.drop(columns=["ConstructorPoints_y"], errors="ignore", inplace=True)

        logger.info(f"Raw data assembled: {len(driver_form)} drivers")
        return driver_form

    # ------------------------------------------------------------------
    # B) Train on existing CSVs in data/race_data/
    # ------------------------------------------------------------------

    def train_from_existing_data(self, force_retrain: bool = False) -> float:
        """
        Train the Ridge model using all *GP.csv files in data/race_data/.
        If TrainingData.csv already exists and force_retrain=False, uses that directly.
        Returns Spearman score on test set.
        """
        model_path = os.path.join(self.model_dir, "next_race_ridge.joblib")

        training_csv = os.path.join(self.data_dir, "TrainingData.csv")
        if os.path.exists(training_csv) and not force_retrain:
            logger.info(f"Loading existing TrainingData.csv from {training_csv}")
            training_df = pd.read_csv(training_csv)
            # Apply feature engineering to the combined CSV
            # TrainingData.csv from F1-2026-Prediction-main already has features
            # Just ensure all model features are present; if not, rebuild
            if all(f in training_df.columns for f in MODEL_FEATURES):
                logger.info("All model features present in TrainingData.csv, using directly.")
            else:
                logger.info("Some features missing, rebuilding from per-race CSVs...")
                training_df = self._build_training_from_csvs()
        else:
            training_df = self._build_training_from_csvs()

        self.model, self.spearman = train_model(training_df, model_save_path=model_path)
        return self.spearman

    def _build_training_from_csvs(self) -> pd.DataFrame:
        """Collect all *GP.csv from data_dir, apply feature engineering, concatenate."""
        csv_files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith("GP.csv")
        ])
        if not csv_files:
            raise FileNotFoundError(f"No *GP.csv files found in {self.data_dir}")
        logger.info(f"Found {len(csv_files)} race CSVs for training")
        return build_training_data(csv_files)

    def _load_model(self):
        """Load saved model from disk if not already in memory."""
        if self.model is not None:
            return
        model_path = os.path.join(self.model_dir, "next_race_ridge.joblib")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.info("No saved model found. Training from existing data first...")
            self.train_from_existing_data()

    # ------------------------------------------------------------------
    # C) Predict next race (full live pipeline)
    # ------------------------------------------------------------------

    def predict(self, race_name: str, save_csv: bool = True) -> pd.DataFrame:
        """
        Full pipeline: fetch live data → engineer features → predict → print results.

        Args:
            race_name: FastF1 race name string, e.g. "Singapore", "Japanese"
            save_csv:  Whether to save PredictedData.csv to results/

        Returns:
            DataFrame with Predicted_Pos column (sorted 1 to N)
        """
        self._load_model()

        # 1. Fetch raw data
        raw_df = self.fetch_race_data(race_name)

        # 2. Feature engineering
        pred_df = engineer_features(raw_df, has_result=False)

        # 3. Predict
        results = predict_from_df(self.model, pred_df)

        # 4. Display
        self._print_results(results, race_name)

        # 5. Save
        if save_csv:
            out_path = os.path.join(self.results_dir, f"next_race_{race_name}_prediction.csv")
            results.to_csv(out_path, index=False)
            logger.info(f"Saved predictions to {out_path}")

        return results

    # ------------------------------------------------------------------
    # D) Predict using a pre-fetched CSV  (offline / testing mode)
    # ------------------------------------------------------------------

    def predict_from_csv(self, race_csv_path: str, race_name: str = "Race") -> pd.DataFrame:
        """
        Run prediction using a pre-saved *GP.csv (no live FastF1 call needed).
        Useful for testing or replaying past predictions.
        """
        self._load_model()
        raw_df = pd.read_csv(race_csv_path)
        pred_df = engineer_features(raw_df, has_result=False)
        results = predict_from_df(self.model, pred_df)
        self._print_results(results, race_name)
        out_path = os.path.join(self.results_dir, f"next_race_{race_name}_prediction.csv")
        results.to_csv(out_path, index=False)
        logger.info(f"Saved predictions to {out_path}")
        return results

    # ------------------------------------------------------------------
    # E) Pretty print
    # ------------------------------------------------------------------

    @staticmethod
    def _print_results(results: pd.DataFrame, race_name: str):
        print()
        print("=" * 60)
        print(f"  F1 2026 NEXT RACE PREDICTION — {race_name.upper()} GP")
        print("=" * 60)
        podium_labels = {1: "P1 (Winner)", 2: "P2", 3: "P3"}
        for _, row in results.iterrows():
            pos   = int(row["Predicted_Pos"])
            drv   = row["Driver"]
            team  = DRIVER_TEAM_MAP.get(drv) or str(row.get("Constructor", "")).strip()
            if team == "nan":
                team = "Unknown"
            label = podium_labels.get(pos, f"P{pos}")
            print(f"  {label:<12} {drv:<6}  ({team})")
        print("=" * 60)
        print()


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    """
    Run from the command line:
        python next_race_predictor.py --race Singapore
        python next_race_predictor.py --retrain
        python next_race_predictor.py --race Singapore --retrain
        python next_race_predictor.py --csv ../data/race_data/ItalyGP.csv --race Italy
    """
    import argparse
    parser = argparse.ArgumentParser(description="F1 2026 Next Race Predictor")
    parser.add_argument("--race",    type=str, help="Race name (e.g. Singapore, Japanese)")
    parser.add_argument("--retrain", action="store_true", help="Force retrain model from CSVs")
    parser.add_argument("--csv",     type=str, help="Path to a pre-fetched *GP.csv for offline prediction")
    args = parser.parse_args()

    predictor = NextRacePredictor()

    if args.retrain or args.race is None and args.csv is None:
        print("\nTraining model from existing race data...")
        spearman = predictor.train_from_existing_data(force_retrain=args.retrain)
        print(f"Model trained. Spearman Rank on test set: {spearman:.3f}\n")

    if args.csv:
        race_name = args.race or os.path.splitext(os.path.basename(args.csv))[0].replace("GP", "")
        predictor.predict_from_csv(args.csv, race_name=race_name)
    elif args.race:
        predictor.predict(args.race)


if __name__ == "__main__":
    main()
