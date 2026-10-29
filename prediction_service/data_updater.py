"""
Live data fetcher for both prediction models.

When the frontend "UPDATE DATA & RETRAIN" button is clicked, the service calls
:func:`fetch_new_data`, which:

  * detects the latest completed round from the live FastF1/Ergast data,
  * refreshes the championship grid CSV (``data/f1_2026_grid.csv``) when new
    rounds are available,
  * refreshes the per-race training CSV for the latest completed round — adding
    its ``Race_Result`` so the next-race model trains on the newest data,
  * advances the season constants (``COMPLETED_ROUNDS`` / ``REMAINING_ROUNDS`` /
    next-race info) in the runtime modules.

Every step degrades gracefully: if the network fetch fails, the previous local
data is kept and the retrain continues on it.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any

import fastf1
import pandas as pd

from fastf1.ergast import Ergast

logger = logging.getLogger(__name__)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_ROOT_DIR, "data")
_RACE_DATA_DIR = os.path.join(_DATA_DIR, "race_data")
_SCRIPTS_DIR = os.path.join(_ROOT_DIR, "scripts")
_CACHE_DIR = os.path.join(_SCRIPTS_DIR, ".fastf1_cache")

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_SEASON = 2026
_TOTAL_ROUNDS = 24

# Round number -> file base used for the saved *GP.csv files
ROUND_FILE_BASE = {
    1: "Australia", 2: "China", 3: "Japan", 4: "Miami",
    5: "Canada", 6: "Monaco", 7: "Barcelona", 8: "Austria",
    9: "British", 10: "Belgian", 11: "Hungarian", 12: "Netherlands",
    13: "Italy", 14: "Spain", 15: "Azerbaijan", 16: "Bahrain",
    17: "Singapore", 18: "United States", 19: "Mexico", 20: "Brazil",
    21: "Las Vegas", 22: "Qatar", 23: "Abu Dhabi",
}

# Short labels used in NEXT_RACE_MAP
ROUND_SHORT = {
    14: "Spain", 15: "Azerbaijan", 16: "Bahrain", 17: "Singapore",
    18: "United States", 19: "Mexico", 20: "Brazil", 21: "Las Vegas",
    22: "Qatar", 23: "Abu Dhabi",
}

# Round -> (Grand Prix name, country code) for rounds beyond the static map
ROUND_INFO = {
    14: ("Spanish Grand Prix", "ESP"),
    15: ("Azerbaijan Grand Prix", "AZE"),
    16: ("Bahrain Grand Prix", "BHR"),
    17: ("Singapore Grand Prix", "SIN"),
    18: ("United States Grand Prix", "USA"),
    19: ("Mexico City Grand Prix", "MEX"),
    20: ("Sao Paulo Grand Prix", "BRA"),
    21: ("Las Vegas Grand Prix", "USA"),
    22: ("Qatar Grand Prix", "QAT"),
    23: ("Abu Dhabi Grand Prix", "UAE"),
}


def discover_race_names() -> Dict[str, tuple]:
    """
    Build a ``{BaseGP: (round, Grand Prix name, country code)}`` map from the
    *GP.csv files currently on disk.  Self-heals for any newly completed round.
    """
    base_to_round = {v: k for k, v in ROUND_FILE_BASE.items()}
    out: Dict[str, tuple] = {}
    if not os.path.isdir(_RACE_DATA_DIR):
        return out
    for f in os.listdir(_RACE_DATA_DIR):
        if not f.endswith("GP.csv"):
            continue
        base = f[: -len("GP.csv")]
        rnd = base_to_round.get(base)
        if not rnd:
            continue
        name, cc = ROUND_INFO.get(rnd, (f"{base} Grand Prix", ""))
        out[f"{base}GP"] = (rnd, name, cc)
    return out


def _enable_cache() -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(_CACHE_DIR)


def detect_latest_completed_round(season: int = _SEASON,
                                  from_round: int = 1) -> int:
    """
    Return the highest round number that has live standings data, probing
    upward from ``from_round``.  Stops at the first round with no data.
    Returns ``from_round - 1`` when no new rounds are available.
    """
    _enable_cache()
    ergast = Ergast()
    latest = 0
    for r in range(max(from_round, 1), _TOTAL_ROUNDS + 1):
        try:
            resp = ergast.get_driver_standings(season=season, round=r)
            rows = resp.content[0]
            if rows is not None and len(rows) > 0:
                latest = r
            else:
                break
        except Exception:
            break
    if latest == 0:
        latest = from_round - 1
    logger.info(f"Latest completed round detected: {latest}")
    return latest


def update_championship_grid(season: int, latest_round: int,
                             completed_rounds: int,
                             force: bool = False) -> Dict[str, Any]:
    """
    Rebuild data/f1_2026_grid.csv from live race results if a new round
    completed.  When no new rounds are available the existing file is kept
    unless ``force`` is set.  Returns a summary dict.
    """
    if not force and latest_round <= completed_rounds:
        return {"updated": False, "reason": "no new completed rounds"}

    try:
        from update_2026_standings import (
            aggregate_standings_from_races,
            build_grid_csv,
        )
    except Exception as e:
        logger.warning(f"Could not import standings helpers: {e}")
        return {"updated": False, "reason": f"import failed: {e}"}

    try:
        driver_data, constructor_data, completed = aggregate_standings_from_races(
            season, latest_round
        )
        if not driver_data:
            return {"updated": False, "reason": "no standings aggregated"}

        new_df, _ = build_grid_csv(driver_data, constructor_data, completed, season)

        # Merge with the existing grid so drivers absent from live results
        # (e.g. mid-season additions) are preserved.
        grid_path = os.path.join(_DATA_DIR, "f1_2026_grid.csv")
        old_df = pd.read_csv(grid_path) if os.path.exists(grid_path) else pd.DataFrame()
        if len(old_df):
            known = set(new_df["driver_code"].astype(str))
            extra = old_df[~old_df["driver_code"].astype(str).isin(known)]
            new_df = pd.concat([new_df, extra], ignore_index=True)
            new_df = new_df.sort_values(
                ["final_points", "driver_code"], ascending=[False, True]
            ).reset_index(drop=True)

        new_df.to_csv(grid_path, index=False)
        logger.info(f"Grid CSV updated: {len(new_df)} drivers through round {completed}")
        return {
            "updated": True,
            "completed_round": int(completed),
            "drivers": int(len(new_df)),
        }
    except Exception as e:
        logger.warning(f"Championship grid refresh failed: {e}")
        return {"updated": False, "reason": str(e)}


def _has_result(csv_path: str) -> bool:
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if "Race_Result" not in df.columns:
            return False
        return int(df["Race_Result"].notna().sum()) > 0
    except Exception:
        return False


def fetch_race_data_with_result(race_name: str, round_num: int) -> pd.DataFrame:
    """
    Fetch the raw feature rows for a completed race weekend (FP/Q/form/
    standings) and merge the actual finishing order into ``Race_Result`` plus
    the starting grid into ``Starting_Pos`` (matching the *GP.csv training
    format).
    """
    _enable_cache()
    from src.next_race_predictor import NextRacePredictor

    nrp = NextRacePredictor()
    raw = nrp.fetch_race_data(race_name)

    race = fastf1.get_session(_SEASON, round_num, "R")
    race.load(telemetry=False, weather=False, messages=False)
    res = race.results[["Abbreviation", "Position", "GridPosition"]].copy()
    res.rename(
        columns={"Abbreviation": "Driver", "Position": "Race_Result",
                 "GridPosition": "Starting_Pos"},
        inplace=True,
    )
    raw = raw.merge(res, on="Driver", how="left")
    # Drivers absent from the race results (e.g. last-minute withdrawals) get a
    # conservative starting slot so interaction features stay finite.
    raw["Starting_Pos"] = raw["Starting_Pos"].fillna(22.0).astype(float)
    logger.info(f"Fetched {len(raw)} rows for round {round_num} with race result")
    return raw


def update_next_race_data(latest_round: int) -> Dict[str, Any]:
    """
    Ensure every completed round up to ``latest_round`` has a *GP.csv training
    file that includes its Race_Result, so the next-race model retrains on the
    newest data.  Backfills any previously-missed rounds (e.g. a race completed
    before a round that already has a file).
    """
    from src.next_race_predictor import RACE_ROUND_MAP

    if latest_round <= 0:
        return {"updated": False, "reason": "no completed rounds"}

    round_to_name = {}
    for name, rnd in RACE_ROUND_MAP.items():
        if rnd not in round_to_name:
            round_to_name[rnd] = name

    created = []
    for r in range(1, latest_round + 1):
        base = ROUND_FILE_BASE.get(r)
        if base is None:
            continue
        csv_path = os.path.join(_RACE_DATA_DIR, f"{base}GP.csv")
        if _has_result(csv_path):
            continue
        race_key = round_to_name.get(r)
        if race_key is None:
            logger.warning(f"No FastF1 name mapping for round {r}; skipping")
            continue
        try:
            df = fetch_race_data_with_result(race_key, r)
            # Keep the original column order of existing *GP.csv files
            col_order = [
                "Driver", "Constructor", "AveragePositionFromLast3Races",
                "AveragePointsFromLast3Races", "ConstructorAveragePointFromLast3Races",
                "FP1_BestTime(s)", "FP2_BestTime(s)", "FP3_BestTime(s)",
                "Sector1Time(s)", "Sector2Time(s)", "Sector3Time(s)",
                "Average_Laptime(s)", "Qualifying_Time(s)", "Starting_Pos",
                "Race_Result", "DriverPoints", "ConstructorPoints",
            ]
            cols = [c for c in col_order if c in df.columns]
            df[cols].to_csv(csv_path, index=False)
            logger.info(f"Wrote training CSV {csv_path}")
            created.append({"round": r, "file": os.path.basename(csv_path),
                            "rows": int(len(df))})
        except Exception as e:
            logger.warning(f"Next-race data fetch failed for round {r}: {e}")

    if created:
        return {"updated": True, "files": created}
    return {"updated": False, "reason": "all completed rounds already have result"}


def sync_season_constants(latest_round: int) -> Dict[str, Any]:
    """
    Advance COMPLETED_ROUNDS / REMAINING_ROUNDS and the next-race mapping in
    prediction_service, src.f1_2026_predictor so the whole app agrees with the
    live season progress.
    """
    import prediction_service as ps
    import src.f1_2026_predictor as f1p

    # Always keep the race registry in sync with the CSVs on disk, so the
    # driver race-history view reflects every completed round even when the
    # season has not advanced since the last update.
    merged = dict(ps.RACE_NAMES)
    for _k, _v in discover_race_names().items():
        merged.setdefault(_k, _v)
    ps.RACE_NAMES = merged

    if latest_round <= ps.COMPLETED_ROUNDS:
        return {"advanced": False, "completed_rounds": ps.COMPLETED_ROUNDS}

    remaining = _TOTAL_ROUNDS - latest_round

    ps.COMPLETED_ROUNDS = latest_round
    ps.REMAINING_ROUNDS = remaining
    ps.MAX_POINTS_REMAINING = remaining * 25

    ps.NEXT_RACE_MAP = dict(ps.NEXT_RACE_MAP)
    nxt = latest_round + 1
    if nxt <= _TOTAL_ROUNDS and nxt not in ps.NEXT_RACE_MAP:
        short = ROUND_SHORT.get(nxt, f"Round{nxt}")
        ps.NEXT_RACE_MAP[latest_round] = (
            f"{short} Grand Prix", short, nxt,
        )

    f1p.COMPLETED_ROUNDS = latest_round
    f1p.REMAINING_ROUNDS = remaining
    f1p.MAX_POINTS_REMAINING = remaining * 25

    # Also persist to the championship predictor's own source constants so a
    # fresh process starts from the correct round.
    pred_path = os.path.join(_ROOT_DIR, "src", "f1_2026_predictor.py")
    try:
        with open(pred_path, "r", encoding="utf-8") as f:
            src = f.read()
        import re

        src = re.sub(r"COMPLETED_ROUNDS = \d+",
                     f"COMPLETED_ROUNDS = {latest_round}", src)
        src = re.sub(r"REMAINING_ROUNDS = \d+",
                     f"REMAINING_ROUNDS = {remaining}", src)
        src = re.sub(r"MAX_POINTS_REMAINING = \d+ \* 25",
                     f"MAX_POINTS_REMAINING = {remaining} * 25", src)
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write(src)
    except Exception as e:
        logger.warning(f"Could not persist constants: {e}")

    # Keep prediction_service's own module constants in sync for fresh starts
    svc_path = os.path.join(_ROOT_DIR, "prediction_service", "__init__.py")
    try:
        with open(svc_path, "r", encoding="utf-8") as f:
            svc_src = f.read()
        svc_src = re.sub(r"COMPLETED_ROUNDS = \d+",
                         f"COMPLETED_ROUNDS = {latest_round}", svc_src)
        with open(svc_path, "w", encoding="utf-8") as f:
            f.write(svc_src)
    except Exception as e:
        logger.warning(f"Could not persist prediction_service constants: {e}")

    logger.info(
        f"Season constants advanced: COMPLETED_ROUNDS={latest_round}, "
        f"REMAINING_ROUNDS={remaining}"
    )
    return {"advanced": True, "completed_rounds": latest_round}


def next_race_training_csv(round_num: int) -> str:
    """Path to the *GP.csv used as prediction input for the upcoming race.

    This is the latest *completed* round's weekend data.  Falls back to the
    existing GP file with the highest round number, then to ItalyGP.csv.
    """
    if round_num in ROUND_FILE_BASE:
        path = os.path.join(_RACE_DATA_DIR, f"{ROUND_FILE_BASE[round_num]}GP.csv")
        if os.path.exists(path):
            return path

    base_to_round = {v: k for k, v in ROUND_FILE_BASE.items()}
    best_round = 0
    best_path = ""
    for f in os.listdir(_RACE_DATA_DIR):
        if not f.endswith("GP.csv"):
            continue
        base = f[: -len("GP.csv")]
        rnd = base_to_round.get(base, 0)
        if rnd > best_round:
            best_round = rnd
            best_path = os.path.join(_RACE_DATA_DIR, f)
    if best_path:
        return best_path
    return os.path.join(_RACE_DATA_DIR, "ItalyGP.csv")


def fetch_new_data(completed_rounds: int) -> Dict[str, Any]:
    """
    Fetch live data for both models.  Returns a summary dict.  Never raises:
    failures are reported and local data is kept.
    """
    summary: Dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "ok": True,
        "errors": [],
        "championship": {"updated": False},
        "next_race": {"updated": False},
        "season": {"advanced": False},
    }

    try:
        latest = detect_latest_completed_round(from_round=completed_rounds + 1)
    except Exception as e:
        latest = completed_rounds
        summary["errors"].append(f"round detection: {e}")

    summary["latest_completed_round"] = latest

    try:
        s = sync_season_constants(latest)
        summary["season"] = s
    except Exception as e:
        summary["errors"].append(f"constants: {e}")

    try:
        summary["championship"] = update_championship_grid(
            _SEASON, latest, completed_rounds, force=True
        )
    except Exception as e:
        summary["championship"] = {"updated": False, "error": str(e)}
        summary["errors"].append(f"championship: {e}")

    try:
        summary["next_race"] = update_next_race_data(latest)
    except Exception as e:
        summary["next_race"] = {"updated": False, "error": str(e)}
        summary["errors"].append(f"next_race: {e}")

    summary["training_csv"] = next_race_training_csv(latest)
    summary["finished_at"] = datetime.now().isoformat()
    summary["ok"] = len(summary["errors"]) == 0
    return summary