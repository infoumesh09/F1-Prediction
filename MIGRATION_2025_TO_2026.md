# Migration from 2025 to 2026 Season

This document summarizes the changes made to adapt the F1 Championship Predictor from the 2025 season to the 2026 season.

## What Changed

### Data Files

1. **`data/f1_2026_grid.csv`** (NEW)
   - Created with 22 drivers from the 2026 grid
   - Populated with Round 13 standings (after Italy GP)
   - Data sourced from `F1-2026-Prediction-main/Race/13.ItalyGP/Data/ItalyGP.csv`
   - Current leader: Andrea Kimi Antonelli (Mercedes) with 242 points

2. **`data/f1_championship_data.csv`** (UNCHANGED)
   - Still contains 2020-2025 historical data for training
   - 2025 final standings already correct (Norris P1, McLaren)

### Source Code

3. **`src/f1_2026_predictor.py`** (NEW)
   - Main prediction interface for 2026 season
   - Key constants:
     - `COMPLETED_ROUNDS = 13`
     - `REMAINING_ROUNDS = 11`
     - `TOTAL_ROUNDS = 24`
     - `MAX_REMAINING_POINTS = 275` (11 races × 25 pts)
   - Loads `data/f1_2026_grid.csv`
   - Saves model to `models/f1_championship_predictor_2026.joblib`
   - Outputs to `results/f1_2026_*` files

4. **`src/feature_engineering_simple.py`** (UPDATED)
   - All year filters changed: `year < 2026` for training
   - Includes `year == 2026` rows in prediction dataset
   - All docstrings updated to reference 2026

5. **`src/ml_model.py`** (UPDATED)
   - Added `prediction_year` parameter to `predict_champion_probabilities_with_history()`
   - Allows both 2025 and 2026 predictors to work without code duplication

6. **`scripts/generate_readme_images.py`** (UPDATED)
   - Updated to use 2026 data files
   - Changed `REMAINING_RACES = 11`
   - All plot titles and references updated to 2026

7. **`scripts/update_2026_standings.py`** (NEW)
   - **Incremental update script** using FastF1
   - Reads current `f1_2026_grid.csv` to detect last completed round
   - Only fetches **new race data** after that round (not all historical races)
   - Updates standings incrementally with new points, wins, podiums
   - Automatically updates `COMPLETED_ROUNDS` in `f1_2026_predictor.py`
   - **Efficient:** Only fetches what's needed (e.g., 1 race instead of 13)

8. **`requirements.txt`** (UPDATED)
   - Added `fastf1>=3.0.0` for optional live data updates

### 2026 Grid Changes

The 2026 season features significant grid changes:

**New Teams:**
- Audi (replacing Sauber)
- Cadillac (new 11th team)

**New Drivers:**
- Andrea Kimi Antonelli (Mercedes) - Rookie
- Isack Hadjar (Red Bull Racing) - Rookie
- Arvid Lindblad (Racing Bulls) - Rookie
- Oliver Bearman (Haas)
- Franco Colapinto (Alpine)
- Gabriel Bortoleto (Audi) - Rookie

**Notable Transfers:**
- Lewis Hamilton: Mercedes → Ferrari
- Carlos Sainz: Ferrari → Williams
- Sergio Perez: Red Bull → Cadillac
- Alexander Albon: stays at Williams

## How to Use

### Run 2026 Predictions

```bash
cd src
python f1_2026_predictor.py
```

### Update After New Race (Incremental - Fast!)

```bash
cd scripts
python update_2026_standings.py  # Only fetches NEW races!
cd ../src
python f1_2026_predictor.py
```

### Update Visualizations

```bash
python scripts/generate_readme_images.py
```

## Key Benefits of Incremental Update

**Before (inefficient):**
- Fetched all 13 races every time (1-2 minutes)
- Re-processed all historical data even if already up-to-date

**After (efficient):**
- Reads current round from CSV
- Only checks next 4-5 rounds for new data
- Only fetches and processes NEW races
- Example: If updated through Round 13 and Round 14 completes, only fetches Round 14 (~10 seconds)

## Files Not Changed

- `src/f1_2025_predictor.py` - Kept as legacy reference
- `src/data_collector.py` - Still used for historical data
- `src/model_evaluator.py` - Still used for model evaluation
- `notebooks/f1_2025_prediction.ipynb` - Kept as exploratory reference

## Season Status (Current)

- **Last completed race:** Round 13 - Italy GP
- **Current leader:** Andrea Kimi Antonelli (242 pts)
- **Races remaining:** 11
- **Maximum points remaining:** 275 (11 races × 25 pts per race)
- **Next race:** Round 14 - Spanish GP (data not yet available)

## Technical Notes

### Handling Missing Prior-Year Data

New drivers and teams (Antonelli, Hadjar, Lindblad, Bearman, Colapinto, Bortoleto, Audi, Cadillac) have no 2025 data. Feature engineering handles this by:
- Setting lagged features (`prev_year_position`, `prev_year_points`, `prev_year_wins`) to median values
- Using `driver_experience = 0` for rookies
- Relying on current season performance and team strength features

### Unicode Handling

All Unicode characters (🏎️, ✓, ✗, •, →) removed from `f1_2026_predictor.py` console output to avoid Windows cp1252 encoding errors.

## Testing Checklist

- [x] `f1_2026_predictor.py` runs without errors
- [x] Model trains on 2020-2025 data (excludes 2026)
- [x] Predictions generate for 2026 grid
- [x] CSV output saved to `results/f1_2026_championship_predictions.csv`
- [x] Report saved to `results/f1_2026_prediction_report.txt`
- [x] Model saved to `models/f1_championship_predictor_2026.joblib`
- [x] `update_2026_standings.py` reads current round correctly
- [x] Update script only fetches new races (incremental)
- [x] Update script auto-updates predictor constants
- [x] README.md updated with 2026 information

---

*Created: September 7, 2026*
