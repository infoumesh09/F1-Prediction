"""
F1 2026 Championship Prediction Interface
Main interface for predicting the 2026 F1 World Championship winner
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from data_collector import F1DataCollector
from feature_engineering_simple import F1FeatureEngineer
from ml_model import F1ChampionshipPredictor
from model_evaluator import F1ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Keep all project file locations independent of the directory from which the
# script is launched.  This allows both `python src/f1_2026_predictor.py` and
# `cd src; python f1_2026_predictor.py` to work.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# ── 2026 season constants ──────────────────────────────────────────────────────
SEASON_YEAR = 2026
TOTAL_ROUNDS = 24   # 2026 calendar length
COMPLETED_ROUNDS = 15  # rounds completed as of last data update (Italy GP)
REMAINING_ROUNDS = TOTAL_ROUNDS - COMPLETED_ROUNDS
MAX_POINTS_REMAINING = REMAINING_ROUNDS * 25  # theoretical max (25 pts per win)
# ──────────────────────────────────────────────────────────────────────────────


class F12026Predictor:
    """Main interface for F1 2026 championship prediction"""

    def __init__(self):
        self.data_collector = F1DataCollector()
        self.predictor = F1ChampionshipPredictor()
        self.evaluator = F1ModelEvaluator()

        # 2026 F1 season data (current standings after Round 13)
        self.f1_2026_drivers = self.get_2026_season_data()

    # ── Data loading ────────────────────────────────────────────────────────────

    def get_2026_season_data(self) -> pd.DataFrame:
        """Load current 2026 F1 season driver and team data"""

        grid_file = os.path.join(DATA_DIR, 'f1_2026_grid.csv')
        df_2026 = pd.read_csv(grid_file)

        logger.info(f"Loaded 2026 F1 grid with {len(df_2026)} drivers")
        return df_2026

    def collect_and_prepare_training_data(self) -> pd.DataFrame:
        """Collect historical F1 data (2020-2025) and prepare for training"""

        logger.info("Collecting historical F1 data for training...")

        data_file = os.path.join(DATA_DIR, 'f1_championship_data.csv')
        if os.path.exists(data_file):
            logger.info("Loading existing F1 data...")
            df = pd.read_csv(data_file)
        else:
            logger.info("Collecting new F1 data from API...")
            df = self.data_collector.get_seasons_data(2010, 2025)

            if not df.empty:
                os.makedirs(DATA_DIR, exist_ok=True)
                self.data_collector.save_data(df, "f1_championship_data.csv")
            else:
                raise ValueError("Failed to collect F1 data")

        logger.info(
            f"Loaded data for {df['year'].nunique()} seasons "
            f"({df['year'].min()}–{df['year'].max()}) with {len(df)} driver records"
        )
        return df

    # ── Training ────────────────────────────────────────────────────────────────

    def train_championship_model(
        self, df: pd.DataFrame, optimize_hyperparameters: bool = False
    ) -> Dict:
        """Train the F1 championship prediction model on historical data"""

        logger.info("Training F1 championship prediction model...")

        # The data set contains only 70 historical driver-season records.
        # A large hyperparameter search is slow and unstable at this size, so
        # use the reproducible baseline by default.  Optimisation remains
        # available to callers who explicitly request it.
        results = self.predictor.train(
            df, optimize_hyperparameters=optimize_hyperparameters
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'f1_championship_predictor_2026.joblib')
        self.predictor.save_model(model_path)

        logger.info(f"Model trained and saved to {model_path}")
        return results

    # ── Prediction ──────────────────────────────────────────────────────────────

    def predict_2026_champion(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Predict the 2026 F1 World Championship winner"""

        logger.info("Making 2026 F1 championship predictions...")

        if not self.predictor.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Combine historical data with 2026 grid so lagged features can be built
        combined_data = pd.concat(
            [training_data, self.f1_2026_drivers],
            ignore_index=True
        )

        predictions_df = self.predictor.predict_champion_probabilities_with_history(
            combined_data, prediction_year=2026
        )

        logger.info(f"Predictions returned {len(predictions_df)} rows for {len(self.f1_2026_drivers)} drivers")

        # ── Extra analysis columns ────────────────────────────────────────────
        predictions_df['odds'] = 1 / predictions_df['championship_probability']
        predictions_df['confidence_level'] = pd.cut(
            predictions_df['championship_probability'],
            bins=[0, 0.05, 0.15, 0.35, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        # Merge current standings info for richer reporting
        # Some drivers may not appear in predictions_df if filtered out during feature engineering
        grid_info = self.f1_2026_drivers[
            ['driver_name', 'final_points', 'wins', 'races_completed',
             'races_started', 'podiums']
        ].copy()
        grid_info.rename(columns={
            'final_points': 'current_points',
            'wins':         'current_wins',
            'races_completed': 'races_done',
            'races_started':   'total_races',
            'podiums':         'current_podiums'
        }, inplace=True)

        # Left join to keep all predictions, fill missing grid data with 0
        predictions_df = predictions_df.merge(
            grid_info, on='driver_name', how='left'
        )
        
        # Fill any missing values with 0 (for drivers not in grid or vice versa)
        predictions_df['current_points'] = predictions_df['current_points'].fillna(0)
        predictions_df['current_wins'] = predictions_df['current_wins'].fillna(0)
        predictions_df['current_podiums'] = predictions_df['current_podiums'].fillna(0)

        # Points needed to mathematically win the championship
        leader_pts = predictions_df['current_points'].max()
        predictions_df['points_gap_to_leader'] = (
            leader_pts - predictions_df['current_points']
        )
        predictions_df['max_possible_points'] = (
            predictions_df['current_points'] + MAX_POINTS_REMAINING
        )
        predictions_df['still_mathematically_alive'] = (
            predictions_df['max_possible_points'] >= leader_pts
        )

        return predictions_df

    # ── Reporting ───────────────────────────────────────────────────────────────

    def generate_prediction_report(
        self,
        predictions_df: pd.DataFrame,
        training_results: Dict
    ) -> str:
        """Generate comprehensive 2026 championship prediction report"""

        report_lines = []
        sep = "=" * 100
        sub = "-" * 50

        report_lines += [
            sep,
            "F1 2026 WORLD CHAMPIONSHIP PREDICTION REPORT",
            sep,
            f"Generated on:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Prediction Model:  F1ChampionshipPredictor (Ensemble ML)",
            f"Season Progress:   Round {COMPLETED_ROUNDS} of {TOTAL_ROUNDS} "
            f"({REMAINING_ROUNDS} races remaining)",
            f"Max pts remaining: {MAX_POINTS_REMAINING} per driver",
            "",
        ]

        # ── Top 10 contenders ─────────────────────────────────────────────────
        report_lines += ["TOP 10 CHAMPIONSHIP CONTENDERS", sub]
        top_10 = predictions_df.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            prob_pct = row['championship_probability'] * 100
            alive    = "YES" if row.get('still_mathematically_alive', True) else "NO"
            cur_pts  = int(row.get('current_points', 0) or 0)
            report_lines.append(
                f"{i:2d}. {row['driver_name']:<22} ({row['constructor_name']:<15}) "
                f"{prob_pct:6.2f}%  Odds: {row['odds']:5.1f}/1  "
                f"Pts: {cur_pts:>3}  Alive: {alive}"
            )
        report_lines.append("")

        # ── Championship favourite ─────────────────────────────────────────────
        favourite = predictions_df.iloc[0]
        report_lines += [
            "CHAMPIONSHIP FAVOURITE ANALYSIS",
            "-" * 40,
            f"Predicted Champion:        {favourite['driver_name']}",
            f"Team:                      {favourite['constructor_name']}",
            f"Championship Probability:  {favourite['championship_probability']:.1%}",
            f"Betting Odds Equivalent:   {favourite['odds']:.1f}/1",
            f"Confidence Level:          {favourite['confidence_level']}",
            f"Current Points:            {int(favourite.get('current_points', 0) or 0)}",
            f"Current Wins:              {int(favourite.get('current_wins', 0) or 0)}",
            f"Points Gap to Leader:      {int(favourite.get('points_gap_to_leader', 0) or 0)}",
            "",
        ]

        # ── Mathematically eliminated ──────────────────────────────────────────
        eliminated = predictions_df[~predictions_df.get(
            'still_mathematically_alive', pd.Series([True] * len(predictions_df))
        )]
        if not eliminated.empty:
            report_lines += [
                "MATHEMATICALLY ELIMINATED DRIVERS",
                "-" * 40,
            ]
            for _, row in eliminated.iterrows():
                report_lines.append(
                    f"  * {row['driver_name']:<22} ({row['constructor_name']})"
                )
            report_lines.append("")

        # ── Constructor outlook ────────────────────────────────────────────────
        report_lines += ["CONSTRUCTOR CHAMPIONSHIP OUTLOOK", "-" * 45]
        constructor_probs = (
            predictions_df
            .groupby('constructor_name')['championship_probability']
            .sum()
            .sort_values(ascending=False)
        )
        for constructor, total_prob in constructor_probs.head(8).items():
            report_lines.append(f"  {constructor:<20}: {total_prob:.1%}")
        report_lines.append("")

        # ── Model performance ──────────────────────────────────────────────────
        report_lines += ["MODEL PERFORMANCE SUMMARY", "-" * 35]
        if 'test_results' in training_results and 'ensemble' in training_results['test_results']:
            em = training_results['test_results']['ensemble']
            report_lines += [
                f"  Model Accuracy:  {em.get('accuracy', 0):.1%}",
                f"  Precision:       {em.get('precision', 0):.1%}",
                f"  Recall:          {em.get('recall', 0):.1%}",
                f"  F1-Score:        {em.get('f1', 0):.1%}",
            ]
            if em.get('auc'):
                report_lines.append(f"  AUC-ROC:         {em['auc']:.3f}")
        report_lines.append(
            f"  Training Data:   {training_results.get('training_data_shape', 'N/A')}"
        )
        report_lines.append("")

        # ── Key insights ───────────────────────────────────────────────────────
        report_lines += ["KEY INSIGHTS & FACTORS", "-" * 30]
        top_3_prob = predictions_df.head(3)['championship_probability'].sum()

        if favourite['championship_probability'] > 0.4:
            report_lines.append(
                "* Strong favourite detected — championship battle may already be decided"
            )
        elif top_3_prob < 0.6:
            report_lines.append(
                "* Very open championship — multiple drivers with genuine winning chances"
            )
        else:
            report_lines.append(
                "* Competitive championship with a small group of main contenders"
            )

        top_constructor_prob = constructor_probs.iloc[0]
        if top_constructor_prob > 0.6:
            report_lines.append("* One team shows clear dominance in championship odds")
        elif constructor_probs.head(3).sum() > 0.8:
            report_lines.append(
                "* Championship likely to be decided between the top 3 constructors"
            )

        prediction_entropy = -np.sum(
            predictions_df['championship_probability'] *
            np.log(predictions_df['championship_probability'] + 1e-10)
        )
        if prediction_entropy > 2.5:
            report_lines.append(
                "* High model uncertainty — season outcome hard to call from statistics alone"
            )
        elif prediction_entropy < 1.5:
            report_lines.append(
                "* Low model uncertainty — statistics strongly favour the leading driver"
            )

        remaining_races_note = (
            f"* {REMAINING_ROUNDS} races remain; max {MAX_POINTS_REMAINING} points still on offer"
        )
        report_lines.append(remaining_races_note)
        report_lines.append("")

        # ── Disclaimers ────────────────────────────────────────────────────────
        report_lines += [
            "IMPORTANT DISCLAIMERS",
            "-" * 25,
            "* Predictions are based on historical data (2020-2025) and current 2026 season stats",
            f"* 2026 season data reflects standings after Round {COMPLETED_ROUNDS} (Italy GP)",
            "* Actual results may vary due to:",
            "  - Car development trajectories and regulation interpretations",
            "  - Driver form, team orders, and strategic decisions",
            "  - Injuries, penalties, retirements, and race incidents",
            "  - Weather conditions and circuit-specific performance",
            "* Model will be updated as 2026 season data becomes available",
            "",
            "For entertainment and analytical purposes only — not financial advice!",
            sep,
        ]

        return "\n".join(report_lines)

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def run_full_prediction_pipeline(
        self, retrain_model: bool = True
    ) -> Tuple[pd.DataFrame, str]:
        """Run the complete 2026 championship prediction pipeline"""

        logger.info("Starting F1 2026 championship prediction pipeline...")

        try:
            # Step 1: Load training data (historical seasons up to 2025)
            training_df = self.collect_and_prepare_training_data()

            # Step 2: Train model (or load existing)
            model_path = os.path.join(MODEL_DIR, 'f1_championship_predictor_2026.joblib')

            if retrain_model or not os.path.exists(model_path):
                training_results = self.train_championship_model(training_df)
            else:
                logger.info("Loading existing trained model...")
                self.predictor.load_model(model_path)
                training_results = {"message": "Loaded pre-trained model"}

            # Step 3: Predict 2026 champion
            predictions_df = self.predict_2026_champion(training_df)

            # Step 4: Generate report
            report = self.generate_prediction_report(predictions_df, training_results)

            # Step 5: Save results
            os.makedirs(RESULTS_DIR, exist_ok=True)

            predictions_path = os.path.join(RESULTS_DIR, 'f1_2026_championship_predictions.csv')
            predictions_df.to_csv(predictions_path, index=False)

            report_path = os.path.join(RESULTS_DIR, 'f1_2026_prediction_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"Predictions saved to {predictions_path}")
            logger.info(f"Report saved to {report_path}")

            return predictions_df, report

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            raise


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Main function to run F1 2026 championship prediction"""

    print("F1 2026 World Championship Prediction System")
    print("=" * 60)

    try:
        f1_predictor = F12026Predictor()
        predictions_df, report = f1_predictor.run_full_prediction_pipeline(
            retrain_model=True
        )

        print("\n" + report)

        print("\nPrediction Summary:")
        print("-" * 30)
        for i, (_, row) in enumerate(predictions_df.head(5).iterrows(), 1):
            pts = int(row.get('current_points', 0) or 0)
            print(
                f"{i}. {row['driver_name']} ({row['constructor_name']}) "
                f"— {row['championship_probability']:.1%}  [{pts} pts]"
            )

        print("\n[SUCCESS] Prediction complete! Check the 'results' folder for detailed outputs.")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("Please ensure the data files are present in the 'data' folder.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
