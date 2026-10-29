"""
F1 Championship Prediction Machine Learning Model
Implements multiple algorithms to predict F1 championship winners
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, List, Optional
import logging
from datetime import datetime

# ML algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Model evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Custom modules
from feature_engineering_simple import F1FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1ChampionshipPredictor:
    """ML model for predicting F1 championship winners"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.ensemble_model = None
        self.feature_engineer = F1FeatureEngineer()
        self.is_trained = False
        self.feature_importance = None
        
    def initialize_models(self) -> Dict:
        """Initialize all ML models with default parameters"""
        
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                learning_rate=0.1,
                max_depth=6
            ),
            
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced',
                kernel='rbf'
            ),
            
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1
            ),
            
            'xgboost': XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=10  # Handle class imbalance
            ),
            
            'lightgbm': LGBMClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
        }
        
        return models
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                model_name: str, cv_folds: int = 3) -> Dict:
        """Optimize hyperparameters for a specific model"""
        
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            },
            
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [6, 8, 10]
            },
            
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'poly']
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return {}
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        # Use StratifiedKFold for cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=cv, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              optimize: bool = False) -> Dict[str, float]:
        """Train individual ML models"""
        
        self.models = self.initialize_models()
        model_scores = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Optimize hyperparameters if requested
                if optimize and model_name in ['random_forest', 'xgboost', 'gradient_boosting', 'svm']:
                    best_params = self.optimize_hyperparameters(X_train, y_train, model_name)
                    if best_params:
                        model.set_params(**best_params)
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
                model_scores[model_name] = cv_scores.mean()
                
                logger.info(f"{model_name} CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                model_scores[model_name] = 0.0
        
        return model_scores
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            top_models: int = 5) -> VotingClassifier:
        """Create an ensemble model from the best performing individual models"""
        
        logger.info("Creating ensemble model...")
        
        # Select top performing models
        model_scores = {}
        for name, model in self.models.items():
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
                model_scores[name] = cv_scores.mean()
            except:
                model_scores[name] = 0.0
        
        # Sort models by performance
        top_model_names = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:top_models]
        
        logger.info(f"Top {top_models} models for ensemble:")
        for name, score in top_model_names:
            logger.info(f"  {name}: {score:.4f}")
        
        # Create ensemble
        ensemble_estimators = [(name, self.models[name]) for name, _ in top_model_names if name in self.models]
        
        ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'  # Use probability-based voting
        )
        
        # Train ensemble
        ensemble_model.fit(X_train, y_train)
        
        return ensemble_model
    
    def calculate_feature_importance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature importance from tree-based models"""
        
        importance_data = []
        
        # Get importance from tree-based models
        tree_models = ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
        
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    for i, feature in enumerate(X_train.columns):
                        importance_data.append({
                            'model': model_name,
                            'feature': feature,
                            'importance': importance[i]
                        })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            # Average importance across models
            avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
            
            self.feature_importance = avg_importance
            return avg_importance.to_frame('importance').reset_index()
        
        return pd.DataFrame()
    
    def train(self, df: pd.DataFrame, target_col: str = 'is_champion', 
              optimize_hyperparameters: bool = False) -> Dict:
        """Train the complete F1 championship prediction model"""
        
        logger.info("Starting F1 championship prediction model training...")
        
        # Feature engineering
        logger.info("Engineering features...")
        features_df = self.feature_engineer.create_features(df)
        
        # Prepare training data
        X, y, feature_names = self.feature_engineer.prepare_training_data(features_df)
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train individual models
        model_scores = self.train_individual_models(X_train, y_train, optimize_hyperparameters)
        
        # Create ensemble model
        self.ensemble_model = self.create_ensemble_model(X_train, y_train)
        
        # Calculate feature importance
        feature_importance_df = self.calculate_feature_importance(X_train)
        
        # Evaluate on test set
        test_results = self.evaluate_models(X_test, y_test)
        
        self.is_trained = True
        
        results = {
            'model_scores': model_scores,
            'test_results': test_results,
            'feature_importance': feature_importance_df,
            'training_data_shape': X_train.shape,
            'test_data_shape': X_test.shape
        }
        
        logger.info("Model training completed successfully!")
        
        return results
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate all trained models on test set"""
        
        results = {}
        
        # Evaluate individual models
        for model_name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                results[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
                }
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        # Evaluate ensemble model
        if self.ensemble_model:
            try:
                y_pred_ensemble = self.ensemble_model.predict(X_test)
                y_proba_ensemble = self.ensemble_model.predict_proba(X_test)[:, 1]
                
                results['ensemble'] = {
                    'accuracy': accuracy_score(y_test, y_pred_ensemble),
                    'precision': precision_score(y_test, y_pred_ensemble),
                    'recall': recall_score(y_test, y_pred_ensemble),
                    'f1': f1_score(y_test, y_pred_ensemble),
                    'auc': roc_auc_score(y_test, y_proba_ensemble)
                }
            except Exception as e:
                logger.error(f"Error evaluating ensemble model: {e}")
        
        return results
    
    def predict_champion_probabilities(self, season_data: pd.DataFrame) -> pd.DataFrame:
        """Predict championship probabilities for drivers in a season"""
        
        if not self.is_trained or self.ensemble_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Feature engineering for prediction data
        features_df = self.feature_engineer.create_features(season_data)
        
        # Prepare prediction data
        X_pred = self.feature_engineer.prepare_prediction_data(features_df)
        
        # Get predictions from ensemble model
        probabilities = self.ensemble_model.predict_proba(X_pred)[:, 1]
        probability_total = probabilities.sum()
        if probability_total <= 0:
            raise ValueError("Model returned no positive championship probability")
        probabilities = probabilities / probability_total
        
        # Create results DataFrame
        results_df = season_data[['driver_name', 'constructor_name']].copy()
        results_df['championship_probability'] = probabilities
        results_df['predicted_champion'] = (probabilities == probabilities.max())
        
        # Sort by probability
        results_df = results_df.sort_values('championship_probability', ascending=False)
        
        return results_df
    
    def predict_champion_probabilities_with_history(self, combined_data: pd.DataFrame, prediction_year: int = 2025) -> pd.DataFrame:
        """Predict championship probabilities for drivers in prediction_year using historical data for context"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Feature engineering using the full dataset (historical + prediction year)
        features_df = self.feature_engineer.create_features(combined_data)
        
        # Prepare prediction data (this will filter to prediction_year only)
        X_pred = self.feature_engineer.prepare_prediction_data(features_df)
        
        # Get prediction year driver data for results
        season_data = combined_data[combined_data['year'] == prediction_year]
        
# Get predictions from ensemble model
        if self.ensemble_model is None:
            raise ValueError("Ensemble model is not available")
        probabilities = self.ensemble_model.predict_proba(X_pred)[:, 1]

        # ── Standings-driven correction (Option C) ─────────────────────────
        # The raw ensemble probabilities are near-noise for every driver but
        # the current leader (tiny 70-row training set, extreme class
        # imbalance), so they can rank e.g. a 0-point Aston Martin pair above
        # front-runners.  Two fixes applied here:
        #   1) Points-prior blend: anchor a softmax prior on current points
        #      and blend it with the model output so standings dominate.
        #   2) Monotonic re-rank: a driver with more points can never have a
        #      lower probability than a driver with fewer points.
        points = (
            season_data["final_points"].astype(float)
            if "final_points" in season_data.columns
            else season_data.get(
                "current_points",
                pd.Series(0.0, index=season_data.index),
            ).astype(float)
        ).to_numpy()

        max_pts = float(np.max(points)) if len(points) else 1.0
        temperature = max_pts / 4.0
        prior = np.exp((points - max_pts) / temperature)
        prior = prior / (prior.sum() or 1.0)

        MODEL_WEIGHT = 0.35
        blended = np.power(np.clip(probabilities, 1e-12, None), MODEL_WEIGHT) * np.power(
            prior, 1.0 - MODEL_WEIGHT
        )

        order = np.argsort(-points, kind="stable")
        mono = blended[order].copy()
        for i in range(1, len(mono)):
            if mono[i] > mono[i - 1]:
                mono[i] = mono[i - 1]
        blended = np.empty_like(mono)
        blended[order] = mono

        probability_total = blended.sum()
        if probability_total <= 0:
            raise ValueError("Model returned no positive championship probability")
        probabilities = blended / probability_total
        
        # Create results DataFrame
        results_df = season_data[['driver_name', 'constructor_name']].copy().reset_index(drop=True)
        results_df['championship_probability'] = probabilities
        results_df['predicted_champion'] = (probabilities == probabilities.max())
        
        # Sort by probability
        results_df = results_df.sort_values('championship_probability', ascending=False)
        
        return results_df
    
    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'individual_models': self.models,
            'feature_engineer': self.feature_engineer,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'training_timestamp': datetime.now()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        
        model_data = joblib.load(filepath)
        
        self.ensemble_model = model_data['ensemble_model']
        self.models = model_data['individual_models']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_importance = model_data.get('feature_importance')
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Training timestamp: {model_data.get('training_timestamp', 'Unknown')}")
    
    def plot_feature_importance(self, top_n: int = 15, figsize: Tuple[int, int] = (12, 8)):
        """Plot feature importance"""
        
        if self.feature_importance is None:
            logger.warning("No feature importance data available")
            return
        
        plt.figure(figsize=figsize)
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(f'Top {top_n} Most Important Features for F1 Championship Prediction')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()

    def predict_championship(self, season_data: pd.DataFrame) -> pd.DataFrame:
        """Predict championship probabilities for drivers in given season data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Use existing method
        return self.predict_champion_probabilities(season_data)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame"""
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train the model first.")
        
        if isinstance(self.feature_importance, pd.Series):
            return self.feature_importance.to_frame('importance')
        return self.feature_importance

if __name__ == "__main__":
    # Example usage
    try:
        # Load data
        df = pd.read_csv('../data/f1_championship_data.csv')
        
        # Initialize and train model
        predictor = F1ChampionshipPredictor()
        
        # Train the model
        results = predictor.train(df, optimize_hyperparameters=True)
        
        print("Training Results:")
        print("================")
        
        print("\nModel Performance (Cross-Validation F1 Scores):")
        for model, score in results['model_scores'].items():
            print(f"{model}: {score:.4f}")
        
        print("\nTest Set Performance:")
        for model, metrics in results['test_results'].items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
        
        print(f"\nTop 10 Most Important Features:")
        if not results['feature_importance'].empty:
            for idx, row in results['feature_importance'].head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save the model
        predictor.save_model('../models/f1_championship_predictor.joblib')
        
    except FileNotFoundError:
        print("No data file found. Run data_collector.py first to collect F1 data.")
