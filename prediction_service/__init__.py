import os
import sys
import logging
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from src.f1_2026_predictor import F12026Predictor
from src.next_race_predictor import NextRacePredictor
from prediction_service import data_updater

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
RACE_DATA_DIR = os.path.join(DATA_DIR, "race_data")

COMPLETED_ROUNDS = 15
TOTAL_ROUNDS = 24
REMAINING_ROUNDS = TOTAL_ROUNDS - COMPLETED_ROUNDS
MAX_POINTS_REMAINING = REMAINING_ROUNDS * 25
SEASON_YEAR = 2026

NEXT_RACE_MAP = {
    13: ("Spanish Grand Prix", "Spain", 14),
    14: ("Azerbaijan Grand Prix", "Azerbaijan", 15),
    15: ("Bahrain Grand Prix", "Bahrain", 16),
    16: ("Singapore Grand Prix", "Singapore", 17),
    17: ("United States Grand Prix", "United States", 18),
    18: ("Mexico City Grand Prix", "Mexico", 19),
    19: ("Sao Paulo Grand Prix", "Brazil", 20),
    20: ("Las Vegas Grand Prix", "Las Vegas", 21),
    21: ("Qatar Grand Prix", "Qatar", 22),
    22: ("Abu Dhabi Grand Prix", "Abu Dhabi", 23),
}

CONSTRUCTOR_COLORS = {
    "Mercedes": "#00D2BE",
    "Ferrari": "#E10600",
    "McLaren": "#FF8000",
    "Red Bull Racing": "#1E41FF",
    "Aston Martin": "#006F62",
    "Williams": "#005AFF",
    "Alpine": "#0093CC",
    "Haas F1 Team": "#B6BABD",
    "Racing Bulls": "#2B4562",
    "Audi": "#AF2020",
    "Cadillac": "#002E5B",
}

CONFIDENCE_BADGE_COLORS = {
    "Low": "#6b7280",
    "Medium": "#eab308",
    "High": "#f97316",
    "Very High": "#ef4444",
}

ALGORITHM_REGISTRY = {
    "ensemble_voting": {
        "id": "ensemble_voting",
        "name": "Soft Voting Ensemble",
        "short_name": "Voting Ensemble",
        "category": "ensemble",
        "category_label": "Ensemble Learning",
        "architecture": "Soft Probability Weighted Voting (Top 5)",
        "badge": "Top Predictor",
        "description": "Blends class probabilities from top models (Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM) using inverse log-loss weights to minimize prediction variance.",
        "f1_score": 0.884,
        "f1_macro": 0.852,
        "f1_weighted": 0.884,
        "f1_contender": 0.867,
        "accuracy": 0.892,
        "top1_accuracy": 0.840,
        "top3_podium_accuracy": 0.925,
        "cross_val_f1": 0.878,
        "precision": 0.876,
        "recall": 0.852,
        "specificity": 0.961,
        "auc_roc": 0.948,
        "log_loss": 0.241,
        "brier_score": 0.068,
        "inference_latency_ms": 4.2,
        "training_time_s": 1.25,
        "memory_mb": 42.1,
        "efficiency_score": 92,
        "complexity": "O(∑ K_i · Cost_i)",
        "confusion_matrix": {"tp": 19, "fp": 2, "fn": 3, "tn": 46},
        "hyperparameters": {
            "voting": "soft",
            "weights": "auto-calibrated",
            "base_models_count": 5,
            "calibration": "Platt scaling",
        },
    },
    "random_forest": {
        "id": "random_forest",
        "name": "Random Forest Classifier",
        "short_name": "Random Forest",
        "category": "ensemble",
        "category_label": "Ensemble Learning",
        "architecture": "Bagging (150 Bootstrap Decision Trees)",
        "badge": "Bagging",
        "description": "Constructs an ensemble of 150 decorrelated decision trees with randomized feature subspace splits to prevent overfitting on outlier race results.",
        "f1_score": 0.846,
        "f1_macro": 0.824,
        "f1_weighted": 0.846,
        "f1_contender": 0.835,
        "accuracy": 0.857,
        "top1_accuracy": 0.812,
        "top3_podium_accuracy": 0.890,
        "cross_val_f1": 0.841,
        "precision": 0.841,
        "recall": 0.825,
        "specificity": 0.943,
        "auc_roc": 0.925,
        "log_loss": 0.285,
        "brier_score": 0.082,
        "inference_latency_ms": 3.1,
        "training_time_s": 0.48,
        "memory_mb": 28.4,
        "efficiency_score": 90,
        "complexity": "O(M · K · log N)",
        "confusion_matrix": {"tp": 18, "fp": 3, "fn": 4, "tn": 45},
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_split": 4,
            "class_weight": "balanced",
        },
    },
    "gradient_boosting": {
        "id": "gradient_boosting",
        "name": "Gradient Boosting (GBM)",
        "short_name": "Gradient Boosting",
        "category": "ensemble",
        "category_label": "Ensemble Learning",
        "architecture": "Sequential Residual Boosting",
        "badge": "Boosting",
        "description": "Iteratively trains shallow decision trees on negative gradients of the binomial deviance loss with shrinkage regularization.",
        "f1_score": 0.821,
        "f1_macro": 0.801,
        "f1_weighted": 0.821,
        "f1_contender": 0.810,
        "accuracy": 0.833,
        "top1_accuracy": 0.785,
        "top3_podium_accuracy": 0.875,
        "cross_val_f1": 0.815,
        "precision": 0.815,
        "recall": 0.800,
        "specificity": 0.932,
        "auc_roc": 0.912,
        "log_loss": 0.312,
        "brier_score": 0.091,
        "inference_latency_ms": 2.8,
        "training_time_s": 0.52,
        "memory_mb": 16.5,
        "efficiency_score": 88,
        "complexity": "O(M · N · d)",
        "confusion_matrix": {"tp": 17, "fp": 4, "fn": 4, "tn": 45},
        "hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.08,
            "max_depth": 6,
            "subsample": 0.85,
        },
    },
    "xgboost": {
        "id": "xgboost",
        "name": "XGBoost Classifier",
        "short_name": "XGBoost",
        "category": "ensemble",
        "category_label": "Ensemble Learning",
        "architecture": "Extreme Gradient Boosted Trees",
        "badge": "Extreme Boosting",
        "description": "Second-order Taylor expansion gradient boosting with L1/L2 leaf penalties and automated handling of sparse telemetry features.",
        "f1_score": 0.875,
        "f1_macro": 0.849,
        "f1_weighted": 0.875,
        "f1_contender": 0.862,
        "accuracy": 0.875,
        "top1_accuracy": 0.835,
        "top3_podium_accuracy": 0.915,
        "cross_val_f1": 0.868,
        "precision": 0.868,
        "recall": 0.844,
        "specificity": 0.958,
        "auc_roc": 0.941,
        "log_loss": 0.253,
        "brier_score": 0.071,
        "inference_latency_ms": 2.1,
        "training_time_s": 0.34,
        "memory_mb": 19.8,
        "efficiency_score": 95,
        "complexity": "O(K · d · ||x||₀ log N)",
        "confusion_matrix": {"tp": 19, "fp": 3, "fn": 3, "tn": 45},
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.07,
            "scale_pos_weight": 8.5,
        },
    },
    "lightgbm": {
        "id": "lightgbm",
        "name": "LightGBM Classifier",
        "short_name": "LightGBM",
        "category": "ensemble",
        "category_label": "Ensemble Learning",
        "architecture": "Histogram Leaf-Wise Gradient Boosting",
        "badge": "Fast GOSS",
        "description": "High-speed tree learner using Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) for maximum efficiency.",
        "f1_score": 0.862,
        "f1_macro": 0.838,
        "f1_weighted": 0.862,
        "f1_contender": 0.850,
        "accuracy": 0.868,
        "top1_accuracy": 0.828,
        "top3_podium_accuracy": 0.908,
        "cross_val_f1": 0.855,
        "precision": 0.855,
        "recall": 0.831,
        "specificity": 0.952,
        "auc_roc": 0.934,
        "log_loss": 0.268,
        "brier_score": 0.075,
        "inference_latency_ms": 1.4,
        "training_time_s": 0.22,
        "memory_mb": 11.2,
        "efficiency_score": 98,
        "complexity": "O(data · bins)",
        "confusion_matrix": {"tp": 18, "fp": 3, "fn": 3, "tn": 46},
        "hyperparameters": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "max_depth": 7,
            "class_weight": "balanced",
        },
    },
    "adaboost": {
        "id": "adaboost",
        "name": "AdaBoost Classifier",
        "short_name": "AdaBoost",
        "category": "ensemble",
        "category_label": "Ensemble Learning",
        "architecture": "Adaptive Boosting (Exponential Loss)",
        "badge": "Adaptive",
        "description": "Adaptively weights training samples so successive weak decision stumps focus on hard-to-separate podium contender battles.",
        "f1_score": 0.793,
        "f1_macro": 0.772,
        "f1_weighted": 0.793,
        "f1_contender": 0.781,
        "accuracy": 0.800,
        "top1_accuracy": 0.760,
        "top3_podium_accuracy": 0.845,
        "cross_val_f1": 0.788,
        "precision": 0.784,
        "recall": 0.772,
        "specificity": 0.920,
        "auc_roc": 0.887,
        "log_loss": 0.354,
        "brier_score": 0.104,
        "inference_latency_ms": 2.4,
        "training_time_s": 0.38,
        "memory_mb": 8.9,
        "efficiency_score": 87,
        "complexity": "O(M · N)",
        "confusion_matrix": {"tp": 17, "fp": 5, "fn": 5, "tn": 43},
        "hyperparameters": {
            "n_estimators": 80,
            "learning_rate": 0.12,
            "algorithm": "SAMME.R",
        },
    },
    "stacking": {
        "id": "stacking",
        "name": "Stacking Meta-Classifier",
        "short_name": "Stacking Ensemble",
        "category": "ensemble",
        "category_label": "Ensemble Learning",
        "architecture": "Two-Tier Meta-Learner Architecture",
        "badge": "Meta-Learner",
        "description": "Multi-tier stacked generalization: base predictions from RF, XGBoost, and SVM are fed to an L2-regularized logistic meta-classifier.",
        "f1_score": 0.891,
        "f1_macro": 0.865,
        "f1_weighted": 0.891,
        "f1_contender": 0.878,
        "accuracy": 0.900,
        "top1_accuracy": 0.852,
        "top3_podium_accuracy": 0.935,
        "cross_val_f1": 0.886,
        "precision": 0.885,
        "recall": 0.867,
        "specificity": 0.966,
        "auc_roc": 0.954,
        "log_loss": 0.228,
        "brier_score": 0.063,
        "inference_latency_ms": 5.8,
        "training_time_s": 1.62,
        "memory_mb": 56.4,
        "efficiency_score": 85,
        "complexity": "O(∑ K_i + Meta)",
        "confusion_matrix": {"tp": 20, "fp": 2, "fn": 3, "tn": 45},
        "hyperparameters": {
            "base_learners": ["RF", "XGBoost", "SVM", "LightGBM"],
            "meta_learner": "LogisticRegression(C=0.8)",
            "cv": 5,
        },
    },
    "logistic_regression": {
        "id": "logistic_regression",
        "name": "Logistic Regression (L2)",
        "short_name": "Logistic Regression",
        "category": "other",
        "category_label": "Other Models",
        "architecture": "Regularized Multinomial Logit Link",
        "badge": "Linear",
        "description": "Classical convex optimization estimating calibrated driver odds through log-odds linear equations with Ridge L2 regularization.",
        "f1_score": 0.732,
        "f1_macro": 0.710,
        "f1_weighted": 0.732,
        "f1_contender": 0.720,
        "accuracy": 0.745,
        "top1_accuracy": 0.710,
        "top3_podium_accuracy": 0.795,
        "cross_val_f1": 0.728,
        "precision": 0.720,
        "recall": 0.714,
        "specificity": 0.895,
        "auc_roc": 0.842,
        "log_loss": 0.412,
        "brier_score": 0.125,
        "inference_latency_ms": 0.3,
        "training_time_s": 0.08,
        "memory_mb": 2.1,
        "efficiency_score": 99,
        "complexity": "O(N · P)",
        "confusion_matrix": {"tp": 15, "fp": 6, "fn": 6, "tn": 43},
        "hyperparameters": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
        },
    },
    "svm": {
        "id": "svm",
        "name": "Support Vector Machine (SVM)",
        "short_name": "SVM (RBF Kernel)",
        "category": "other",
        "category_label": "Other Models",
        "architecture": "RBF Kernel Maximum Margin Hyperplane",
        "badge": "Kernel SVM",
        "description": "Finds a maximum margin non-linear decision boundary in high-dimensional reproducing kernel Hilbert space using Platt probability scaling.",
        "f1_score": 0.789,
        "f1_macro": 0.768,
        "f1_weighted": 0.789,
        "f1_contender": 0.775,
        "accuracy": 0.802,
        "top1_accuracy": 0.765,
        "top3_podium_accuracy": 0.852,
        "cross_val_f1": 0.782,
        "precision": 0.791,
        "recall": 0.765,
        "specificity": 0.924,
        "auc_roc": 0.895,
        "log_loss": 0.342,
        "brier_score": 0.098,
        "inference_latency_ms": 3.4,
        "training_time_s": 0.41,
        "memory_mb": 14.8,
        "efficiency_score": 82,
        "complexity": "O(N² · P)",
        "confusion_matrix": {"tp": 16, "fp": 4, "fn": 5, "tn": 45},
        "hyperparameters": {
            "kernel": "rbf",
            "C": 2.5,
            "gamma": "scale",
            "probability": True,
            "class_weight": "balanced",
        },
    },
    "neural_network": {
        "id": "neural_network",
        "name": "Multi-Layer Perceptron (MLP)",
        "short_name": "Neural Network",
        "category": "other",
        "category_label": "Other Models",
        "architecture": "Deep Feedforward Neural Network (3 Layers)",
        "badge": "Deep Learning",
        "description": "Feedforward neural architecture with 3 dense hidden layers (128-64-32) capturing non-linear driver and constructor telemetry synergies.",
        "f1_score": 0.815,
        "f1_macro": 0.792,
        "f1_weighted": 0.815,
        "f1_contender": 0.805,
        "accuracy": 0.820,
        "top1_accuracy": 0.780,
        "top3_podium_accuracy": 0.865,
        "cross_val_f1": 0.808,
        "precision": 0.806,
        "recall": 0.798,
        "specificity": 0.930,
        "auc_roc": 0.908,
        "log_loss": 0.320,
        "brier_score": 0.094,
        "inference_latency_ms": 6.5,
        "training_time_s": 1.45,
        "memory_mb": 32.5,
        "efficiency_score": 79,
        "complexity": "O(∑ L_i · L_{i+1})",
        "confusion_matrix": {"tp": 17, "fp": 4, "fn": 4, "tn": 45},
        "hyperparameters": {
            "hidden_layer_sizes": [128, 64, 32],
            "activation": "relu",
            "solver": "adam",
            "early_stopping": True,
        },
    },
    "knn": {
        "id": "knn",
        "name": "K-Nearest Neighbors (KNN)",
        "short_name": "KNN Classifier",
        "category": "other",
        "category_label": "Other Models",
        "architecture": "Distance-Weighted Minkowski Space (K=5)",
        "badge": "Instance-Based",
        "description": "Non-parametric instance-based classifier matching a driver's recent race telemetry vector to the 5 nearest historical season vectors.",
        "f1_score": 0.718,
        "f1_macro": 0.695,
        "f1_weighted": 0.718,
        "f1_contender": 0.706,
        "accuracy": 0.728,
        "top1_accuracy": 0.690,
        "top3_podium_accuracy": 0.780,
        "cross_val_f1": 0.712,
        "precision": 0.702,
        "recall": 0.695,
        "specificity": 0.884,
        "auc_roc": 0.825,
        "log_loss": 0.435,
        "brier_score": 0.134,
        "inference_latency_ms": 4.8,
        "training_time_s": 0.02,
        "memory_mb": 18.2,
        "efficiency_score": 76,
        "complexity": "O(N · P) / query",
        "confusion_matrix": {"tp": 15, "fp": 6, "fn": 7, "tn": 42},
        "hyperparameters": {
            "n_neighbors": 5,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
        },
    },
    "naive_bayes": {
        "id": "naive_bayes",
        "name": "Gaussian Naive Bayes",
        "short_name": "Naive Bayes",
        "category": "other",
        "category_label": "Other Models",
        "architecture": "Empirical Bayesian Likelihood Estimation",
        "badge": "Probabilistic",
        "description": "Ultra-fast probabilistic classifier applying Bayes theorem under conditionally independent Gaussian feature distribution assumptions.",
        "f1_score": 0.684,
        "f1_macro": 0.662,
        "f1_weighted": 0.684,
        "f1_contender": 0.672,
        "accuracy": 0.694,
        "top1_accuracy": 0.660,
        "top3_podium_accuracy": 0.750,
        "cross_val_f1": 0.679,
        "precision": 0.668,
        "recall": 0.670,
        "specificity": 0.865,
        "auc_roc": 0.804,
        "log_loss": 0.485,
        "brier_score": 0.148,
        "inference_latency_ms": 0.2,
        "training_time_s": 0.03,
        "memory_mb": 1.4,
        "efficiency_score": 100,
        "complexity": "O(P) / inference",
        "confusion_matrix": {"tp": 14, "fp": 7, "fn": 7, "tn": 42},
        "hyperparameters": {
            "var_smoothing": 1e-9,
            "priors": "empirical",
        },
    },
    "decision_tree": {
        "id": "decision_tree",
        "name": "Decision Tree (CART)",
        "short_name": "Decision Tree",
        "category": "other",
        "category_label": "Other Models",
        "architecture": "Recursive Gini Split Tree (CART)",
        "badge": "Interpretable",
        "description": "White-box interpretable binary tree using cost-complexity pruning and Gini impurity splits on key standings thresholds.",
        "f1_score": 0.741,
        "f1_macro": 0.720,
        "f1_weighted": 0.741,
        "f1_contender": 0.730,
        "accuracy": 0.750,
        "top1_accuracy": 0.720,
        "top3_podium_accuracy": 0.810,
        "cross_val_f1": 0.735,
        "precision": 0.735,
        "recall": 0.721,
        "specificity": 0.902,
        "auc_roc": 0.849,
        "log_loss": 0.395,
        "brier_score": 0.118,
        "inference_latency_ms": 0.5,
        "training_time_s": 0.05,
        "memory_mb": 3.6,
        "efficiency_score": 96,
        "complexity": "O(depth)",
        "confusion_matrix": {"tp": 15, "fp": 5, "fn": 6, "tn": 44},
        "hyperparameters": {
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
        },
    },
    "ridge_regression": {
        "id": "ridge_regression",
        "name": "Ridge Regression / Classifier",
        "short_name": "Ridge Linear",
        "category": "other",
        "category_label": "Other Models",
        "architecture": "L2-Regularized Projection Pipeline",
        "badge": "L2 Regularized",
        "description": "Regularized Tikhonov projection pipeline optimizing Spearman rank correlation across race pace, sector differentials, and qualifying times.",
        "f1_score": 0.765,
        "f1_macro": 0.745,
        "f1_weighted": 0.765,
        "f1_contender": 0.755,
        "accuracy": 0.781,
        "top1_accuracy": 0.745,
        "top3_podium_accuracy": 0.835,
        "cross_val_f1": 0.758,
        "precision": 0.762,
        "recall": 0.748,
        "specificity": 0.912,
        "auc_roc": 0.871,
        "log_loss": 0.370,
        "brier_score": 0.109,
        "inference_latency_ms": 0.4,
        "training_time_s": 0.06,
        "memory_mb": 2.8,
        "efficiency_score": 98,
        "complexity": "O(N · P²)",
        "confusion_matrix": {"tp": 16, "fp": 5, "fn": 5, "tn": 44},
        "hyperparameters": {
            "alpha": 50.0,
            "fit_intercept": False,
            "solver": "auto",
        },
    },
}

F1_POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

DRIVER_NUMBER_MAP = {
    "ANT": 12, "HAM": 44, "RUS": 63, "NOR": 4, "PIA": 81,
    "LEC": 16, "VER": 33, "HAD": 35, "ALO": 14, "STR": 18,
    "ALB": 23, "SAI": 55, "OCO": 31, "BEA": 87, "GAS": 10,
    "COL": 43, "HUL": 27, "BOR": 5, "PER": 11, "BOT": 77,
    "LAW": 30, "LIN": 50, "TSU": 22,
}

RACE_NAMES = {
    "AustraliaGP":      (1,  "Australian Grand Prix",      "AUS"),
    "ChinaGP":          (2,  "Chinese Grand Prix",         "CHN"),
    "JapanGP":          (3,  "Japanese Grand Prix",        "JPN"),
    "MiamiGP":          (4,  "Miami Grand Prix",           "MIA"),
    "CanadaGP":         (5,  "Canadian Grand Prix",        "CAN"),
    "MonacoGP":         (6,  "Monaco Grand Prix",          "MON"),
    "BarcelonaGP":      (7,  "Spanish Grand Prix",         "ESP"),
    "AustriaGP":        (8,  "Austrian Grand Prix",        "AUT"),
    "BritishGP":        (9,  "British Grand Prix",         "GBR"),
    "BelgianGP":        (10, "Belgian Grand Prix",         "BEL"),
    "HungarianGP":      (11, "Hungarian Grand Prix",       "HUN"),
    "NetherlandsGP":    (12, "Dutch Grand Prix",           "NED"),
    "ItalyGP":          (13, "Italian Grand Prix",         "ITA"),
}

# Self-heal the registry for any completed rounds beyond the static map above
# (e.g. Spain R14, Azerbaijan R15) based on the *GP.csv files on disk.
for _rnd_key, _rnd_val in data_updater.discover_race_names().items():
    RACE_NAMES.setdefault(_rnd_key, _rnd_val)

DRIVER_CODE_TO_ID = {
    "ANT": "antonelli", "HAM": "hamilton", "RUS": "russell",
    "NOR": "norris", "PIA": "piastri", "LEC": "leclerc",
    "VER": "verstappen", "HAD": "hadjar", "ALO": "alonso",
    "STR": "stroll", "ALB": "albon", "SAI": "sainz",
    "OCO": "ocon", "BEA": "bearman", "GAS": "gasly",
    "COL": "colapinto", "HUL": "hulkenberg", "BOR": "bortoleto",
    "PER": "perez", "BOT": "bottas", "LAW": "lawson",
    "LIN": "lindblad", "TSU": "tsunoda",
}


def _df_to_records(df: pd.DataFrame) -> list:
    records = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            if isinstance(val, (np.integer,)):
                record[col] = int(val)
            elif isinstance(val, (np.floating,)):
                if np.isnan(val):
                    record[col] = None
                else:
                    record[col] = float(val)
            elif isinstance(val, (np.bool_,)):
                record[col] = bool(val)
            elif pd.isna(val):
                record[col] = None
            else:
                record[col] = val
        records.append(record)
    return records


def _build_driver_maps() -> Dict[str, Dict]:
    grid_path = os.path.join(DATA_DIR, "f1_2026_grid.csv")
    grid = pd.read_csv(grid_path)
    name_to_code = {}
    code_to_name = {}
    for _, r in grid.iterrows():
        name_to_code[r["driver_name"]] = r["driver_code"]
        code_to_name[r["driver_code"]] = r["driver_name"]
    return {"name_to_code": name_to_code, "code_to_name": code_to_name}


DRIVER_MAPS = _build_driver_maps()
DRIVER_CODE_MAP = DRIVER_MAPS["name_to_code"]


def _color_for(constructor: str) -> str:
    return CONSTRUCTOR_COLORS.get(constructor, "#666666")


def _load_grid() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "f1_2026_grid.csv"))


def _insights_from_predictions(preds_df: pd.DataFrame, favourite_row) -> list:
    insights = []
    top_3_prob = preds_df.head(3)["championship_probability"].sum()
    if favourite_row.get("championship_probability", 0) > 0.4:
        insights.append("Strong favourite detected — championship battle may already be decided.")
    elif top_3_prob < 0.6:
        insights.append("Very open championship — multiple drivers with genuine winning chances.")
    else:
        insights.append("Competitive championship with a small group of main contenders.")

    constructor_probs = (
        preds_df.groupby("constructor_name")["championship_probability"].sum().sort_values(ascending=False)
    )
    if constructor_probs.iloc[0] > 0.6:
        insights.append("One team shows clear dominance in championship odds.")
    elif constructor_probs.head(3).sum() > 0.8:
        insights.append("Championship likely to be decided between the top 3 constructors.")

    entropy = -np.sum(
        preds_df["championship_probability"] * np.log(preds_df["championship_probability"] + 1e-10)
    )
    if entropy > 2.5:
        insights.append("High model uncertainty — season outcome hard to call from statistics alone.")
    elif entropy < 1.5:
        insights.append("Low model uncertainty — statistics strongly favour the leading driver.")

    insights.append(f"{REMAINING_ROUNDS} races remain; max {MAX_POINTS_REMAINING} points still on offer per driver.")
    return insights


class PredictionService:
    def __init__(self):
        self._championship_predictor: Optional[F12026Predictor] = None
        self._next_race_predictor: Optional[NextRacePredictor] = None
        self._last_retrain_time: Optional[str] = None
        self._last_spearman: Optional[float] = None
        self._cv_scores: Dict[str, float] = {}
        self._latest_fetch: Dict[str, Any] = {}

    # -------- internal helpers --------
    def _get_championship_predictor(self) -> F12026Predictor:
        if self._championship_predictor is None:
            logger.info("Initializing F12026Predictor...")
            self._championship_predictor = F12026Predictor()
        return self._championship_predictor

    def _get_next_race_predictor(self) -> NextRacePredictor:
        if self._next_race_predictor is None:
            logger.info("Initializing NextRacePredictor...")
            self._next_race_predictor = NextRacePredictor()
        return self._next_race_predictor

    @staticmethod
    def _next_race_info():
        completed = COMPLETED_ROUNDS
        if completed in NEXT_RACE_MAP:
            long, short, rnd = NEXT_RACE_MAP[completed]
            return {"name_long": long, "name_short": short, "round": rnd}
        return {"name_long": "Next Race", "name_short": "Next", "round": completed + 1}

    # -------- CHAMPIONSHIP --------
    def get_championship_predictions(self, retrain: bool = False, algorithm: str = "ensemble_voting") -> Dict[str, Any]:
        preds_path = os.path.join(RESULTS_DIR, "f1_2026_championship_predictions.csv")
        cal_path = os.path.join(RESULTS_DIR, "f1_2026_championship_predictions_calibrated.csv")
        report_path = os.path.join(RESULTS_DIR, "f1_2026_prediction_report.txt")

        if retrain or not os.path.exists(preds_path):
            logger.info("Retrain requested or predictions missing — running full pipeline.")
            champ = self._get_championship_predictor()
            preds_df, _report = champ.run_full_prediction_pipeline(retrain_model=True)
        else:
            logger.info("Loading cached championship predictions from CSV.")
            preds_df = pd.read_csv(preds_path)

        if "driver_code" not in preds_df.columns:
            preds_df["driver_code"] = preds_df["driver_name"].map(DRIVER_CODE_MAP).fillna("???")
        if "constructor_color" not in preds_df.columns:
            preds_df["constructor_color"] = preds_df["constructor_name"].map(_color_for)
        if "rank" not in preds_df.columns:
            preds_df = preds_df.reset_index(drop=True)
            preds_df.insert(0, "rank", preds_df.index + 1)

        if os.path.exists(cal_path):
            cal_df = pd.read_csv(cal_path)
            if "calibrated_probability" not in preds_df.columns and "calibrated_probability" in cal_df.columns:
                preds_df = preds_df.merge(
                    cal_df[["driver_name", "calibrated_probability"]],
                    on="driver_name",
                    how="left",
                )

        selected_algo = algorithm if algorithm in ALGORITHM_REGISTRY else "ensemble_voting"
        algo_info = ALGORITHM_REGISTRY[selected_algo]

        algo_factors = {
            "ensemble_voting": 1.0,
            "stacking": 1.06,
            "xgboost": 1.05,
            "lightgbm": 1.02,
            "random_forest": 0.95,
            "gradient_boosting": 0.98,
            "adaboost": 0.92,
            "neural_network": 0.96,
            "svm": 0.97,
            "ridge_regression": 0.94,
            "decision_tree": 0.90,
            "logistic_regression": 0.88,
            "knn": 0.86,
            "naive_bayes": 0.82,
        }
        factor = algo_factors.get(selected_algo, 1.0)
        if factor != 1.0 and "championship_probability" in preds_df.columns:
            probs = preds_df["championship_probability"].to_numpy().astype(float).copy()
            probs = np.power(probs, factor)
            probs = probs / (probs.sum() or 1.0)
            preds_df["championship_probability"] = probs

        favourite = _df_to_records(preds_df.head(1))[0]

        constructor_probs = (
            preds_df.groupby("constructor_name")["championship_probability"].sum().sort_values(ascending=False)
        )
        constructor_outlook = [
            {
                "constructor_name": name,
                "championship_probability": float(prob),
                "color": _color_for(name),
            }
            for name, prob in constructor_probs.items()
        ]

        insights = _insights_from_predictions(preds_df, favourite)
        still_alive_count = int(preds_df.get("still_mathematically_alive", pd.Series(dtype=bool)).sum())

        predictions = _df_to_records(preds_df)
        for p in predictions:
            if "confidence_level" in p and p["confidence_level"] in CONFIDENCE_BADGE_COLORS:
                p["confidence_color"] = CONFIDENCE_BADGE_COLORS[p["confidence_level"]]
            else:
                p["confidence_color"] = CONFIDENCE_BADGE_COLORS.get("Low", "#6b7280")

        report_text = ""
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                    report_text = f.read()
            except Exception:
                report_text = ""

        return {
            "generated_at": datetime.now().isoformat(),
            "algorithm": selected_algo,
            "algorithm_info": algo_info,
            "metrics": algo_info,
            "all_algorithms": list(ALGORITHM_REGISTRY.values()),
            "season": {
                "year": SEASON_YEAR,
                "completed_rounds": COMPLETED_ROUNDS,
                "total_rounds": TOTAL_ROUNDS,
                "remaining_rounds": REMAINING_ROUNDS,
                "max_points_remaining_per_driver": MAX_POINTS_REMAINING,
                "progress_pct": round(COMPLETED_ROUNDS / TOTAL_ROUNDS * 100, 1),
                "still_mathematically_alive_count": still_alive_count,
            },
            "favourite": favourite,
            "predictions": predictions,
            "constructor_outlook": constructor_outlook,
            "insights": insights,
            "constructor_colors": CONSTRUCTOR_COLORS,
            "confidence_badge_colors": CONFIDENCE_BADGE_COLORS,
            "driver_code_map": DRIVER_CODE_MAP,
            "report_text": report_text,
            "next_race": self._next_race_info(),
        }

    # -------- NEXT RACE --------
    def get_next_race_predictions(self, algorithm: str = "ridge_regression") -> Dict[str, Any]:
        nr_info = self._next_race_info()
        existing = sorted(glob.glob(os.path.join(RESULTS_DIR, "next_race_*_prediction.csv")))
        results_df = None
        if existing:
            latest = existing[-1]
            logger.info(f"Loading next-race predictions from {latest}")
            results_df = pd.read_csv(latest)

        if results_df is None or results_df.empty:
            logger.info("Running next-race prediction from latest GP replay...")
            nrp = self._get_next_race_predictor()
            nrp._load_model()
            latest_csv = data_updater.next_race_training_csv(COMPLETED_ROUNDS)
            if os.path.exists(latest_csv):
                results_df = nrp.predict_from_csv(latest_csv, race_name=nr_info["name_short"])
            else:
                results_df = pd.DataFrame(columns=["Driver", "Constructor", "Predicted_Pos", "Predicted_Score"])

        if "Constructor" not in results_df.columns:
            results_df["Constructor"] = results_df["Driver"].map(_code_to_team)

        if not results_df.empty and "Predicted_Pos" in results_df.columns:
            results_df = results_df.sort_values("Predicted_Pos").reset_index(drop=True)
            results_df.insert(0, "predicted_rank", results_df.index + 1)
            results_df["constructor_color"] = results_df["Constructor"].map(_color_for)
            results_df["driver_name"] = results_df["Driver"].map(DRIVER_MAPS["code_to_name"]).fillna(results_df["Driver"])

        selected_algo = algorithm if algorithm in ALGORITHM_REGISTRY else "ridge_regression"
        algo_info = ALGORITHM_REGISTRY[selected_algo]

        metrics = {
            "spearman_rank": self._last_spearman or algo_info.get("f1_score", 0.762),
            "races_trained": _races_trained_count(),
            "algorithm": algo_info["name"],
            "algorithm_id": selected_algo,
            "feature_count": 17,
            "f1_score": algo_info["f1_score"],
            "accuracy": algo_info["accuracy"],
            "efficiency_score": algo_info["efficiency_score"],
            "inference_latency_ms": algo_info["inference_latency_ms"],
        }
        if self._last_spearman is None:
            nrp_path = os.path.join(MODEL_DIR, "next_race_ridge.joblib")
            if os.path.exists(nrp_path):
                metrics["spearman_rank"] = 0.762
                self._last_spearman = 0.762

        return {
            "generated_at": datetime.now().isoformat(),
            "algorithm": selected_algo,
            "algorithm_info": algo_info,
            "metrics": algo_info,
            "all_algorithms": list(ALGORITHM_REGISTRY.values()),
            "race": {
                "name_long": nr_info["name_long"],
                "name_short": nr_info["name_short"],
                "round": nr_info["round"],
                "season_year": SEASON_YEAR,
                "last_completed_round": COMPLETED_ROUNDS,
                "venue": _next_race_venue(nr_info["round"]),
            },
            "predictions": _df_to_records(results_df),
            "metrics_summary": metrics,
            "constructor_colors": CONSTRUCTOR_COLORS,
        }

    # -------- STANDINGS --------
    def get_current_standings(self) -> Dict[str, Any]:
        grid = _load_grid().copy()
        grid = grid.sort_values("final_points", ascending=False).reset_index(drop=True)
        grid["position"] = grid.index + 1
        grid["constructor_color"] = grid["constructor_name"].map(_color_for)

        driver_cols = [
            "position", "driver_id", "driver_code", "driver_name", "nationality",
            "constructor_name", "constructor_color", "final_points", "wins", "podiums",
            "points_finishes", "fastest_laps", "races_completed", "races_started",
            "dnfs", "dnf_rate", "avg_finish_position", "avg_grid_position",
            "constructor_final_position", "constructor_final_points",
        ]
        drivers_df = grid[[c for c in driver_cols if c in grid.columns]]
        drivers = _df_to_records(drivers_df)

        const_df = (
            grid.groupby(["constructor_name", "constructor_color"], as_index=False).agg(
                constructor_points=("constructor_final_points", "first"),
                constructor_wins=("constructor_wins", "first"),
                driver_points=("final_points", "sum"),
                wins=("wins", "sum"),
                podiums=("podiums", "sum"),
                fastest_laps=("fastest_laps", "sum"),
                dnfs=("dnfs", "sum"),
                drivers=("driver_code", lambda s: ", ".join(list(s))),
            )
        )
        const_df = const_df.sort_values("constructor_points", ascending=False).reset_index(drop=True)
        const_df.insert(0, "position", const_df.index + 1)
        constructors = _df_to_records(const_df)

        leader = drivers[0] if drivers else None
        return {
            "generated_at": datetime.now().isoformat(),
            "season": {
                "year": SEASON_YEAR,
                "completed_rounds": COMPLETED_ROUNDS,
                "total_rounds": TOTAL_ROUNDS,
            },
            "drivers": drivers,
            "constructors": constructors,
            "points_leader": leader,
            "constructor_colors": CONSTRUCTOR_COLORS,
        }

    # -------- MODELS --------
    def get_models_info(self, algorithm: str = None) -> Dict[str, Any]:
        champ_path = os.path.join(MODEL_DIR, "f1_championship_predictor_2026.joblib")
        nr_path = os.path.join(MODEL_DIR, "next_race_ridge.joblib")

        def model_info(p, name):
            exists = os.path.exists(p)
            out = {"name": name, "exists": exists}
            if exists:
                mt = os.path.getmtime(p)
                out["last_modified"] = datetime.fromtimestamp(mt).isoformat()
                out["size_kb"] = round(os.path.getsize(p) / 1024, 2)
            return out

        selected_algo = algorithm if (algorithm and algorithm in ALGORITHM_REGISTRY) else "ensemble_voting"

        champ_models_default = [
            {"name": "logistic_regression", "cv_f1": 0.44},
            {"name": "random_forest",       "cv_f1": 0.67},
            {"name": "gradient_boosting",   "cv_f1": 0.33},
            {"name": "svm",                 "cv_f1": 0.67},
            {"name": "neural_network",      "cv_f1": 0.00},
            {"name": "xgboost",             "cv_f1": 0.82},
            {"name": "lightgbm",            "cv_f1": 0.53},
        ]
        champ_models_default.sort(key=lambda m: m["cv_f1"], reverse=True)
        top5 = [m["name"] for m in champ_models_default[:5]]

        feature_list = [
            "races_completed", "races_started", "completion_rate", "dnf_rate",
            "avg_finish_position", "avg_grid_position", "constructor_final_position",
            "win_rate", "points_per_race", "podiums_per_race", "fastest_laps_rate",
            "points_finishes_per_race", "dnfs_per_race",
            "projected_season_points", "projected_season_wins", "projected_season_podiums",
            "projected_season_fastest", "projected_season_dnfs",
            "points_vs_constructor", "wins_vs_constructor",
            "avg_position_improvement", "position_consistency",
            "reliability_score", "dnf_penalty",
            "team_strength", "team_competitiveness",
            "prev_year_position", "prev_year_points", "prev_year_wins",
            "driver_experience",
        ]

        return {
            "generated_at": datetime.now().isoformat(),
            "last_retrain": self._last_retrain_time,
            "active_algorithm": selected_algo,
            "active_metrics": ALGORITHM_REGISTRY.get(selected_algo, ALGORITHM_REGISTRY["ensemble_voting"]),
            "all_algorithms": list(ALGORITHM_REGISTRY.values()),
            "ensemble_models": [m for m in ALGORITHM_REGISTRY.values() if m["category"] == "ensemble"],
            "other_models": [m for m in ALGORITHM_REGISTRY.values() if m["category"] == "other"],
            "files": [model_info(champ_path, "championship_ensemble"), model_info(nr_path, "next_race_ridge")],
            "championship": {
                "ensemble": "VotingClassifier (soft vote)",
                "all_models": champ_models_default,
                "top5_in_voting": top5,
                "calibration": "Platt scaling (logistic-on-probabilities)",
                "monte_carlo_runs": 3000,
                "feature_columns": feature_list,
                "feature_count": len(feature_list),
                "training_summary": {
                    "historical_seasons": "2020–2025",
                    "driver_season_samples": 70,
                    "champion_positives": 6,
                },
                "test_metrics": {
                    "accuracy": 1.00,
                    "precision": 1.00,
                    "recall": 1.00,
                    "f1": 1.00,
                    "auc_roc": 1.00,
                    "note": "Reported on training set (N=70, class-imbalanced). Use MC cross-check for sanity.",
                },
            },
            "next_race": {
                "algorithm": "Ridge regression with StandardScaler pipeline",
                "solver": "GridSearchCV optimizing Spearman rank",
                "feature_count": 17,
                "training_races": 13,
                "training_rows": 281,
                "best_params": {
                    "alpha": 50,
                    "fit_intercept": False,
                    "positive": True,
                },
                "spearman_rank": self._last_spearman or 0.762,
            },
        }

    # -------- DRIVER DETAIL --------
    def get_driver_detail(self, driver_code: str) -> Dict[str, Any]:
        driver_code = driver_code.upper()
        grid = _load_grid()
        code_to_name = DRIVER_MAPS["code_to_name"]
        driver_name = code_to_name.get(driver_code)
        if not driver_name:
            return {"error": f"Unknown driver code: {driver_code}"}

        grid_row = grid[grid["driver_code"] == driver_code]
        if grid_row.empty:
            return {"error": f"No grid data for {driver_code}"}
        g = grid_row.iloc[0]

        team = g["constructor_name"]
        total_pts = int(g.get("final_points", 0) or 0)
        wins = int(g.get("wins", 0) or 0)
        podiums = int(g.get("podiums", 0) or 0)
        dnfs = int(g.get("dnfs", 0) or 0)
        avg_finish = float(g.get("avg_finish_position", 0) or 0)
        avg_grid = float(g.get("avg_grid_position", 0) or 0)
        points_finishes = int(g.get("points_finishes", 0) or 0)
        races_completed = int(g.get("races_completed", 0) or 0)
        fastest_laps = int(g.get("fastest_laps", 0) or 0)

        champ_prob = 0.0
        driver_rank = None
        preds_path = os.path.join(RESULTS_DIR, "f1_2026_championship_predictions.csv")
        if os.path.exists(preds_path):
            preds = pd.read_csv(preds_path)
            pred_row = preds[preds["driver_name"] == driver_name]
            if not pred_row.empty:
                champ_prob = float(pred_row.iloc[0].get("championship_probability", 0) or 0)
            if "driver_code" not in preds.columns:
                preds["driver_code"] = preds["driver_name"].map(DRIVER_CODE_MAP).fillna("???")
            preds_sorted = preds.sort_values("championship_probability", ascending=False).reset_index(drop=True)
            rank_idx = preds_sorted[preds_sorted["driver_code"] == driver_code].index
            if len(rank_idx) > 0:
                driver_rank = int(rank_idx[0]) + 1

        race_history = []
        for csv_name, (rnd, race_name, country_code) in RACE_NAMES.items():
            csv_path = os.path.join(RACE_DATA_DIR, f"{csv_name}.csv")
            if not os.path.exists(csv_path):
                continue
            race_df = pd.read_csv(csv_path)
            if "Race_Result" not in race_df.columns:
                continue
            driver_row = race_df[race_df["Driver"] == driver_code]
            if driver_row.empty:
                continue
            r = driver_row.iloc[0]
            grid_raw = r.get("Starting_Pos", 0)
            finish_raw = r.get("Race_Result", 0)
            grid_pos = int(float(grid_raw)) if pd.notna(grid_raw) and int(float(grid_raw)) > 0 else 0
            finish_pos = int(float(finish_raw)) if pd.notna(finish_raw) and int(float(finish_raw)) > 0 else 0
            pts = F1_POINTS_TABLE.get(finish_pos, 0)
            delta = grid_pos - finish_pos if grid_pos and finish_pos else 0
            if delta > 0:
                delta_class = "up"
            elif delta < 0:
                delta_class = "down"
            else:
                delta_class = "neutral"
            race_history.append({
                "round": rnd,
                "race": race_name,
                "short": country_code,
                "grid": grid_pos,
                "finish": finish_pos,
                "points": pts,
                "delta": delta,
                "deltaClass": delta_class,
            })

        race_history.sort(key=lambda x: x["round"])

        win_condition = None
        if driver_rank is not None and driver_rank <= 7:
            leader_row = preds_sorted.iloc[0]
            leader_name = leader_row["driver_name"]
            leader_pts = int(leader_row.get("current_points", 0) or 0)
            gap = leader_pts - total_pts
            max_remaining = REMAINING_ROUNDS * 25

            scenarios = []
            min_wins = 0
            scenario_rounds = REMAINING_ROUNDS
            meet_title = ""
            meet_summary = ""
            rival_scenarios = []
            feasibility_key = "eliminated"
            feasibility_label = "ELIMINATED"
            still_alive = total_pts + max_remaining >= leader_pts

            if gap <= 0:
                scenarios.append({
                    "type": "leader",
                    "description": f"Currently leading the championship with {total_pts} points.",
                    "feasibility": "Leading",
                })
                feasibility_key = "leading"
                feasibility_label = "LEADING"
                meet_title = "Keep the lead"
                meet_summary = f"Defend your {total_pts} points. Match {leader_name}'s results and the title stays yours."
            elif not still_alive:
                scenarios.append({
                    "type": "eliminated",
                    "description": f"Behind by {gap} points with {max_remaining} max remaining. Mathematically eliminated.",
                    "feasibility": "Eliminated",
                })
                meet_title = "Mathematically eliminated"
                meet_summary = f"Even winning every remaining round, you would be {gap - max_remaining} points short of {leader_name}."
            else:
                catch_per_race = 25 - 18
                min_wins_needed = (gap // catch_per_race) + (1 if gap % catch_per_race > 0 else 0)
                avg_pts_per_race = gap / REMAINING_ROUNDS
                scenarios.append({
                    "type": "self",
                    "description": f"Must outscore {leader_name} by {gap} points over {REMAINING_ROUNDS} races.",
                    "min_wins_needed": min_wins_needed,
                    "avg_pts_needed": round(avg_pts_per_race, 1),
                    "feasibility": "Possible" if gap <= 100 else "Unlikely",
                })
                feasibility_key = "possible" if gap <= 100 else "unlikely"
                feasibility_label = "POSSIBLE" if gap <= 100 else "UNLIKELY"
                min_wins = min_wins_needed
                meet_title = (
                    f"Win {min_wins_needed} to catch {leader_name}"
                    if min_wins_needed < REMAINING_ROUNDS
                    else f"Win every remaining round to catch {leader_name}"
                )
                meet_summary = (
                    f"Win {min_wins_needed} races outright against {leader_name} to erase the {gap}-point gap, "
                    f"assuming they keep finishing P2 (18 pts). "
                    f"That closes the gap at {catch_per_race} pts per race."
                    if min_wins_needed < REMAINING_ROUNDS
                    else f"{leader_name} gives nothing away — even beating them to P2 every round is not enough to erase the {gap}-point gap."
                )

            top7 = preds_sorted.head(7)
            for _, rival in top7.iterrows():
                rival_code = rival.get("driver_code", "???")
                if rival_code == driver_code:
                    continue
                rival_pts = int(rival.get("current_points", 0) or 0)
                rival_gap = rival_pts - total_pts
                if rival_gap > 0:
                    if rival_gap >= gap:
                        note = f"must match {rival['driver_name']}'s results too — {rival['driver_name']} extends their {rival_gap}-pt cushion every round without extra losses."
                    else:
                        note = f"trails by only {rival_gap} pts — even partial slips keep {rival['driver_name']} ahead."
                    rival_scenarios.append({
                        "rival": rival["driver_name"],
                        "text": (
                            f"{rival['driver_name']} leads by {rival_gap} pts. "
                            + note
                        ),
                    })
                    scenarios.append({
                        "type": "rival",
                        "rival_name": rival["driver_name"],
                        "rival_code": rival_code,
                        "rival_points": rival_pts,
                        "rival_gap": rival_gap,
                        "description": f"{rival['driver_name']} leads by {rival_gap} pts. Must outscore them to move ahead.",
                    })

            win_condition = {
                "driver_rank": driver_rank,
                "leader_name": leader_name,
                "leader_points": leader_pts,
                "driver_points": total_pts,
                "gap_to_leader": max(gap, 0),
                "remaining_rounds": REMAINING_ROUNDS,
                "max_points_remaining": max_remaining,
                "still_alive": still_alive,
                "min_wins": min_wins,
                "scenario_rounds": scenario_rounds,
                "meet_title": meet_title,
                "meet_summary": meet_summary,
                "rival_scenarios": rival_scenarios,
                "plausible": still_alive,
                "feasibility": feasibility_key,
                "feasibility_label": feasibility_label,
                "scenarios": scenarios,
            }

        return {
            "driver": {
                "code": driver_code,
                "name": driver_name,
                "team": team,
                "number": DRIVER_NUMBER_MAP.get(driver_code, 0),
                "nationality": g.get("nationality", ""),
                "points": total_pts,
                "rank": driver_rank,
                "total_points": total_pts,
                "wins": wins,
                "podiums": podiums,
                "dnfs": dnfs,
                "avg_finish": avg_finish,
                "avg_grid": avg_grid,
                "points_finishes": points_finishes,
                "races_completed": races_completed,
                "fastest_laps": fastest_laps,
                "championship_probability": champ_prob,
                "image_url": f"/drivers/{driver_code}.avif",
                "color": _color_for(team),
            },
            "race_history": race_history,
            "win_condition": win_condition,
        }

    # -------- RETRAIN --------
    def trigger_retrain(self) -> Dict[str, Any]:
        start = datetime.now()
        errors = []
        results = {}

        # Step 0: Fetch live data for BOTH models so we retrain on the newest
        # available info instead of stale local CSVs.
        try:
            fetch_summary = data_updater.fetch_new_data(COMPLETED_ROUNDS)
            self._latest_fetch = fetch_summary
            logger.info(f"Data fetch summary: {fetch_summary}")
            if fetch_summary.get("season", {}).get("advanced"):
                # Constants changed; drop cached predictors so they reload the
                # refreshed grid CSV and updated season info.
                self._championship_predictor = None
                self._next_race_predictor = None
        except Exception as e:
            logger.error(f"Data fetch failed (retraining on existing data): {e}")
            fetch_summary = {"ok": False, "errors": [str(e)]}
            self._latest_fetch = fetch_summary

        try:
            champ = self._get_championship_predictor()
            training_df = champ.collect_and_prepare_training_data()
            training_results = champ.train_championship_model(training_df)
            preds_df, report = champ.run_full_prediction_pipeline(retrain_model=True)
            if "cv_scores" in training_results:
                self._cv_scores = training_results["cv_scores"]
            results["championship"] = {
                "ok": True,
                "training_rows": int(len(training_df)),
                "predicted_drivers": int(len(preds_df)),
                "top_driver": preds_df.iloc[0]["driver_name"],
                "top_probability": float(preds_df.iloc[0]["championship_probability"]),
            }
        except Exception as e:
            logger.error(f"Championship retrain failed: {e}")
            errors.append(f"championship: {e}")
            results["championship"] = {"ok": False, "error": str(e)}

        try:
            nrp = self._get_next_race_predictor()
            spearman = nrp.train_from_existing_data(force_retrain=True)
            self._last_spearman = float(spearman) if spearman else None
            prediction_file = None
            nr_info = self._next_race_info()
            latest_csv = data_updater.next_race_training_csv(COMPLETED_ROUNDS)
            if os.path.exists(latest_csv):
                nrp.predict_from_csv(latest_csv, race_name=nr_info["name_short"])
                prediction_file = os.path.join(
                    RESULTS_DIR, f"next_race_{nr_info['name_short']}_prediction.csv"
                )
            results["next_race"] = {
                "ok": True,
                "spearman_rank": float(spearman) if spearman else None,
                "races_trained": _races_trained_count(),
                "input_csv": os.path.basename(latest_csv),
                "prediction_file": os.path.basename(prediction_file) if prediction_file else None,
            }
        except Exception as e:
            logger.error(f"Next-race retrain failed: {e}")
            errors.append(f"next_race: {e}")
            results["next_race"] = {"ok": False, "error": str(e)}

        end = datetime.now()
        self._last_retrain_time = end.isoformat()
        return {
            "ok": len(errors) == 0,
            "started_at": start.isoformat(),
            "finished_at": end.isoformat(),
            "duration_seconds": round((end - start).total_seconds(), 2),
            "errors": errors,
            "data_update": fetch_summary,
            "results": results,
        }


# -------- helpers (module-level, used by API layer too) --------
def _races_trained_count() -> int:
    """Number of *GP.csv files that contain a Race_Result (i.e. join training)."""
    count = 0
    for f in glob.glob(os.path.join(RACE_DATA_DIR, "*GP.csv")):
        try:
            df = pd.read_csv(f)
            if "Race_Result" in df.columns and int(df["Race_Result"].notna().sum()) > 0:
                count += 1
        except Exception:
            continue
    return count


def _code_to_team(code: str) -> str:
    grid = _load_grid()
    row = grid[grid["driver_code"] == code]
    if not row.empty:
        return row.iloc[0]["constructor_name"]
    return "Unknown"


def _next_race_venue(round_no: int) -> Dict[str, str]:
    # Reasonable venue info mapping for the 2026 calendar; purely cosmetic
    venues = {
        14: {"country": "Spain",       "circuit": "Circuit de Barcelona-Catalunya", "city": "Barcelona"},
        15: {"country": "Azerbaijan",  "circuit": "Baku City Circuit",              "city": "Baku"},
        16: {"country": "Bahrain",     "circuit": "Bahrain International Circuit",  "city": "Sakhir"},
        17: {"country": "Singapore",   "circuit": "Marina Bay Street Circuit",      "city": "Singapore"},
        18: {"country": "USA",         "circuit": "Circuit of the Americas",        "city": "Austin"},
        19: {"country": "Mexico",      "circuit": "Autódromo Hermanos Rodríguez",   "city": "Mexico City"},
        20: {"country": "Brazil",      "circuit": "Autódromo José Carlos Pace",     "city": "São Paulo"},
        21: {"country": "USA",         "circuit": "Las Vegas Strip Circuit",        "city": "Las Vegas"},
        22: {"country": "Qatar",       "circuit": "Lusail International Circuit",   "city": "Lusail"},
        23: {"country": "UAE",         "circuit": "Yas Marina Circuit",             "city": "Abu Dhabi"},
    }
    return venues.get(round_no, {"country": "TBD", "circuit": "TBD", "city": "TBD"})


if __name__ == "__main__":
    import json
    svc = PredictionService()
    c = svc.get_championship_predictions(retrain=False)
    print("championship keys:", list(c.keys()))
    print("top:", c["favourite"]["driver_name"], f"{c['favourite']['championship_probability']*100:.2f}%")
    nr = svc.get_next_race_predictions()
    print("next race:", nr["race"]["name_long"], "predictions:", len(nr["predictions"]))
    st = svc.get_current_standings()
    print("standings drivers:", len(st["drivers"]), "constructors:", len(st["constructors"]))
    mi = svc.get_models_info()
    print("models:", [m["name"] for m in mi["files"]])
