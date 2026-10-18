from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.models.features import build_preprocessor

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "neg_mean_absolute_error"


def ridge_refinement(best: dict) -> dict:
    alpha = best["model__alpha"]
    return {"model__alpha": [alpha / 3, alpha / 1.5, alpha, alpha * 1.5, alpha * 3]}


def forest_refinement(best: dict) -> dict:
    leaf = best["model__min_samples_leaf"]
    features = best["model__max_features"]
    return {
        **{key: [value] for key, value in best.items()},
        "model__min_samples_leaf": sorted({max(1, leaf - 2), leaf, leaf + 2, leaf + 5}),
        "model__max_features": sorted({round(min(1.0, max(0.1, features + step)), 2) for step in (-0.1, 0.0, 0.1)}),
    }


def boosting_refinement(best: dict) -> dict:
    rate = best["model__learning_rate"]
    leaves = best["model__num_leaves"]
    child = best["model__min_child_samples"]
    return {
        **{key: [value] for key, value in best.items()},
        "model__learning_rate": [round(rate * factor, 4) for factor in (0.7, 1.0, 1.4)],
        "model__num_leaves": sorted({max(3, leaves // 2), leaves, leaves * 2}),
        "model__min_child_samples": sorted({max(5, child // 2), child, child * 2}),
    }


@dataclass(frozen=True)
class ModelFamily:
    name: str
    scale_numeric: bool
    estimator: Callable[[], object]
    coarse_space: dict
    coarse_iterations: int
    refinement: Callable[[dict], dict]

    def pipeline(self) -> Pipeline:
        return Pipeline([("prepare", build_preprocessor(self.scale_numeric)), ("model", self.estimator())])


MODEL_FAMILIES = [
    ModelFamily(
        name="ridge",
        scale_numeric=True,
        estimator=lambda: Ridge(),
        coarse_space={"model__alpha": loguniform(1e-3, 1e3)},
        coarse_iterations=20,
        refinement=ridge_refinement,
    ),
    ModelFamily(
        name="random_forest",
        scale_numeric=False,
        estimator=lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        coarse_space={
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 3, 5, 10, 20],
            "model__max_features": [0.3, 0.5, 0.8, 1.0],
        },
        coarse_iterations=16,
        refinement=forest_refinement,
    ),
    ModelFamily(
        name="lightgbm",
        scale_numeric=False,
        estimator=lambda: LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
        coarse_space={
            "model__objective": ["l1", "l2", "huber"],
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__num_leaves": [7, 15, 31, 63],
            "model__min_child_samples": [10, 20, 40, 80],
            "model__n_estimators": [200, 400, 800],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__subsample_freq": [1],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [0.0, 1.0, 5.0],
        },
        coarse_iterations=30,
        refinement=boosting_refinement,
    ),
]


@dataclass(frozen=True)
class SearchStage:
    family: str
    stage: str
    results: pd.DataFrame
    best_params: dict

    @property
    def best_mae_points(self) -> float:
        return float(-100 * self.results["mean_test_score"].max())


def cross_validator() -> GroupKFold:
    return GroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def native(value):
    return value.item() if isinstance(value, np.generic) else value


def run_search(search, family: str, stage: str, X, y, institutions) -> SearchStage:
    search.fit(X, y, groups=institutions)
    best_params = {key: native(value) for key, value in search.best_params_.items()}
    return SearchStage(family, stage, pd.DataFrame(search.cv_results_), best_params)


def coarse_search(family: ModelFamily, X, y, institutions, iterations: int | None = None) -> SearchStage:
    search = RandomizedSearchCV(
        family.pipeline(),
        family.coarse_space,
        n_iter=iterations or family.coarse_iterations,
        scoring=SCORING,
        cv=cross_validator(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return run_search(search, family.name, "coarse", X, y, institutions)


def fine_search(family: ModelFamily, coarse: SearchStage, X, y, institutions) -> SearchStage:
    search = GridSearchCV(family.pipeline(), family.refinement(coarse.best_params), scoring=SCORING,
                          cv=cross_validator(), n_jobs=-1)
    return run_search(search, family.name, "fine", X, y, institutions)


def tuned_pipeline(family: ModelFamily, params: dict) -> Pipeline:
    return family.pipeline().set_params(**params)
