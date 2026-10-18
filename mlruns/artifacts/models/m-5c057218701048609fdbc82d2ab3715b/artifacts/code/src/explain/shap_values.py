import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge

from src.models.features import FEATURE_LABELS, FEATURES, prepare_features, source_feature
from src.models.forecaster import YieldForecaster
from src.models.mitigation import BoundedGroupLossRegressor

BACKGROUND_SIZE = 200


def _encoded(forecaster: YieldForecaster, frame: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    pipeline = forecaster.change_model_
    prepare = pipeline.named_steps["prepare"]
    encoded = pd.DataFrame(
        prepare.transform(prepare_features(frame)), columns=prepare.get_feature_names_out(), index=frame.index
    )
    return encoded, pipeline.named_steps["model"]


def _explain_estimator(estimator, encoded: pd.DataFrame, background: pd.DataFrame) -> tuple[np.ndarray, float]:
    if isinstance(estimator, Ridge):
        explainer = shap.LinearExplainer(estimator, background)
    else:
        explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(encoded))
    return values, float(np.ravel(explainer.expected_value)[0])


def background_sample(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sample(min(BACKGROUND_SIZE, len(frame)), random_state=0)[FEATURES]


def encoded_shap_values(forecaster: YieldForecaster, frame: pd.DataFrame, background_frame: pd.DataFrame):
    encoded, model = _encoded(forecaster, frame)
    background, _ = _encoded(forecaster, background_sample(background_frame))
    if isinstance(model, BoundedGroupLossRegressor):
        parts = [_explain_estimator(p, encoded, background) for p in model.predictors_]
        values = np.sum([w * v for w, (v, _) in zip(model.weights_, parts)], axis=0)
        base = float(np.sum([w * b for w, (_, b) in zip(model.weights_, parts)]))
    else:
        values, base = _explain_estimator(model, encoded, background)
    return pd.DataFrame(values, columns=encoded.columns, index=frame.index), base


def shap_explanation(forecaster: YieldForecaster, frame: pd.DataFrame, background_frame: pd.DataFrame) -> shap.Explanation:
    encoded_values, base = encoded_shap_values(forecaster, frame, background_frame)
    by_feature = encoded_values.T.groupby(encoded_values.columns.map(source_feature)).sum().T[FEATURES]
    return shap.Explanation(
        values=by_feature.to_numpy(),
        base_values=np.full(len(frame), base),
        data=prepare_features(frame).to_numpy(),
        feature_names=[FEATURE_LABELS[f] for f in FEATURES],
    )


def top_drivers(explanation: shap.Explanation, row: int, count: int = 3) -> list[dict]:
    values = explanation.values[row]
    order = np.argsort(-np.abs(values))[:count]
    return [
        {
            "feature": explanation.feature_names[i],
            "value": None if np.isnan(explanation.data[row, i]) else float(explanation.data[row, i]),
            "contribution_points": round(100 * float(values[i]), 2),
        }
        for i in order
    ]
