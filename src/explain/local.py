import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.models.features import CATEGORICAL_FEATURES, FEATURE_LABELS, FEATURES, prepare_features
from src.models.forecaster import YieldForecaster

LIME_SAMPLES = 5000
CATEGORY_NAMES = {
    "ADMCON7": {1.0: "required", 3.0: "neither required nor recommended", 5.0: "considered but not required"},
    "CONTROL": {1.0: "public", 2.0: "private nonprofit", 3.0: "private for-profit"},
    "INSTSIZE": {1.0: "under 1,000", 2.0: "1,000-4,999", 3.0: "5,000-9,999", 4.0: "10,000-19,999", 5.0: "20,000+"},
}


def filled_features(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    features = prepare_features(frame)
    fill = {c: reference[c].mode().iloc[0] if c in CATEGORICAL_FEATURES else reference[c].median() for c in FEATURES}
    return features.fillna(fill)


class LocalExplainer:
    def __init__(self, forecaster: YieldForecaster, train_frame: pd.DataFrame, random_state: int = 42):
        self.forecaster = forecaster
        self.reference = prepare_features(train_frame)
        training = filled_features(train_frame, self.reference)
        categorical_index = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]
        self.category_values = {i: sorted(training[FEATURES[i]].unique()) for i in categorical_index}
        self.explainer = LimeTabularExplainer(
            training_data=self._encode_categories(training).to_numpy(),
            feature_names=[FEATURE_LABELS[f] for f in FEATURES],
            categorical_features=categorical_index,
            categorical_names={
                i: [CATEGORY_NAMES.get(FEATURES[i], {}).get(v, f"code {v:g}") for v in values]
                for i, values in self.category_values.items()
            },
            mode="regression",
            discretize_continuous=True,
            random_state=random_state,
        )

    def _encode_categories(self, features: pd.DataFrame) -> pd.DataFrame:
        encoded = features.copy()
        for i, values in self.category_values.items():
            column = FEATURES[i]
            encoded[column] = encoded[column].map({v: k for k, v in enumerate(values)}).fillna(0)
        return encoded

    def _decode_categories(self, array: np.ndarray) -> pd.DataFrame:
        decoded = pd.DataFrame(array, columns=FEATURES)
        for i, values in self.category_values.items():
            positions = np.clip(np.rint(decoded[FEATURES[i]]).astype(int), 0, len(values) - 1)
            decoded[FEATURES[i]] = np.asarray(values)[positions]
        return decoded

    def predict(self, array: np.ndarray) -> np.ndarray:
        return self.forecaster.predict(self._decode_categories(array))

    def explain(self, row: pd.DataFrame, num_features: int = 6) -> dict:
        instance = self._encode_categories(filled_features(row, self.reference)).to_numpy()[0]
        explanation = self.explainer.explain_instance(instance, self.predict, num_features=num_features,
                                                      num_samples=LIME_SAMPLES)
        return {
            "forecast_yield": float(self.forecaster.predict(row)[0]),
            "local_model_intercept": float(next(iter(explanation.intercept.values()))),
            "local_model_r2": float(explanation.score),
            "weights": [{"rule": rule, "weight_points": round(100 * weight, 2)} for rule, weight in explanation.as_list()],
        }
