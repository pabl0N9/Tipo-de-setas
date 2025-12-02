import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

COLUMN_NAMES = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


class MushroomClassifier:
    """Wrapper para preprocesar y predecir con el modelo entrenado."""

    def __init__(self, model, feature_names: List[str], label_encoder: LabelEncoder):
        self.model = model
        self.feature_names = feature_names
        self.label_encoder = label_encoder

    def predict(self, input_data: Dict) -> Dict:
        input_df = pd.DataFrame([input_data])
        encoded = pd.get_dummies(input_df, prefix_sep="_", drop_first=False)

        for col in self.feature_names:
            if col not in encoded.columns:
                encoded[col] = 0

        encoded = encoded[self.feature_names]

        pred_idx = self.model.predict(encoded)[0]
        proba = self.model.predict_proba(encoded)[0]

        return {
            "prediction": self.label_encoder.inverse_transform([pred_idx])[0],
            "probability_edible": float(proba[0]),
            "probability_poisonous": float(proba[1]),
            "confidence": float(np.max(proba)),
        }


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    if "stalk-root" in df_clean.columns:
        missing_mask = df_clean["stalk-root"] == "?"
        if missing_mask.any():
            mode_value = df_clean.loc[~missing_mask, "stalk-root"].mode().iloc[0]
            df_clean.loc[missing_mask, "stalk-root"] = mode_value
    return df_clean


def encode_features(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["class"])
    X = pd.get_dummies(df.drop(columns=["class"]), prefix_sep="_", drop_first=False)
    return X, y, label_encoder


def split_data(X: pd.DataFrame, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def train_model() -> MushroomClassifier:
    df = load_data()
    df_clean = clean_dataset(df)
    X, y, label_encoder = encode_features(df_clean)
    X_train, _, y_train, _ = split_data(X, y)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    feature_names = X.columns.tolist()
    return MushroomClassifier(model, feature_names=feature_names, label_encoder=label_encoder)


def create_app(classifier: MushroomClassifier):
    app = Flask(__name__)

    @app.route("/health")
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["GET", "POST"])
    @app.route("/api/predict", methods=["GET", "POST"])
    def predict_endpoint():
        if request.method == "GET":
            return jsonify({"status": "ok", "message": "POST to this endpoint with mushroom features."})

        payload = request.get_json(force=True)
        result = classifier.predict(payload)
        return jsonify({"status": "success", "prediction": result})

    @app.route("/")
    def index():
        index_path = Path(__file__).parent / "public" / "index.html"
        return send_file(index_path)

    return app


classifier = train_model()
app = create_app(classifier)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
