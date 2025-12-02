"""HTTP predictor handler for the mushroom classifier API."""

import json
from http.server import BaseHTTPRequestHandler
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
API_VERSION = "v1-no-drop-first"

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


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Codifica sin drop_first para que todas las categorías estén disponibles en inferencia."""
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["class"])
    X = pd.get_dummies(df.drop(columns=["class"]), prefix_sep="_")
    return X, y, label_encoder


def train_model() -> Tuple[RandomForestClassifier, List[str], LabelEncoder]:
    df = load_data()
    df_clean = clean_dataset(df)
    X, y, label_encoder = encode_features(df_clean)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X, y)
    feature_names = X.columns.tolist()
    return model, feature_names, label_encoder


MODEL, FEATURE_NAMES, LABEL_ENCODER = train_model()


def predict_one(input_data: Dict) -> Dict:
    input_df = pd.DataFrame([input_data])
    # No usamos drop_first para preservar las columnas usadas en entrenamiento
    encoded = pd.get_dummies(input_df, prefix_sep="_")

    for col in FEATURE_NAMES:
        if col not in encoded.columns:
            encoded[col] = 0

    encoded = encoded[FEATURE_NAMES]

    pred_idx = MODEL.predict(encoded)[0]
    probas = MODEL.predict_proba(encoded)[0]

    return {
        "prediction": LABEL_ENCODER.inverse_transform([pred_idx])[0],
        "probability_edible": float(probas[0]),
        "probability_poisonous": float(probas[1]),
        "confidence": float(np.max(probas)),
    }


class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        self._set_headers(200)
        message = {
            "status": "ok",
            "message": "POST to this endpoint with mushroom features.",
            "version": API_VERSION,
        }
        self.wfile.write(json.dumps(message).encode())

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length else b""

        try:
            payload = json.loads(raw_body) if raw_body else {}
            result = predict_one(payload)
            response = {"status": "success", "version": API_VERSION, "prediction": result}
            self._set_headers(200)
            self.wfile.write(json.dumps(response).encode())
        except Exception as exc:  # noqa: BLE001
            self._set_headers(400)
            error_message = {"status": "error", "version": API_VERSION, "message": str(exc)}
            self.wfile.write(json.dumps(error_message).encode())
