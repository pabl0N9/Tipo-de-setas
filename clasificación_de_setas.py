import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

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


@dataclass
class ModelResult:
    name: str
    model: object
    accuracy: float
    y_pred: np.ndarray


class MushroomClassifier:
    """Wrapper para preprocesar y predecir con el modelo entrenado."""

    def __init__(self, model, feature_names: List[str], label_encoder: LabelEncoder):
        self.model = model
        self.feature_names = feature_names
        self.label_encoder = label_encoder

    def predict(self, input_data: Dict) -> Dict:
        """Recibe un diccionario con las caracteristicas del hongo y devuelve prediccion."""
        input_df = pd.DataFrame([input_data])
        encoded = pd.get_dummies(input_df, prefix_sep="_", drop_first=True)

        # Asegurar todas las columnas esperadas
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


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url, header=None, names=COLUMN_NAMES)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza valores '?' en stalk-root por la moda."""
    df_clean = df.copy()
    if "stalk-root" in df_clean.columns:
        missing_mask = df_clean["stalk-root"] == "?"
        if missing_mask.any():
            mode_value = df_clean.loc[~missing_mask, "stalk-root"].mode().iloc[0]
            df_clean.loc[missing_mask, "stalk-root"] = mode_value
            print(f"Valores '?' en stalk-root reemplazados por: '{mode_value}'")
    return df_clean


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Codifica la columna objetivo y aplica one-hot encoding a las features."""
    if "class" not in df.columns:
        raise ValueError("La columna 'class' no esta en el DataFrame.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["class"])
    X = pd.get_dummies(df.drop(columns=["class"]), prefix_sep="_", drop_first=True)

    print(f"Features despues de one-hot: {X.shape[1]}")
    print(f"Distribucion de clases: {np.bincount(y) / len(y)}")
    return X, y, label_encoder


def split_data(
    X: pd.DataFrame, y: np.ndarray, test_size: float = TEST_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def build_models() -> Dict[str, object]:
    """Modelos base a evaluar."""
    return {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Decision Tree": RandomForestClassifier(random_state=RANDOM_STATE, max_depth=5),
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "SVM": SVC(random_state=RANDOM_STATE, probability=True),
    }


def train_and_evaluate(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> List[ModelResult]:
    results: List[ModelResult] = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append(ModelResult(name=name, model=model, accuracy=acc, y_pred=y_pred))
        print(f"{name}: accuracy={acc:.4f}")

    results.sort(key=lambda r: r.accuracy, reverse=True)
    return results


def print_detailed_report(result: ModelResult, y_test: np.ndarray, target_names=None):
    """Imprime matriz de confusion y reporte de clasificacion."""
    if target_names is None:
        target_names = ["edible", "poisonous"]

    cm = confusion_matrix(y_test, result.y_pred)
    report = classification_report(
        y_test, result.y_pred, target_names=target_names, digits=4
    )

    print("\nMatriz de confusion:")
    print(cm)
    print("\nReporte de clasificacion:")
    print(report)


def print_feature_importance(model, feature_names: List[str], top_n: int = 15):
    """Muestra las features mas importantes si el modelo lo soporta."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    df_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    print(f"\nTop {top_n} caracteristicas por importancia:")
    print(df_importance.to_string(index=False, float_format="%.4f"))


def create_app(classifier: MushroomClassifier):
    """Crea una app Flask sencilla para servir predicciones y la interfaz web."""
    from flask import Flask, jsonify, request, send_file

    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        payload = request.get_json(force=True)
        result = classifier.predict(payload)
        return jsonify({"status": "success", "prediction": result})

    @app.route("/")
    def index():
        index_path = Path(__file__).parent / "index.html"
        return send_file(index_path)

    return app


def run_api(classifier: MushroomClassifier, port: int = 5000, use_ngrok: bool = False):
    """Inicia la API Flask. Ngrok es opcional y requiere flask-ngrok instalado."""
    app = create_app(classifier)
    if use_ngrok:
        try:
            from flask_ngrok import run_with_ngrok
        except ImportError:
            raise ImportError("Instala flask-ngrok o ejecuta sin --use-ngrok.")
        run_with_ngrok(app)
    app.run(port=port)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificador de setas (entrenamiento y API opcional)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Inicia la API Flask despues de entrenar el modelo",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Puerto para la API Flask"
    )
    parser.add_argument(
        "--use-ngrok",
        action="store_true",
        help="Usa flask-ngrok si esta instalado (solo para entornos Colab/ngrok)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_data()
    df_clean = clean_dataset(df)
    X, y, label_encoder = encode_features(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = build_models()
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    best_result = results[0]
    print(f"\nMejor modelo: {best_result.name} (accuracy={best_result.accuracy:.4f})")

    best_classifier = MushroomClassifier(
        best_result.model, feature_names=X.columns.tolist(), label_encoder=label_encoder
    )

    print_detailed_report(best_result, y_test, target_names=["edible", "poisonous"])
    print_feature_importance(best_result.model, X.columns.tolist(), top_n=15)

    if args.serve:
        print("\nIniciando API Flask...")
        run_api(best_classifier, port=args.port, use_ngrok=args.use_ngrok)


if __name__ == "__main__":
    main()
