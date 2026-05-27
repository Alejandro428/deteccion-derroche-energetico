from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import keras
import numpy as np
import pandas as pd


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
FEATURE_ORDER = [
    "calefaccion_on",
    "temperature_sensor",
    "temperatura",
    "temp_diff",
    "humidity_sensor",
    "humedad",
    "humidity_diff",
    "viento",
    "direccion_viento",
    "elevation",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
BASE_FIELDS = [
    "calefaccion_on",
    "temperature_sensor",
    "temperatura",
    "humidity_sensor",
    "humedad",
    "viento",
    "direccion_viento",
    "elevation",
]


@dataclass(frozen=True)
class PredictionResult:
    prediction: int
    probability: float
    threshold: float
    model_version: str


class InferenceService:
    @staticmethod
    def list_available_models() -> dict[str, Path]:
        """Lista modelos disponibles en src/model."""
        base_dir = Path(__file__).resolve().parent / "model"
        models = {}
        if base_dir.is_dir():
            for subdir in sorted(base_dir.iterdir()):
                if subdir.is_dir() and (subdir / "config.json").exists():
                    models[subdir.name] = subdir
        return models

    @staticmethod
    def select_model(model_name: str | None = None) -> Path:
        """Selecciona modelo: por defecto 'a', sino existe busca primero, o usa el pasado."""
        available = InferenceService.list_available_models()

        if not available:
            raise RuntimeError("No se encontraron modelos en src/model")

        if model_name:
            if model_name not in available:
                raise ValueError(f"Modelo '{model_name}' no existe. Disponibles: {list(available.keys())}")
            return available[model_name]

        if "a" in available:
            return available["a"]

        return available[next(iter(available))]

    def __init__(self, model_name: str | None = None):
        self.model_dir = self.select_model(model_name)
        
        model_path = self.model_dir / "model.keras"
        compat_h5_path = self.model_dir / "model_compat.h5"
        saved_keras_version = self._read_saved_keras_version(model_path)

        try:
            self.model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as exc:
            if compat_h5_path.exists():
                self.model = keras.models.load_model(compat_h5_path, compile=False, safe_mode=False)
            else:
                runtime_keras = getattr(keras, "__version__", "unknown")
                raise RuntimeError(
                    "No se pudo cargar 'model.keras'. "
                    f"Keras guardado: {saved_keras_version or 'desconocido'}, "
                    f"Keras runtime: {runtime_keras}."
                ) from exc

        self.scaler = joblib.load(self.model_dir / "scaler.pkl")

        config_path = self.model_dir / "config.json"
        with config_path.open("r", encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        self.threshold = float(self.config["threshold"])
        architecture = self.config.get("architecture", [])
        architecture_label = "_".join(str(value) for value in architecture)
        self.model_version = f"{self.model_dir.name}:{architecture_label}"

    @staticmethod
    def _read_saved_keras_version(model_path: Path) -> str | None:
        try:
            with zipfile.ZipFile(model_path, "r") as model_zip:
                metadata = json.loads(model_zip.read("metadata.json").decode("utf-8"))
                return metadata.get("keras_version")
        except Exception:
            return None

    @staticmethod
    def _parse_timestamp(raw_timestamp: Any) -> datetime:
        if not isinstance(raw_timestamp, str):
            raise ValueError("El campo 'timestamp' debe ser un string con formato YYYY-MM-DD HH:mm:ss.")

        try:
            return datetime.strptime(raw_timestamp, TIMESTAMP_FORMAT)
        except ValueError as exc:
            raise ValueError("Formato invalido en 'timestamp'. Usa YYYY-MM-DD HH:mm:ss.") from exc

    @staticmethod
    def _to_number(payload: dict[str, Any], field: str) -> float:
        if field not in payload:
            raise ValueError(f"Falta el campo requerido '{field}'.")

        value = payload[field]
        if isinstance(value, bool):
            raise ValueError(f"El campo '{field}' debe ser numerico, no booleano.")

        if not isinstance(value, (int, float)):
            raise ValueError(f"El campo '{field}' debe ser numerico (int o float).")

        return float(value)

    def _build_features(self, payload: dict[str, Any]) -> list[float]:
        timestamp = self._parse_timestamp(payload.get("timestamp"))

        values = {field: self._to_number(payload, field) for field in BASE_FIELDS}

        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        temp_diff = values["temperature_sensor"] - values["temperatura"]
        humidity_diff = values["humidity_sensor"] - values["humedad"]

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)

        feature_map = {
            "calefaccion_on": values["calefaccion_on"],
            "temperature_sensor": values["temperature_sensor"],
            "temperatura": values["temperatura"],
            "temp_diff": temp_diff,
            "humidity_sensor": values["humidity_sensor"],
            "humedad": values["humedad"],
            "humidity_diff": humidity_diff,
            "viento": values["viento"],
            "direccion_viento": values["direccion_viento"],
            "elevation": values["elevation"],
            "hour_sin": float(hour_sin),
            "hour_cos": float(hour_cos),
            "dow_sin": float(dow_sin),
            "dow_cos": float(dow_cos),
        }

        return [feature_map[name] for name in FEATURE_ORDER]

    def predict_from_payload(self, payload: dict[str, Any]) -> PredictionResult:
        if not isinstance(payload, dict):
            raise ValueError("El body debe ser un JSON valido.")

        features = self._build_features(payload)
        features_df = pd.DataFrame([features], columns=FEATURE_ORDER)
        scaled_features = self.scaler.transform(features_df)
        probability = float(self.model.predict(scaled_features, verbose=0)[0][0])
        prediction = int(probability > self.threshold)

        return PredictionResult(
            prediction=prediction,
            probability=probability,
            threshold=self.threshold,
            model_version=self.model_version,
        )

