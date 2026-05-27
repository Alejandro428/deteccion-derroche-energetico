from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request
from astral import LocationInfo
from astral.sun import elevation, sun

try:
    from .inference_service import BASE_FIELDS, InferenceService, TIMESTAMP_FORMAT
except ImportError:
    from inference_service import BASE_FIELDS, InferenceService, TIMESTAMP_FORMAT

try:
    import requests
except ImportError:
    requests = None


# Coordenadas de Mislata, Valencia
MISLATA_LAT = 39.475
MISLATA_LON = -0.418
MISLATA_ELEVATION = 31.0  # metros sobre el nivel del mar
MET_NO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
MISLATA_TZ = ZoneInfo("Europe/Madrid")
MISLATA_CITY = LocationInfo(name="Mislata", region="Valencia", timezone="Europe/Madrid", latitude=MISLATA_LAT, longitude=MISLATA_LON)


def _parse_target_timestamp(raw_timestamp: Any, *, default_next_hour: bool) -> datetime:
    if raw_timestamp is None:
        now_local = datetime.now(MISLATA_TZ)
        if default_next_hour:
            return (now_local + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return now_local

    if not isinstance(raw_timestamp, str):
        raise ValueError("El campo 'timestamp' debe ser string con formato YYYY-MM-DD HH:mm:ss.")

    try:
        return datetime.strptime(raw_timestamp, TIMESTAMP_FORMAT).replace(tzinfo=MISLATA_TZ)
    except ValueError as exc:
        raise ValueError("Formato invalido en 'timestamp'. Usa YYYY-MM-DD HH:mm:ss.") from exc


def _fmt_ts_local(value: datetime) -> str:
    return value.astimezone(MISLATA_TZ).strftime(TIMESTAMP_FORMAT)


def _fetch_mislata_timeseries() -> list[dict[str, Any]]:
    if not requests:
        raise RuntimeError("requests library not available. Install it.")

    params = {"lat": MISLATA_LAT, "lon": MISLATA_LON}
    headers = {"User-Agent": "CalefaccionPredictor/1.0"}
    response = requests.get(MET_NO_URL, params=params, headers=headers, timeout=10)
    response.raise_for_status()

    data = response.json()
    timeseries = data.get("properties", {}).get("timeseries", [])
    if not timeseries:
        raise ValueError("No timeseries data found in response")
    return timeseries


def _pick_timeseries_by_target(timeseries: list[dict[str, Any]], target_dt: datetime) -> dict[str, Any]:
    def _row_dt(row: dict[str, Any]) -> datetime:
        return datetime.fromisoformat(row["time"].replace("Z", "+00:00")).astimezone(MISLATA_TZ)

    return min(timeseries, key=lambda row: abs(_row_dt(row) - target_dt))


def get_mislata_weather(target_dt: datetime) -> dict[str, Any]:
    """Obtiene datos meteorológicos de Mislata para la hora más cercana al target."""
    try:
        timeseries = _fetch_mislata_timeseries()
        selected = _pick_timeseries_by_target(timeseries, target_dt)
        selected_dt = datetime.fromisoformat(selected["time"].replace("Z", "+00:00")).astimezone(MISLATA_TZ)
        details = selected["data"]["instant"]["details"]
        return {
            "target_timestamp": _fmt_ts_local(target_dt),
            "weather_timestamp": _fmt_ts_local(selected_dt),
            "temperatura": details.get("air_temperature"),
            "viento": details.get("wind_speed"),
            "direccion_viento": details.get("wind_from_direction"),
            "elevation": MISLATA_ELEVATION,
            "humedad": details.get("relative_humidity"),
            "presion": details.get("air_pressure_at_sea_level"),
        }
    except requests.RequestException as e:
        raise RuntimeError(f"Error fetching data from Met.no: {str(e)}")


def get_mislata_sun(target_dt: datetime) -> dict[str, Any]:
    day_sun = sun(MISLATA_CITY.observer, date=target_dt.date(), tzinfo=MISLATA_TZ)
    return {
        "target_timestamp": _fmt_ts_local(target_dt),
        "sunrise": _fmt_ts_local(day_sun["sunrise"]),
        "noon": _fmt_ts_local(day_sun["noon"]),
        "sunset": _fmt_ts_local(day_sun["sunset"]),
        "dawn": _fmt_ts_local(day_sun["dawn"]),
        "dusk": _fmt_ts_local(day_sun["dusk"]),
        "elevation": float(elevation(MISLATA_CITY.observer, target_dt)),
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[Any, int]:
        try:
            inference_service = InferenceService()
        except Exception as exc:
            return jsonify({"error": str(exc), "status": "error"}), 500

        return (
            jsonify(
                {
                    "status": "ok",
                    "model_version": inference_service.model_version,
                    "model": inference_service.model_dir.name,
                    "available_models": list(InferenceService.list_available_models().keys()),
                }
            ),
            200,
        )

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json."}), 415

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "JSON invalido en el body."}), 400

        model_name = payload.pop("model", None)

        try:
            inference_service = InferenceService(model_name)
        except (ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            result = inference_service.predict_from_payload(payload)
            target_timestamp = payload.get("timestamp")
            return (
                jsonify(
                    {
                        "target_timestamp": target_timestamp,
                        "prediction": result.prediction,
                        "derroche": bool(result.prediction),
                        "probability": result.probability,
                        "threshold": result.threshold,
                        "model_version": result.model_version,
                    }
                ),
                200,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422

    @app.get("/predict")
    def predict_help() -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "message": "Usa POST /predict con JSON.",
                    "required_fields": ["timestamp", *BASE_FIELDS],
                    "timestamp_format": TIMESTAMP_FORMAT,
                    "optional_fields": ["model"],
                    "available_models": list(InferenceService.list_available_models().keys()),
                }
            ),
            200,
        )

    @app.get("/models")
    def models() -> tuple[Any, int]:
        available = InferenceService.list_available_models()
        return (
            jsonify(
                {
                    "available_models": list(available.keys()),
                    "default_model": "a" if "a" in available else next(iter(available)) if available else None,
                }
            ),
            200,
        )

    @app.get("/mislata")
    def mislata() -> tuple[Any, int]:
        """Obtiene datos meteorológicos actuales de Mislata (Valencia)."""
        try:
            target_timestamp = request.args.get("timestamp")
            target_dt = _parse_target_timestamp(target_timestamp, default_next_hour=False)
            weather_data = get_mislata_weather(target_dt)
            return (jsonify(weather_data), 200)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/sun")
    def sun_data() -> tuple[Any, int]:
        """Obtiene datos solares de Mislata para timestamp dado o actual."""
        try:
            target_timestamp = request.args.get("timestamp")
            target_dt = _parse_target_timestamp(target_timestamp, default_next_hour=False)
            return jsonify(get_mislata_sun(target_dt)), 200
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/predict/mislata")
    def predict_mislata() -> tuple[Any, int]:
        """Predice usando sensores interiores + meteo Mislata + datos solares."""
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json."}), 415

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "JSON invalido en el body."}), 400

        model_name = payload.pop("model", None)
        raw_target_timestamp = payload.pop("timestamp", None)

        try:
            target_dt = _parse_target_timestamp(raw_target_timestamp, default_next_hour=True)
            target_timestamp = _fmt_ts_local(target_dt)
            weather_data = get_mislata_weather(target_dt)
            sun_data_obj = get_mislata_sun(target_dt)

            full_payload = {
                "timestamp": target_timestamp,
                "calefaccion_on": payload["calefaccion_on"],
                "temperature_sensor": payload["temperature_sensor"],
                "temperatura": weather_data["temperatura"],
                "humidity_sensor": payload["humidity_sensor"],
                "humedad": weather_data["humedad"],
                "viento": weather_data["viento"],
                "direccion_viento": weather_data["direccion_viento"],
                "elevation": sun_data_obj["elevation"],
            }

            inference_service = InferenceService(model_name)
            result = inference_service.predict_from_payload(full_payload)

            return (
                jsonify(
                    {
                        "target_timestamp": target_timestamp,
                        "prediction": result.prediction,
                        "derroche": bool(result.prediction),
                        "probability": result.probability,
                        "threshold": result.threshold,
                        "model_version": result.model_version,
                        "inputs_used": {
                            "calefaccion_on": full_payload["calefaccion_on"],
                            "temperature_sensor": full_payload["temperature_sensor"],
                            "humidity_sensor": full_payload["humidity_sensor"],
                            "temperatura": full_payload["temperatura"],
                            "humedad": full_payload["humedad"],
                            "viento": full_payload["viento"],
                            "direccion_viento": full_payload["direccion_viento"],
                            "elevation": full_payload["elevation"],
                            "weather_timestamp": weather_data["weather_timestamp"],
                        },
                    }
                ),
                200,
            )
        except KeyError as exc:
            return jsonify({"error": f"Falta campo requerido '{exc.args[0]}'. Usa: calefaccion_on, temperature_sensor, humidity_sensor."}), 422
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)

