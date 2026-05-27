from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class ApiClientError(RuntimeError):
    pass


@dataclass
class ApiClient:
    base_url: str = "http://localhost:8000"
    timeout: int = 12

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _raise_from_response(self, response: requests.Response) -> None:
        try:
            payload = response.json()
            message = payload.get("error") or payload.get("message") or str(payload)
        except ValueError:
            message = response.text or f"HTTP {response.status_code}"
        raise ApiClientError(message)

    def get_models(self) -> tuple[list[str], str | None]:
        response = requests.get(self._url("/models"), timeout=self.timeout)
        if response.status_code != 200:
            self._raise_from_response(response)
        data = response.json()
        return data.get("available_models", []), data.get("default_model")

    def get_mislata(self, timestamp: datetime) -> dict[str, Any]:
        response = requests.get(
            self._url("/mislata"),
            params={"timestamp": timestamp.strftime(TIMESTAMP_FORMAT)},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            self._raise_from_response(response)
        return response.json()

    def get_sun(self, timestamp: datetime) -> dict[str, Any]:
        response = requests.get(
            self._url("/sun"),
            params={"timestamp": timestamp.strftime(TIMESTAMP_FORMAT)},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            self._raise_from_response(response)
        return response.json()

    def predict_simple(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(self._url("/predict/mislata"), json=payload, timeout=self.timeout)
        if response.status_code != 200:
            self._raise_from_response(response)
        return response.json()

    def predict_advanced(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(self._url("/predict"), json=payload, timeout=self.timeout)
        if response.status_code != 200:
            self._raise_from_response(response)
        return response.json()

