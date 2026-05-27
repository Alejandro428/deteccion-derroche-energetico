from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import flet as ft

from ..services.api_client import ApiClient, ApiClientError, TIMESTAMP_FORMAT
from ..services.config import load_config, save_config
from ..services.scheduler import WEEKDAY_SHORT, is_heating_on_prediction_hour, parse_blocks

DEFAULT_INTERIOR_TEMP = 21.0
DEFAULT_INTERIOR_HUM = 55.0
DEFAULT_OBTAINED_VALUES: dict[str, float] = {
    "temperatura": 20.0,
    "humedad": 60.0,
    "viento": 10.0,
    "direccion_viento": 90.0,
    "elevation": 35.0,
}
DAY_LABELS = [
    ("lun", "Lunes"),
    ("mar", "Martes"),
    ("mie", "Miercoles"),
    ("jue", "Jueves"),
    ("vie", "Viernes"),
    ("sab", "Sabado"),
    ("dom", "Domingo"),
]


@dataclass
class AppState:
    config: dict[str, Any]
    selected_date: date = field(default_factory=date.today)
    heating_program: dict[str, Any] = field(default_factory=lambda: {"programacion": []})
    heating_blocks: list[Any] = field(default_factory=list)
    last_valid_heating_program: dict[str, Any] | None = None
    auto_fetch_ok: bool = False


def _parse_float(value: str, label: str) -> float:
    try:
        return float(value.replace(",", ".").strip())
    except Exception as exc:
        raise ValueError(f"{label} debe ser numerico.") from exc


def _hour_options() -> list[ft.dropdown.Option]:
    return [ft.dropdown.Option(key=f"{hour:02d}:00", text=f"{hour:02d}:00") for hour in range(24)]


def _parse_hour(hour_text: str) -> time:
    hh = int(hour_text.split(":", 1)[0])
    return time(hour=hh, minute=0)


def _format_selected_date(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def _program_rows(program: dict[str, Any]) -> list[dict[str, Any]]:
    rows = program.get("programacion", [])
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _normalize_days(days: list[str]) -> list[str]:
    normalized: list[str] = []
    for day in days:
        d = str(day).strip().lower()
        if d == "todos":
            return list(WEEKDAY_SHORT)
        if d == "lunes":
            normalized.append("lun")
        elif d == "martes":
            normalized.append("mar")
        elif d == "miercoles":
            normalized.append("mie")
        elif d == "jueves":
            normalized.append("jue")
        elif d == "viernes":
            normalized.append("vie")
        elif d == "sabado":
            normalized.append("sab")
        elif d == "domingo":
            normalized.append("dom")
        elif d in WEEKDAY_SHORT:
            normalized.append(d)
    return normalized


def _bundled_heating_path() -> Path:
    return Path(__file__).resolve().parents[1] / "calefaccion.json"


def _chip(text: str, fg: str, bg: str) -> ft.Control:
    return ft.Container(
        padding=ft.padding.symmetric(horizontal=8, vertical=4),
        border_radius=999,
        bgcolor=bg,
        content=ft.Text(text, color=fg, size=12, weight=ft.FontWeight.W_600, no_wrap=True),
    )


def build_dashboard(page: ft.Page) -> None:
    api = ApiClient(base_url="http://localhost:8000")
    now_plus_1 = datetime.now() + timedelta(hours=1)
    state = AppState(config=load_config(), selected_date=now_plus_1.date())

    model_dropdown = ft.Dropdown(label="Modelo", width=170, options=[])
    date_label = ft.Text(value=_format_selected_date(state.selected_date), size=14)
    hour_dropdown = ft.Dropdown(label="Hora", width=130, options=_hour_options(), value=f"{now_plus_1.hour:02d}:00")

    temp_interior_input = ft.TextField(label="Temperatura interior (C)", width=220, value=f"{DEFAULT_INTERIOR_TEMP:.1f}")
    humidity_interior_input = ft.TextField(label="Humedad interior (%)", width=220, value=f"{DEFAULT_INTERIOR_HUM:.1f}")

    exterior_temp_input = ft.TextField(label="Temperatura exterior (C)", width=220, value=f"{DEFAULT_OBTAINED_VALUES['temperatura']:.1f}")
    exterior_humidity_input = ft.TextField(label="Humedad exterior (%)", width=220, value=f"{DEFAULT_OBTAINED_VALUES['humedad']:.1f}")
    wind_input = ft.TextField(label="Viento (km/h)", width=180, value=f"{DEFAULT_OBTAINED_VALUES['viento']:.1f}")
    wind_dir_input = ft.TextField(label="Direccion viento", width=190, value=f"{DEFAULT_OBTAINED_VALUES['direccion_viento']:.1f}")
    sun_elev_input = ft.TextField(label="Elevacion solar", width=180, value=f"{DEFAULT_OBTAINED_VALUES['elevation']:.1f}")

    auto_fetch_status = ft.Text(value="Esperando carga inicial de meteo/sol.", color=ft.Colors.GREY_500)
    auto_fetch_loader = ft.ProgressRing(width=18, height=18, visible=False)

    refresh_button = ft.ElevatedButton("Actualizar Meteo/Sol")
    clear_button = ft.OutlinedButton("Borrar datos")

    load_heating_button = ft.ElevatedButton("Cargar JSON calefacción")
    browse_heating_button = ft.ElevatedButton("Cargar programacion de calefacción")
    heating_json_path_input = ft.TextField(
        label="Ruta JSON calefacción",
        value=str(_bundled_heating_path())
    )
    heating_status = ft.Text(value="Sin programacion cargada.", color=ft.Colors.GREY_500)
    heating_table_container = ft.Column(controls=[], spacing=0, tight=True)

    predict_button = ft.ElevatedButton("Predecir")
    predict_multi_button = ft.OutlinedButton("Predecir multihora")
    exit_button = ft.IconButton(
        icon=ft.Icons.LOGOUT,
        icon_color=ft.Colors.RED_300,
        tooltip="Salir de la aplicacion",
    )

    result_status = ft.Text(value="Sin resultados.", color=ft.Colors.GREY_500)
    result_table_container = ft.Container(content=ft.Text("Aun no hay predicciones."), padding=ft.padding.only(top=6))

    date_picker = ft.DatePicker()
    page.overlay.append(date_picker)

    def toast(message: str, color: str = ft.Colors.BLUE_700) -> None:
        page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor=color)
        page.snack_bar.open = True
        page.update()

    def selected_dt() -> datetime | None:
        if not hour_dropdown.value:
            return None
        try:
            return datetime.combine(state.selected_date, _parse_hour(hour_dropdown.value))
        except Exception:
            return None

    def save_current_config() -> None:
        state.config["modelo"] = model_dropdown.value
        state.config["heating_program"] = state.last_valid_heating_program
        save_config(state.config)

    def set_fetch_status(message: str, ok: bool | None = None) -> None:
        auto_fetch_status.value = message
        if ok is True:
            auto_fetch_status.color = ft.Colors.GREEN_400
        elif ok is False:
            auto_fetch_status.color = ft.Colors.RED_300
        else:
            auto_fetch_status.color = ft.Colors.GREY_500

    def set_loading(loading: bool) -> None:
        auto_fetch_loader.visible = loading

    def set_obtained_values(values: dict[str, float]) -> None:
        exterior_temp_input.value = f"{values['temperatura']:.1f}"
        exterior_humidity_input.value = f"{values['humedad']:.1f}"
        wind_input.value = f"{values['viento']:.1f}"
        wind_dir_input.value = f"{values['direccion_viento']:.1f}"
        sun_elev_input.value = f"{values['elevation']:.1f}"

    def clear_results() -> None:
        result_status.value = "Sin resultados."
        result_status.color = ft.Colors.GREY_500
        result_table_container.content = ft.Text("Aun no hay predicciones.")

    def can_predict() -> tuple[bool, str]:
        if not model_dropdown.value:
            return False, "Selecciona un modelo."
        if not selected_dt():
            return False, "Fecha u hora invalidas."
        try:
            _parse_float(temp_interior_input.value, "Temperatura interior")
            _parse_float(humidity_interior_input.value, "Humedad interior")
            _parse_float(exterior_temp_input.value, "Temperatura exterior")
            _parse_float(exterior_humidity_input.value, "Humedad exterior")
            _parse_float(wind_input.value, "Viento")
            _parse_float(wind_dir_input.value, "Direccion viento")
            _parse_float(sun_elev_input.value, "Elevacion solar")
        except ValueError as exc:
            return False, str(exc)
        return True, ""

    def update_predict_enabled() -> None:
        enabled, _ = can_predict()
        predict_button.disabled = not enabled
        predict_multi_button.disabled = not enabled

    def read_current_obtained_values() -> dict[str, float]:
        return {
            "temperatura": _parse_float(exterior_temp_input.value, "Temperatura exterior"),
            "humedad": _parse_float(exterior_humidity_input.value, "Humedad exterior"),
            "viento": _parse_float(wind_input.value, "Viento"),
            "direccion_viento": _parse_float(wind_dir_input.value, "Direccion viento"),
            "elevation": _parse_float(sun_elev_input.value, "Elevacion solar"),
        }

    def fetch_auto_data(_: Any = None) -> None:
        dt = selected_dt()
        if not dt:
            set_fetch_status("Fecha u hora invalidas.", ok=False)
            state.auto_fetch_ok = False
            update_predict_enabled()
            page.update()
            return

        set_loading(True)
        set_fetch_status("Actualizando meteo y sol...", ok=None)
        page.update()

        try:
            weather = api.get_mislata(dt)
            sun_data = api.get_sun(dt)

            values = {
                "temperatura": float(weather.get("temperatura", DEFAULT_OBTAINED_VALUES["temperatura"])),
                "humedad": float(weather.get("humedad", DEFAULT_OBTAINED_VALUES["humedad"])),
                "viento": float(weather.get("viento", DEFAULT_OBTAINED_VALUES["viento"])),
                "direccion_viento": float(weather.get("direccion_viento", DEFAULT_OBTAINED_VALUES["direccion_viento"])),
                "elevation": float(sun_data.get("elevation", DEFAULT_OBTAINED_VALUES["elevation"])),
            }
            set_obtained_values(values)
            set_fetch_status("Datos meteo/sol actualizados.", ok=True)
            state.auto_fetch_ok = True
        except (ApiClientError, ValueError) as exc:
            state.auto_fetch_ok = False
            set_fetch_status(f"Fallo auto-fetch: {exc}", ok=False)
        finally:
            set_loading(False)
            update_predict_enabled()
            page.update()

    def on_clear_data(_: Any) -> None:
        temp_interior_input.value = f"{DEFAULT_INTERIOR_TEMP:.1f}"
        humidity_interior_input.value = f"{DEFAULT_INTERIOR_HUM:.1f}"
        set_obtained_values(DEFAULT_OBTAINED_VALUES)
        clear_results()
        set_fetch_status("Datos restablecidos.", ok=None)
        update_predict_enabled()
        page.update()

    def parse_program_or_raise(raw_program: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
        rows = _program_rows(raw_program)
        parsed_blocks = parse_blocks(rows)
        if not rows or not parsed_blocks:
            raise ValueError("JSON sin bloques validos de programacion.")
        clean_program = {"programacion": rows}
        return clean_program, parsed_blocks

    def render_heating_table() -> None:
        def update_heating_card_style() -> None:
            has_table = bool(heating_table_container.controls)
            heating_card.bgcolor = ft.Colors.SURFACE_CONTAINER if has_table else ft.Colors.TRANSPARENT
            heating_card.border = (
                ft.border.all(1, ft.Colors.with_opacity(0.1, ft.Colors.WHITE))
                if not has_table
                else None
            )

        _SHORT = [
            ("lun", "L"), ("mar", "Ma"), ("mie", "Mi"), ("jue", "J"),
            ("vie", "V"), ("sab", "Sa"), ("dom", "Do"),
        ]

        rows = _program_rows(state.heating_program)
        heating_table_container.controls.clear()
        if not rows:
            update_heating_card_style()
            return

        columns = [
            ft.DataColumn(label=ft.Text("Horario", size=12, weight=ft.FontWeight.W_600)),
        ] + [
            ft.DataColumn(label=ft.Text(h, size=12, weight=ft.FontWeight.W_600, text_align=ft.TextAlign.CENTER))
            for _, h in _SHORT
        ]
        rendered_rows: list[ft.DataRow] = []

        for row in rows:
            horario = row.get("horario", {})
            reglas = row.get("reglas", {})
            inicio = str(horario.get("inicio", "--:--"))
            fin = str(horario.get("fin", "--:--"))
            temp_on = reglas.get("encender_si_menor_que", "-")
            temp_off = reglas.get("apagar_si_mayor_que", "-")
            activo = bool(row.get("activo", True))
            days = _normalize_days([str(value) for value in row.get("dias", [])])

            dot_color = ft.Colors.GREEN_400 if activo else ft.Colors.GREY_600
            cells: list[ft.DataCell] = [
                ft.DataCell(
                    ft.Row(
                        [
                            ft.Text(f"{inicio}\n{fin}", size=11, weight=ft.FontWeight.W_600),
                            ft.Container(width=8, height=8, border_radius=999, bgcolor=dot_color),
                        ],
                        spacing=6,
                        tight=True,
                    )
                )
            ]
            for short, _ in _SHORT:
                if short in days:
                    on_col = ft.Colors.BLUE_200 if activo else ft.Colors.GREY_600
                    off_col = ft.Colors.ORANGE_200 if activo else ft.Colors.GREY_600
                    cells.append(
                        ft.DataCell(
                            ft.Column(
                                [
                                    ft.Text(f"↑{temp_on}", size=11, color=on_col),
                                    ft.Text(f"↓{temp_off}", size=11, color=off_col),
                                ],
                                spacing=2,
                                tight=True,
                            )
                        )
                    )
                else:
                    cells.append(ft.DataCell(ft.Text("—", color=ft.Colors.GREY_700, size=12)))
            rendered_rows.append(ft.DataRow(cells=cells))

        table = ft.DataTable(
            columns=columns,
            rows=rendered_rows,
            column_spacing=6,
            heading_row_height=36,
            data_row_min_height=44,
            data_row_max_height=56,
        )
        heating_table_container.controls.append(table)
        update_heating_card_style()

    def apply_heating_program(program: dict[str, Any], blocks: list[Any], message: str, ok: bool) -> None:
        state.heating_program = program
        state.heating_blocks = blocks
        state.last_valid_heating_program = program
        save_current_config()
        render_heating_table()
        heating_status.value = message
        heating_status.color = ft.Colors.GREEN_400 if ok else ft.Colors.RED_300

    def load_program_from_path(path: Path) -> tuple[dict[str, Any], list[Any]]:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise ValueError("El archivo JSON debe contener un objeto.")
        return parse_program_or_raise(raw_data)

    def initialize_heating_program() -> None:
        saved_program = state.config.get("heating_program")
        if isinstance(saved_program, dict):
            try:
                program, blocks = parse_program_or_raise(saved_program)
                apply_heating_program(program, blocks, "Programacion cargada desde configuracion local.", ok=True)
                return
            except ValueError:
                pass

        try:
            bundled_program, bundled_blocks = load_program_from_path(_bundled_heating_path())
            apply_heating_program(bundled_program, bundled_blocks, "Programacion precargada desde calefacción.json.", ok=True)
        except Exception as exc:
            state.heating_program = {"programacion": []}
            state.heating_blocks = []
            state.last_valid_heating_program = None
            heating_status.value = f"No se pudo cargar programacion: {exc}"
            heating_status.color = ft.Colors.RED_300
            render_heating_table()

    def on_load_heating_json(_: Any) -> None:
        json_path = (heating_json_path_input.value or "").strip()
        if not json_path:
            toast("Indica la ruta de un archivo JSON.", color=ft.Colors.RED_700)
            return
        try:
            program, blocks = load_program_from_path(Path(json_path))
            apply_heating_program(program, blocks, "Programacion cargada correctamente.", ok=True)
            toast("JSON de calefacción cargado.", color=ft.Colors.GREEN_700)
        except Exception:
            if state.last_valid_heating_program is not None:
                fallback_rows = _program_rows(state.last_valid_heating_program)
                state.heating_blocks = parse_blocks(fallback_rows)
                render_heating_table()
            heating_status.value = "Carga incorrecta, se cargan los datos por defecto"
            heating_status.color = ft.Colors.RED_300
            toast("Carga incorrecta, se cargan los datos por defecto", color=ft.Colors.RED_700)

        update_predict_enabled()
        page.update()

    def on_browse_heating_json(_: Any) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Selecciona un archivo JSON de calefacción",
                filetypes=[("JSON", "*.json")],
            )
            root.destroy()
        except Exception:
            toast("No se pudo abrir el explorador de archivos.", color=ft.Colors.RED_700)
            return

        if not file_path:
            return

        heating_json_path_input.value = file_path
        on_load_heating_json(None)

    async def on_exit_app(_: Any) -> None:
        await page.window.close()

    def heating_for_hour(target_dt: datetime, temp_interior: float, prev_on: bool = False) -> bool:
        return is_heating_on_prediction_hour(target_dt, temp_interior, state.heating_blocks, prev_on)

    def build_payload(target_dt: datetime, obtained: dict[str, float], heating_on: bool, elevation_value: float) -> dict[str, Any]:
        return {
            "timestamp": target_dt.strftime(TIMESTAMP_FORMAT),
            "model": model_dropdown.value,
            "calefaccion_on": 1 if heating_on else 0,
            "temperature_sensor": _parse_float(temp_interior_input.value, "Temperatura interior"),
            "humidity_sensor": _parse_float(humidity_interior_input.value, "Humedad interior"),
            "temperatura": obtained["temperatura"],
            "humedad": obtained["humedad"],
            "viento": obtained["viento"],
            "direccion_viento": obtained["direccion_viento"],
            "elevation": elevation_value,
        }

    def render_prediction_rows(rows: list[dict[str, Any]]) -> None:
        columns = [
            ft.DataColumn(label=ft.Text("Hora")),
            ft.DataColumn(label=ft.Text("Prediccion")),
            ft.DataColumn(label=ft.Text("Confianza")),
            ft.DataColumn(label=ft.Text("Probabilidad")),
            ft.DataColumn(label=ft.Text("Threshold")),
        ]
        rendered_rows: list[ft.DataRow] = []

        for row in rows:
            if row.get("error"):
                rendered_rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(row["hora"])),
                            ft.DataCell(ft.Text("Error", color=ft.Colors.AMBER_300)),
                            ft.DataCell(ft.Text("-")),
                            ft.DataCell(ft.Text("-")),
                            ft.DataCell(ft.Text("-")),
                        ]
                    )
                )
                continue

            is_alert = bool(row["alerta"])
            label = "Alerta" if is_alert else "Sin incidencia"
            color = ft.Colors.RED_300 if is_alert else ft.Colors.GREEN_300
            chip_bg = ft.Colors.with_opacity(0.2, ft.Colors.RED_700 if is_alert else ft.Colors.GREEN_700)
            probabilidad = float(row.get("probabilidad", 0.0))   # ya en 0-100
            threshold = float(row.get("threshold", 0.0))          # ya en 0-100
            # Confianza: si clase==1 (alerta) → probabilidad; si clase==0 → 100 - probabilidad
            confianza = probabilidad if is_alert else (100.0 - probabilidad)
            rendered_rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(row["hora"])),
                        ft.DataCell(_chip(label, color, chip_bg)),
                        ft.DataCell(
                            ft.Row(
                                [
                                    ft.Text(f"{confianza:.2f}%", weight=ft.FontWeight.W_600, color=color),
                                    ft.ProgressBar(
                                        value=max(0.0, min(confianza / 100.0, 1.0)),
                                        width=90,
                                        color=color,
                                    ),
                                ],
                                spacing=8,
                            )
                        ),
                        ft.DataCell(ft.Text(f"{probabilidad:.2f}%", color=ft.Colors.GREY_400)),
                        ft.DataCell(ft.Text(f"{threshold:.2f}%", color=ft.Colors.GREY_400)),
                    ]
                )
            )

        result_table_container.content = ft.Container(
            border=ft.border.all(1, ft.Colors.with_opacity(0.2, ft.Colors.WHITE)),
            border_radius=12,
            padding=10,
            content=ft.DataTable(columns=columns, rows=rendered_rows, column_spacing=20),
        )

    def run_single_prediction(_: Any) -> None:
        enabled, reason = can_predict()
        if not enabled:
            toast(reason, color=ft.Colors.RED_700)
            return

        dt = selected_dt()
        if not dt:
            toast("Fecha u hora invalidas.", color=ft.Colors.RED_700)
            return

        try:
            obtained = read_current_obtained_values()
            temp_interior = _parse_float(temp_interior_input.value, "Temperatura interior")
            heating_on = heating_for_hour(dt, temp_interior, prev_on=False)
            payload = build_payload(dt, obtained, heating_on, obtained["elevation"])
            response = api.predict_advanced(payload)

            render_prediction_rows(
                [
                    {
                        "hora": f"{dt.hour:02d}:00",
                        "alerta": bool(response.get("derroche")),
                        "probabilidad": float(response.get("probability", 0.0)) * 100,
                        "threshold": float(response.get("threshold", 0.0)) * 100,
                    }
                ]
            )
            result_status.value = f"Prediccion completada para {dt.strftime('%d/%m/%Y %H:00')}"
            result_status.color = ft.Colors.GREEN_400
        except (ApiClientError, ValueError) as exc:
            result_status.value = f"Error en prediccion: {exc}"
            result_status.color = ft.Colors.RED_300
            toast(str(exc), color=ft.Colors.RED_700)

        page.update()

    def run_multi_prediction(_: Any) -> None:
        enabled, reason = can_predict()
        if not enabled:
            toast(reason, color=ft.Colors.RED_700)
            return

        dt = selected_dt()
        if not dt:
            toast("Fecha u hora invalidas.", color=ft.Colors.RED_700)
            return

        rows: list[dict[str, Any]] = []
        obtained = read_current_obtained_values()
        temp_interior = _parse_float(temp_interior_input.value, "Temperatura interior")
        prev_on = False

        for offset in range(6):
            point_dt = dt + timedelta(hours=offset)
            try:
                sun_data = api.get_sun(point_dt)
                elevation_value = float(sun_data.get("elevation", obtained["elevation"]))
            except (ApiClientError, ValueError):
                elevation_value = obtained["elevation"]

            try:
                heating_on = heating_for_hour(point_dt, temp_interior, prev_on=prev_on)
                prev_on = heating_on
                payload = build_payload(point_dt, obtained, heating_on, elevation_value)
                response = api.predict_advanced(payload)
                rows.append(
                    {
                        "hora": f"{point_dt.hour:02d}:00",
                        "alerta": bool(response.get("derroche")),
                        "probabilidad": float(response.get("probability", 0.0)) * 100,
                        "threshold": float(response.get("threshold", 0.0)) * 100,
                    }
                )
            except (ApiClientError, ValueError):
                rows.append({"hora": f"{point_dt.hour:02d}:00", "error": True})

        render_prediction_rows(rows)
        result_status.value = f"Prediccion multihora completada desde {dt.strftime('%d/%m/%Y %H:00')}"
        result_status.color = ft.Colors.GREEN_400
        page.update()

    def load_models() -> None:
        try:
            models, default_model = api.get_models()
            model_dropdown.options = [ft.dropdown.Option(key=value, text=value) for value in models]
            preferred = state.config.get("modelo")
            if preferred in models:
                model_dropdown.value = preferred
            elif default_model in models:
                model_dropdown.value = default_model
            elif models:
                model_dropdown.value = models[0]
            else:
                model_dropdown.value = None
            save_current_config()
        except ApiClientError as exc:
            toast(f"No se pudieron cargar modelos: {exc}", color=ft.Colors.RED_700)

    def on_date_change(_: Any) -> None:
        if date_picker.value:
            v = date_picker.value
            # Flutter envía la fecha como medianoche UTC → Flet la deserializa
            # como datetime naive un día anterior en UTC+1/+2.
            # Sumamos 12h para moverla a mediodía UTC, lejos de cualquier
            # frontera de día independientemente de la zona horaria local.
            corrected = v + timedelta(hours=12)
            state.selected_date = date(corrected.year, corrected.month, corrected.day)
            date_label.value = _format_selected_date(state.selected_date)
            fetch_auto_data()
            page.update()

    def open_date_picker(_: Any) -> None:
        date_picker.open = True
        page.update()

    pick_date_button = ft.OutlinedButton("Seleccionar fecha", on_click=open_date_picker)

    def on_model_change(_: Any) -> None:
        save_current_config()
        update_predict_enabled()
        page.update()

    for control in [
        temp_interior_input,
        humidity_interior_input,
        exterior_temp_input,
        exterior_humidity_input,
        wind_input,
        wind_dir_input,
        sun_elev_input,
    ]:
        control.on_blur = lambda _: (update_predict_enabled(), page.update())

    date_picker.on_change = on_date_change
    hour_dropdown.on_change = lambda _: fetch_auto_data()
    model_dropdown.on_change = on_model_change
    refresh_button.on_click = fetch_auto_data
    clear_button.on_click = on_clear_data
    load_heating_button.on_click = on_load_heating_json
    browse_heating_button.on_click = on_browse_heating_json
    predict_button.on_click = run_single_prediction
    predict_multi_button.on_click = run_multi_prediction
    exit_button.on_click = on_exit_app

    inputs_card = ft.Container(
        padding=20,
        border_radius=14,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        content=ft.Column(
            [
                ft.Text("Introduce los datos", size=20, weight=ft.FontWeight.BOLD),
                ft.Row([pick_date_button, date_label, hour_dropdown, model_dropdown], wrap=True),
                ft.Row([temp_interior_input, humidity_interior_input], wrap=True),
                ft.Row([exterior_temp_input, exterior_humidity_input, wind_input], wrap=True),
                ft.Row([wind_dir_input, sun_elev_input], wrap=True),
                ft.Row([refresh_button, clear_button, auto_fetch_loader, auto_fetch_status], wrap=True),
            ],
            tight=True,
        ),
    )

    heating_card = ft.Container(
        padding=20,
        border_radius=14,
        bgcolor=ft.Colors.TRANSPARENT,
        border=ft.border.all(1, ft.Colors.with_opacity(0.1, ft.Colors.WHITE)),
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("Programacion de calefacción", size=18, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Row([browse_heating_button], wrap=True),
                heating_status,
                heating_table_container,
            ],
            tight=True,
        ),
    )

    actions_card = ft.Container(
        padding=20,
        border_radius=14,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        content=ft.Row([predict_button, predict_multi_button], alignment=ft.MainAxisAlignment.END),
    )

    result_card = ft.Container(
        padding=20,
        border_radius=14,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        content=ft.Column(
            [
                ft.Text("Resultados", size=18, weight=ft.FontWeight.BOLD),
                result_status,
                result_table_container,
            ],
            tight=True,
        ),
    )

    header = ft.Container(
        padding=ft.padding.only(top=20, left=8, right=8, bottom=4),
        content=ft.Row(
            [ft.Text("Prediccion de derroche energetico", size=30, weight=ft.FontWeight.BOLD)],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
    )

    main_column = ft.Column(
        controls=[inputs_card, heating_card, actions_card, result_card],
        spacing=14,
        width=960,
        tight=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    page.appbar = ft.AppBar(
        title=ft.Text("Prediccion de derroche energetico", weight=ft.FontWeight.BOLD),
        center_title=True,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        elevation=2,
        actions=[exit_button],
    )

    page.add(
        ft.Container(
            width=960,
            padding=ft.padding.only(bottom=24, top=12),
            content=main_column,
        )
    )

    load_models()
    initialize_heating_program()
    clear_results()
    fetch_auto_data()
    update_predict_enabled()
    page.update()
