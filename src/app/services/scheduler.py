from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

_DAY_PAIRS = [
    ("lunes", "lun"),
    ("martes", "mar"),
    ("miercoles", "mie"),
    ("jueves", "jue"),
    ("viernes", "vie"),
    ("sabado", "sab"),
    ("domingo", "dom"),
    ("lun", "lun"),
    ("mar", "mar"),
    ("mie", "mie"),
    ("jue", "jue"),
    ("vie", "vie"),
    ("sab", "sab"),
    ("dom", "dom"),
    ("todos", "todos"),
]
DAY_MAP = dict(_DAY_PAIRS)

WEEKDAY_SHORT = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]


@dataclass
class HeatingBlock:
    dias: list[str]
    start: str
    end: str
    temp_on: float
    temp_off: float
    activo: bool = True


def normalize_day_name(day: str) -> str | None:
    return DAY_MAP.get(day.strip().lower())


def _expand_days(raw_days: list[str]) -> list[str]:
    normalized = [normalize_day_name(day) for day in raw_days]
    clean = [day for day in normalized if day]
    if "todos" in clean:
        return list(WEEKDAY_SHORT)
    return clean


def parse_blocks(raw_blocks: list[dict[str, Any]]) -> list[HeatingBlock]:
    blocks: list[HeatingBlock] = []
    for row in raw_blocks:
        try:
            # Formato nuevo: {horario:{inicio,fin}, dias:[], reglas:{encender..., apagar...}, activo}
            if "horario" in row and "reglas" in row:
                horario = row.get("horario", {})
                reglas = row.get("reglas", {})
                dias = _expand_days([str(value) for value in row.get("dias", [])])
                if not dias:
                    continue
                blocks.append(
                    HeatingBlock(
                        dias=dias,
                        start=str(horario["inicio"]),
                        end=str(horario["fin"]),
                        temp_on=float(reglas["encender_si_menor_que"]),
                        temp_off=float(reglas["apagar_si_mayor_que"]),
                        activo=bool(row.get("activo", True)),
                    )
                )
                continue

            # Formato legado: {dias,start,end,temp_on,temp_off}
            legacy_days = _expand_days([str(value) for value in row.get("dias", [])])
            if not legacy_days:
                continue
            blocks.append(
                HeatingBlock(
                    dias=legacy_days,
                    start=str(row["start"]),
                    end=str(row["end"]),
                    temp_on=float(row["temp_on"]),
                    temp_off=float(row["temp_off"]),
                    activo=bool(row.get("activo", True)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return blocks


def _to_minutes(hhmm: str) -> int:
    parts = hhmm.strip().split(":")
    if len(parts) != 2:
        raise ValueError("Formato de hora invalido. Usa HH:MM.")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("Hora fuera de rango.")
    return hour * 60 + minute


def _block_matches_time(block: HeatingBlock, now_minutes: int, day_short: str) -> bool:
    if not block.activo:
        return False
    if day_short not in block.dias:
        return False

    start_minutes = _to_minutes(block.start)
    end_minutes = _to_minutes(block.end)

    if start_minutes <= end_minutes:
        return start_minutes <= now_minutes <= end_minutes

    return now_minutes >= start_minutes or now_minutes <= end_minutes


def _intersects(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or b_end < a_start)


def _block_active_in_prediction_hour(block: HeatingBlock, timestamp: datetime) -> bool:
    if not block.activo:
        return False

    target_day = WEEKDAY_SHORT[timestamp.weekday()]
    window_start = timestamp.hour * 60
    window_end = window_start + 59

    start_minutes = _to_minutes(block.start)
    end_minutes = _to_minutes(block.end)

    for block_day in block.dias:
        day_index = WEEKDAY_SHORT.index(block_day)
        next_day = WEEKDAY_SHORT[(day_index + 1) % 7]

        if start_minutes <= end_minutes:
            if target_day == block_day and _intersects(start_minutes, end_minutes, window_start, window_end):
                return True
            continue

        # Tramo que cruza medianoche: [start..23:59] en el dia del bloque,
        # y [00:00..end] en el dia siguiente.
        if target_day == block_day and _intersects(start_minutes, 1439, window_start, window_end):
            return True
        if target_day == next_day and _intersects(0, end_minutes, window_start, window_end):
            return True

    return False


def is_heating_on_prediction_hour(
    timestamp: datetime,
    temp_interior: float,
    blocks: list[HeatingBlock],
    previously_on: bool = False,
) -> bool:
    active_blocks = [block for block in blocks if _block_active_in_prediction_hour(block, timestamp)]
    if not active_blocks:
        return False

    temp_on = min(block.temp_on for block in active_blocks)
    temp_off = min(block.temp_off for block in active_blocks)

    if temp_interior <= temp_on:
        return True
    if temp_interior >= temp_off:
        return False
    return previously_on


def is_heating_on(
    timestamp: datetime,
    temp_interior: float,
    blocks: list[HeatingBlock],
    previously_on: bool = False,
) -> bool:
    day_short = WEEKDAY_SHORT[timestamp.weekday()]
    now_minutes = timestamp.hour * 60 + timestamp.minute

    active_blocks = [block for block in blocks if _block_matches_time(block, now_minutes, day_short)]
    if not active_blocks:
        return False

    temp_on = min(block.temp_on for block in active_blocks)
    temp_off = min(block.temp_off for block in active_blocks)

    if temp_interior <= temp_on:
        return True
    if temp_interior >= temp_off:
        return False
    return previously_on
