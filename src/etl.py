import os
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from etl_logger import ETLLogger

DATA_PATH = '../data/lake'
BRONZE_PATH = DATA_PATH + '/bronze'
SILVER_PATH = DATA_PATH + '/silver'
GOLD_PATH = DATA_PATH + '/gold'

SENSORS_PATH = DATA_PATH + '/sensors'
SENSORS_CALEFACTION = BRONZE_PATH + '/sensor_calefaccion'
SENSORS_CALEFACTION_RAW_NAME = 'History_2025-09-01_100-2026-03-02_1400.csv'

MODEL_PATH = SILVER_PATH + '/models'
MODEL_CALEFACTION_NAME = 'modelo_regresion_temperatura.pkl'

def _format_seconds(seconds: float) -> str:
    return f"{int(round(max(seconds, 0)))}sec"




@contextmanager
def log_step(logger: ETLLogger, capa: str, fase: str, etapa: str, mensaje_inicio: str = ''):
    start = time.perf_counter()
    logger.emit(capa, fase, etapa, 'Start', '0sec', 0, mensaje_inicio)
    data = {'longitud': 0, 'mensaje_fin': ''}
    try:
        yield data
    except Exception as exc:
        logger.emit(
            capa,
            fase,
            etapa,
            'ERROR',
            _format_seconds(time.perf_counter() - start),
            int(data.get('longitud', 0)),
            f"Error: {exc}",
        )
        raise
    else:
        logger.emit(
            capa,
            fase,
            etapa,
            'End',
            _format_seconds(time.perf_counter() - start),
            int(data.get('longitud', 0)),
            data.get('mensaje_fin', ''),
        )


#########################
# UTILITIES
##########################

# Carga variables desde src/.env si existen y no sobrescribe variables ya exportadas.
def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_db_config() -> dict:
    env_path = Path(__file__).with_name('.env')
    load_env_file(env_path)

    required = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise ValueError(f"Faltan variables de entorno requeridas: {', '.join(missing)}")

    return {k: os.environ[k] for k in required}


def get_log_config() -> dict:
    env_path = Path(__file__).with_name('.env')
    load_env_file(env_path)

    target = os.getenv('ETL_LOG_TARGET', 'consola').strip().lower()
    if target not in {'consola', 'db', 'all'}:
        raise ValueError("ETL_LOG_TARGET debe ser 'consola', 'db' o 'all'.")

    mysql_keys = ['MYSQL_LOG_USER', 'MYSQL_LOG_PASSWORD', 'MYSQL_LOG_HOST', 'MYSQL_LOG_PORT', 'MYSQL_LOG_DB']
    mysql_values = {k: os.getenv(k, '').strip() for k in mysql_keys}
    sqlite_buffer_path = os.getenv('ETL_LOG_SQLITE_BUFFER_PATH', f"{DATA_PATH}/logs/etl_log_buffer.sqlite")

    return {
        'ETL_LOG_TARGET': target,
        'MYSQL_CONFIG': mysql_values,
        'SQLITE_BUFFER_PATH': sqlite_buffer_path,
    }


def init_logger() -> ETLLogger:
    config = get_log_config()
    mysql_config = config['MYSQL_CONFIG']
    has_mysql = all(mysql_config.values())
    if config['ETL_LOG_TARGET'] in {'db', 'all'} and not has_mysql:
        # Si faltan credenciales se degrada a consola, manteniendo buffer disponible.
        fallback_target = 'consola' if config['ETL_LOG_TARGET'] == 'db' else 'all'
        return ETLLogger(fallback_target, None, config['SQLITE_BUFFER_PATH'])

    return ETLLogger(config['ETL_LOG_TARGET'], mysql_config if has_mysql else None, config['SQLITE_BUFFER_PATH'])


def connect_to_db(user, password, host, port, db_name):
    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    )


def get_db_data(engine):
    tabla = "ltss"

    query = f"""
    SELECT
        time AS timestamp,
        entity_id,
        state,
        attributes
    FROM {tabla}
    """
    df = pd.read_sql(query, engine)
    return df


def save_df_to_csv(df, path, name):
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_csv(path + '/' + name + '.csv', index=False)


def normalize_attributes_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'attributes' not in df.columns:
        return df

    def _serialize_attribute(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list, tuple, bool, int, float)):
            return json.dumps(value, ensure_ascii=True, default=str)
        return str(value)

    # Fuerza un tipo homogéneo para que pyarrow no intente inferir tipos incompatibles.
    df = df.copy()
    df['attributes'] = df['attributes'].map(_serialize_attribute).astype('string')
    return df


def save_df_to_parquet(df, path, name):
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_parquet(path + '/' + name + '.parquet', index=False)


#########################
# LAYERS MAIN
##########################

def extract_db_to_bronze(path, name, logger: ETLLogger):
    with log_step(logger, 'BRONZE', 'GLOBAL', 'GLOBAL', 'Inicio de capa BRONZE') as phase:
        with log_step(logger, 'BRONZE', 'EXTRACT', 'extract_db_to_bronze', 'Cargando datos de DDBB') as step:
            db_config = get_db_config()
            engine = connect_to_db(
                db_config['DB_USER'],
                db_config['DB_PASSWORD'],
                db_config['DB_HOST'],
                db_config['DB_PORT'],
                db_config['DB_NAME'],
            )
            df = get_db_data(engine)
            step['longitud'] = len(df)
            step['mensaje_fin'] = f"Cargados {len(df)} registros desde PostgreSQL"

        with log_step(logger, 'BRONZE', 'TRANSFORM', 'normalize_attributes_column', 'Normalizando columna attributes') as step:
            df = normalize_attributes_column(df)
            step['longitud'] = len(df)
            step['mensaje_fin'] = f"Normalizados {len(df)} registros"

        with log_step(logger, 'BRONZE', 'LOAD', 'save_df_to_parquet', f"Guardando {name}.parquet") as step:
            save_df_to_parquet(df, path, name)
            step['longitud'] = len(df)
            step['mensaje_fin'] = f"Guardado {name}.parquet"

        with log_step(logger, 'BRONZE', 'LOAD', 'save_df_to_csv', f"Guardando {name}.csv") as step:
            save_df_to_csv(df, path, name)
            step['longitud'] = len(df)
            step['mensaje_fin'] = f"Guardado {name}.csv"

        phase['mensaje_fin'] = 'Capa BRONZE completada'


def prepare_silver(sensors_path, name_raw_sensors, catefaccion_path, silver_path, model_path, model_name,
                   calefaccion_filename, logger: ETLLogger):
    with log_step(logger, 'SILVER', 'GLOBAL', 'GLOBAL', 'Inicio de capa SILVER') as phase:
        with log_step(logger, 'SILVER', 'EXTRACT', 'prepare_silver.read_csv', 'Leyendo dataset base de sensores') as step:
            df = pd.read_csv(sensors_path + '/' + name_raw_sensors + '.csv')
            step['longitud'] = len(df)
            step['mensaje_fin'] = f"Leidos {len(df)} registros desde {name_raw_sensors}.csv"

        process_temperature(df, silver_path, logger)
        process_humidity(df, silver_path, logger)
        process_openclose(df, silver_path, logger)
        process_sun(df, silver_path, logger)
        process_calefaccion('temperatura.csv', silver_path, model_path, model_name, logger)
        process_weather(df, silver_path, logger)

        phase['mensaje_fin'] = 'Capa SILVER completada'

def prepare_gold(silver_path, gold_path, logger: ETLLogger):
    with log_step(logger, 'GOLD', 'GLOBAL', 'GLOBAL', 'Inicio de capa GOLD') as phase:
        phase['mensaje_fin'] = 'Capa GOLD sin transformaciones implementadas'
    return


#########################
# SILVER UTILITIES
##########################

def get_calefaccion_thresholds(timestamp):
    weekday = timestamp.weekday()
    current_hour = timestamp.hour

    if weekday > 4:
        return None, None

    # temperatura.csv está agregado por horas en punto, así que usamos
    # grupos horarios adecuados para el inicio y el fin del algoritmo
    # ya que el CSV de temperaturas ya está agrupado por horas cuando lo procesamos con esta
    # función para ello modificamos [inicio, fin) para que 11:00, 16:00 y
    # 18:00 entren ya en la siguiente franja.
    if 7 <= current_hour < 11:
        return 21.9, 22.0

    if 11 <= current_hour < 16:
        return 20.9, 21.0

    if 16 <= current_hour < 18:
        return 21.5, 21.6

    if 18 <= current_hour < 21:
        return 21.9, 22.0

    return None, None

def process_calefaccion(temperature_filename, silver_path, model_path, model_name, logger: ETLLogger | None = None):
    logger = logger or init_logger()

    with log_step(logger, 'SILVER', 'EXTRACT', 'process_calefaccion.read', f"Leyendo {temperature_filename} y modelo") as step:
        df = pd.read_csv(silver_path + '/' + temperature_filename)
        model = joblib.load(model_path + '/' + model_name)
        step['longitud'] = len(df)
        step['mensaje_fin'] = f"Leidos {len(df)} registros de temperatura"

    with log_step(logger, 'SILVER', 'TRANSFORM', 'process_calefaccion.transform', 'Calculando estado de calefaccion') as step:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601')

        mask = df["value"].notna()
        df["predicted_temperature"] = np.nan
        df.loc[mask, "predicted_temperature"] = model.predict(df.loc[mask, ["value"]])

        df = df.sort_values("timestamp")

        calefaccion = []
        estado = 0
        for _, row in df.iterrows():
            low, high = get_calefaccion_thresholds(row["timestamp"])
            temp = row["predicted_temperature"]

            if low is None:
                estado = 0
                calefaccion.append(estado)
                continue

            if pd.isna(temp):
                calefaccion.append(estado)
                continue

            if estado == 0 and temp < low:
                estado = 1
            elif estado == 1 and temp > high:
                estado = 0

            calefaccion.append(estado)

        df["calefaccion_status"] = calefaccion
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.weekday

        df_final = df[["timestamp", "calefaccion_status", "date", "hour", "weekday"]]
        df_final = df_final.rename(columns={"calefaccion_status": "value"})

        step['longitud'] = len(df_final)
        step['mensaje_fin'] = f"Transformados {len(df_final)} registros de calefaccion"

    with log_step(logger, 'SILVER', 'LOAD', 'process_calefaccion.save', 'Guardando calefaccion.csv') as step:
        df_final.to_csv(silver_path + '/calefaccion.csv', index=False)
        step['longitud'] = len(df_final)
        step['mensaje_fin'] = f"Guardados {len(df_final)} registros en calefaccion.csv"

def process_weather(df, silver_path, logger: ETLLogger | None = None):
    logger = logger or init_logger()

    with log_step(logger, 'SILVER', 'EXTRACT', 'process_weather.extract', 'Extrayendo sensores meteorologicos de Mislata') as step:
        weather_sensor = df["entity_id"].str.contains("mislata")
        df_mislata = df[weather_sensor].copy()
        step['longitud'] = len(df_mislata)
        step['mensaje_fin'] = f"Extraidos {len(df_mislata)} registros meteorologicos"

    with log_step(logger, 'SILVER', 'TRANSFORM', 'process_weather.transform', 'Transformando serie meteorologica') as step:
        df_mislata["timestamp"] = pd.to_datetime(df_mislata["timestamp"], format='ISO8601')
        df_mislata["state"] = pd.to_numeric(df_mislata["state"], errors="coerce")

        df_mislata_wide = df_mislata.pivot(
            index="timestamp",
            columns="entity_id",
            values="state"
        )

        df_mislata_wide = df_mislata_wide.resample("1h").mean()

        if "weather.forecast_mislata" in df_mislata_wide.columns:
            df_mislata_wide = df_mislata_wide.drop(columns=["weather.forecast_mislata"])

        df_mislata_wide = df_mislata_wide.rename(columns={
            "sensor.mislata_temperatura": "temperatura",
            "sensor.mislata_humedad": "humedad",
            "sensor.mislata_presion": "presion",
            "sensor.mislata_viento": "viento",
            "sensor.mislata_direccion_viento": "direccion_viento",
            "sensor.mislata_nubosidad": "nubosidad"
        })

        sensor_cols = df_mislata_wide.columns

        rows_all_nan = df_mislata_wide[sensor_cols].isna().all(axis=1)

        gap_groups = (rows_all_nan != rows_all_nan.shift()).cumsum()
        gap_sizes = rows_all_nan.groupby(gap_groups).transform("sum")

        df_mislata_wide = df_mislata_wide[~(rows_all_nan & (gap_sizes > 24))]

        df_mislata_wide[sensor_cols] = df_mislata_wide[sensor_cols].interpolate(
            method="time",
            limit=3
        )

        if "nubosidad" in df_mislata_wide.columns:
            df_mislata_wide = df_mislata_wide.drop(columns=["nubosidad"])

        df_mislata_wide = df_mislata_wide.dropna()

        df_mislata_wide = df_mislata_wide.reset_index()

        df_mislata_wide["date"] = df_mislata_wide["timestamp"].dt.date
        df_mislata_wide["hour"] = df_mislata_wide["timestamp"].dt.hour
        df_mislata_wide["weekday"] = df_mislata_wide["timestamp"].dt.weekday

        df_mislata_wide = df_mislata_wide.set_index("timestamp")

        df_mislata_wide = df_mislata_wide.round(1)
        step['longitud'] = len(df_mislata_wide)
        step['mensaje_fin'] = f"Transformados {len(df_mislata_wide)} registros meteorologicos"

    with log_step(logger, 'SILVER', 'LOAD', 'process_weather.save', 'Guardando weather_mislata.csv') as step:
        if not os.path.exists(silver_path):
            os.makedirs(silver_path, exist_ok=True)

        df_mislata_wide.to_csv(os.path.join(silver_path, "weather_mislata.csv"))
        step['longitud'] = len(df_mislata_wide)
        step['mensaje_fin'] = f"Guardados {len(df_mislata_wide)} registros en weather_mislata.csv"



def hourly_weighted_mean(df, max_gap_hours=24):
    start = df.index.min().floor('h')
    end = df['next_time'].max().ceil('h')

    hours = pd.date_range(start, end, freq='h')

    results = []

    max_gap = pd.Timedelta(hours=max_gap_hours)

    for h_start in hours[:-1]:

        h_end = h_start + pd.Timedelta(hours=1)

        mask = (df.index < h_end) & (df['next_time'] > h_start)
        subset = df.loc[mask]

        if subset.empty:
            results.append(np.nan)
            continue

        total_weight = 0
        weighted_sum = 0

        for t, row in subset.iterrows():

            if (row['next_time'] - t) > max_gap:
                continue

            start_i = max(t, h_start)
            end_i = min(row['next_time'], h_end)

            seconds = (end_i - start_i).total_seconds()

            weighted_sum += row['state'] * seconds
            total_weight += seconds

        if total_weight == 0:
            results.append(np.nan)
        else:
            results.append(weighted_sum / total_weight)

    return pd.DataFrame(
        {'state': results},
        index=hours[:-1]
    )


def hourly_weighted_open(df, weights, max_gap_hours=48):
    start = df["timestamp"].min().floor("h")
    end = df["next_timestamp"].max().ceil("h")

    hours = pd.date_range(start, end, freq="h")

    results = []

    max_gap = pd.Timedelta(hours=max_gap_hours)
    total_weight = sum(weights.values())

    for h_start in hours[:-1]:

        h_end = h_start + pd.Timedelta(hours=1)

        mask = (df["timestamp"] < h_end) & (df["next_timestamp"] > h_start)
        subset = df.loc[mask]

        if subset.empty:
            results.append(np.nan)
            continue

        weighted_sum = 0

        for _, row in subset.iterrows():

            if (row["next_timestamp"] - row["timestamp"]) > max_gap:
                continue

            start_i = max(row["timestamp"], h_start)
            end_i = min(row["next_timestamp"], h_end)

            seconds = (end_i - start_i).total_seconds()

            if row["state"] == 1:
                weight = weights.get(row["entity_id"], 0)

                weighted_sum += seconds * weight

        equivalent_open_seconds = weighted_sum / total_weight

        results.append(equivalent_open_seconds)

    return pd.DataFrame(
        {"open_seconds": results},
        index=hours[:-1]
    )


def process_sensor_data(df, sensors_names, decimals, filtermin, filtermax, sensor) -> DataFrame:
    df_sensor = df[sensors_names].copy()

    df_sensor["timestamp"] = pd.to_datetime(df_sensor["timestamp"], format='ISO8601')
    df_sensor["state"] = pd.to_numeric(df_sensor["state"], errors="coerce")

    sensor_virtual = df_sensor[
        df_sensor["entity_id"] == sensor
        ].copy()

    sensor_virtual["timestamp"] = pd.to_datetime(sensor_virtual["timestamp"], format='ISO8601')
    sensor_virtual["state"] = pd.to_numeric(sensor_virtual["state"], errors="coerce")

    sensor_virtual = sensor_virtual.set_index("timestamp")

    # quedarte solo con la columna numérica
    sensor_virtual = sensor_virtual[["state"]]

    # eliminar valores imposibles
    sensor_virtual = sensor_virtual[
        (sensor_virtual['state'] >= filtermin) &
        (sensor_virtual['state'] <= filtermax)
        ]

    # ordenar
    sensor_virtual = sensor_virtual.sort_index()

    # --- 2. quedarnos solo con temperatura ---
    df_sensor_virtual = sensor_virtual[["state"]].copy()

    # eliminar nulos
    df_sensor_virtual = df_sensor_virtual.dropna()

    # --- 3. construir intervalos de validez ---
    df_sensor_virtual['next_time'] = df_sensor_virtual.index.to_series().shift(-1)

    # último registro dura hasta el final de la última hora
    end_time = df_sensor_virtual.index.max().ceil('h')
    df_sensor_virtual.loc[df_sensor_virtual['next_time'].isna(), 'next_time'] = end_time

    # --- calcular valor horario ---
    sensor_hourly = hourly_weighted_mean(df_sensor_virtual, max_gap_hours=24)

    # convertir a dataframe final
    sensor_virtual = sensor_hourly.reset_index()
    sensor_virtual = sensor_virtual.rename(columns={'index': 'timestamp'})

    sensor_virtual["date"] = sensor_virtual["timestamp"].dt.date
    sensor_virtual["hour"] = sensor_virtual["timestamp"].dt.hour
    sensor_virtual["weekday"] = sensor_virtual["timestamp"].dt.weekday
    sensor_virtual["timestamp"] = pd.to_datetime(sensor_virtual["timestamp"])

    sensor_virtual = sensor_virtual.set_index("timestamp")

    # state redondear a 1 decimal
    sensor_virtual["state"] = sensor_virtual["state"].round(decimals)
    sensor_virtual = sensor_virtual.rename(columns={"state": "value"})
    return sensor_virtual


def process_temperature(df, silver_path, logger: ETLLogger | None = None) -> DataFrame:
    logger = logger or init_logger()

    with log_step(logger, 'SILVER', 'EXTRACT', 'process_temperature.extract', 'Filtrando registros de temperatura') as step:
        sensors_names = (
            ((df["entity_id"].str.contains("sensor.sensor_temperatura_2_temperature")) &
             ~df["entity_id"].str.contains("humidity") &
             ~df["entity_id"].str.contains("pressure") &
             ~df["entity_id"].str.contains("battery") &
             ~df["entity_id"].str.contains("manises") &
             ~df["entity_id"].str.contains("mislata") &
             ~df["entity_id"].str.contains("device")
             ))
        step['longitud'] = int(sensors_names.sum())
        step['mensaje_fin'] = f"Extraidos {int(sensors_names.sum())} registros de temperatura"

    with log_step(logger, 'SILVER', 'TRANSFORM', 'process_temperature.transform', 'Calculando media ponderada horaria') as step:
        sensor_virtual = process_sensor_data(df, sensors_names, 1, 5, 45, "sensor.sensor_temperatura_2_temperature")
        step['longitud'] = len(sensor_virtual)
        step['mensaje_fin'] = f"Transformados {len(sensor_virtual)} registros de temperatura"

    with log_step(logger, 'SILVER', 'LOAD', 'process_temperature.save', 'Guardando temperatura.csv') as step:
        if not os.path.exists(silver_path):
            os.makedirs(silver_path, exist_ok=True)
        sensor_virtual.to_csv(os.path.join(silver_path, "temperatura.csv"), index=True)
        step['longitud'] = len(sensor_virtual)
        step['mensaje_fin'] = f"Guardados {len(sensor_virtual)} registros en temperatura.csv"
    return sensor_virtual


def process_humidity(df, silver_path, logger: ETLLogger | None = None):
    logger = logger or init_logger()

    with log_step(logger, 'SILVER', 'EXTRACT', 'process_humidity.extract', 'Filtrando registros de humedad') as step:
        sensors_names = (
            ((df["entity_id"].str.contains("sensor.sensor_temperatura_2_humidity")) &
             ~df["entity_id"].str.contains("temperature") &
             ~df["entity_id"].str.contains("pressure") &
             ~df["entity_id"].str.contains("battery") &
             ~df["entity_id"].str.contains("manises") &
             ~df["entity_id"].str.contains("mislata") &
             ~df["entity_id"].str.contains("device")
             ))
        step['longitud'] = int(sensors_names.sum())
        step['mensaje_fin'] = f"Extraidos {int(sensors_names.sum())} registros de humedad"

    with log_step(logger, 'SILVER', 'TRANSFORM', 'process_humidity.transform', 'Calculando media ponderada horaria') as step:
        sensor_virtual = process_sensor_data(df, sensors_names, 2, 0, 100, "sensor.sensor_temperatura_2_humidity")
        step['longitud'] = len(sensor_virtual)
        step['mensaje_fin'] = f"Transformados {len(sensor_virtual)} registros de humedad"

    with log_step(logger, 'SILVER', 'LOAD', 'process_humidity.save', 'Guardando humedad.csv') as step:
        if not os.path.exists(silver_path):
            os.makedirs(silver_path, exist_ok=True)
        sensor_virtual.to_csv(os.path.join(silver_path, "humedad.csv"), index=True)
        step['longitud'] = len(sensor_virtual)
        step['mensaje_fin'] = f"Guardados {len(sensor_virtual)} registros en humedad.csv"
    return sensor_virtual


def process_openclose(df, silver_path, logger: ETLLogger | None = None):
    logger = logger or init_logger()

    with log_step(logger, 'SILVER', 'EXTRACT', 'process_openclose.extract', 'Filtrando sensores de puertas y ventanas') as step:
        door_sensors = (
                (df["entity_id"].str.contains("ventana") | df["entity_id"].str.contains("puerta")) &
                ~df["entity_id"].str.contains("battery") &
                ~df["entity_id"].str.contains("temperature") &
                ~df["entity_id"].str.contains("binary_sensor.sensor_ventana_11_contact")
        )
        df_doors = df[door_sensors].copy()
        step['longitud'] = len(df_doors)
        step['mensaje_fin'] = f"Extraidos {len(df_doors)} registros de aperturas"

    with log_step(logger, 'SILVER', 'TRANSFORM', 'process_openclose.transform', 'Calculando indicadores horarios de apertura') as step:
        df_doors["timestamp"] = pd.to_datetime(df_doors["timestamp"], format='ISO8601')

        df_doors["state"] = df_doors["state"].map({
            "on": 1,
            "off": 0
        }).astype(float)

        df_doors = (
            df_doors[["timestamp", "entity_id", "state"]]
            .dropna()
            .sort_values(["entity_id", "timestamp"])
        )

        df_doors["next_timestamp"] = (
            df_doors.groupby("entity_id")["timestamp"].shift(-1)
        )

        end_time = df_doors["timestamp"].max().ceil("h")
        df_doors.loc[df_doors["next_timestamp"].isna(), "next_timestamp"] = end_time

        weights = {
        "binary_sensor.sensor_puerta_1_contact": 2,
        "binary_sensor.sensor_ventana_1_contact": 1,
        "binary_sensor.sensor_ventana_2_contact": 0.5,
        "binary_sensor.sensor_ventana_3_contact": 1,
        "binary_sensor.sensor_ventana_4_contact": 0.5,
        "binary_sensor.sensor_ventana_5_contact": 1,
        "binary_sensor.sensor_ventana_6_contact": 0.5,
        "binary_sensor.sensor_ventana_7_contact": 1,
        "binary_sensor.sensor_ventana_8_contact": 0.5,
        "binary_sensor.sensor_ventana_9_contact": 1,
        "binary_sensor.sensor_ventana_10_contact": 0.5,
        "binary_sensor.sensor_ventana_12_contact": 0.5,
        }

        door_hour = hourly_weighted_open(df_doors, weights)
        door_hour["percent_open_time"] = door_hour["open_seconds"] / 3600 * 100
        door_hour["open_flag"] = (door_hour["open_seconds"] > 0).astype(int)
        door_hour = door_hour.reset_index()
        door_hour = door_hour.rename(columns={"index": "timestamp"})
        door_hour["date"] = door_hour["timestamp"].dt.date
        door_hour["hour"] = door_hour["timestamp"].dt.hour
        door_hour["weekday"] = door_hour["timestamp"].dt.weekday
        door_hour["open_seconds"] = door_hour["open_seconds"].round(decimals=2)
        door_hour["percent_open_time"] = door_hour["percent_open_time"].round(decimals=2)

        door_hour_final = door_hour[
            ["timestamp", "date", "hour", "weekday", "percent_open_time", "open_seconds", "open_flag"]
        ]
        door_hour_final = door_hour_final.set_index("timestamp")
        step['longitud'] = len(door_hour_final)
        step['mensaje_fin'] = f"Transformados {len(door_hour_final)} registros de aperturas"

    with log_step(logger, 'SILVER', 'LOAD', 'process_openclose.save', 'Guardando puertas_ventanas.csv') as step:
        if not os.path.exists(silver_path):
            os.makedirs(silver_path, exist_ok=True)
        door_hour_final.to_csv(os.path.join(silver_path, "puertas_ventanas.csv"))
        step['longitud'] = len(door_hour_final)
        step['mensaje_fin'] = f"Guardados {len(door_hour_final)} registros en puertas_ventanas.csv"

    return door_hour_final

#########################
# NUEVO ALEJANDRO
##########################

def process_sun(df, silver_path, logger: ETLLogger | None = None):
    logger = logger or init_logger()

    ####################################
    ###### SOL #########################
    ####################################

    with log_step(logger, 'SILVER', 'EXTRACT', 'process_sun.extract', 'Filtrando registros del sensor solar') as step:
        sun_sensor = df["entity_id"] == "sun.sun"
        df_sun = df[sun_sensor].copy()
        step['longitud'] = len(df_sun)
        step['mensaje_fin'] = f"Extraidos {len(df_sun)} registros solares"

    with log_step(logger, 'SILVER', 'TRANSFORM', 'process_sun.transform', 'Calculando agregacion solar horaria') as step:
        df_sun = df_sun.reset_index(drop=True)

        df_sun["attributes"] = df_sun["attributes"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        attrs = pd.json_normalize(df_sun["attributes"])

        df_sun["azimuth"] = attrs["azimuth"]
        df_sun["elevation"] = attrs["elevation"]

        df_sun["daynight"] = df_sun["state"].map({
            "below_horizon": 0,
            "above_horizon": 1
        })

        sun_virtual = df_sun.copy()

        sun_virtual["timestamp"] = pd.to_datetime(sun_virtual["timestamp"], format='ISO8601')

        sun_virtual = sun_virtual.set_index("timestamp")

        sun_virtual = sun_virtual[[
            "daynight",
            "azimuth",
            "elevation"
        ]]

        sun_virtual = sun_virtual.sort_index()

        df_sun_virtual = sun_virtual.copy()

        df_sun_virtual = df_sun_virtual.dropna()

        df_sun_virtual["next_time"] = df_sun_virtual.index.to_series().shift(-1)

        end_time = df_sun_virtual.index.max().ceil("h")
        df_sun_virtual.loc[df_sun_virtual["next_time"].isna(), "next_time"] = end_time

        sun_hourly = (
            df_sun_virtual
            .resample("h")
            .agg({
                "daynight": "max",
                "azimuth": "mean",
                "elevation": "mean"
            })
        )

        sun_hourly["daynight"] = sun_hourly["daynight"].astype("Int64")

        sun_hourly = sun_hourly.reset_index()

        if "next_time" in sun_hourly.columns:
            sun_hourly = sun_hourly.drop(columns=["next_time"])

        sun_hourly["date"] = sun_hourly["timestamp"].dt.date
        sun_hourly["hour"] = sun_hourly["timestamp"].dt.hour
        sun_hourly["weekday"] = sun_hourly["timestamp"].dt.weekday

        sun_hourly[["azimuth", "elevation"]] = sun_hourly[
            ["azimuth", "elevation"]
        ].round(2)

        sun_hourly = sun_hourly.set_index("timestamp")
        step['longitud'] = len(sun_hourly)
        step['mensaje_fin'] = f"Transformados {len(sun_hourly)} registros solares"

    with log_step(logger, 'SILVER', 'LOAD', 'process_sun.save', 'Guardando sol.csv') as step:
        os.makedirs(silver_path, exist_ok=True)
        sun_hourly.to_csv(os.path.join(silver_path, "sol.csv"))
        step['longitud'] = len(sun_hourly)
        step['mensaje_fin'] = f"Guardados {len(sun_hourly)} registros en sol.csv"



#########################
# MAIN
##########################
def run_main():
    logger = init_logger()
    logger.drain_buffer_to_mysql()

    extract_db_to_bronze(SENSORS_PATH, 'ltss_sensores', logger)
    prepare_silver( SENSORS_PATH, 'ltss_sensores', SENSORS_CALEFACTION, SILVER_PATH, MODEL_PATH, MODEL_CALEFACTION_NAME, SENSORS_CALEFACTION_RAW_NAME, logger)
    prepare_gold(SILVER_PATH, GOLD_PATH, logger)

    logger.drain_buffer_to_mysql()

    return


if __name__ == "__main__":
    run_main()
