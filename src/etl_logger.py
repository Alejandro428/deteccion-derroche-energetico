import json
import sqlite3
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, text

LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS etl_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    capa VARCHAR(16) NOT NULL,
    fase VARCHAR(16) NOT NULL,
    etapa VARCHAR(128) NOT NULL,
    estado VARCHAR(16) NOT NULL,
    tiempo_consumido VARCHAR(32) NOT NULL,
    longitud BIGINT NOT NULL,
    mensaje TEXT
)
"""


def _now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _sanitize_text(value: object) -> str:
    # Normaliza texto para que sea seguro al serializar en JSON/CSV/tabla.
    text_value = str(value)
    text_value = text_value.replace('\\', '\\\\')
    text_value = text_value.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
    text_value = text_value.replace('\t', '\\t').replace('\x00', '')
    text_value = text_value.replace('"', '""')
    return text_value


class ETLLogger:
    def __init__(self, target: str, mysql_config: dict | None, sqlite_buffer_path: str):
        self.target = target.lower()
        self.mysql_config = mysql_config
        self.sqlite_buffer_path = sqlite_buffer_path
        self.mysql_engine = None
        self.mysql_available = False
        self._console_header_printed = False

        self._init_sqlite_buffer()
        if self.target in {'db', 'all'}:
            self._init_mysql_storage()

    def _init_sqlite_buffer(self):
        Path(self.sqlite_buffer_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_buffer_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS log_buffer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    creado_en TEXT NOT NULL
                )
                """
            )

    def _init_mysql_storage(self):
        if not self.mysql_config:
            self._console_fallback("Configuracion MySQL de logs no disponible; se usa solo consola y buffer.")
            return

        user = self.mysql_config['MYSQL_LOG_USER']
        password = self.mysql_config['MYSQL_LOG_PASSWORD']
        host = self.mysql_config['MYSQL_LOG_HOST']
        port = self.mysql_config['MYSQL_LOG_PORT']
        db_name = self.mysql_config['MYSQL_LOG_DB']

        try:
            server_engine = create_engine(
                f"mysql+pymysql://{user}:{password}@{host}:{port}/",
                pool_pre_ping=True,
            )
            with server_engine.begin() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))

            self.mysql_engine = create_engine(
                f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}",
                pool_pre_ping=True,
            )
            with self.mysql_engine.begin() as conn:
                conn.execute(text(LOG_SCHEMA))

            self.mysql_available = True
            self.drain_buffer_to_mysql()
        except Exception as exc:
            self.mysql_available = False
            self._console_fallback(f"Error al preparar MySQL de logs: {exc}. Se encolara en SQLite.")

    def _console_fallback(self, mensaje: str):
        self._print_console_header_if_needed()
        print(self._format_row(
            _now_str(),
            'SILVER',
            'GLOBAL',
            'logging',
            'ERROR',
            '0sec',
            0,
            _sanitize_text(mensaje),
        ))

    def _format_cell(self, value: object, width: int, align: str = 'left') -> str:
        text_value = str(value)
        if len(text_value) > width:
            text_value = text_value[:width - 3] + '...'
        if align == 'right':
            return f"{text_value:>{width}}"
        return f"{text_value:<{width}}"

    def _format_row(self, timestamp: str, capa: str, fase: str, etapa: str,
                    estado: str, tiempo: str, longitud: int, mensaje: str) -> str:
        return (
            f"| {self._format_cell(timestamp, 19)} "
            f"| {self._format_cell(capa, 7)} "
            f"| {self._format_cell(fase, 9)} "
            f"| {self._format_cell(etapa, 28)} "
            f"| {self._format_cell(estado, 6)} "
            f"| {self._format_cell(tiempo, 8)} "
            f"| {self._format_cell(longitud, 8, 'right')} "
            f"| {mensaje}"
        )

    def _print_console_header_if_needed(self):
        if self._console_header_printed:
            return

        header = self._format_row(
            'timestamp',
            'CAPA',
            'FASE',
            'ETAPA',
            'ESTADO',
            'TIEMPO',
            'LONGITUD',
            'MENSAJE',
        )
        separator = '-' * len(header)
        print(separator)
        print(header)
        print(separator)
        self._console_header_printed = True

    def _print_console(self, event: dict):
        self._print_console_header_if_needed()
        print(self._format_row(
            event['timestamp'],
            event['capa'],
            event['fase'],
            event['etapa'],
            event['estado'],
            event['tiempo_consumido'],
            event['longitud'],
            event['mensaje'],
        ))

    def _insert_mysql(self, event: dict):
        if not self.mysql_available or self.mysql_engine is None:
            raise RuntimeError('MySQL de logs no disponible')

        sql = text(
            """
            INSERT INTO etl_logs
                (timestamp, capa, fase, etapa, estado, tiempo_consumido, longitud, mensaje)
            VALUES
                (:timestamp, :capa, :fase, :etapa, :estado, :tiempo_consumido, :longitud, :mensaje)
            """
        )
        payload = event.copy()
        payload['timestamp'] = datetime.strptime(event['timestamp'], '%Y-%m-%d %H:%M:%S')
        with self.mysql_engine.begin() as conn:
            conn.execute(sql, payload)

    def _enqueue_buffer(self, event: dict):
        with sqlite3.connect(self.sqlite_buffer_path) as conn:
            conn.execute(
                "INSERT INTO log_buffer(payload, creado_en) VALUES (?, ?)",
                (json.dumps(event, ensure_ascii=True), _now_str()),
            )

    def drain_buffer_to_mysql(self):
        if not self.mysql_available:
            return

        with sqlite3.connect(self.sqlite_buffer_path) as conn:
            rows = conn.execute("SELECT id, payload FROM log_buffer ORDER BY id").fetchall()
            for row_id, payload in rows:
                event = json.loads(payload)
                try:
                    self._insert_mysql(event)
                    conn.execute("DELETE FROM log_buffer WHERE id = ?", (row_id,))
                except Exception:
                    # Si vuelve a fallar, se mantiene en cola para drenado posterior.
                    break

    def emit(self, capa: str, fase: str, etapa: str, estado: str,
             tiempo_consumido: str = '0sec', longitud: int = 0, mensaje: str = ''):
        event = {
            'timestamp': _now_str(),
            'capa': _sanitize_text(capa),
            'fase': _sanitize_text(fase),
            'etapa': _sanitize_text(etapa),
            'estado': _sanitize_text(estado),
            'tiempo_consumido': _sanitize_text(tiempo_consumido),
            'longitud': int(longitud),
            'mensaje': _sanitize_text(mensaje),
        }

        if self.target in {'consola', 'all'}:
            self._print_console(event)

        if self.target in {'db', 'all'}:
            try:
                self._insert_mysql(event)
            except Exception as exc:
                self._enqueue_buffer(event)
                self._console_fallback(
                    f"Error al escribir en MySQL: {exc}. Evento encolado en SQLite temporal."
                )

