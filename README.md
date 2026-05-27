# Deteccion de Derroche Energetico en Aula

> Sistema IoT + ML para detectar derroche energetico en aula a partir de sensores de Home Assistant (temperatura, humedad, apertura de puertas/ventanas, datos meteorologicos y solares).

**Proyecto en equipo del Curso de Especializacion en Inteligencia Artificial y Big Data — CIPFP Mislata, 2025-2026.**

---

## Sobre el proyecto

El objetivo es **detectar situaciones de derroche energetico en un aula**, por ejemplo:
- Calefaccion encendida con puertas o ventanas abiertas.
- Condiciones ambientales incompatibles con un uso eficiente.

Para ello se construye un pipeline de extremo a extremo que va desde la ingesta de datos crudos de los sensores hasta la prediccion y visualizacion del derroche.

## Pipeline de datos

```
+-------------------+      +----------+      +----------+      +-----------+      +----------+
| Sensores IoT      |      | Postgres |      |  Bronze  |      |  Silver   |      |  Modelo  |
| (Home Assistant,  | ---> | + LTSS   | ---> |  (raw)   | ---> | (limpio)  | ---> |  ML + NN |
|  Zigbee, MQTT)    |      |          |      |          |      |           |      |          |
+-------------------+      +----------+      +----------+      +-----------+      +----------+
                                                                                        |
                                                                                        v
                                                                                  +-----------+
                                                                                  | API       |
                                                                                  | + Grafana |
                                                                                  +-----------+
```

1. **Ingesta**: Home Assistant escribe los estados de los sensores en Postgres (extension LTSS).
2. **EDA**: notebooks de exploracion en `notebooks/EDA/` para entender cada fuente (temperatura, humedad, sol, meteo Mislata, puertas/ventanas).
3. **ETL Bronze/Silver**: `src/etl.py` transforma los datos crudos en datasets horarios limpios.
4. **Modelado**:
   - Regresion lineal para inferir cuando esta encendida la calefaccion.
   - Red neuronal (`notebooks/EDA/RedNeuronal.ipynb`) para predecir derroche.
5. **Exposicion**: API de inferencia + dashboard en Grafana.

## Stack tecnico

| Capa | Tecnologia |
|---|---|
| Lenguaje | Python 3 |
| Datos | pandas, numpy, SQLAlchemy |
| ML / DL | scikit-learn, TensorFlow / Keras |
| Almacenamiento | PostgreSQL + LTSS, MySQL (logs ETL) |
| Infraestructura | Docker Compose |
| Visualizacion | Grafana, matplotlib, seaborn |
| UI | Flet (cliente Python) |
| Integracion IoT | Home Assistant, Zigbee2MQTT, MQTT (Mosquitto) |

## Mi contribucion al proyecto

Proyecto en equipo en el que **participe en varias areas del pipeline**:

- Analisis exploratorio de los datos de sensores (humedad, sol, meteorologia de Mislata).
- Pipeline ETL Bronze -> Silver con pandas, incluyendo el procesamiento del sensor solar (`src/etl.py` -> `process_sun`): parseo de atributos JSON, mapeo de estados below/above horizon, agregacion horaria y enriquecimiento temporal.
- Participacion en el modelado y en la integracion de las fuentes externas (sol y meteorologia).

## Estructura del repositorio

```
.
|-- data/
|   |-- Sensores/          # CSV con el historico exportado de sensores
|   `-- lake/              # Salidas del ETL (bronze, silver, modelos entrenados)
|-- docker/                # docker-compose.yml + Dockerfiles
|-- grafana/               # Dashboards exportados (JSON)
|-- notebooks/EDA/         # Notebooks de exploracion y red neuronal
|-- src/
|   |-- etl.py             # Pipeline Bronze/Silver
|   |-- etl_logger.py      # Logger del ETL (SQLite buffer + MySQL persistente)
|   |-- api.py             # API de inferencia (Flask)
|   |-- inference_service.py
|   |-- model/             # Modelos entrenados (Keras)
|   `-- app/               # Dashboard cliente con Flet
|-- assets/                # Imagenes embebidas en la documentacion tecnica
|-- requirements.txt
|-- environment.tensorflow.yml
|-- .env.example
`-- README_TECNICO.md      # Documentacion tecnica detallada con EDA paso a paso
```

## Como reproducirlo localmente

### 1. Requisitos

- Python 3.10+ (o el entorno conda `environment.tensorflow.yml`).
- Docker y Docker Compose.

### 2. Clonar el repositorio

```bash
git clone https://github.com/Alejandro428/deteccion-derroche-energetico.git
cd deteccion-derroche-energetico
```

> **Nota sobre credenciales**: el proyecto trae unas credenciales por defecto cableadas en `docker-compose.yml` y en los notebooks (`DB_PASSWORD=dfer4X4d5`, `MYSQL_LOG_PASSWORD=dfer4X4d5`, `GRAFANA=admin/admin`). Esto es intencionado: son **passwords solo para el entorno Docker local**, no protegen ningun sistema real y permiten que el proyecto arranque sin configuracion. Si quieres cambiarlas, copia `.env.example` a `.env` y edita lo que quieras — los notebooks y `docker-compose.yml` priorizan las variables de entorno frente a los defaults.

### 3. Levantar la infraestructura

```bash
cd docker
docker compose up -d
```

Esto arranca:
- **Postgres** (puerto 5432) — base de datos historica de sensores.
- **MySQL** (puerto 3306) — logs persistentes del ETL.
- **Grafana** (puerto 3000) — dashboard, usuario `admin` / `admin`.
- **API de inferencia** (puerto 8000).

### 4. Cargar los datos historicos

Los datos crudos estan en `data/Sensores/History_2025-09-01_100-2026-03-02_1400.csv`. Hay que cargarlos en la tabla `ltss` de Postgres para que los notebooks de EDA puedan consultarlos.

### 5. Instalar dependencias Python y arrancar los notebooks

```bash
pip install -r requirements.txt
jupyter notebook notebooks/EDA/
```

### 6. Documentacion tecnica detallada

Para el detalle del EDA, decisiones de limpieza, justificaciones de los filtros y resultados del modelo: ver [`README_TECNICO.md`](README_TECNICO.md).

## Estado del proyecto

Proyecto academico entregado en marzo de 2026. La infraestructura desplegada en el centro educativo durante el curso ha sido desmontada al finalizar el periodo lectivo; el repositorio se conserva como referencia del trabajo realizado.

## Presentacion

[`20260330 Deteccion de Derroche Energetico.pdf`](20260330%20Detecci%C3%B3n%20de%20Derroche%20Energ%C3%A9tico.pdf) — presentacion final del proyecto.

## Licencia

Proyecto academico desarrollado durante el Curso de Especializacion en IA y Big Data del CIPFP Mislata.
