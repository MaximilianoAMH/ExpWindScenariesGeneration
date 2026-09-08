# Experimentos TimesFM — Generación de escenarios de viento

Repositorio para experimentar con **TimesFM 2.5** sobre velocidad media diaria del viento y construir **trayectorias probabilísticas temporalmente coherentes** mediante **Schaake Shuffle**.

La prioridad actual del proyecto es el pipeline **zero-shot TimesFM + Schaake Shuffle**. El fine-tuning con LoRA se conserva como experimento separado para una etapa posterior.

## Objetivo

TimesFM entrega pronósticos marginales por día, en este proyecto Q10, Q20, ..., Q90. Por sí solos esos cuantiles no forman trayectorias temporales. Schaake Shuffle utiliza patrones de ranks de trayectorias históricas completas para imponer una estructura de dependencia temporal sobre las marginales futuras.

El flujo principal es:

```text
Contexto observado (32 días)
          |
          v
TimesFM 2.5 zero-shot
Q10,...,Q90 para 28 días
          |
          +-----------------------+
                                  |
Histórico 1980-2013               |
trayectorias completas de 28 días |
          |                       |
          v                       |
ranks históricos por día          |
          |                       |
          +-----------+-----------+
                      v
               Schaake Shuffle
                      |
                      v
                L escenarios x 28
```

El target observado del período pronosticado **no participa en la generación**. Se utiliza solamente después, para evaluación ex-post y visualización.

## Estructura del repositorio

```text
ExperimentosTimesFM/
├── README.md
├── pyproject.toml
├── .gitignore
│
├── data/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── docs/
│   └── Metodologia_Schaake_Shuffle.pdf
│
├── src/
│   └── wind_scenarios/
│       ├── __init__.py
│       ├── data.py
│       ├── forecasting.py
│       ├── historical.py
│       ├── schaake.py
│       └── cli/
│           ├── generate.py
│           └── visualize.py
│
├── scripts/
│   ├── preprocess_wind_ab.py
│   └── diagnostics/
│       ├── inspect_data.py
│       └── test_timesfm_channels.py
│
├── experiments/
│   └── finetuning/
│       ├── train_lora.py
│       └── evaluate_test.py
│
└── tests/
    └── test_schaake.py
```

### Módulos principales

- `data.py`: carga del dataset diario, validación de continuidad temporal y definición de splits/ventanas.
- `forecasting.py`: extracción de todos los cuantiles disponibles de TimesFM en una matriz `H x K`.
- `historical.py`: construcción y selección flexible de trayectorias históricas para Schaake Shuffle.
- `schaake.py`: cálculo de ranks, discretización de marginales y construcción final de escenarios.
- `cli/generate.py`: pipeline completo de generación zero-shot.
- `cli/visualize.py`: visualización de fan plots o escenarios individuales; siempre muestra el contexto utilizado por TimesFM.

Los scripts de fine-tuning se mantienen en `experiments/finetuning/` para no mezclar el pipeline estable actual con experimentos futuros.

## Datos

El código espera por defecto:

```text
data/wind_ab_daily.csv
```

con columnas:

```text
DATE,WIND
```

La serie diaria usada en los experimentos actuales cubre 1980-2017. La configuración existente separa:

- desarrollo: 1980-2013;
- test: 2014-2017.

Los CSV de datos están excluidos de Git mediante `.gitignore`; no deberían subirse al repositorio salvo que exista una razón explícita para publicarlos.

Si se parte del archivo horario `data/wind_hourly.csv`, el preprocesamiento se ejecuta con:

```powershell
python scripts\preprocess_wind_ab.py
```

El script calcula medias diarias sólo para días observados completos de 24 horas y reporta discontinuidades de calendario sin interpolarlas silenciosamente.

## Instalación

Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Para trabajar también con el experimento de fine-tuning:

```powershell
pip install -e ".[finetuning]"
```

Para ejecutar tests:

```powershell
pip install -e ".[dev]"
pytest
```

## Generación de escenarios

### Caso A: una trayectoria histórica por año

Para pronosticar el 1-28 de abril de 2014 usando los 34 abriles disponibles entre 1980 y 2013:

```powershell
generate-wind-scenarios --origin 2014-04-01 --historical_method same_calendar_date
```

También puede ejecutarse directamente como módulo:

```powershell
python -m wind_scenarios.cli.generate --origin 2014-04-01 --historical_method same_calendar_date
```

El contexto suministrado a TimesFM es siempre el bloque de 32 días inmediatamente anterior al origen. Para `2014-04-01` es `2014-02-28` a `2014-03-31`.

### Caso B: 100 escenarios con ventanas móviles

```powershell
generate-wind-scenarios `
    --origin 2014-04-01 `
    --historical_method moving_window `
    --n_scenarios 100 `
    --window_radius_days 15 `
    --min_start_distance_days 7
```

La selección histórica es independiente del núcleo de Schaake Shuffle; cambiar `L` o el método histórico no modifica la implementación del shuffle.

## Salidas

Cada origen crea una carpeta bajo:

```text
outputs/schaake/YYYY-MM-DD/
```

por ejemplo:

```text
outputs/schaake/2014-04-01/
├── context.csv
├── timesfm_quantiles.csv
├── historical_trajectories.csv
├── historical_metadata.csv
├── historical_ranks.csv
├── empirical_pseudo_observations.csv
├── assigned_probabilities.csv
├── timesfm_discretized_marginals.csv
└── scenarios.csv
```

`context.csv` guarda exactamente las observaciones entregadas a TimesFM, lo que permite auditar que no hubo leakage del target.

## Visualización

### Escenario individual con todos los cuantiles

```powershell
visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --scenario_number 1 `
    --all_quantiles
```

El gráfico muestra:

- contexto observado utilizado por TimesFM;
- origen del forecast;
- Q10, Q20, ..., Q90;
- escenario Schaake seleccionado;
- trayectoria histórica fuente;
- realización observada, sólo para evaluación.

### Fan plot

```powershell
visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode fan `
    --all_quantiles
```

### Exportar todos los escenarios individualmente

```powershell
visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --all_quantiles `
    --export_all
```

## Decisiones metodológicas actuales

### 1. Zero-shot primero

La versión principal usa `google/timesfm-2.5-200m-transformers` sin ajuste sobre WIND_AB. Los experimentos LoRA se mantienen separados hasta validar completamente el pipeline zero-shot.

Esto significa que el modelo no se fine-tunea con el período 1980-2013 dentro de este proyecto. No implica por sí solo demostrar que la serie no pudiera haber formado parte del pretraining original de TimesFM.

### 2. Horizonte y contexto

Configuración actual:

```text
contexto = 32 días
horizonte = 28 días
```

### 3. Dependencia temporal

Para cada día `t`, los valores de las `L` trayectorias históricas se ordenan y se calcula el rank de cada trayectoria. Aunque el rank se calcula por columna, cada fila conserva su vector completo de ranks a través de los 28 días. Esa secuencia conjunta es la plantilla ordinal que se transfiere al escenario correspondiente.

La trayectoria histórica aporta **dependencia/ranks**, mientras que TimesFM aporta **magnitudes marginales futuras**.

### 4. Marginales Q10-Q90

TimesFM entrega actualmente Q10-Q90. El modo recomendado es:

```text
--probability_grid_mode timesfm_support
```

que discretiza `L` posiciones dentro de `[0.1, 0.9]` y evita extrapolar automáticamente colas no entregadas por TimesFM.

El modo clásico:

```text
--probability_grid_mode schaake
```

usa `k/(L+1)`, pero el código falla explícitamente si alguna probabilidad requerida cae fuera del soporte suministrado por TimesFM.

### 5. Quantile crossing

Por defecto:

```text
--crossing_policy error
```

La generación se detiene si se detecta crossing. Existe una opción `sort`, pero cualquier corrección debe ser una decisión metodológica explícita.

## Tests

`tests/test_schaake.py` reproduce el ejemplo pequeño de la metodología y verifica que el resultado final preserve exactamente las marginales discretizadas mediante permutación.

```powershell
pytest
```

## Próximos pasos

- validación cuantitativa de los escenarios generados;
- comparación `same_calendar_date` vs `moving_window`;
- Analog Schaake Shuffle;
- evaluación de autocorrelación, ramps, persistencia y eventos conjuntos;
- comparación zero-shot vs TimesFM fine-tuned;
- tratamiento explícito de colas fuera de Q10-Q90.
