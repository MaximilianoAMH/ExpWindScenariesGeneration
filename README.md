# TimesFM + Schaake Shuffle — escenarios de potencia eólica

Repositorio para generar escenarios probabilísticos de **potencia media diaria** con **TimesFM 2.5** y **Schaake Shuffle**.

La velocidad del viento se transforma a potencia antes del modelo probabilístico:

```text
viento horario [m/s]
        ↓
curva de potencia Alto Baguales
        ↓
potencia horaria [MW]
        ↓
agregación de días completos
        ↓
potencia media diaria [MW]
        ↓
TimesFM 2.5 zero-shot
        ↓
Q10, Q20, ..., Q90 de potencia
        ↓
Schaake Shuffle
        ↓
escenarios de potencia [MW]
```

El target de TimesFM es `POWER_MW`. El futuro observado no participa en la generación; se utiliza posteriormente como ground truth para evaluación.

## 1. Estructura principal

```text
ExpWindScenariesGeneration/
├── README.md
├── pyproject.toml
├── uv.lock
│
├── data/
│   ├── wind_hourly.csv
│   ├── power_ab_hourly.csv
│   ├── power_ab_daily.csv
│   ├── power_ab_daily.metadata.json
│   └── power_curves/
│       └── alto_baguales_3p5mw.csv
│
├── outputs/
│
├── notebooks/
│   └── TimesFM_Schaake_Quickstart.ipynb
│
├── docs/
│   ├── Metodologia_Schaake_Shuffle.pdf
│   └── Metodologia_Schaake_Shuffle.tex
│
├── scripts/
│   ├── preprocess_power_ab.py
│   └── diagnostics/
│       ├── check_timesfm_installation.py
│       ├── check_timesfm_channels.py
│       ├── inspect_data.py
│       ├── plot_data_overview.py
│       └── plot_wind_power_conversion.py
│
├── src/
│   └── wind_scenarios/
│       ├── data.py
│       ├── forecasting.py
│       ├── historical.py
│       ├── schaake.py
│       ├── evaluation.py
│       ├── power.py
│       └── cli/
│           ├── generate.py
│           └── visualize.py
│
└── tests/
```

## 2. Instalación

El proyecto usa **uv**.

Desde la raíz:

```powershell
uv sync
```

Ejecuta los comandos con `uv run`.

Para desarrollo y tests:

```powershell
uv sync --extra dev
uv run pytest
```

Para los experimentos de fine-tuning:

```powershell
uv sync --extra finetuning
```

## 3. Preparar el dataset de potencia

El punto de partida es:

```text
data/wind_hourly.csv
```

Por defecto se esperan:

```text
source_timestamp
wind_ab
```

Ejecuta:

```powershell
uv run python scripts\preprocess_power_ab.py
```

Para cada velocidad horaria \(v_{d,h}\) se calcula primero:

\[
P_{d,h}=f(v_{d,h}),
\]

donde \(f\) es la curva de potencia de Alto Baguales.

Sólo se utilizan días completos de 24 observaciones. Luego:

\[
POWER\_MW_d=
\frac{1}{24}
\sum_{h=1}^{24}P_{d,h}
\]

y:

\[
ENERGY\_MWH_d=
\sum_{h=1}^{24}P_{d,h}.
\]

El script genera:

```text
data/power_ab_hourly.csv
data/power_ab_daily.csv
data/power_ab_daily.metadata.json
```

`power_ab_hourly.csv` permite auditar la transformación horaria.

`power_ab_daily.csv` es el dataset de TimesFM:

```text
DATE,POWER_MW,ENERGY_MWH
```

`power_ab_daily.metadata.json` registra la configuración física y la capacidad efectiva.

### Días incompletos

Por defecto:

```text
--incomplete_day_policy error
```

detiene el preprocesamiento si encuentra un día con un número distinto de 24 observaciones.

Para descartarlos explícitamente:

```powershell
uv run python scripts\preprocess_power_ab.py `
    --incomplete_day_policy drop
```

Los gaps resultantes no se interpolan.

## 4. Curva de potencia

La curva por defecto está en:

```text
data/power_curves/alto_baguales_3p5mw.csv
```

Configuración base:

```text
turbine_model = Enercon E-138 EP3
rated_power_mw = 3.5
hub_height_m = 120
n_turbines = 1
availability = 1.0
electrical_efficiency = 1.0
```

Entre dos puntos consecutivos \((v_k,P_k)\) y \((v_{k+1},P_{k+1})\):

\[
P(v)=h_{0,k}+h_kv
\]

con:

\[
h_k=
\frac{P_{k+1}-P_k}
     {v_{k+1}-v_k},
\qquad
h_{0,k}=P_k-h_kv_k.
\]

Para velocidades superiores al último punto de la curva:

```text
v > 28 m/s -> P = 0
```

A nivel de planta:

\[
P_{\text{plant}}(v)=
P_{\text{turbine}}(v)
N_{\text{turbines}}
A
\eta_{\text{el}}.
\]

## 5. Generar escenarios de potencia

El generador lee:

```text
data/power_ab_daily.csv
```

y usa:

```text
POWER_MW
```

como variable objetivo.

Caso base:

```powershell
uv run python -m wind_scenarios.cli.generate `
    --origin 2014-04-01 `
    --historical_method same_calendar_date
```

Defaults:

```text
context_len = 32
horizon_len = 28
device = auto
probability_grid_mode = timesfm_support
power_bounds_policy = clip
```

Para cambiar contexto y horizonte:

```powershell
uv run python -m wind_scenarios.cli.generate `
    --origin 2014-04-01 `
    --historical_method same_calendar_date `
    --context_len 60 `
    --horizon_len 60
```

### Límites físicos

Por defecto:

```text
--power_bounds_policy clip
--power_min_mw 0
```

El límite superior se obtiene de `power_ab_daily.metadata.json` cuando existe.

También puede definirse manualmente:

```powershell
uv run python -m wind_scenarios.cli.generate `
    --origin 2014-04-01 `
    --historical_method same_calendar_date `
    --power_max_mw 3.5
```

Políticas disponibles:

```text
clip
error
none
```

## 6. Biblioteca histórica

### `same_calendar_date`

Selecciona una trayectoria completa por año histórico con el mismo mes y día que el origen.

Para:

```text
origin = 2014-04-01
horizon_len = 28
```

se buscan:

```text
1980-04-01 -> 1980-04-28
1981-04-01 -> 1981-04-28
...
2013-04-01 -> 2013-04-28
```

Con 1980-2013 se obtienen idealmente 34 trayectorias.

### `moving_window`

Permite aumentar la biblioteca desplazando el inicio alrededor de la fecha estacional.

Ejemplo:

```powershell
uv run python -m wind_scenarios.cli.generate `
    --origin 2014-04-01 `
    --historical_method moving_window `
    --n_scenarios 100 `
    --window_radius_days 15 `
    --min_start_distance_days 7
```

`window_radius_days` controla el rango de inicios candidatos.

`min_start_distance_days` limita la redundancia entre ventanas seleccionadas dentro del mismo año.

Cada trayectoria debe contener `horizon_len` observaciones diarias consecutivas y válidas.

## 7. Schaake Shuffle

TimesFM entrega marginales futuras de potencia:

```text
Q10, Q20, ..., Q90
```

Las trayectorias históricas de `POWER_MW` aportan los ranks y la estructura de dependencia temporal.

```text
TimesFM   -> marginales de potencia
Histórico -> dependencia ordinal de potencia
Schaake   -> escenarios temporales de potencia
```

El modo recomendado es:

```text
--probability_grid_mode timesfm_support
```

que utiliza el soporte `[0.1, 0.9]`.

La política por defecto frente a quantile crossing es:

```text
--crossing_policy error
```

## 8. Archivos de salida

Cada origen se guarda en:

```text
outputs/schaake/YYYY-MM-DD/
```

Ejemplo:

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
├── power_scenarios.csv
├── run_config.json
└── ground_truth.csv
```

| Archivo | Contenido |
|---|---|
| `context.csv` | potencia observada entregada a TimesFM |
| `timesfm_quantiles.csv` | Q10-Q90 de potencia futura |
| `historical_trajectories.csv` | trayectorias históricas de potencia |
| `historical_ranks.csv` | ranks utilizados por Schaake Shuffle |
| `power_scenarios.csv` | escenarios finales de potencia |
| `run_config.json` | configuración de la ejecución |
| `ground_truth.csv` | potencia observada para evaluación ex-post |

`ground_truth.csv` no participa en la generación.

## 9. Notebook

El notebook recomendado es:

```text
notebooks/TimesFM_Schaake_Quickstart.ipynb
```

En VS Code selecciona:

```text
.venv\Scripts\python.exe
```

Caso base:

```python
result = generate_power_scenarios(
    origin="2014-04-01",
    historical_method="same_calendar_date",
    context_len=32,
    horizon_len=28,
)
```

100 escenarios:

```python
result = generate_power_scenarios(
    origin="2014-04-01",
    historical_method="moving_window",
    context_len=32,
    horizon_len=28,
    n_scenarios=100,
    window_radius_days=15,
    min_start_distance_days=7,
)
```

El notebook también permite preprocesar los datos, revisar los CSV, generar gráficos y exportar todas las figuras sin volver a ejecutar TimesFM.

## 10. Visualización

### Escenario individual con Q10, Q50 y Q90

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --scenario_number 1
```

### Escenario individual con Q10-Q90

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --scenario_number 1 `
    --all_quantiles
```

### Fan plot

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode fan `
    --all_quantiles
```

### Exportar todos con Q10, Q50 y Q90

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --export_all
```

Se guardan en:

```text
power_scenario_plots/
```

### Exportar todos con Q10-Q90

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --all_quantiles `
    --export_all
```

Se guardan en:

```text
power_scenario_plots_all_quantiles/
```

## 11. Diagnóstico viento → potencia

Ejecuta:

```powershell
uv run python scripts\diagnostics\plot_wind_power_conversion.py
```

Genera:

```text
outputs/diagnostics/hourly_wind_vs_power.png
outputs/diagnostics/hourly_points_on_power_curve.png
```

El primero compara viento horario y potencia calculada en el tiempo.

El segundo muestra las observaciones transformadas sobre la curva de potencia.

## 12. Diagnósticos y tests

Verificar TimesFM:

```powershell
uv run python scripts\diagnostics\check_timesfm_installation.py
```

Inspeccionar canales:

```powershell
uv run python scripts\diagnostics\check_timesfm_channels.py
```

Inspeccionar el dataset diario de potencia:

```powershell
uv run python scripts\diagnostics\inspect_data.py
```

Graficar la serie diaria:

```powershell
uv run python scripts\diagnostics\plot_data_overview.py
```

Tests:

```powershell
uv run pytest
```

## 13. Documentación metodológica

```text
docs/Metodologia_Schaake_Shuffle.pdf
docs/Metodologia_Schaake_Shuffle.tex
```

## 14. Fine-tuning

El pipeline principal utiliza:

```text
google/timesfm-2.5-200m-transformers
```

en modo zero-shot.

Los experimentos de fine-tuning se mantienen separados en:

```text
experiments/finetuning/
```
