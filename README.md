# TimesFM + Schaake Shuffle — escenarios de generación eólica diaria

Repositorio para generar escenarios probabilísticos de **energía total generada por día** usando **TimesFM 2.5** y **Schaake Shuffle**.

> En términos físicos, la generación acumulada durante un día es **energía**, no potencia. Por eso el target del modelo es `ENERGY_MWH`.

```text
velocidad del viento horaria [m/s]
              ↓
curva de potencia Alto Baguales
              ↓
potencia horaria [MW]
              ↓
integración de 24 intervalos de 1 h
              ↓
ENERGY_MWH diaria
              ↓
TimesFM
              ↓
Schaake Shuffle
              ↓
escenarios de generación diaria [MWh]
```

Para un día completo:

\[
E_d=\sum_{h=1}^{24}P_{d,h}\Delta t,
\qquad \Delta t=1\,h.
\]

Por tanto, con datos horarios:

\[
E_d[\mathrm{MWh}]
=
\sum_{h=1}^{24}
P_{d,h}[\mathrm{MW}]
\times 1[\mathrm{h}].
\]

## 1. Instalación

Desde la raíz:

```powershell
uv sync
```

Ejecuta los comandos del repositorio mediante `uv run`.

Para tests:

```powershell
uv sync --extra dev
uv run pytest
```

## 2. Datos de entrada

El archivo horario esperado es:

```text
data/wind_hourly.csv
```

con:

```text
source_timestamp
wind_ab
```

La curva de potencia está en:

```text
data/power_curves/alto_baguales_3p5mw.csv
```

## 3. Construir la serie diaria de generación

Ejecuta:

```powershell
uv run python scripts\preprocess_power_ab.py
```

El procesamiento sigue este orden:

```text
WIND_MS horaria
      ↓
POWER_MW horaria
      ↓
sólo días completos de 24 h
      ↓
ENERGY_MWH diaria
```

Se generan:

```text
data/power_ab_hourly.csv
data/power_ab_daily.csv
data/power_ab_daily.metadata.json
```

El dataset diario contiene:

```text
DATE,POWER_MW,ENERGY_MWH
```

`POWER_MW` es la potencia media del día y se conserva como variable auxiliar.

`ENERGY_MWH` es la suma de la energía producida durante las 24 horas y es la variable utilizada por TimesFM.

### Días incompletos

Por defecto:

```text
--incomplete_day_policy error
```

Para descartarlos explícitamente:

```powershell
uv run python scripts\preprocess_power_ab.py `
    --incomplete_day_policy drop
```

Los gaps no se rellenan ni interpolan.

## 4. Límites físicos

La curva base alcanza:

```text
3.5 MW por turbina
```

Para una turbina funcionando a potencia nominal durante 24 horas, el máximo diario teórico es:

\[
E_{\max}=3.5\times24=84\;\mathrm{MWh/día}.
\]

Con varias turbinas y factores de disponibilidad/eficiencia:

\[
E_{\max,d}
=
P_{\mathrm{rated}}
N_{\mathrm{turbines}}
A
\eta_{\mathrm{el}}
\times24.
\]

El preprocesamiento guarda este valor como:

```text
max_daily_energy_mwh
```

en:

```text
data/power_ab_daily.metadata.json
```

## 5. Generar escenarios

Caso base:

```powershell
uv run python -m wind_scenarios.cli.generate `
    --origin 2014-04-01 `
    --historical_method same_calendar_date
```

Defaults:

```text
target_column = ENERGY_MWH
context_len = 32
horizon_len = 28
device = auto
probability_grid_mode = timesfm_support
energy_bounds_policy = clip
energy_min_mwh = 0
```

El límite superior se lee automáticamente desde el metadata del preprocesamiento.

También puede fijarse manualmente:

```powershell
uv run python -m wind_scenarios.cli.generate `
    --origin 2014-04-01 `
    --historical_method same_calendar_date `
    --energy_max_mwh 84
```

Para cambiar contexto y horizonte:

```powershell
uv run python -m wind_scenarios.cli.generate `
    --origin 2014-04-01 `
    --historical_method same_calendar_date `
    --context_len 60 `
    --horizon_len 60
```

## 6. Selección histórica

### `same_calendar_date`

Selecciona una trayectoria completa de `ENERGY_MWH` por año histórico con el mismo mes y día que el origen.

Ejemplo para un origen `2014-04-01` y `H=28`:

```text
1980-04-01 -> 1980-04-28
1981-04-01 -> 1981-04-28
...
2013-04-01 -> 2013-04-28
```

### `moving_window`

Ejemplo con 100 escenarios:

```powershell
uv run python -m wind_scenarios.cli.generate `
    --origin 2014-04-01 `
    --historical_method moving_window `
    --n_scenarios 100 `
    --window_radius_days 15 `
    --min_start_distance_days 7
```

Cada trayectoria histórica continúa siendo una secuencia completa y diaria de longitud `H`.

## 7. Schaake Shuffle

TimesFM genera cuantiles marginales diarios de `ENERGY_MWH`.

Las trayectorias históricas de `ENERGY_MWH` aportan los ranks utilizados para transferir dependencia temporal.

```text
TimesFM
    -> marginales de energía diaria

Histórico
    -> dependencia ordinal

Schaake Shuffle
    -> escenarios de generación diaria
```

El modo recomendado es:

```text
--probability_grid_mode timesfm_support
```

La política por defecto frente a quantile crossing es:

```text
--crossing_policy error
```

## 8. Salidas

Cada origen genera:

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
├── energy_scenarios.csv
├── run_config.json
└── ground_truth.csv
```

`energy_scenarios.csv` contiene los escenarios finales en MWh/día.

`ground_truth.csv` contiene:

```text
DATE,ENERGY_MWH,LEAD_DAY
```

y se utiliza sólo para evaluación ex-post.

## 9. Notebook

El notebook es:

```text
notebooks/TimesFM_Schaake_Quickstart.ipynb
```

En VS Code selecciona:

```text
.venv\Scripts\python.exe
```

Caso base:

```python
result = generate_energy_scenarios(
    origin="2014-04-01",
    historical_method="same_calendar_date",
    context_len=32,
    horizon_len=28,
)
```

## 10. Visualización

Escenario individual con Q10, Q50 y Q90:

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --scenario_number 1
```

Con Q10-Q90:

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --scenario_number 1 `
    --all_quantiles
```

Fan plot:

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode fan `
    --all_quantiles
```

Exportar todos con Q10, Q50 y Q90:

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --export_all
```

Se guardan en:

```text
energy_scenario_plots/
```

Exportar todos con Q10-Q90:

```powershell
uv run python -m wind_scenarios.cli.visualize `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --all_quantiles `
    --export_all
```

Se guardan en:

```text
energy_scenario_plots_all_quantiles/
```

El eje Y de los gráficos es:

```text
Daily generated energy [MWh]
```

y el ground truth aparece como:

```text
Daily Generation Ground Truth
```

## 11. Diagnósticos

Conversión horaria viento → potencia:

```powershell
uv run python scripts\diagnostics\plot_wind_power_conversion.py
```

Inspeccionar la serie diaria:

```powershell
uv run python scripts\diagnostics\inspect_data.py
```

Graficar primeros y últimos cinco años:

```powershell
uv run python scripts\diagnostics\plot_data_overview.py
```

Verificar TimesFM:

```powershell
uv run python scripts\diagnostics\check_timesfm_installation.py
```

Revisar canales de TimesFM:

```powershell
uv run python scripts\diagnostics\check_timesfm_channels.py
```

## 12. Tests

```powershell
uv run pytest
```

## 13. Documentación metodológica

```text
docs/Metodologia_Schaake_Shuffle.pdf
docs/Metodologia_Schaake_Shuffle.tex
```
