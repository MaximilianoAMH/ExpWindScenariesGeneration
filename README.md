# Experimentos TimesFM — Generación de escenarios de viento y potencia

Repositorio para generar escenarios probabilísticos de velocidad media diaria del viento con **TimesFM 2.5** y **Schaake Shuffle**, y convertir posteriormente esos escenarios a potencia eólica para **Alto Baguales**.

El pipeline principal es:

```text
Contexto observado
        |
        v
TimesFM 2.5 zero-shot
Q10, Q20, ..., Q90
        |
        v
Schaake Shuffle
        |
        v
Escenarios de viento [m/s]
        |
        v
Curva de potencia Alto Baguales
        |
        v
Escenarios de potencia [MW]
```

TimesFM determina las marginales futuras y Schaake Shuffle transfiere una estructura de dependencia temporal basada en trayectorias históricas completas.

El ground truth del período futuro se usa únicamente para evaluación ex-post y no participa en la generación.

---

## 1. Estructura del repositorio

```text
ExpWindScenariesGeneration/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── data/
│   ├── wind_ab_daily.csv
│   └── power_curves/
│       └── alto_baguales_3p5mw.csv
│
├── outputs/
│
├── docs/
│   ├── Metodologia_Schaake_Shuffle.pdf
│   └── Metodologia_Schaake_Shuffle.tex
│
├── notebooks/
│   └── TimesFM_Schaake_Quickstart.ipynb
│
├── src/
│   └── wind_scenarios/
│       ├── __init__.py
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
├── scripts/
│   ├── preprocess_wind_ab.py
│   └── diagnostics/
│       ├── check_timesfm_installation.py
│       ├── check_timesfm_channels.py
│       ├── inspect_data.py
│       └── plot_data_overview.py
│
├── experiments/
│   └── finetuning/
│       ├── train_lora.py
│       └── evaluate_test.py
│
└── tests/
```

Los módulos principales son:

- `data.py`: carga y validación del dataset diario.
- `forecasting.py`: inferencia zero-shot con TimesFM y extracción de cuantiles.
- `historical.py`: selección de trayectorias históricas.
- `schaake.py`: ranks, discretización de marginales y Schaake Shuffle.
- `evaluation.py`: lectura y exportación del ground truth.
- `power.py`: conversión viento → potencia.
- `cli/generate.py`: pipeline completo de generación.
- `cli/visualize.py`: generación de gráficos.

---

## 2. Instalación

El repositorio usa **uv** para gestionar entorno y dependencias.

Desde la raíz del proyecto:

```powershell
uv sync
```

No es necesario activar manualmente `.venv`. Los comandos deben ejecutarse con `uv run`.

Ejemplos:

```powershell
uv run generate-wind-scenarios ...
uv run visualize-wind-scenarios ...
uv run pytest
```

Para dependencias de desarrollo:

```powershell
uv sync --extra dev
```

Para los experimentos de fine-tuning:

```powershell
uv sync --extra finetuning
```

---

## 3. Datos de entrada

El pipeline espera por defecto:

```text
data/wind_ab_daily.csv
```

con columnas:

```text
DATE,WIND
```

La variable `WIND` corresponde a velocidad media diaria del viento.

La serie usada actualmente cubre:

```text
1980-2017
```

con la separación:

```text
1980-2013 -> biblioteca histórica
2014-2017 -> período de test
```

Si se parte de datos horarios:

```powershell
uv run python scripts\preprocess_wind_ab.py
```

El preprocesamiento calcula medias diarias sólo para días con 24 observaciones completas y conserva los gaps de calendario.

---

## 4. Generación de escenarios

El comando principal es:

```powershell
uv run generate-wind-scenarios `
    --origin 2014-04-01 `
    --historical_method same_calendar_date
```

Los valores por defecto son:

```text
context_len = 32 días
horizon_len = 28 días
device = auto
```

Para cambiar contexto y horizonte:

```powershell
uv run generate-wind-scenarios `
    --origin 2014-04-01 `
    --historical_method same_calendar_date `
    --context_len 60 `
    --horizon_len 60
```

`context_len` es el número de días observados entregados a TimesFM.

`horizon_len` es el número de días futuros generados. El mismo horizonte se usa en TimesFM, en las trayectorias históricas y en la matriz final de escenarios.

Por ejemplo:

```text
origin = 2014-04-01
context_len = 60
horizon_len = 60
```

produce:

```text
contexto: 2014-01-31 -> 2014-03-31
forecast: 2014-04-01 -> 2014-05-30
```

### Selección del dispositivo

```text
--device auto
--device cpu
--device cuda
```

`auto` usa CUDA cuando está disponible y CPU en caso contrario.

---

## 5. Selección de trayectorias históricas

### `same_calendar_date`

Selecciona una trayectoria por año histórico con el mismo mes y día que el origen.

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

Con 1980-2013, el máximo ideal es:

```text
L = 34 escenarios
```

Ejemplo:

```powershell
uv run generate-wind-scenarios `
    --origin 2014-04-01 `
    --historical_method same_calendar_date
```

### `moving_window`

Permite generar más trayectorias históricas desplazando la fecha de inicio alrededor de la fecha ancla de cada año.

Si:

```text
window_radius_days = 15
```

los candidatos de cada año pueden comenzar entre:

```text
anchor - 15 días
```

y:

```text
anchor + 15 días
```

Cada candidato sigue siendo una trayectoria completa de `H = horizon_len` días consecutivos.

Ejemplo para 100 escenarios:

```powershell
uv run generate-wind-scenarios `
    --origin 2014-04-01 `
    --historical_method moving_window `
    --n_scenarios 100 `
    --window_radius_days 15 `
    --min_start_distance_days 7
```

`min_start_distance_days` controla la separación mínima entre fechas de inicio seleccionadas dentro del mismo año histórico.

Con:

```text
--min_start_distance_days 7
```

no se seleccionan dos ventanas del mismo año cuyos inicios estén separados por menos de siete días.

Los candidatos inválidos se descartan si:

- no contienen `H` observaciones;
- atraviesan un gap de calendario;
- contienen fechas ausentes;
- quedan fuera del período histórico permitido.

---

## 6. Schaake Shuffle

Para cada día del horizonte, las trayectorias históricas se ordenan y se calcula el rank de cada trayectoria.

Aunque los ranks se calculan por columna, cada fila conserva su secuencia completa de ranks a través de los `H` días. Esa secuencia ordinal determina la dependencia temporal del escenario final.

En el pipeline:

```text
TimesFM      -> magnitudes marginales futuras
Histórico    -> estructura ordinal de dependencia
Schaake      -> escenarios temporalmente coherentes
```

### Probabilidades usadas para discretizar las marginales

El modo recomendado es:

```text
--probability_grid_mode timesfm_support
```

que usa niveles dentro de:

```text
[0.1, 0.9]
```

porque TimesFM entrega Q10-Q90.

El modo clásico:

```text
--probability_grid_mode schaake
```

usa:

```text
k / (L + 1)
```

y falla explícitamente si requiere probabilidades fuera del soporte disponible.

### Quantile crossing

La política por defecto es:

```text
--crossing_policy error
```

Si se detecta crossing entre cuantiles, la generación se detiene.

---

## 7. Conversión de viento a potencia

Después de generar los escenarios de viento, el pipeline aplica la curva de potencia de Alto Baguales:

```text
data/power_curves/alto_baguales_3p5mw.csv
```

La conversión produce:

```text
power_scenarios_alto_baguales.csv
```

Los valores por defecto son:

```text
turbine_model = Enercon E-138 EP3
hub_height_m = 120
rated_power_mw = 3.5
n_turbines = 1
availability = 1.0
electrical_efficiency = 1.0
```

La curva alcanza:

```text
3500 kW = 3.5 MW por turbina
```

### Función de potencia

Entre dos puntos consecutivos de la curva:

$$
(v_k,P_k),\qquad(v_{k+1},P_{k+1})
$$

se usa interpolación lineal:

$$
P(v)=h_{0,k}+h_kv
$$

con:

$$
h_k=
\frac{P_{k+1}-P_k}
     {v_{k+1}-v_k}
$$

y:

$$
h_{0,k}=P_k-h_kv_k.
$$

Para velocidades superiores al último punto de la curva:

```text
v > 28 m/s -> P = 0
```

A nivel de planta:

$$
P_{\text{plant}}(v)=
P_{\text{turbine}}(v)
N_{\text{turbines}}
A
\eta_{\text{el}}.
$$

La implementación actual transforma velocidad media diaria en potencia equivalente diaria. No representa todavía una integración horaria exacta de energía.

---

## 8. Evaluación y ground truth

El ground truth se obtiene con:

```text
src/wind_scenarios/evaluation.py
```

y contiene:

```text
DATE,WIND,LEAD_DAY
```

Para:

```text
origin = 2014-04-01
horizon_len = 28
```

se exportan exactamente las observaciones:

```text
2014-04-01 -> 2014-04-28
```

El ground truth se lee después de generar los escenarios y se utiliza únicamente para evaluación.

---

## 9. Uso desde Jupyter / VS Code

El notebook recomendado es:

```text
notebooks/TimesFM_Schaake_Quickstart.ipynb
```

Puede abrirse con:

```powershell
uv run --with jupyter jupyter lab
```

o directamente desde VS Code seleccionando como kernel el Python del entorno del proyecto:

```text
.venv\Scripts\python.exe
```

La función principal es:

```python
result = generate_wind_scenarios(
    origin="2014-04-01",
    historical_method="same_calendar_date",
    context_len=32,
    horizon_len=28,
)
```

Ejemplo de 60 días:

```python
result = generate_wind_scenarios(
    origin="2014-04-01",
    historical_method="same_calendar_date",
    context_len=60,
    horizon_len=60,
)
```

Ejemplo con 100 escenarios:

```python
result = generate_wind_scenarios(
    origin="2014-04-01",
    historical_method="moving_window",
    context_len=32,
    horizon_len=28,
    n_scenarios=100,
    window_radius_days=15,
    min_start_distance_days=7,
)
```

---

## 10. Archivos de salida

Cada origen genera una carpeta:

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
├── scenarios.csv
├── power_scenarios_alto_baguales.csv
└── ground_truth.csv
```

Los archivos principales son:

| Archivo | Contenido |
|---|---|
| `context.csv` | contexto exacto entregado a TimesFM |
| `timesfm_quantiles.csv` | Q10-Q90 para cada día futuro |
| `historical_trajectories.csv` | trayectorias históricas seleccionadas |
| `historical_ranks.csv` | ranks históricos por día |
| `scenarios.csv` | escenarios finales de viento |
| `power_scenarios_alto_baguales.csv` | escenarios de potencia en MW |
| `ground_truth.csv` | observaciones futuras para evaluación |

---

## 11. Visualización

El comando general es:

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode <modo>
```

Los modos disponibles son:

```text
single
fan
wind_power
power_curve
```

### Escenario individual con Q10, Q50 y Q90

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --scenario_number 1
```

### Escenario individual con Q10-Q90

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --scenario_number 1 `
    --all_quantiles
```

### Fan plot

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode fan `
    --all_quantiles
```

### Viento generado vs potencia calculada

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode wind_power `
    --scenario_number 1
```

Genera:

```text
scenario_001_wind_power.png
```

El eje izquierdo muestra velocidad del viento `[m/s]` y el eje derecho potencia `[MW]`.

### Escenario sobre la curva de potencia

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode power_curve `
    --scenario_number 1
```

Genera:

```text
scenario_001_power_curve.png
```

Este gráfico muestra los pares `(velocidad, potencia)` del escenario sobre la curva de Alto Baguales.

---

## 12. Exportar todos los gráficos

### Todos los escenarios con Q10, Q50 y Q90

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --export_all
```

Los PNG se guardan en:

```text
scenario_plots/
```

Desde el notebook:

```python
visualize_result(
    result,
    mode="single",
    all_quantiles=False,
    export_all=True,
    display_plot=False,
)
```

### Todos los escenarios con Q10-Q90

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode single `
    --all_quantiles `
    --export_all
```

Los PNG se guardan en:

```text
scenario_plots_all_quantiles/
```

Desde el notebook:

```python
visualize_result(
    result,
    mode="single",
    all_quantiles=True,
    export_all=True,
    display_plot=False,
)
```

### Todos los gráficos viento vs potencia

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode wind_power `
    --export_all
```

Se guardan en:

```text
wind_power_plots/
```

### Todos los gráficos sobre la curva de potencia

```powershell
uv run visualize-wind-scenarios `
    --input_dir outputs\schaake\2014-04-01 `
    --mode power_curve `
    --export_all
```

Se guardan en:

```text
power_curve_plots/
```

---

## 13. Diagnósticos

Verificar instalación de TimesFM:

```powershell
uv run python scripts\diagnostics\check_timesfm_installation.py
```

Inspeccionar los canales del modelo:

```powershell
uv run python scripts\diagnostics\check_timesfm_channels.py
```

Revisar el dataset:

```powershell
uv run python scripts\diagnostics\inspect_data.py
```

Graficar la serie histórica:

```powershell
uv run python scripts\diagnostics\plot_data_overview.py
```

---

## 14. Tests

Ejecutar todos los tests:

```powershell
uv run pytest
```

Los tests cubren la lógica de Schaake Shuffle y la conversión viento → potencia.

---

## 15. Documentación metodológica

La metodología de Schaake Shuffle se encuentra en:

```text
docs/Metodologia_Schaake_Shuffle.pdf
docs/Metodologia_Schaake_Shuffle.tex
```

El archivo `.tex` puede utilizarse directamente en Overleaf.

---
