# Reorganización propuesta

La versión anterior mezclaba scripts de producción, diagnósticos, fine-tuning y visualización dentro de `src/`, y además podía crear `src/outputs/` si los comandos se ejecutaban desde esa carpeta.

La reorganización aplica cuatro reglas:

1. `src/wind_scenarios/` contiene únicamente código reusable del pipeline principal.
2. `experiments/` contiene código experimental, actualmente LoRA/fine-tuning.
3. `scripts/` contiene utilidades de preprocesamiento y diagnóstico.
4. todos los resultados se escriben en `outputs/` en la raíz del proyecto.

## Mapeo desde la estructura anterior

| Antes | Ahora |
|---|---|
| `src/dataset.py` | `src/wind_scenarios/data.py` |
| `src/scenarios/timesfm_forecast.py` | `src/wind_scenarios/forecasting.py` |
| `src/scenarios/historical_trajectories.py` | `src/wind_scenarios/historical.py` |
| `src/scenarios/schaake_shuffle.py` | `src/wind_scenarios/schaake.py` |
| `src/scenarios/run_schaake.py` | `src/wind_scenarios/cli/generate.py` |
| tres scripts de visualización | `src/wind_scenarios/cli/visualize.py` |
| `src/train.py` | `experiments/finetuning/train_lora.py` |
| `src/evaluate_test.py` | `experiments/finetuning/evaluate_test.py` |
| `src/test_2.py` | `scripts/diagnostics/test_timesfm_channels.py` |
| `src/inspect_data.py` | `scripts/diagnostics/inspect_data.py` |
| `src/preprocess_wind_ab.py` | `scripts/preprocess_wind_ab.py` |

Los archivos `test_timesfm.py` y `visualize_data.py` visibles en el repositorio local no se incorporaron a esta copia porque su contenido no estaba disponible en los archivos compartidos durante la reorganización. Conviene revisarlos y moverlos a `scripts/diagnostics/` si siguen siendo útiles, o eliminarlos si ya son redundantes.
