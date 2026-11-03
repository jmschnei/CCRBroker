# routing_data_kit

Kit para construir/reutilizar datasets de LLM routing y evaluarlos (E1–E8). Ver `PLAN.md`.

## Instalación mínima
    pip install numpy pandas scipy scikit-learn pyyaml pyarrow   # + vllm y datasets en tu máquina GPU
    python tests/test_synthetic.py                               # 7 pruebas sin GPU ni red

## Flujo típico
1. `catalog/dataset_catalog.yaml`: qué reutilizar y qué verificar.
2. `routedata/adapters/*`: convertir datasets públicos al esquema (primero `inspect`).
3. `routedata/metrics.py` + `stats.py` + `splits.py`: evaluar con baselines triviales, IC y pruebas pareadas.
4. `generate/configroute.py`: generar ConfigRoute con vLLM (`VLLMBackend`), reanudable, por modelo.
5. `generate/driftroute.py`, `agentroute.py`, `advroute.py`: fases 3–5.
6. `estimate_budget.py`: estima GPU-horas (calibra antes con un piloto).

## Avisos
- No probado con datos reales (sin red/GPU en el entorno de desarrollo).
- RouterArena es solo de evaluación (el código lo bloquea para entrenamiento).
- `run_code_tests` ejecuta código generado: solo en sandbox aislado.
- Cargar pickles (RouterBench) puede ejecutar código: verifica origen y hash.
