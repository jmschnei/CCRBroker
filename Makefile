# Makefile de routing_data_kit
# Uso típico: make install && make test && make phase1

PYTHON ?= python
VENV   ?= .venv
PY     ?= $(VENV)/bin/python
PIP    ?= $(PY) -m pip
PHASE  ?= 1          # fase de ingestión disponible (1..5)
OUT    ?= outputs
N_QUERIES ?= 20000
N_CFGS    ?= 60
AVG_IN    ?= 400
AVG_OUT   ?= 600
DECODE_TPS ?= 2500
PREFILL_TPS ?= 20000
N_GPUS     ?= 1
USD_GPU_H  ?= 1.8
UTIL       ?= 0.6
MODEL      ?= Qwen/Qwen3-0.6B
REQUESTS   ?= 64
BATCH_SIZE ?= 8
WARMUP     ?= 8
GPU_USD_H  ?=
GPU_PRICE_SOURCE ?=

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Comandos disponibles:"
	@echo "  make venv        - crea .venv si no existe"
	@echo "  make install     - instala paquete + dependencias de desarrollo/pruebas"
	@echo "  make install-gpu - instala desarrollo, evaluación y vLLM"
	@echo "  make test        - pytest"
	@echo "  make validate    - valida la estructura de data/raw/"
	@echo "  make validate-unified - valida manifiesto, filas y claves del dataset procesado"
	@echo "  make lint        - ruff check"
	@echo "  make format      - ruff format"
	@echo "  make typecheck   - mypy"
	@echo "  make budget      - estima GPU-horas (estimate_budget.py)"
	@echo "  make phase0-measure MODEL=... GPU_USD_H=... GPU_PRICE_SOURCE=... - mide tok/s (Linux/NVIDIA CUDA)"
	@echo "  make phase1      - regenera el unificado con RouterBench/SPROUT/LLMRouterBench"
	@echo "  make phase1-audit - audita pool oficial, 15 datasets, labels y cobertura"
	@echo "  make phase1-eval - ejecuta routers × 5 semillas en los 15 datasets"
	@echo "  make phase1-arena-setup - clona RouterArena oficial (commit fijado) y prepara sub_10"
	@echo "  make phase1-arena-generate - respuestas del pool a sub_10 (GPU; BACKEND=vllm|openai|mock)"
	@echo "  make phase1-arena-candidates - evalúa cada candidato con el evaluador oficial"
	@echo "  make phase2..5   - aún no tienen CLI ejecutable (son APIs/prototipos)"
	@echo "  make run         - ejecuta la fase seleccionada (PHASE=1 disponible)"
	@echo "  make clean       - borra artefactos locales (outputs/, .venv/)"
	@echo "  make distclean   - clean + pycache"

.PHONY: venv
venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

.PHONY: install
install: venv
	@$(PIP) install -U pip wheel
	@$(PIP) install -e '.[dev]'
	@echo "Paquete y dependencias de desarrollo instalados."
	@echo "Para GPU:  $(PIP) install -e '.[vllm]'"
	@echo "Para eval: $(PIP) install -e '.[eval]'"

.PHONY: install-gpu
install-gpu: venv
	@$(PIP) install -e '.[dev,vllm,eval]'

.PHONY: test
test: venv
	@$(PY) -m pytest tests/ -v

.PHONY: validate
validate: venv
	@$(PY) -m routedata.validate

.PHONY: validate-unified
validate-unified: venv
	@$(PY) -m routedata.validate --unified

.PHONY: lint
lint: venv
	@$(PIP) install -q 'ruff>=0.4,<1.0' 2>/dev/null || true
	@$(PY) -m ruff check routedata generate scripts tests estimate_budget.py

.PHONY: format
format: venv
	@$(PIP) install -q 'ruff>=0.4,<1.0' 2>/dev/null || true
	@$(PY) -m ruff format routedata generate scripts tests estimate_budget.py
	@$(PY) -m ruff check --fix routedata generate scripts tests estimate_budget.py

.PHONY: typecheck
typecheck: venv
	@$(PIP) install -q 'mypy>=1.8,<2.0' 2>/dev/null || true
	@$(PY) -m mypy routedata generate scripts

.PHONY: budget
budget: venv
	@$(PY) -m estimate_budget --n-queries $(N_QUERIES) --n-configs $(N_CFGS) \
		--avg-in $(AVG_IN) --avg-out $(AVG_OUT) --decode-tps $(DECODE_TPS) \
		--prefill-tps $(PREFILL_TPS) --n-gpus $(N_GPUS) --usd-gpu-hour $(USD_GPU_H) \
		--utilization $(UTIL)

.PHONY: phase0-measure
phase0-measure: venv
	@$(PY) scripts/benchmark_vllm.py --model "$(MODEL)" --requests $(REQUESTS) \
		--batch-size $(BATCH_SIZE) --warmup $(WARMUP) --gpu-usd-hour "$(GPU_USD_H)" \
		--gpu-price-source "$(GPU_PRICE_SOURCE)"

# ---- Fases ------------------------------------------------------------------
.PHONY: phase1
phase1: venv
	@$(PY) regenerate_combined.py

.PHONY: phase1-eval
phase1-eval: venv
	@$(PY) scripts/run_phase1_multidataset.py

BACKEND ?= vllm

.PHONY: phase1-arena-setup phase1-arena-generate phase1-arena-candidates
phase1-arena-setup: venv
	@$(PY) scripts/run_routerarena.py setup

phase1-arena-generate: venv
	@$(PY) scripts/run_routerarena.py generate --backend $(BACKEND)

phase1-arena-candidates: venv
	@$(PY) scripts/run_routerarena.py candidates --backend $(BACKEND)

.PHONY: phase1-audit
phase1-audit: venv
	@$(PY) scripts/run_phase1_multidataset.py --audit-only

.PHONY: phase2
phase2: venv
	@echo "ConfigRoute no tiene CLI autónoma: requiere queries, configs y backend_factory." >&2
	@echo "Use generate.configroute.run(...) desde un experimento configurado." >&2
	@exit 2

.PHONY: phase3
phase3: venv
	@echo "DriftRoute contiene funciones de comparación/replay, pero no un CLI de generación." >&2
	@exit 2

.PHONY: phase4
phase4: venv
	@echo "AgentRoute requiere adaptadores Env y backends; todavía no tiene CLI autónoma." >&2
	@exit 2

.PHONY: phase5
phase5: venv
	@echo "AdvRoute contiene utilidades de evaluación; todavía no tiene CLI autónoma." >&2
	@exit 2

.PHONY: run
run: venv
	@case "$(PHASE)" in \
		1) $(MAKE) phase1 ;; \
		2) $(MAKE) phase2 ;; \
		3) $(MAKE) phase3 ;; \
		4) $(MAKE) phase4 ;; \
		5) $(MAKE) phase5 ;; \
		*) echo "PHASE debe ser un entero entre 1 y 5" >&2; exit 2 ;; \
	esac

# ---- Limpieza ----------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf $(OUT) *.egg-info .pytest_cache .mypy_cache .ruff_cache
	rm -f .coverage

.PHONY: distclean
distclean: clean
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
