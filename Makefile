# Makefile de routing_data_kit
# Uso tipico: make install && make test && make run-phase2

PYTHON ?= python
VENV   ?= .venv
PY     ?= $(VENV)/bin/python
PIP    ?= $(PY) -m pip
PHASE  ?= 2          # fase por defecto (1..5)
OUT    ?= outputs
N      ?= 200        # n de muestras por celda
WARMUP ?= 5          # warmup de vLLM por modelo

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Comandos disponibles:"
	@echo "  make venv        - crea .venv si no existe"
	@echo "  make install     - instala dependencias base (+ --extra vllm para GPU)"
	@echo "  make test        - pytest"
	@echo "  make lint        - ruff check"
	@echo "  make format      - ruff format"
	@echo "  make typecheck   - mypy"
	@echo "  make budget      - estima GPU-horas (estimate_budget.py)"
	@echo "  make phase1      - reutiliza datasets publicos (LLMRouterBench/...)"
	@echo "  make phase2      - ConfigRoute"
	@echo "  make phase3      - DriftRoute"
	@echo "  make phase4      - AgentRoute"
	@echo "  make phase5      - AdvRoute"
	@echo "  make run         - genera la fase $(PHASE)  (override: PHASE=3 make run)"
	@echo "  make clean       - borra artefactos locales (outputs/, .venv/)"
	@echo "  make distclean   - clean + pycache"

.PHONY: venv
venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

.PHONY: install
install: venv
	@$(PIP) install -U pip wheel
	@$(PIP) install -e .
	@echo "Dependencias instaladas."
	@echo "Para GPU:  $(PIP) install -e '.[vllm]'"
	@echo "Para eval: $(PIP) install -e '.[eval]'"

.PHONY: install-gpu
install-gpu: venv
	@$(PIP) install -e '.[vllm,eval]'

.PHONY: test
test: venv
	@$(PY) -m pytest tests/ -v

.PHONY: lint
lint: venv
	@$(PIP) install -q 'ruff>=0.4,<1.0' 2>/dev/null || true
	@$(PY) -m ruff check routedata generate tests estimate_budget.py

.PHONY: format
format: venv
	@$(PIP) install -q 'ruff>=0.4,<1.0' 2>/dev/null || true
	@$(PY) -m ruff format routedata generate tests estimate_budget.py
	@$(PY) -m ruff check --fix routedata generate tests estimate_budget.py

.PHONY: typecheck
typecheck: venv
	@$(PIP) install -q 'mypy>=1.8,<2.0' 2>/dev/null || true
	@$(PY) -m mypy routedata generate

.PHONY: budget
budget: venv
	@$(PY) estimate_budget.py --n $(N) --warmup $(WARMUP)

# ---- Fases ------------------------------------------------------------------
.PHONY: phase1
phase1: venv
	@echo "Fase 1: reutilizacion de LLMRouterBench / RouterBench / SPROUT / RouterArena"
	@$(PY) -c "from routedata.adapters import LLMRouterBench, RouterBench, SPROUT, RouterArena; \
print('LLMRouterBench ok, n =', LLMRouterBench().load().n); \
print('RouterBench  ok, n =', RouterBench().load().n); \
print('RouterArena  ok, n =', RouterArena().load().n)"

.PHONY: phase2
phase2: venv
	@$(PY) -m generate.configroute --n $(N) --out $(OUT)

.PHONY: phase3
phase3: venv
	@$(PY) -m generate.driftroute --n $(N) --out $(OUT)

.PHONY: phase4
phase4: venv
	@$(PY) -m generate.agentroute --n $(N) --out $(OUT)

.PHONY: phase5
phase5: venv
	@$(PY) -m generate.advroute --n $(N) --out $(OUT)

.PHONY: run
run: venv
	@$(PY) -c "import generate; \
import importlib; \
mod = importlib.import_module(generate.PHASES[$(PHASE)].module); \
mod.run(n=$(N), out='$(OUT)')" 2>&1 || \
$(PY) -m generate.configroute --n $(N) --out $(OUT)

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
