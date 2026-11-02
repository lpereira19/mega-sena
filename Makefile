#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mega-sena
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download incremental + ETL dos resultados (data/raw -> data/processed)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mega_sena.dataset

## Features por concurso (pares, quadrantes, soma...)
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mega_sena.features

## Testes de hipótese (qui-quadrado + Monte Carlo)
.PHONY: hypothesis
hypothesis:
	$(PYTHON_INTERPRETER) -m mega_sena.hypothesis

## Backtest walk-forward das estratégias
.PHONY: backtest
backtest:
	$(PYTHON_INTERPRETER) -m mega_sena.modeling.backtest

## Figuras da EDA e do backtest (reports/figures)
.PHONY: figures
figures:
	$(PYTHON_INTERPRETER) -m mega_sena.plots

## Pipeline completo
.PHONY: all
all: data features hypothesis backtest figures


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
