VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
RUFF=$(VENV)/bin/ruff

COMPOSE=docker compose
LLM_BASE=hf.co/Qwen/Qwen3-8B-GGUF:Q8_0
LLM=qwen3-8b-rag
EMBEDDING=hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0

.PHONY: venv env install up down models check-config cli app lint lint-check

venv:
	@python -m venv $(VENV)

env:
	@test -f .env && echo ".env already exists, leaving it untouched." || cp .env.sample .env

install:
	@$(PIP) install --upgrade pip && $(PIP) install -e ".[dev]"
	@# NLTK ignores data others can write to, and llama-index keeps its NLTK data inside the venv
	@chmod -R go-w $(VENV)

up:
	@$(COMPOSE) up -d

down:
	@$(COMPOSE) down

models: up
	@$(COMPOSE) exec ollama ollama pull $(LLM_BASE)
	@$(COMPOSE) exec ollama ollama pull $(EMBEDDING)
	@$(COMPOSE) cp Modelfile ollama:/Modelfile
	@$(COMPOSE) exec ollama ollama create $(LLM) -f /Modelfile
	@$(COMPOSE) exec ollama ollama list

check-config:
	@$(PYTHON) test_config.py

cli:
	@$(PYTHON) main.py --mock

app:
	@$(PYTHON) app.py

lint:
	@$(RUFF) check --fix .
	@$(RUFF) format .

lint-check:
	@$(RUFF) check .
	@$(RUFF) format --check .
