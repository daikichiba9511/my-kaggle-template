.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash
COMPE :=
PYTHONPATH := $(shell pwd)

.PHONY: setup
setup: ## setup install packages
	@python -m pip install --upgrade pip setuptools wheel
	@python -m pip install -e .
	@python -m pip install -e .[dev] --no-warn-script-location
	# if command -v uv &> /dev/null; then \
	# 	echo bootstrap of uv; \
	# 	curl -LsSf https://astral.sh/uv/install.sh | sh; \
	# fi
	# @uv python pin 3.10
	# @uv sync
	@echo "Setup Done ✅"

.PHONY: download_data
download_data: ## download data from competition page
	@kaggle competitions download -c "${COMPE}" -p ./input;
	@unzip "./input/${COMPE}.zip" -d "./input/${COMPE}"

.PHONY: upload
upload: ## upload dataset
	@kaggle datasets version --dir-mode zip -p ./src -m "update $(date +'%Y-%m-%d %H:%M:%S')"
	@kaggle datasets version --dir-mode zip -p ./output/submit -m "update $(date +'%Y-%m-%d %H:%M:%S')"

.PHONY: lint
lint: ## lint code
	@ruff check scripts src

.PHONY: mypy
mypy: ## typing check
	@mypy --config-file pyproject.toml scripts src

.PHONY: fmt
fmt: ## auto format
	@ruff check --fix scripts src
	@ruff format scripts src

.PHONY: test
test: ## run test with pytest
	@pytest -c tests

.PHONY: setup-dev
setup-dev: ## setup my dev env by installing my dotfiles
	[ ! -d ~/dotfiles ] && git@github.com:daikichiba9511/dotfiles.git ~/dotfiles ;\
	cd ~/dotfiles && bash setup.sh && cd -

.PHONY: lock
lock: ## lock dependencies
	if command -v uv &> /dev/null; then \
		uv lock ; \
	else \
		pip freeze > requirements.txt; \
	fi

.PHONY: clean
clean: ## clean outputs
	@rm -rf ./output/*
	@rm -rf ./wandb
	@rm -rf ./debug
	@rm -rf ./.venv

%:
	@echo 'command "$@" is not found.'
	@$(MAKE) help
	@exit 1

help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
