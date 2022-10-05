SHELL := /bin/bash

down: ## Download click datasets
	rsync -Pr --include="*/" --include "*.parquet" --include "*.yaml" --exclude="*" ilps:/home/phager/Developer/ultr-cm-vs-ips/results .

.PHONY: help
help: ## List all commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.DEFAULT_GOAL := help
