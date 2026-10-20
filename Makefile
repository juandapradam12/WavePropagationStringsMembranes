PYTHON ?= python3
DATA_DIR := data
FIGURE_DIR := figures
RESULTS_DIR := results

.PHONY: all run figures clean test

all: run

run:
	$(PYTHON) scripts/run_pipeline.py \
		--data-dir $(DATA_DIR) \
		--figure-dir $(FIGURE_DIR) \
		--results-dir $(RESULTS_DIR)

figures: run

test:
	$(PYTHON) -m pytest -q tests

clean:
	rm -rf $(RESULTS_DIR)/* $(FIGURE_DIR)/*.png
	find . -type d -name __pycache__ -exec rm -rf {} +
