PYTHON ?= python3

.PHONY: build smoke

build:
	$(PYTHON) -m pip install -v -e . --no-build-isolation

smoke:
	$(PYTHON) -m ampere_kv.smoke
