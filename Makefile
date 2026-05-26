PYTHON ?= python3
COMMON_SRC ?= ../renquant-common/src
export PYTHONPATH := $(COMMON_SRC):src:$(PYTHONPATH)

.PHONY: test doctor

test:
	$(PYTHON) -m pytest -q

doctor:
	$(PYTHON) -c "from renquant_base_data import DataManifestValidationPipeline, validate_data_manifest; from renquant_common import Pipeline; print('renquant-base-data ok')"
