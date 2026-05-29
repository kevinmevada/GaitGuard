IMAGE ?= gaitguard:latest
CONFIG ?= configs/pipeline_config.yaml
STAGE ?= all

.PHONY: help install pipeline stage api docker-build docker-run docker-stage docker-api clean

help:
	@echo "Targets:"
	@echo "  make install       Install pipeline + API dependencies locally"
	@echo "  make pipeline      Run full pipeline locally"
	@echo "  make stage STAGE=evaluate   Run a single stage locally"
	@echo "  make api           Start API locally on port 8001"
	@echo "  make docker-build  Build reproducible container image"
	@echo "  make docker-run    Run full pipeline in container"
	@echo "  make docker-stage STAGE=evaluate   Run one stage in container"
	@echo "  make docker-api    Start API in container on port 8001"

install:
	pip install -r fall_risk_pipeline/requirements.txt
	pip install -r api/requirements.txt

pipeline:
	cd fall_risk_pipeline && python main.py --config $(CONFIG)

stage:
	cd fall_risk_pipeline && python main.py --config $(CONFIG) --stage $(STAGE)

api:
	cd api && python -m uvicorn main:app --host 0.0.0.0 --port 8001

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -it \
		-v "$(CURDIR)/fall_risk_pipeline/data:/app/fall_risk_pipeline/data" \
		-v "$(CURDIR)/fall_risk_pipeline/results:/app/fall_risk_pipeline/results" \
		-v "$(CURDIR)/fall_risk_pipeline/logs:/app/fall_risk_pipeline/logs" \
		$(IMAGE)

docker-stage:
	docker run --rm -it \
		-v "$(CURDIR)/fall_risk_pipeline/data:/app/fall_risk_pipeline/data" \
		-v "$(CURDIR)/fall_risk_pipeline/results:/app/fall_risk_pipeline/results" \
		-v "$(CURDIR)/fall_risk_pipeline/logs:/app/fall_risk_pipeline/logs" \
		$(IMAGE) python main.py --config $(CONFIG) --stage $(STAGE)

docker-api:
	docker run --rm -it -p 8001:8001 \
		-v "$(CURDIR)/fall_risk_pipeline/data:/app/fall_risk_pipeline/data" \
		-v "$(CURDIR)/fall_risk_pipeline/results:/app/fall_risk_pipeline/results" \
		$(IMAGE) bash -lc "cd /app/api && python -m uvicorn main:app --host 0.0.0.0 --port 8001"

clean:
	rm -rf fall_risk_pipeline/__pycache__ api/__pycache__
