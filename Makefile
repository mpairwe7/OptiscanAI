.PHONY: install dev backend frontend train plots clean

# ==============================================================================
#  MLOps 2026 Pipeline - Retinal Disease Classification
#  Backend: FastAPI + UV | Frontend: Next.js 16 + Bun + Zustand + TanStack Query
#  Training: PyTorch DDP on 8x RTX A6000
# ==============================================================================

# --- Installation ---
install: install-backend install-frontend
	@echo "All dependencies installed."

install-backend:
	uv sync
	@echo "Backend (uv) ready."

install-frontend:
	cd frontend && bun install
	@echo "Frontend (bun) ready."

# --- Development ---
dev:
	@echo "Starting backend + frontend..."
	@make -j2 backend frontend

backend:
	PYTHONPATH=. uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 --reload

frontend:
	cd frontend && bun dev

# --- Training (Multi-GPU) ---
train:
	bash scripts/train_multigpu.sh configs/train.yaml 8

train-4gpu:
	bash scripts/train_multigpu.sh configs/train.yaml 4

train-1gpu:
	PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python3 train.py --config configs/train.yaml

# --- Plot Generation ---
plots:
	PYTHONPATH=. python3 scripts/generate_all_plots.py

plots-eda:
	PYTHONPATH=. python3 scripts/generate_all_plots.py --stages eda

# --- Full Pipeline ---
pipeline: train plots
	@echo "Full pipeline complete!"

# --- Build ---
build-frontend:
	cd frontend && bun run build

# --- Testing ---
test:
	PYTHONPATH=. pytest tests/ -v --tb=short

test-fast:
	PYTHONPATH=. pytest tests/ -v --tb=short -x -q

test-gate:
	PYTHONPATH=. pytest tests/test_fundus_gate_v2.py tests/test_fundus_gate_v2_adversarial.py -v --tb=short

benchmark-gate:
	PYTHONPATH=. python3 scripts/benchmark_gate.py

# --- Data Validation ---
validate-data:
	PYTHONPATH=. python3 scripts/validate_data.py --config configs/train.yaml

# --- Model Export ---
export:
	PYTHONPATH=. python3 scripts/export_model.py --config configs/train.yaml --checkpoint outputs/checkpoints/best.pt

# --- Governance ---
model-card:
	PYTHONPATH=. python3 scripts/generate_model_card.py --config configs/train.yaml

# --- HPO ---
hpo:
	PYTHONPATH=. python3 scripts/run_hpo.py --config configs/train.yaml --n-trials 20

# --- Retraining Check ---
check-retrain:
	PYTHONPATH=. python3 scripts/check_retraining.py

# --- DVC Pipeline ---
dvc-repro:
	dvc repro

# --- Full MLOps Pipeline ---
mlops-pipeline: validate-data train export model-card
	@echo "Full MLOps pipeline complete!"

# --- Hugging Face Spaces Deployment ---
# Dockerfile.hf: python:3.11-slim-bookworm, CPU PyTorch, supervisord (nginx + backend + frontend)
deploy-hf:
	bash scripts/deploy_hf.sh

hf-login:
	huggingface-cli login --token $(HF_TOKEN)

hf-local:
	docker compose --profile hf up --build

# --- 2026 Production Infrastructure ---
up-phase1:
	docker compose -f docker-compose.yml -f docker-compose.otel.yml -f docker-compose.mlflow.yml up -d
	@echo "Phase 1 stack: OTEL + Jaeger + Prometheus + MLflow"

up-phase2:
	docker compose -f docker-compose.yml -f docker-compose.otel.yml -f docker-compose.mlflow.yml -f docker-compose.2026.yml up -d
	@echo "Phase 2 stack: Phase 1 + Ray Serve + Kafka"

up-full:
	docker compose -f docker-compose.yml -f docker-compose.2026.yml up -d
	@echo "Full 2026 stack running"

down-full:
	docker compose -f docker-compose.yml -f docker-compose.2026.yml down

sbom:
	bash scripts/generate_sbom.sh retinalai:latest

export-all:
	PYTHONPATH=. python3 scripts/export_all_formats.py --model-path models/model_vignn_rank1.pth --output-dir models/export

# --- Phase 5: Quantization, Offline, Mobile ---
quantize:
	PYTHONPATH=. python3 scripts/quantize_models.py --model-path models/model_vignn_rank1.pth --output-dir outputs/quantized --formats gguf onnx

quantize-all:
	PYTHONPATH=. python3 scripts/quantize_models.py --model-path models/model_vignn_rank1.pth --output-dir outputs/quantized --formats gguf awq gptq onnx tensorrt

quality-gate:
	PYTHONPATH=. python3 scripts/quantization_quality_gate.py --manifest-path outputs/quantized/quantization_manifest.json --max-faithfulness-drop 0.04 --max-p95-latency-ms 100

mobile-bundle:
	PYTHONPATH=. python3 scripts/export_mobile_bundle.py --output-dir outputs/mobile_bundle --include-voice

mobile-bundle-validate:
	PYTHONPATH=. python3 scripts/export_mobile_bundle.py --output-dir outputs/mobile_bundle --no-archive --max-bundle-mb 800

flutter-scaffold:
	PYTHONPATH=. python3 scripts/export_mobile_bundle.py --output-dir outputs/mobile_bundle --generate-flutter --no-archive

# --- Phase 5 Docker ---
up-offline:
	OFFLINE_RAG__ENABLED=true docker compose up -d
	@echo "API with offline RAG enabled"

up-quantized:
	QUANTIZATION__ENABLED=true QUANTIZATION__TORCH_COMPILE_ENABLED=true docker compose up -d
	@echo "API with quantized models + torch.compile"

up-full-v2:
	OFFLINE_RAG__ENABLED=true QUANTIZATION__ENABLED=true VOICE_FIRST__ENABLED=true docker compose -f docker-compose.yml -f docker-compose.otel.yml up -d
	@echo "Full Phase 5 stack: offline + quantized + voice + OTEL"

# --- Phase 1: Mobile Distillation ---
distill:
	PYTHONPATH=. python3 scripts/distill_mobile_student.py --config configs/distillation_mobile_2026.yaml

export-mobile:
	PYTHONPATH=. python3 scripts/export_mobile_student.py && \
	PYTHONPATH=. python3 scripts/export_fundus_gate_onnx.py && \
	PYTHONPATH=. python3 scripts/extract_clinical_kg_json.py && \
	PYTHONPATH=. python3 scripts/build_screening_bundle.py

# --- Phase 4: Governance + Production ---
bias-audit-uganda:
	PYTHONPATH=. python3 scripts/run_bias_audit.py --uganda --f1-threshold 0.08

federated-sim:
	PYTHONPATH=. python3 scripts/simulate_federation.py --clients 5 --rounds 10

moh-package:
	PYTHONPATH=. python3 scripts/generate_moh_package.py

pilot-readiness:
	PYTHONPATH=. python3 scripts/validate_pilot_readiness.py

up-phase4:
	VOICE_FIRST__ENABLED=true DHIS2__ENABLED=true docker compose up -d

# --- Docker Hub ---
docker-login:
	docker login -u $(DOCKERHUB_USERNAME)

docker-build:
	docker build -t landwind/optiscan-ai:latest -f Dockerfile .

docker-build-cpu:
	docker build -t landwind/optiscan-ai:cpu -f Dockerfile.cpu .

docker-push: docker-build
	docker push landwind/optiscan-ai:latest

docker-push-cpu: docker-build-cpu
	docker push landwind/optiscan-ai:cpu

docker-push-all: docker-push docker-push-cpu

# --- Clean ---
clean:
	rm -rf outputs/checkpoints/epoch_*.pt
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
