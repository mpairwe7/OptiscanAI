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

# --- Clean ---
clean:
	rm -rf outputs/checkpoints/epoch_*.pt
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
