#!/usr/bin/env bash
# End-to-end smoke test of the training workflow: infra stack up, data versioned in LakeFS,
# training tracked in MLflow, champion registered and exported. Usage:
#   model_training/test/smoke_train.sh [work-dir]
# Needs: docker compose v2, python venv with model_training/requirements.txt active, lakectl.
set -euo pipefail
export GIT_AUTHOR_NAME=smoke GIT_AUTHOR_EMAIL=smoke@local GIT_COMMITTER_NAME=smoke GIT_COMMITTER_EMAIL=smoke@local
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT="$(cd "$HERE/.." && pwd)"
WORK="${1:-$(mktemp -d /tmp/smoke-train-XXXX)}"
PROJECT="smoke-train-$$"
export COMPOSE_PROJECT_NAME="$PROJECT"

cleanup() { ( cd "$MT/infra" && docker compose --profile local-s3 down -v --remove-orphans ) >/dev/null 2>&1 || true; }
trap cleanup EXIT

# A copy of model_training with data, inside a git repository (lakectl local needs one).
mkdir -p "$WORK"
cp -r "$MT" "$WORK/model_training"
rm -rf "$WORK/model_training/model" "$WORK/model_training/metrics" "$WORK/model_training/output" "$WORK/model_training/data/.lakefs_ref.yaml"
cp "$MT/.env.example" "$WORK/model_training/.env"
( cd "$WORK" && git init -q && git add -A && git -c user.name=smoke -c user.email=smoke@local commit -qm "smoke workspace" )

( cd "$MT/infra" && docker compose --profile local-s3 up -d --wait )

cd "$WORK"
python model_training/setup_lakefs.py
python model_training/prepare_lakefs_branch.py
python model_training/data_augmentation.py
python model_training/commit_lakefs.py
python model_training/train.py
python model_training/mlflow_registry.py
python model_training/export_champion.py --output-dir "$WORK/output"

test -s "$WORK/output/model.keras"
test -s "$WORK/output/classes.json"
test -s "$WORK/output/MODEL_CARD.md"
python - <<'PY'
import json, os, urllib.request
base = os.environ.get("TRACKING_URI", "http://127.0.0.1:5000")
q = urllib.request.urlopen(f"{base}/api/2.0/mlflow/registered-models/alias?name=screw_classifier&alias=champion")
mv = json.load(q)["model_version"]
assert mv["run_id"], "alias does not point at a run"
run = json.load(urllib.request.urlopen(f"{base}/api/2.0/mlflow/runs/get?run_id={mv['run_id']}"))["run"]
sources = [d["dataset"]["source"] for d in run.get("inputs", {}).get("dataset_inputs", [])]
assert any("lakefs://" in s for s in sources), f"no lakefs dataset source logged: {sources}"
print("smoke ok: champion version", mv["version"], "run", mv["run_id"], "data", sources[0])
PY
