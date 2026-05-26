#!/bin/bash
# Sync the local repo to a TPU VM and launch a command inside a tmux session
# so the run survives SSH disconnects.
#
# Usage:
#   ./sync_and_run.sh <tpu-name> <zone> "<command>"
#
# Reads GCP_PROJECT from .env or falls back to `gcloud config get-value project`.
# Set WITH_WANDB=1 to also forward your local ~/.netrc (W&B credentials).

set -eo pipefail

if [ -f "$(dirname "$0")/../.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$(dirname "$0")/../.env"
    set +a
fi

if [ -z "$3" ]; then
    echo "Usage: ./sync_and_run.sh <tpu-name> <zone> \"<command>\""
    echo "Example: ./sync_and_run.sh utm-run-1 us-south1-ai1b \"python train.py --data_paths data/sudoku-extreme-full ...\""
    exit 1
fi

TPU_NAME=$1
ZONE=$2
COMMAND=$3
REMOTE_DIR=${REMOTE_DIR:-"~/Work/utm-jax"}

PROJECT=${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo "")}
if [ -z "$PROJECT" ]; then
    echo "ERROR: No GCP project set. Export GCP_PROJECT or run 'gcloud config set project <id>'."
    exit 1
fi

GCLOUD_BASE=(gcloud compute tpus tpu-vm --project="$PROJECT" --zone="$ZONE")

echo "Project: $PROJECT  TPU: $TPU_NAME  Zone: $ZONE"
echo "Packaging local workspace..."
PAYLOAD=$(mktemp -t utm_sync.XXXXXX.tar.gz)
trap 'rm -f "$PAYLOAD"' EXIT

tar --exclude=".git" --exclude=".venv" --exclude=".pytest_cache" \
    --exclude=".env" --exclude="__pycache__" --exclude="checkpoints" \
    --exclude="wandb" --exclude="data" --exclude="analysis_output" \
    -czf "$PAYLOAD" -C . . || [[ $? -eq 1 ]] || exit $?

echo "Ensuring remote directory exists..."
"${GCLOUD_BASE[@]}" ssh "$TPU_NAME" --command="mkdir -p $REMOTE_DIR"

echo "Uploading payload..."
"${GCLOUD_BASE[@]}" scp "$PAYLOAD" "$TPU_NAME:$REMOTE_DIR/_sync.tar.gz"

echo "Extracting on TPU..."
"${GCLOUD_BASE[@]}" ssh "$TPU_NAME" \
    --command="cd $REMOTE_DIR && tar -xzf _sync.tar.gz && rm _sync.tar.gz"

if [ "${WITH_WANDB:-0}" = "1" ] && [ -f ~/.netrc ]; then
    echo "Forwarding W&B credentials (~/.netrc)..."
    "${GCLOUD_BASE[@]}" scp ~/.netrc "$TPU_NAME:~/"
fi

echo "Installing dependencies..."
"${GCLOUD_BASE[@]}" ssh "$TPU_NAME" \
    --command="cd $REMOTE_DIR && pip install -r requirements.txt --break-system-packages"

echo "Launching in tmux session 'utm_run'..."
"${GCLOUD_BASE[@]}" ssh "$TPU_NAME" \
    --command="cd $REMOTE_DIR && tmux new-session -d -s utm_run \"$COMMAND\" && echo 'Started. Attach with: tmux a -t utm_run'"

echo "Done."
