#!/bin/bash
# Sync checkpoints from a TPU VM to a GCS bucket.
#
# Usage:
#   ./upload_checkpoints.sh <tpu-name> <zone> [run-name-glob]
#
# Reads GCP_PROJECT and GCS_CHECKPOINT_BUCKET from .env (or override via env).
# `run-name-glob` defaults to '*' (all runs).

set -eo pipefail

if [ -f "$(dirname "$0")/../.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$(dirname "$0")/../.env"
    set +a
fi

if [ -z "$2" ]; then
    echo "Usage: ./upload_checkpoints.sh <tpu-name> <zone> [run-name-glob]"
    echo "Example: ./upload_checkpoints.sh utm-run-1 us-south1-ai1b 'utm-T16-*'"
    exit 1
fi

TPU_NAME=$1
ZONE=$2
GLOB=${3:-"*"}
REMOTE_DIR=${REMOTE_DIR:-"~/Work/utm-jax"}

PROJECT=${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo "")}
if [ -z "$PROJECT" ]; then
    echo "ERROR: No GCP project set."
    exit 1
fi
if [ -z "${GCS_CHECKPOINT_BUCKET:-}" ]; then
    echo "ERROR: GCS_CHECKPOINT_BUCKET not set. Add it to .env or export it."
    echo "Example: GCS_CHECKPOINT_BUCKET=gs://my-bucket/utm-checkpoints"
    exit 1
fi

echo "Project: $PROJECT  TPU: $TPU_NAME  Zone: $ZONE"
echo "Uploading checkpoints matching '$GLOB' to $GCS_CHECKPOINT_BUCKET..."

# Run gsutil rsync from inside the TPU VM (it has GCE service account creds).
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" \
    --command="cd $REMOTE_DIR/checkpoints && for d in $GLOB; do \
                 if [ -d \"\$d\" ]; then \
                   echo \"-> rsync \$d\"; \
                   gsutil -m rsync -r \"\$d\" \"$GCS_CHECKPOINT_BUCKET/\$d\"; \
                 fi; \
               done"

echo "Done."
