#!/bin/bash
# Delete a TPU VM and its associated queued-resource request (if any).
#
# Usage:
#   ./teardown_tpu.sh <tpu-name> <zone>

set -eo pipefail

if [ -f "$(dirname "$0")/../.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$(dirname "$0")/../.env"
    set +a
fi

if [ -z "$2" ]; then
    echo "Usage: ./teardown_tpu.sh <tpu-name> <zone>"
    exit 1
fi

TPU_NAME=$1
ZONE=$2

PROJECT=${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo "")}
if [ -z "$PROJECT" ]; then
    echo "ERROR: No GCP project set."
    exit 1
fi

echo "Project: $PROJECT  TPU: $TPU_NAME  Zone: $ZONE"
echo "Deleting TPU VM..."
set +e
gcloud compute tpus tpu-vm delete "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" --quiet 2>/dev/null

# Also kill the queued-resource request if one exists, otherwise it can
# respawn the VM after deletion.
gcloud alpha compute tpus queued-resources delete "${TPU_NAME}-req" \
    --project="$PROJECT" --zone="$ZONE" --force --quiet 2>/dev/null
set -e

echo "Done."
