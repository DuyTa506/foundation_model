#!/bin/bash
# Provision a TPU VM with smart zone fallback.
#
# Tries Queued Resources flex-start first (when supported), then falls back to
# synchronous polling across all zones that stock the requested accelerator,
# preferring the user's primary zone.
#
# Usage:
#   ./create_tpu.sh <tpu-name> [tpu-type] [primary-zone] [runtime-version]
#
# Reads GCP_PROJECT from .env or falls back to `gcloud config get-value project`.

set -eo pipefail

# Load .env if present (supports GCP_PROJECT, etc.)
if [ -f "$(dirname "$0")/../.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$(dirname "$0")/../.env"
    set +a
fi

if [ -z "$1" ]; then
    echo "Usage: ./create_tpu.sh <tpu-name> [tpu-type] [primary-zone] [runtime-version]"
    echo "Example: ./create_tpu.sh utm-run-1 v6e-1 us-south1-ai1b v6e-ubuntu-2404"
    exit 1
fi

TPU_NAME=$1
TPU_TYPE=${2:-"v6e-1"}

# Sensible defaults per accelerator family
if [[ "$TPU_TYPE" == v5p* ]]; then
    DEFAULT_ZONE="us-east5-a"
    DEFAULT_RUNTIME="v2-alpha-tpuv5"
elif [[ "$TPU_TYPE" == v6e* ]]; then
    DEFAULT_ZONE="us-south1-ai1b"
    DEFAULT_RUNTIME="v6e-ubuntu-2404"
else
    DEFAULT_ZONE="us-west4-a"
    DEFAULT_RUNTIME="tpu-ubuntu2204-base"
fi

PRIMARY_ZONE=${3:-$DEFAULT_ZONE}
RUNTIME_VERSION=${4:-$DEFAULT_RUNTIME}

# Resolve project: env var GCP_PROJECT > gcloud config
PROJECT=${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo "")}
if [ -z "$PROJECT" ]; then
    echo "ERROR: No GCP project set. Either export GCP_PROJECT or run 'gcloud config set project <id>'."
    exit 1
fi
echo "Using GCP project: $PROJECT"
echo "Provisioning TPU $TPU_NAME ($TPU_TYPE)..."

# Phase 1: Queued Resources flex-start (skip on v6e — Google's backend has historically
# been flaky for v6e queued requests; the polling fallback is more reliable).
if [[ "$TPU_TYPE" != v6e* ]]; then
    echo "[Phase 1] Attempting Queued Resources flex-start in $PRIMARY_ZONE..."
    set +e
    gcloud alpha compute tpus queued-resources create "${TPU_NAME}-req" \
        --project="$PROJECT" \
        --zone="$PRIMARY_ZONE" \
        --accelerator-type="$TPU_TYPE" \
        --runtime-version="$RUNTIME_VERSION" \
        --node-id="$TPU_NAME" \
        --provisioning-model=flex-start \
        --max-run-duration=4h \
        --valid-until-duration=4h \
        --labels=purpose=flex-start \
        --quiet
    Q_STATUS=$?
    set -e

    if [ $Q_STATUS -eq 0 ]; then
        echo "Queued Resource request submitted in $PRIMARY_ZONE."
        echo "TPU will be instantiated asynchronously when capacity allows."
        exit 0
    fi
    echo "Queued resources unavailable. Falling back to polling."
fi

# Phase 2: Synchronous polling across zones that physically stock the type
echo "[Phase 2] Discovering zones that stock $TPU_TYPE..."
ZONES=($(gcloud compute tpus accelerator-types list \
    --project="$PROJECT" --zone=- \
    --filter="type=$TPU_TYPE" \
    --format="value(name)" --quiet 2>/dev/null \
    | awk -F/ '{print $4}' | sort | uniq))

if [ ${#ZONES[@]} -eq 0 ]; then
    echo "ERROR: No zones found that support $TPU_TYPE in project $PROJECT."
    echo "Check the type name and your project's TPU quota."
    exit 1
fi
echo "Capacity potentially available in ${#ZONES[@]} zones: ${ZONES[*]}"

# Move PRIMARY_ZONE to the front
PRIORITY_ZONES=()
for Z in "${ZONES[@]}"; do
    if [ "$Z" == "$PRIMARY_ZONE" ]; then
        PRIORITY_ZONES=("$PRIMARY_ZONE" "${PRIORITY_ZONES[@]}")
    else
        PRIORITY_ZONES+=("$Z")
    fi
done

CREATED=false
while [ "$CREATED" = false ]; do
    for ZONE in "${PRIORITY_ZONES[@]}"; do
        echo "[$(date +'%H:%M:%S')] Requesting allocation in $ZONE..."
        set +e
        gcloud compute tpus tpu-vm create "$TPU_NAME" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --accelerator-type="$TPU_TYPE" \
            --version="$RUNTIME_VERSION" \
            --preemptible \
            --quiet
        STATUS=$?
        set -e

        if [ $STATUS -eq 0 ]; then
            echo "Allocated $TPU_TYPE in $ZONE."
            CREATED=true
            break
        fi
    done
    if [ "$CREATED" = false ]; then
        echo "No capacity in any viable zone. Retrying in 60s..."
        sleep 60
    fi
done

echo "TPU $TPU_NAME ready."
