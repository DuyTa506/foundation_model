# ADR 004: TPU Deployment Pipeline

## Status
Accepted

## Context
Running experiments on Google Cloud TPUs requires managing VM lifecycle, syncing the workspace, surviving SSH disconnects during long runs, and handling preemption / capacity shortages. We want a simple, scriptable pipeline that:

- Doesn't hardcode project IDs or zones (so the repo can be cloned and used by anyone with a GCP account)
- Tries the cheapest provisioning model (flex-start, preemptible) first
- Falls back gracefully when the requested zone is out of capacity
- Persists checkpoints to GCS so a preempted run is recoverable

## Decision

Four shell scripts in `scripts/` cover the workflow:

1. **`create_tpu.sh <name> [type] [zone] [runtime]`**
   - Reads `GCP_PROJECT` from `.env` or falls back to `gcloud config get-value project`.
   - For non-`v6e` types: tries the Queued Resources flex-start API first.
   - On failure (or for `v6e` directly): polls all zones that stock the accelerator type, preferring the user's primary zone, until allocation succeeds. Uses preemptible VMs.

2. **`sync_and_run.sh <name> <zone> "<command>"`**
   - Tars and uploads the workspace (excluding `.git`, `.venv`, `checkpoints`, `wandb`, `data`, `__pycache__`).
   - Installs `requirements.txt` on the remote.
   - Launches the command inside a detached `tmux` session named `utm_run` so it survives SSH drops.
   - W&B credential forwarding (`~/.netrc`) is opt-in via `WITH_WANDB=1` rather than automatic.

3. **`upload_checkpoints.sh <name> <zone> [glob]`**
   - Runs `gsutil -m rsync -r` from inside the TPU VM (which has GCE service-account credentials by default) to a bucket configured by `GCS_CHECKPOINT_BUCKET`.
   - This is intentionally a shell wrapper rather than native Orbax-to-GCS in `train.py` — keeps `train.py` focused on training and lets users substitute their own storage backend.

4. **`teardown_tpu.sh <name> <zone>`**
   - Deletes both the TPU VM and any associated queued-resource request (otherwise the QR can respawn the VM after deletion).

## Consequences

**Pros**
- No project/zone lock-in. `.env.example` documents the few configurable values.
- The polling loop in `create_tpu.sh` reflects the practical reality that single-zone TPU capacity is often unavailable; falling back across zones in priority order is significantly more reliable than retrying the same zone.

**Cons**
- Hard dependency on a working `gcloud` CLI and the user being authenticated.
- The polling loop can run for hours if global capacity is tight. Users should run it in a screen/tmux locally or wrap with their own timeout.
