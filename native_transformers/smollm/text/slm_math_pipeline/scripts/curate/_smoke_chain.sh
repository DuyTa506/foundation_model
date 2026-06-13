#!/usr/bin/env bash
# Clean, uninterrupted smoke run of the full curation chain 00->07.
# Investigates whether empty/corrupt parquet files appear when NOTHING is killed.
# Uses workers=2 to exercise the multi-worker sharding the user is worried about.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/tmp/smoke_hf_cache
PY=.venv/bin/python
W=2
ROOT=outputs/_smoke2
SMOKE_TARGET_TOKENS=${SMOKE_TARGET_TOKENS:-1000000}
rm -rf "$ROOT"
mkdir -p "$ROOT"

health() {  # report parquet file health under a dir
  local d="$1"
  echo "---- health: $d ----"
  if [ ! -d "$d" ]; then echo "  (missing)"; return; fi
  local count=0
  while IFS=$'\t' read -r sz p; do
    count=$((count + 1))
    status=$($PY - "$p" <<'EOF'
import sys
import pyarrow.parquet as pq
p=sys.argv[1]
try:
    f=pq.ParquetFile(p); print(f"OK rows={f.metadata.num_rows}")
except Exception as e:
    print(f"CORRUPT {type(e).__name__}")
    sys.exit(1)
EOF
)
    echo "  ${sz}B  ${p##*/}  -> $status"
  done < <(find "$d" -name '*.parquet' -printf '%s\t%p\n')
  if [ "$count" -eq 0 ]; then
    echo "  (no parquet files)"
  fi
}

echo "######## STAGE 00 materialize (workers=$W) ########"
$PY scripts/curate/00_materialize.py --source_ids wikipedia_vi \
  --cache_dir /tmp/smoke_hf_cache --output_dir "$ROOT/raw" \
  --max_rows_per_source ${SMOKE_ROWS_PER_SOURCE:-2000} --workers $W
health "$ROOT/raw"

echo "######## STAGE 02 language_id ########"
$PY scripts/curate/02_language_id.py \
  --input_dir "$ROOT/raw" --output_dir "$ROOT/lang_filtered" --workers $W
health "$ROOT/lang_filtered"

echo "######## STAGE 01 quality_filter ########"
$PY scripts/curate/01_quality_filter.py \
  --input_dir "$ROOT/lang_filtered" --output_dir "$ROOT/quality_filtered" --workers $W
health "$ROOT/quality_filtered"

echo "######## STAGE 04 dedup_minhash ########"
$PY scripts/curate/04_dedup_minhash.py \
  --input_dir "$ROOT/quality_filtered" --output_dir "$ROOT/deduped" --workers $W
health "$ROOT/deduped"

echo "######## STAGE 03 ultraclean (skip VI train) ########"
$PY scripts/curate/03_ultraclean_filter.py \
  --input_dir "$ROOT/deduped" --output_dir "$ROOT/ultraclean" \
  --skip_train_vi --workers $W
health "$ROOT/ultraclean"

echo "######## STAGE 05 decontaminate ########"
$PY scripts/curate/05_decontaminate.py \
  --input_dir "$ROOT/ultraclean" --output_dir "$ROOT/decontaminated" \
  --allow_missing_benchmarks --workers $W
health "$ROOT/decontaminated"

echo "######## STAGE 06 pii_redact ########"
$PY scripts/curate/06_pii_redact.py \
  --input_dir "$ROOT/decontaminated" --output_dir "$ROOT/pii_clean" --workers $W
health "$ROOT/pii_clean"

echo "######## STAGE 6.5 build_mixed_corpus ########"
$PY scripts/curate/build_mixed_corpus.py \
  --input_dir "$ROOT/pii_clean" --output_dir "$ROOT/mixed" \
  --target_tokens "$SMOKE_TARGET_TOKENS"
health "$ROOT/mixed"

echo "######## STAGE 07 tokenize_pack ########"
$PY scripts/curate/07_tokenize_pack.py \
  --input_dir "$ROOT/mixed" --output_dir "$ROOT/tokenized" \
  --tokenizer_path outputs/_smoke/tokenizer --workers $W
echo "---- tokenized shards ----"
find "$ROOT/tokenized" -name '*.ds' -printf '%s\t%p\n' || true

echo "######## CHAIN COMPLETE ########"
