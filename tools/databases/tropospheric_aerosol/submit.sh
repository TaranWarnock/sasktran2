#!/bin/bash
set -euo pipefail

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/scratch/output/sasktran}"
JOB_SCRIPT="${JOB_SCRIPT:-tools/databases/tropospheric_aerosol/build.slurm}"
mkdir -p "$OUTPUT_ROOT"

cd "$REPOSITORY"
exec sbatch \
    --output="${OUTPUT_ROOT}/%x-%j.out" \
    "$@" \
    "$JOB_SCRIPT"
