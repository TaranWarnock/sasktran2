#!/bin/bash
#SBATCH --job-name=s2-merra2-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
TEST_DATE="${TEST_DATE:-2024-01-01}"
OPTICS_ROOT="${OPTICS_ROOT:?Set OPTICS_ROOT to the MERRA-2 optical tables directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/scratch/output/sasktran/smoke-${TEST_DATE}}"
PYTHON="${PYTHON:-${REPOSITORY}/.venv/bin/python}"

cd "$REPOSITORY"
"$PYTHON" tools/databases/tropospheric_aerosol/build.py \
    --optics-root "$OPTICS_ROOT" \
    --output "$OUTPUT_DIR" \
    --smoke-date "$TEST_DATE" \
    --verbose
