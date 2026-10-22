#!/bin/bash
#SBATCH --job-name=s2-merra2-aerosol
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=merra2-aerosol-%j.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
START_YEAR="${START_YEAR:-2024}"
END_YEAR="${END_YEAR:-$START_YEAR}"
WORKERS="${SLURM_CPUS_PER_TASK:-4}"
OPTICS_ROOT="${OPTICS_ROOT:?Set OPTICS_ROOT to the MERRA-2 optical tables directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/scratch/output/sasktran/${START_YEAR}-${END_YEAR}}"

cd "$REPOSITORY"
source tools/databases/tropospheric_aerosol/setup_job_env.sh
"$PYTHON" tools/databases/tropospheric_aerosol/build.py \
    --optics-root "$OPTICS_ROOT" \
    --output "$OUTPUT_DIR" \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR" \
    --workers "$WORKERS" \
    --verbose
