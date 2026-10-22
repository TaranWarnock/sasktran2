#!/bin/bash
set -euo pipefail

: "${SLURM_TMPDIR:?SLURM_TMPDIR must be set by Slurm}"

module load StdEnv/2023
module load python/3.12.4
module load scipy-stack

VENV_PATH="${SLURM_TMPDIR}/sasktran2-merra2-venv"
virtualenv --no-download --system-site-packages "$VENV_PATH"
source "${VENV_PATH}/bin/activate"

python -m pip install --no-index --upgrade pip
python -m pip install --no-index \
    "fsspec[http]" \
    h5netcdf \
    h5py \
    requests \
    xarray

python -c "import aiohttp, fsspec, h5py, h5netcdf, numpy, requests, scipy, xarray; from fsspec.implementations.http import HTTPFileSystem; HTTPFileSystem(); print('MERRA2 job environment ready')"

export PYTHON="${VENV_PATH}/bin/python"
