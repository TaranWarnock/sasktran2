# Building the MERRA2 tropospheric aerosol catalogue

The runtime API loads a compact, versioned NetCDF catalogue. Build it only when
reproducing or revising the reference cases:

```bash
pip install -e ".[merra2-build]"
python tools/databases/tropospheric_aerosol/build.py \
  --optics-root /path/to/MERRA-2-Versions/v0.0.0
```

By default, the builder streams 1980 through the last complete calendar year
from GES DISC using HTTP range reads. Configure Earthdata Login credentials in
the user's netrc; no full source granules are saved locally. To use existing
granules instead, pass `--data-root` with `M2I1NXASM`, `M2I3NVAER`, and
`M2I3NVASM` subdirectories. Pass `--start-year` and `--end-year` to pin the
analysis period. Local builds fail on the first missing product day so that
partial coverage is never mistaken for a climatology.

## Set up Python with uv on DRAC

From the repository checkout on the cluster login node, create an isolated
environment with only the dependencies needed by the builder and cloud-
streaming smoke test:

```bash
cd /path/to/sasktran2
uv venv --python 3.12
uv pip install --python .venv/bin/python numpy xarray fsspec h5netcdf
.venv/bin/python -c "import fsspec, h5netcdf, numpy, xarray; print('MERRA2 build environment ready')"
```

Use `uv pip install` rather than `uv sync` here: syncing the project resolves
its full declared Python support range and optional extras. The `complete`
extra includes `astropy>=7`, which requires Python 3.11+, while this project
declares Python 3.10+ support. That cross-version resolution can fail even
though the smoke-test builder itself does not need `astropy` or the SASKTRAN2
extension.

If `uv` is not already available, install it using the cluster's approved
method, or its official user-local installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After setting up the environment, the Slurm scripts default to
`.venv/bin/python`; `PYTHON` can be set to another interpreter path if desired.
Ensure dependencies can be installed on the login node before submitting,
since compute nodes may have different network/package access.

For GES DISC authentication, configure Earthdata Login credentials in
`~/.netrc` on the cluster and restrict its permissions:

```text
machine urs.earthdata.nasa.gov login YOUR_EARTHDATA_USERNAME password YOUR_EARTHDATA_PASSWORD
```

```bash
chmod 600 ~/.netrc
```

Do not commit or share the credential file. Test the environment and access
with the one-day smoke job before submitting the longer yearly build.

For the first DRAC Slurm test, submit one complete year (2024 by default) with
the optical tables available on the cluster:

```bash
bash tools/databases/tropospheric_aerosol/submit.sh \
  --export=ALL,OPTICS_ROOT=/path/to/MERRA-2-Versions/v0.0.0
```

For a short one-day connectivity and processing check, submit the smoke job
instead. It reads all aerosol timestamps and one sample column from each
product, calculates 550 nm extinction and tropospheric AOD, and writes a JSON report.
The default test date is 2024-01-01:

```bash
JOB_SCRIPT=tools/databases/tropospheric_aerosol/smoke.slurm \
  bash tools/databases/tropospheric_aerosol/submit.sh \
  --export=ALL,OPTICS_ROOT=/path/to/MERRA-2-Versions/v0.0.0,TEST_DATE=2024-01-01
```

The smoke job requests one CPU, 8 GB, and 30 minutes. Its output and report go
under `~/scratch/output/sasktran/smoke-YYYY-MM-DD` by default. Change
`TEST_DATE` to a date with available MERRA2 inputs.

The job uses four concurrent month workers and streams GES DISC granules
directly. Set `START_YEAR` and `END_YEAR` to change the period, and set
`OUTPUT_DIR` to choose where the catalogue and build report are written. For
example, a complete-history run can be submitted with
`START_YEAR=1980,END_YEAR=2025`; retain a separate output directory for this
one-year test. The Python environment used by the job must have the optional
`merra2-build` dependencies installed, and Earthdata credentials must be
available in the submitting user's netrc file. The submission helper creates
`~/scratch/output/sasktran` before submission and puts the Slurm log there;
the catalogue and report are written to its year-specific subdirectory by
default.

```bash
bash tools/databases/tropospheric_aerosol/submit.sh \
  --export=ALL,OPTICS_ROOT=/path/to/MERRA-2-Versions/v0.0.0,START_YEAR=1980,END_YEAR=2025
```

For local runs, set `--workers` to process the four climatology months
concurrently. Keep the worker count within the memory and CPU allocation.
Set `OUTPUT_ROOT` before invoking `submit.sh` to use another location for logs;
set `OUTPUT_DIR` via `--export` to override the default catalogue directory.

The five required optical files are:

- `optics_BC.v1_3.nc`
- `optics_DU.v14_2.nc`
- `optics_OC.v1_3.nc`
- `optics_SS.v3_3.nc`
- `optics_SU.v1_3.nc`

The component-to-bin mapping, mass-mixing-ratio conversion, relative-humidity
interpolation, and `bext` use follow
`sasktran-accel/scripts/monthly_tropospheric_aerosol.py`. Extinction is
integrated over hydrostatic model-layer thickness where `PL >= TROPPB`.
Individual columns are area-weighted by cosine latitude. A deterministic
weighted reservoir bounds memory while estimating the monthly percentile
windows; the seed, reservoir size, candidate counts, and selected source
columns are recorded in the build report.

The tool writes `tropospheric_aerosol_v1.nc` and `build_report_v1.json`. Review
the report, selected profiles, and spectral calculations before shipping the
catalogue. Pin the reviewed catalogue SHA-256 in
`src/sasktran2/database/tropospheric_aerosol.py`, then run the climatology
tests and docs build. Do not replace an existing catalogue version in place.
