# Building the MERRA2 tropospheric aerosol catalogue

The runtime API loads a compact, versioned NetCDF catalogue. Build it only when
reproducing or revising the reference cases:

```bash
pip install -e ".[merra2-build]"
python tools/databases/tropospheric_aerosol/build.py \
  --optics-root /path/to/MERRA-2-Versions/v0.0.0
```

By default, the builder reads 2016 through the last complete calendar year;
the recent decade represents the current aerosol climate, and MERRA-2 goes
back to 1980 if a longer period is needed. Data are read from GES DISC,
downloading each day's granules to temporary storage and deleting them after processing (see "Daily staging" below). Configure
Earthdata Login credentials in the user's netrc. To use existing
granules instead, pass `--data-root` with `M2I1NXASM`, `M2I3NVAER`, and
`M2I3NVASM` subdirectories. Pass `--start-year` and `--end-year` to pin the
analysis period. Local builds fail on the first missing product day so that
partial coverage is never mistaken for a climatology.

## Python environment on DRAC

Each Slurm job creates its own virtual environment on the compute node under
`$SLURM_TMPDIR`. The shared setup script loads the Alliance modules in order:
`StdEnv/2023`, `python/3.12.4`, then the default (latest) `scipy-stack` module.
The venv uses `--system-site-packages` to reuse NumPy, SciPy, and other packages
from `scipy-stack`; pip installs the remaining packages from Alliance's local
wheelhouse with `--no-index`. The environment is temporary and is recreated
for each job.

No login-node environment setup is needed. The smoke and catalogue Slurm
scripts source `tools/databases/tropospheric_aerosol/setup_job_env.sh` before
running the builder. It verifies the HTTP and NetCDF backends are importable
before beginning the MERRA2 work.

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
Like the catalogue build, it downloads that day's granules whole by default
(`--no-stage-daily` streams them with range requests instead).
The default test date is 2024-01-01:

```bash
JOB_SCRIPT=tools/databases/tropospheric_aerosol/smoke.slurm \
  bash tools/databases/tropospheric_aerosol/submit.sh \
  --export=ALL,OPTICS_ROOT=/path/to/MERRA-2-Versions/v0.0.0,TEST_DATE=2024-01-01
```

The smoke job requests one CPU, 8 GB, and 30 minutes. Its output and report go
under `~/scratch/output/sasktran/smoke-YYYY-MM-DD` by default. Change
`TEST_DATE` to a date with available MERRA2 inputs.

The job uses four concurrent month workers and downloads GES DISC granules
one day at a time. Transient HTTP failures (429 and common 5xx responses, connection
errors, and timeouts) get up to five attempts with backoff. If a GES DISC
range request returns 403, the builder resolves a fresh signed URL and retries
the range request. Persistent failures stop the job with the original
exception. The Slurm log includes timestamped startup and input-preparation
milestones, month/checkpoint startup, each day started and completed, granule
opens, checkpoint-validation progress, and timestamp processing progress. After
each fully processed day, the builder atomically saves the month workers'
weighted-reservoir state under
`OUTPUT_DIR/.checkpoints/<build-id>/`. If the job fails, resubmit with the same
`OUTPUT_DIR`, year range, seed, reservoir size, optical tables, and data
source; the builder resumes each month after its last checkpointed day.
The checkpoint identity includes the algorithm version. Before resuming, local
granules are checked by SHA-256 and remote granules by their stable source URL,
ETag, last-modified time, and size; a remote ETag is sufficient, otherwise
both last-modified time and size are required. Temporary signed redirect URLs
are not stored. A mismatch stops the resume rather than mixing source data.
The checkpoint directory key includes the algorithm version, so incrementing
`_CHECKPOINT_ALGORITHM_VERSION` in `build.py` automatically starts a separate
checkpoint set whenever processing changes can alter reservoir sampling or
selected profiles.
Checkpoints are deleted after the catalogue and report are written
successfully. They are stored in the output directory, so choose a persistent
scratch location rather than `$SLURM_TMPDIR`.

### Daily staging

By default (`--stage-daily`, or `STAGE_DAILY=1` in the Slurm job), the builder
downloads each day's three granules to a temporary directory on the compute
node (under `$SLURM_TMPDIR` when available), processes them locally, and
removes them after processing. While a day is processed, the next day's
granules download in the background, so downloads overlap with processing.
Checkpoint source identities remain tied to the remote granules, so resume
validation works after temporary files are deleted. Staging downloads each
full granule (about 6.6 GB per day for the three products) and each worker
holds up to two days at once, so four workers need about 55 GB of
`$SLURM_TMPDIR`. Check available capacity before submitting.

If temporary storage is too small, stream the granules with HTTP range reads
instead (`--no-stage-daily`, or `STAGE_DAILY=0` in the Slurm job). Nothing is
saved locally, but range reads are much slower. For example:

```bash
STAGE_DAILY=0 bash tools/databases/tropospheric_aerosol/submit.sh \
  --export=ALL,OPTICS_ROOT=/path/to/MERRA-2-Versions/v0.0.0,START_YEAR=2024,END_YEAR=2024
```

Use the same `OUTPUT_DIR` and build settings when resuming. `STAGE_DAILY` only
changes how remote input files are read; it does not change the climatology or
checkpoint identity.

After sampling, each month reads its selected columns (up to 60: three bands,
five aerosol types, four loadings). Staging applies here too: each day holding
a selected column is downloaded whole while the previous one is read, and
removed afterwards. Each selected column is read once, even when several cases
chose it, and each day is downloaded once. Range reads are much slower for
this pass: a column needs about 1,500 small requests (one per variable and
model level), which took about 4 minutes per column on DRAC, while staging a
day took about 2 minutes. Staging adds up to about 6.6 GB of downloads per
selected day (at most 240 days, about 1.6 TB, for a multi-year build).

Set `START_YEAR` and
`END_YEAR` to change the period, and set
`OUTPUT_DIR` to choose where the catalogue and build report are written. For
example, the reference ten-year run can be submitted with
`START_YEAR=2016,END_YEAR=2025`; keep its output directory separate from the
one-year test. Earthdata credentials must be available in the submitting
user's netrc file. The submission helper creates
`~/scratch/output/sasktran` before submission and puts the Slurm log there;
the catalogue and report are written to its year-specific subdirectory by
default.

```bash
bash tools/databases/tropospheric_aerosol/submit.sh \
  --export=ALL,OPTICS_ROOT=/path/to/MERRA-2-Versions/v0.0.0,START_YEAR=2016,END_YEAR=2025
```

For local runs, install the optional `merra2-build` dependencies and set
`--workers` to process the four climatology months concurrently. Month workers
are separate processes: as threads they serialized on Python's GIL and h5py's
global HDF5 lock (four threads ran about 1.2x faster than one; four processes
about 3.2x). More than four workers gives no further speed-up. Keep the worker
count within the memory and CPU allocation.
Set `OUTPUT_ROOT` before invoking `submit.sh` to use another location for logs;
set `OUTPUT_DIR` via `--export` to override the default catalogue directory.

The five required optical files are:

- `optics_BC.v1_3.nc`
- `optics_DU.v14_2.nc`
- `optics_OC.v1_3.nc`
- `optics_SS.v3_3.nc`
- `optics_SU.v1_3.nc`

The builder maps aerosol species to optical bins using the MERRA2 optical
property tables, then applies the mass-mixing-ratio conversion, relative-
humidity interpolation, and `bext` calculations. Extinction is integrated
over hydrostatic model-layer thickness up to the tropopause height (`TROPPB`
interpolated in pressure between model levels). The layer containing the
tropopause counts only below it: `TROPPB` usually equals a model level's
pressure, so counting that whole layer in or out would put half a layer of AOD
above the tropopause or leave a gap below it. Individual
columns are area-weighted by cosine latitude. A deterministic weighted
reservoir bounds memory while estimating monthly percentile windows; the seed,
reservoir size, candidate counts, and selected source columns are recorded in
the build report.

Loading levels are chosen separately for each of the five aerosol types
(sulfate, dust, sea salt, black carbon, organic carbon): the windows are
percentiles of the type's own tropospheric 550 nm AOD, with the same bands,
months and percentiles for every type. One reservoir per band serves all
types, because its sampling keys depend only on cosine latitude and a random
draw, not on AOD. It keeps each sampled column's AOD per species and its
tropospheric profile per type. Within a window, the selected column (with
positive AOD of the type) is closest to the window's median profile shape of
that type and its median mix of the type's species: the size bins for dust and
sea salt, the hydrophilic and hydrophobic parts for black and organic carbon.
Sulfate has one species, so only its shape counts.

The selected columns are mapped to the catalogue grid conservatively over
their whole height: each layer's AOD is spread over its height range, binned
into cells around the grid points, and AOD below the surface is folded into
the first point above it. The grid is 250 m apart up to 20 km, which holds the
troposphere, then 1 km to 50 km and 2 km to 84 km, above the MERRA2 model top
(0.01 hPa, about 80 km). The profiles therefore hold the whole column,
including aerosol above the tropopause; any AOD above the top point folds into
it. sasktran2 cuts profiles at the tropopause by default
(`cut_at_tropopause=True`). The trapezoid integral of each catalogue profile on
its own grid equals the native-layer column AOD. Selection uses only the
tropospheric profiles, on the 0-20 km part of the grid. `candidate_count` is the
estimated number of valid columns in the loading window, scaled from the
sampled reservoir to all valid columns.

Each case (scenario, aerosol type, month) holds only its type's species. It
stores spectral extinction, `species_extinction_550nm_per_m` (550 nm
extinction per MERRA2 species bin, mapped with the same conservative scheme,
so the species sum to the total; zero for other types' species) and
`relative_humidity` on the grid, clipped to 0-0.99 as in the optical tables.
Together with the MERRA2 optical tables these give each case's single
scattering albedo and phase function. The catalogue schema is version 3. The
wavelength grid is the union of the five tables' wavelengths within their
shared range, rounded to 0.01 nm so float32 copies of one wavelength merge.

MERRA2 `H` and `PHIS` are geopotential quantities. The builder converts their
heights to geometric height using latitude-dependent Earth radius and gravity,
then adds EGM2008 geoid undulation to express them above the WGS84 ellipsoid.
The EGM2008 grid is part of the SASKTRAN2 standard database. On first build,
the builder downloads the grid directly from the public database URL and caches
it under `$SASKTRAN2_DATABASE_ROOT` when set, otherwise under
`${XDG_DATA_HOME:-~/.local/share}/sasktran2/database`. The builder does not
need the SASKTRAN2 Python package installed in its build environment.

The tool writes `tropospheric_aerosol_v1.nc` and `build_report_v1.json`. Review
the report, selected profiles, and spectral calculations before shipping the
catalogue. Pin the reviewed catalogue SHA-256 in
`src/sasktran2/database/tropospheric_aerosol.py`, then run the climatology
tests and docs build. Do not replace an existing catalogue version in place.
