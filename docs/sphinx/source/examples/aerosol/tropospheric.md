---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(_example_tropospheric_aerosol)=
# Add MERRA2 tropospheric aerosol

Use `sk.climatology.tropospheric_aerosol` to inspect a monthly MERRA2 aerosol
reference profile:

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
import sasktran2 as sk

aerosol = sk.climatology.tropospheric_aerosol
profile = aerosol.profile(
    "tropical_typical",
    month=7,
    wavelength_nm=550.,
)

plt.plot(profile.extinction_per_m, profile.altitude_m / 1000)
plt.xlabel("Extinction at 550 nm [m$^{-1}$]")
plt.ylabel("Altitude [km]")
plt.grid(alpha=0.2)
```

## Choose a scenario

Names combine a SAGE-style latitude band with a loading level. The available
bands are southern midlatitudes (`sh_midlat`, 55–35°S), the tropics (`tropical`,
20°S–20°N), and northern midlatitudes (`nh_midlat`, 35–55°N). Each band has
`low`, `typical`, `elevated`, and `extreme` cases.

Month selection is exact and follows the CAIRT scenario months: 1, 4, 7, or 10.
The loading levels use area-weighted monthly tropospheric AOD percentiles. Each
case is an actual MERRA2 column selected from a narrow percentile window. Of
the sampled columns in that window, the one chosen is closest to both the
group's median vertical profile shape (extinction divided by column AOD) and
its median composition (the share of 550 nm AOD from sulfate, dust, sea salt,
black carbon, and organic carbon). Each case therefore has a typical aerosol
mix for its band, month, and loading rather than whatever one column happened
to contain, and the choice does not favour lower or higher loading within the
window. The catalogue records the selected column's shares
(`selection_type_aod_fraction`) and the window's median shares
(`candidate_median_type_aod_fraction`):

| Loading | AOD percentile window |
| --- | --- |
| `low` | P10 ± 3 percentile points |
| `typical` | P50 ± 3 percentile points |
| `elevated` | P90 ± 3 percentile points |
| `extreme` | P99 ± 0.5 percentile points |

The catalogue uses the blended MERRA2 tropopause pressure (`TROPPB`) to define
the tropospheric column. Extinction is calculated from the aerosol mass mixing
ratios, air density, relative humidity, and species/radius-bin mass-extinction
efficiencies. The default reference wavelength is 550 nm; other wavelengths
within the bundled optical-table coverage can be selected:

```{code-cell}
profile = aerosol.profile(
    "nh_midlat_elevated",
    month=1,
    wavelength_nm=870.,
)
profile[["extinction_per_m", "selection_aod_550nm", "tropopause_altitude_m"]]
```

List available cases and their selected-column metadata:

```{code-cell}
aerosol.scenarios(month=10)[
    ["latitude_band", "loading", "selection_aod_550nm", "selected_time"]
]
```

The profile helper returns extinction on its packaged altitude grid or
interpolates to a caller-provided grid. Profile altitudes are geometric heights
above the WGS84 ellipsoid: MERRA2 geopotential heights are converted using
latitude-dependent gravity and Earth radius, then offset by EGM2008 geoid
undulation. The profile does not extend aerosol above the tropopause.

To add a case to an atmosphere, use the constituent. Its extinction follows
the catalogue spectra (250 nm to 40 um), and its single-scattering albedo and
phase function come from the case's MERRA2 species mixture and relative
humidity, using the MERRA2 aerosol optics tables. It supports scalar
calculations only (`num_stokes = 1`).

Coarse dust and sea salt are strongly forward-scattering and need hundreds of
phase-function moments near backscatter. The constituent keeps every table
moment (751), and the engine must keep them too. Truncating sea salt to 128
moments gave radiance errors of tens of percent, or negative radiance, in
testing. Delta-M scaling fixes
the multiple-scatter part but not the single scatter, and the
discrete-ordinates single-scatter source sees only `num_streams` moments.
These settings were within 0.5% of a 64-stream reference. Set them before
creating the `sk.Atmosphere`, which sizes its phase-function storage from the
config:

```python
config = sk.Config()
config.num_singlescatter_moments = 751
config.delta_m_scaling = True
config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
config.num_streams = 16
# Keep the default config.single_scatter_source = sk.SingleScatterSource.Exact

atmosphere = sk.Atmosphere(model_geometry, config, wavelengths_nm=wavelengths_nm)
atmosphere["tropospheric_aerosol"] = aerosol.constituent(
    "tropical_typical",
    month=7,
    altitudes_m=atmosphere.model_geometry.altitudes(),
)
```

Pass `optics_root` to use a local copy of the optics tables. Passing
`optical_property` replaces the MERRA2 optics with your own, normalized to
the catalogue extinction at `wavelength_nm`; its spectral shape may then
disagree with the catalogue's.

## Rebuilding the catalogue

The compact runtime catalogue is built offline from MERRA2 products
`M2I3NVAER`, `M2I3NVASM`, and `M2I1NXASM`, together with the five MERRA2 aerosol
optical tables. By default the builder downloads each day's source granules
from GES DISC to temporary storage and deletes them after processing; with
`--no-stage-daily` it instead streams them using authenticated HTTP range
reads. Create
an Earthdata Login netrc entry. For a local run, install the optional build
dependencies:

```bash
pip install -e ".[merra2-build]"
python tools/databases/tropospheric_aerosol/build.py \
  --optics-root /path/to/MERRA-2-Versions/v0.0.0 \
  --start-year 2024 --end-year 2024 --workers 4
```

Use `--data-root` to process locally available granules instead. The builder
records source years, selected columns, sampling parameters, optical-table
checksums, and the generated catalogue checksum in `build_report_v1.json`.
On DRAC, submit via
`tools/databases/tropospheric_aerosol/submit.sh` to create
`~/scratch/output/sasktran`, save the Slurm log there, and write the catalogue
and report to a year-specific subdirectory. The four-worker job defaults to
one complete year (2024). Set `START_YEAR`, `END_YEAR`, and `OPTICS_ROOT` at
submission time to control the period and cluster optical-table location. Each
job loads `StdEnv/2023`, Python 3.12.4, and the default/latest `scipy-stack`
module, then builds its own virtual environment under `$SLURM_TMPDIR`.
For a quick one-day connectivity and calculation check, use
`JOB_SCRIPT=tools/databases/tropospheric_aerosol/smoke.slurm` and set
`TEST_DATE=YYYY-MM-DD`; it processes all aerosol timestamps for one sample
column and writes an AOD report without attempting to build the climatology.
The versioned output must be scientifically reviewed and its checksum pinned
before it is packaged for runtime use.
