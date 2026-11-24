---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(_example_tropospheric_aerosol)=
# Add tropospheric aerosol

Use `sk.climatology.tropospheric_aerosol` to add MERRA2 aerosol reference
scenarios to an existing atmosphere, one aerosol type at a time:

```python
import sasktran2 as sk

aerosol = sk.climatology.tropospheric_aerosol
atmosphere["dust"] = aerosol.constituent(
    "tropical_elevated", aerosol_type="dust", month=7, ground_altitude_m=0.0
)
atmosphere["sulfate"] = aerosol.constituent(
    "tropical_typical", aerosol_type="sulfate", month=7, ground_altitude_m=0.0
)
```

Each constituent holds one of the five MERRA2 aerosol types, from an actual
MERRA2 column chosen for that type's loading. Build a mixture by adding one
constituent per type, under different names, each at its own loading level. The
helper supplies extinction from 250 nm to 40 microns, with single-scattering
albedo and phase function from the type's MERRA2 species and relative humidity.
It supports scalar calculations only (`num_stokes = 1`), and needs the engine
settings shown in the radiance example below.

You can add these constituents alongside your chosen gases and Rayleigh
scattering. To also include stratospheric aerosol, add
{ref}`stratospheric aerosol <_example_stratospheric_aerosol>` created with
`lower_extension="zero"`. Its default exponential taper below the observations
models stratospheric aerosol reaching into the troposphere, which the
tropospheric cases already contain.

## Choose a scenario

Choose an aerosol type, then combine a latitude band with a loading level and
pick a month, for example dust in `nh_midlat_elevated` in month 1. Pass the type
as `aerosol_type`, by name or code (either case):

| Aerosol type | `aerosol_type` | MERRA2 species |
| --- | --- | --- |
| Sulfate | `"sulfate"` or `"SU"` | `SO4` |
| Dust | `"dust"` or `"DU"` | five size bins, `DU001`–`DU005` |
| Sea salt | `"sea_salt"` or `"SS"` | five size bins, `SS001`–`SS005` |
| Black carbon | `"black_carbon"` or `"BC"` | hydrophilic and hydrophobic |
| Organic carbon | `"organic_carbon"` or `"OC"` | hydrophilic and hydrophobic |

There are twelve scenarios for each type, each available for months 1, 4, 7,
and 10:

| Latitude band | Name prefix | Available loading levels |
| --- | --- | --- |
| Southern midlatitudes, 55–35°S | `sh_midlat` | `low`, `typical`, `elevated`, `extreme` |
| Tropics, 20°S–20°N | `tropical` | `low`, `typical`, `elevated`, `extreme` |
| Northern midlatitudes, 35–55°N | `nh_midlat` | `low`, `typical`, `elevated`, `extreme` |

Start with `typical`, then use the other levels to explore the effect of aerosol
loading. Each type has its own levels, from area-weighted percentiles of that
type's monthly tropospheric 550 nm optical depth within each latitude band:

| Loading | AOD percentile window |
| --- | --- |
| `low` | P10 ± 3 percentile points |
| `typical` | P50 ± 3 percentile points |
| `elevated` | P90 ± 3 percentile points |
| `extreme` | P99 ± 0.5 percentile points |

Of the sampled columns in each window, the one chosen is closest to both the
window's median vertical profile shape of the type (its extinction divided by
its optical depth) and the window's median mix of the type's species: the size
bins for dust and sea salt, or the hydrophilic share for black and organic
carbon. Sulfate has a single species, so only its shape counts. Each case
therefore has a typical profile and particle mix for its type, band, month, and
loading. The catalogue records the selected column's species shares
(`selection_species_aod_fraction`) and the window's median shares
(`candidate_median_species_aod_fraction`).

Because each type is ranked separately, the dust `extreme` column in a band is
usually a different MERRA2 column from the sulfate `extreme` one. Adding several
types combines aerosol from different places and times, each typical for its
own loading level.

You can list the scenarios and their selected columns in Python.
`selection_aod_550nm` is the selected column's optical depth of the type, and
`column_total_aod_550nm` that of all its aerosol:

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
import sasktran2 as sk

aerosol = sk.climatology.tropospheric_aerosol
aerosol.scenarios(month=7, aerosol_type="dust")[
    [
        "latitude_band",
        "loading",
        "selection_aod_550nm",
        "column_total_aod_550nm",
        "selected_time",
    ]
]
```

## Compare the profiles

Use `profile()` to inspect extinction before adding a scenario. The extinction
is at 550 nm by default, in m⁻¹; pass `wavelength_nm` for another catalogue
wavelength. Altitude is in metres.

```{code-cell}
types = {
    "sulfate": "Sulfate",
    "dust": "Dust",
    "sea_salt": "Sea salt",
    "black_carbon": "Black carbon",
    "organic_carbon": "Organic carbon",
}
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
for ax, (aerosol_type, title) in zip(axes, types.items()):
    for loading in ["low", "typical", "elevated", "extreme"]:
        profile = aerosol.profile(
            f"tropical_{loading}",
            aerosol_type=aerosol_type,
            month=7,
            ground_altitude_m=0.0,
        )
        troposphere = profile.region == 1
        ax.semilogx(
            profile.extinction_per_m.where(troposphere),
            profile.altitude_m / 1000,
            label=loading,
        )
    ax.set(title=f"{title}, tropics, July", xlabel="Extinction at 550 nm [m$^{-1}$]",
           ylim=(0, 18))
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Altitude above ground [km]")
axes[-1].legend()
fig.tight_layout()
```

The catalogue stores the whole MERRA2 column, from the surface to above the
model top near 80 km, including aerosol above the tropopause. Its grid is
250 m apart up to 20 km and coarser above. By default (`cut_at_tropopause=True`) profiles stop at
the tropopause, so extinction is zero below the surface and above the
tropopause. The `region` variable flags those levels as -1 and 0 and marks the
troposphere with 1; the plot uses it to leave out the zero levels. Profile
altitudes are geometric heights above the WGS84 ellipsoid, converted from MERRA2
geopotential heights with the EGM2008 geoid. Pass `altitudes_m` to remap the
profile to your own grid. The remap conserves optical depth: each level gets the
catalogue optical depth in its trapezoid cell divided by the cell width, so
`tropospheric_aod`, the trapezoid integral of the returned profile, equals the
catalogue tropospheric optical depth on any grid that spans the troposphere.

## Adjust smoothing and ground altitude

The defaults keep the MERRA2 column as it is:

| Option | Default | Effect |
| --- | --- | --- |
| `ground_altitude_m` | `None` | Keeps the column's real terrain height, which can be several hundred metres. Set it to move the whole profile so its surface lies at that altitude, for example `0.` when the model's ground is at 0 m. |
| `smoothing_fwhm_m` | `0.` | Applies a Gaussian of this full width at half maximum between the surface and the top of the profile, preserving optical depth at every wavelength. It acts on the 250 m part of the grid (up to 20 km); an uncut profile is not smoothed above that. |
| `cut_at_tropopause` | `True` | Stops the profile at the tropopause. Set `False` to keep the whole MERRA2 column; `aod` is then the whole profile's optical depth and `tropospheric_aod` the part below the tropopause. That part overlaps stratospheric aerosol, mostly sulfate, so don't combine it with the stratospheric constituent without care. |
| `altitudes_m` | `None` | Remaps the constituent onto the atmosphere's model altitudes when it is added. Set a grid to fix it yourself, for example to edit `extinction_per_m` before the calculation. |

Smoothing is off by default. Unlike the noisy satellite observations behind the
stratospheric scenarios, the catalogue's structure (boundary-layer top, elevated
dust and smoke layers) is MERRA2's, and widths above a few hundred metres blur
it. The same options work with `profile()`, so you can plot a customised profile
first:

```{code-cell}
for fwhm in (0.0, 1000.0):
    smoothed = aerosol.profile(
        "nh_midlat_extreme",
        aerosol_type="dust",
        month=7,
        smoothing_fwhm_m=fwhm,
        ground_altitude_m=0.0,
    )
    plt.plot(smoothed.extinction_per_m, smoothed.altitude_m / 1000, label=f"{fwhm:g} m")
plt.xlabel("Extinction at 550 nm [m$^{-1}$]")
plt.ylabel("Altitude above ground [km]")
plt.ylim(0, 18)
plt.legend(title="smoothing_fwhm_m")
plt.grid(alpha=0.2)
```

The constituent's remap onto the model grid matters on coarse grids: plain
linear interpolation onto a 1 km grid lost a median 24% of the optical depth
across the catalogue cases, mostly from the lowest few hundred metres.

## Calculate radiance with aerosol

This complete example creates an atmosphere, adds elevated tropical dust and
typical tropical sulfate, and calculates a limb radiance at three wavelengths:

```{code-cell}
config = sk.Config()
config.num_singlescatter_moments = 751
config.delta_m_scaling = True
config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
config.num_streams = 16

model_geometry = sk.Geometry1D(
    cos_sza=0.6,
    solar_azimuth=0.,
    earth_radius_m=6372000.,
    altitude_grid_m=np.arange(0., 65001., 500.),
    interpolation_method=sk.InterpolationMethod.LinearInterpolation,
    geometry_type=sk.GeometryType.Spherical,
)
atmosphere = sk.Atmosphere(
    model_geometry, config, wavelengths_nm=np.array([525., 756., 1021.])
)
sk.climatology.us76.add_us76_standard_atmosphere(atmosphere)
atmosphere["rayleigh"] = sk.constituent.Rayleigh()
atmosphere["dust"] = aerosol.constituent(
    "tropical_elevated", aerosol_type="dust", month=7, ground_altitude_m=0.0
)
atmosphere["sulfate"] = aerosol.constituent(
    "tropical_typical", aerosol_type="sulfate", month=7, ground_altitude_m=0.0
)

viewing_geometry = sk.ViewingGeometry()
viewing_geometry.add_ray(sk.TangentAltitudeSolar(
    tangent_altitude_m=5000.,
    relative_azimuth=0.,
    observer_altitude_m=200000.,
    cos_sza=0.6,
))
engine = sk.Engine(config, model_geometry, viewing_geometry)
result = engine.calculate_radiance(atmosphere)
result.radiance
```

Set these options before creating the `sk.Atmosphere`, which sizes its
phase-function storage from the config. Coarse dust and sea salt are strongly
forward-scattering and need hundreds of phase-function moments near
backscatter. The constituent keeps every MERRA2 table moment (751), and the
engine must keep them too: truncating sea salt to 128 moments gave radiance
errors of tens of percent, or negative radiance, in testing. Delta-M scaling
fixes the multiple-scatter part but not the single scatter, so keep the default
exact single-scatter source; the discrete-ordinates single-scatter source sees
only `num_streams` moments. These settings were within 0.5% of a 64-stream
reference, and the constituent warns when the engine settings would
misrepresent the aerosol.

The first use may download the catalogue and the MERRA2 aerosol optics tables.
Pass `optics_root` to use a local copy of the tables. Passing `optical_property`
replaces the MERRA2 optics with your own, normalized to the catalogue extinction
at `wavelength_nm`; its spectral shape may then disagree with the catalogue's.

## Rebuilding the catalogue

The compact runtime catalogue is built offline from MERRA2 products
`M2I3NVAER`, `M2I3NVASM`, and `M2I1NXASM`, together with the five MERRA2 aerosol
optical tables. By default the builder reads 2016 through the last complete
calendar year, so the scenarios represent the current aerosol climate; pass
`--start-year` and `--end-year` to choose another period. It downloads each
day's source granules from GES DISC to temporary storage and deletes them after
processing; with `--no-stage-daily` it instead streams them using authenticated
HTTP range reads. The days holding the selected columns are read the same way.
Create an Earthdata Login netrc entry. For a quick local run
of one year, install the optional build dependencies:

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
