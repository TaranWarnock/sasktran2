"""MERRA2-derived tropospheric aerosol reference scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid

import sasktran2 as sk
from sasktran2.constituent import ExtinctionScatterer
from sasktran2.database.tropospheric_aerosol import (
    TroposphericAerosolDatabase,
    _sha256,
)
from sasktran2.optical.base import OpticalProperty

_MONTHS = (1, 4, 7, 10)
_BANDS = ("sh_midlat", "tropical", "nh_midlat")
_LOADINGS = ("low", "typical", "elevated", "extreme")
_REQUIRED_METADATA = (
    "latitude_band",
    "loading",
    "selection_aod_550nm",
    "selection_percentile",
    "candidate_count",
    "selected_latitude_degrees",
    "selected_longitude_degrees",
    "selected_time",
    "tropopause_altitude_m",
    "surface_altitude_m",
)


def _validate_month(month: int) -> int:
    if isinstance(month, (bool, np.bool_)) or not isinstance(month, (int, np.integer)):
        msg = f"month must be one of {_MONTHS}, got {month!r}"
        raise ValueError(msg)
    if int(month) not in _MONTHS:
        msg = f"month must be one of {_MONTHS}, got {month!r}"
        raise ValueError(msg)
    return int(month)


def _grid(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 1
        or len(result) < 2
        or not np.isfinite(result).all()
        or np.any(np.diff(result) <= 0)
    ):
        msg = f"{name} must contain at least two finite, strictly increasing values"
        raise ValueError(msg)
    return result


def load_dataset(
    version: str = "v1",
    *,
    path: str | Path | None = None,
    db_root: str | Path | None = None,
) -> xr.Dataset:
    """Load the packaged MERRA2 aerosol catalogue into memory.

    ``path`` loads an explicit local NetCDF file and bypasses the packaged
    checksum. ``db_root`` selects the cache directory. These options are
    mutually exclusive.
    """
    if version != "v1":
        msg = f"Unsupported catalogue version {version!r}; available: ['v1']"
        raise ValueError(msg)
    if path is not None and db_root is not None:
        msg = "Specify either path or db_root, not both"
        raise ValueError(msg)
    source = (
        Path(path)
        if path is not None
        else TroposphericAerosolDatabase(version, db_root).path()
    )
    with xr.open_dataset(source) as data:
        result = data.load()
    _validate_catalogue(result)
    result.attrs.update(
        source_file=source.name,
        source_sha256=_sha256(source),
        source_mode="local" if path is not None else "verified_cache",
    )
    return result


def _validate_catalogue(data: xr.Dataset) -> None:
    if (
        data.attrs.get("catalogue_version") != "v1"
        or data.attrs.get("schema_version") != 1
    ):
        msg = "Incompatible tropospheric aerosol catalogue schema/version"
        raise ValueError(msg)
    for coordinate in ("scenario", "month", "altitude_m", "wavelength_nm"):
        if coordinate not in data.coords or data[coordinate].dims != (coordinate,):
            msg = f"Catalogue requires a one-dimensional {coordinate} coordinate"
            raise ValueError(msg)
    if data.month.values.tolist() != list(_MONTHS):
        msg = f"Catalogue months must be exactly {_MONTHS}"
        raise ValueError(msg)
    if data.scenario.values.tolist() != [
        f"{band}_{loading}" for band in _BANDS for loading in _LOADINGS
    ]:
        msg = "Catalogue scenarios must contain the twelve SAGE band/loading names"
        raise ValueError(msg)
    altitude = _grid(data.altitude_m.values, "Catalogue altitude_m")
    wavelength = _grid(data.wavelength_nm.values, "Catalogue wavelength_nm")
    if altitude[0] < 0 or wavelength[0] <= 0:
        msg = "Catalogue altitudes and wavelengths must be nonnegative and positive"
        raise ValueError(msg)
    if (
        data.extinction_per_m.dims
        != ("scenario", "month", "altitude_m", "wavelength_nm")
        or data.extinction_per_m.attrs.get("units") != "m-1"
    ):
        msg = "Catalogue extinction_per_m has invalid dimensions or units"
        raise ValueError(msg)
    extinction = data.extinction_per_m.values
    if not np.isfinite(extinction).all() or np.any(extinction < 0):
        msg = "Catalogue extinction must be finite and nonnegative"
        raise ValueError(msg)
    for name in _REQUIRED_METADATA:
        if name not in data or data[name].dims != ("scenario", "month"):
            msg = f"Catalogue requires scenario/month metadata field {name!r}"
            raise ValueError(msg)
    if data.tropopause_altitude_m.attrs.get("units") != "m":
        msg = "Catalogue tropopause_altitude_m must use metres"
        raise ValueError(msg)


def scenarios(
    month: int | None = None,
    version: str = "v1",
    *,
    path: str | Path | None = None,
    db_root: str | Path | None = None,
) -> xr.Dataset:
    """List the twelve band/loading cases and their selected-profile metadata.

    If ``month`` is provided, select that exact month from 1, 4, 7, or 10.
    """
    data = load_dataset(version, path=path, db_root=db_root)
    result = data[list(_REQUIRED_METADATA)]
    if month is not None:
        result = result.sel(month=_validate_month(month), drop=True)
    return result


def profile(
    scenario: str,
    *,
    month: int,
    wavelength_nm: float = 550.0,
    altitudes_m: np.ndarray | None = None,
    version: str = "v1",
    path: str | Path | None = None,
    db_root: str | Path | None = None,
) -> xr.Dataset:
    """Return one MERRA2 extinction profile on an optional altitude grid.

    ``scenario`` is an exact SAGE-style name such as ``"tropical_typical"``.
    ``month`` must be one of 1, 4, 7, or 10. The default extinction wavelength
    is 550 nm; other wavelengths may be selected within the catalogue's
    spectral range. No stratospheric extension is applied.
    """
    selected_month = _validate_month(month)
    try:
        wavelength = float(wavelength_nm)
    except (TypeError, ValueError):
        wavelength = np.nan
    if not np.isfinite(wavelength) or wavelength <= 0:
        msg = "wavelength_nm must be finite and positive"
        raise ValueError(msg)
    data = load_dataset(version, path=path, db_root=db_root)
    names = data.scenario.values.tolist()
    if names.count(scenario) != 1:
        msg = f"Select an exact aerosol scenario; got {scenario!r}, available: {names}"
        raise ValueError(msg)
    wavelengths = data.wavelength_nm.values
    if wavelength < wavelengths[0] or wavelength > wavelengths[-1]:
        msg = f"wavelength_nm must be within {wavelengths[0]:g}-{wavelengths[-1]:g} nm"
        raise ValueError(msg)

    case = data.sel(scenario=scenario, month=selected_month)
    source_altitudes = _grid(case.altitude_m.values, "Catalogue altitude_m")
    source_extinction = case.extinction_per_m.interp(wavelength_nm=wavelength).values
    target = (
        source_altitudes if altitudes_m is None else _grid(altitudes_m, "altitudes_m")
    )
    prepared = np.interp(target, source_altitudes, source_extinction, left=0, right=0)
    tropopause = float(case.tropopause_altitude_m)
    surface = float(case.surface_altitude_m)
    prepared[(target < surface) | (target > tropopause)] = 0
    result = xr.Dataset(
        {
            "extinction_per_m": (
                "altitude_m",
                prepared,
                {"units": "m-1", "wavelength_nm": wavelength},
            ),
            "region": (
                "altitude_m",
                ((target >= surface) & (target <= tropopause)).astype(np.int8),
                {
                    "flag_values": [0, 1],
                    "flag_meanings": "above_tropopause troposphere",
                },
            ),
            "tropospheric_aod": xr.DataArray(
                float(trapezoid(source_extinction, source_altitudes)),
                attrs={"units": "1", "wavelength_nm": wavelength},
            ),
        },
        coords={"altitude_m": ("altitude_m", target, {"units": "m"})},
    )
    for name in _REQUIRED_METADATA:
        result[name] = case[name]
    result.attrs.update(data.attrs)
    result.attrs.update(
        scenario=scenario,
        month=selected_month,
        extinction_wavelength_nm=wavelength,
        source_tropopause_field="TROPPB",
        profile_extension="none; extinction is zero above the tropopause",
    )
    return result


def constituent(
    scenario: str,
    *,
    month: int,
    optical_property: OpticalProperty,
    wavelength_nm: float = 550.0,
    altitudes_m: np.ndarray | None = None,
    version: str = "v1",
    path: str | Path | None = None,
    db_root: str | Path | None = None,
) -> ExtinctionScatterer:
    """Create an extinction-normalized aerosol constituent.

    The caller supplies an optical property whose spectral scattering model is
    appropriate for the intended calculation. Its extinction is normalized to
    the MERRA2 profile at ``wavelength_nm``.
    """
    if optical_property is None:
        msg = "optical_property is required for a scattering constituent"
        raise ValueError(msg)
    data = profile(
        scenario,
        month=month,
        wavelength_nm=wavelength_nm,
        altitudes_m=altitudes_m,
        version=version,
        path=path,
        db_root=db_root,
    )
    return sk.constituent.ExtinctionScatterer(
        optical_property,
        data.altitude_m.values,
        data.extinction_per_m.values,
        wavelength_nm,
    )
