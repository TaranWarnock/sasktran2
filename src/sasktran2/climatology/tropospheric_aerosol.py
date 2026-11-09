"""MERRA2-derived tropospheric aerosol reference scenarios."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator

import sasktran2 as sk
from sasktran2.constituent import ExtinctionScatterer
from sasktran2.database.merra2_aerosol import (
    MERRA2_AEROSOL_SPECIES,
    OpticalDatabaseMERRA2Aerosol,
    ScalarOnlyOpticalDatabaseMixin,
)
from sasktran2.database.tropospheric_aerosol import (
    TroposphericAerosolDatabase,
    _sha256,
)
from sasktran2.optical.base import OpticalProperty, OpticalQuantities
from sasktran2.optical.database import OpticalDatabaseGenericScattererRust

_SCHEMA_VERSION = 2
_REFERENCE_WAVELENGTH_NM = 550.0
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

    Each file is read, validated and hashed once per process. Later calls
    return a copy of the in-memory catalogue for as long as the file's
    modification time and size are unchanged.
    """
    if version != "v1":
        msg = f"Unsupported catalogue version {version!r}; available: ['v1']"
        raise ValueError(msg)
    if path is not None and db_root is not None:
        msg = "Specify either path or db_root, not both"
        raise ValueError(msg)
    mode = "local" if path is not None else "verified_cache"
    if path is not None:
        source = Path(path)
    else:
        database = TroposphericAerosolDatabase(version, db_root)
        # A cached copy was verified when it was loaded; verify again only
        # when the file is new or has changed.
        source = database._db_root / database._filename
        if _cached(source, mode) is None:
            source = database.path()
    cached = _cached(source, mode)
    if cached is None:
        signature = _file_signature(source)
        with xr.open_dataset(source) as data:
            cached = data.load()
        _validate_catalogue(cached)
        cached.attrs.update(
            source_file=source.name,
            source_sha256=_sha256(source),
            source_mode=mode,
        )
        _CATALOGUE_CACHE[(str(source.resolve()), mode)] = (signature, cached)
    return cached.copy(deep=True)


_CATALOGUE_CACHE: dict[tuple[str, str], tuple[tuple[int, int], xr.Dataset]] = {}


def _file_signature(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_mtime_ns, status.st_size


def _cached(path: Path, mode: str) -> xr.Dataset | None:
    """Return the cached catalogue for ``path`` if the file is unchanged."""
    try:
        signature = _file_signature(path)
    except OSError:
        return None
    entry = _CATALOGUE_CACHE.get((str(path.resolve()), mode))
    if entry is None or entry[0] != signature:
        return None
    return entry[1]


def _validate_catalogue(data: xr.Dataset) -> None:
    if (
        data.attrs.get("catalogue_version") != "v1"
        or data.attrs.get("schema_version") != _SCHEMA_VERSION
    ):
        msg = "Incompatible tropospheric aerosol catalogue schema/version"
        raise ValueError(msg)
    for coordinate in ("scenario", "month", "species", "altitude_m", "wavelength_nm"):
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
    unknown = set(data.species.values.tolist()) - set(MERRA2_AEROSOL_SPECIES)
    if unknown:
        msg = f"Catalogue has unknown MERRA2 species {sorted(unknown)}"
        raise ValueError(msg)
    for name, dims, units, valid in (
        (
            "species_extinction_550nm_per_m",
            ("scenario", "month", "species", "altitude_m"),
            "m-1",
            lambda values: np.all(values >= 0),
        ),
        (
            "relative_humidity",
            ("scenario", "month", "altitude_m"),
            "1",
            lambda values: np.all((values >= 0) & (values <= 1)),
        ),
    ):
        if (
            name not in data
            or data[name].dims != dims
            or data[name].attrs.get("units") != units
            or not np.isfinite(data[name].values).all()
            or not valid(data[name].values)
        ):
            msg = f"Catalogue {name} is missing or has invalid dimensions, units or values"
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

    ``altitudes_m`` defaults to the catalogue grid, where the catalogue values
    are returned unchanged. Other grids receive a conservative remap: each
    level's extinction is the catalogue AOD within its trapezoid cell divided
    by the cell width, using a monotone spline of the cumulative AOD.
    Extinction is zero below the surface and above the tropopause, and the
    AOD of cells that cross those boundaries moves to the nearest
    tropospheric level. The trapezoid integral of ``extinction_per_m`` over
    ``altitude_m`` therefore equals the catalogue column AOD whenever the grid
    spans the surface to the tropopause; ``tropospheric_aod`` is that integral.
    At least one level must lie between the surface and the tropopause.

    ``region`` flags each level as below the surface (-1), above the
    tropopause (0) or in the troposphere (1).
    """
    return _profile(
        load_dataset(version, path=path, db_root=db_root),
        scenario,
        month,
        wavelength_nm,
        altitudes_m,
    )


def _cell_bounds(altitudes: np.ndarray) -> np.ndarray:
    """Bounds of the trapezoid-rule cells around each level."""
    return np.concatenate(
        ([altitudes[0]], 0.5 * (altitudes[1:] + altitudes[:-1]), [altitudes[-1]])
    )


def _conservative_remap(
    source_altitudes: np.ndarray,
    source_extinction: np.ndarray,
    target: np.ndarray,
    surface: float,
    tropopause: float,
) -> np.ndarray:
    """Remap a catalogue profile to ``target`` via its cumulative AOD.

    Catalogue extinction is AOD per trapezoid cell (the builder's
    representation). Each tropospheric level's cell AOD is placed in its cell,
    with the outermost cells clipped to the surface and tropopause, and the
    cumulative AOD is interpolated with a monotone (PCHIP) spline. Each target
    level receives the AOD in its own trapezoid cell; cells of levels below
    the surface or above the tropopause fold into the nearest tropospheric
    level, whose extinction is the cell AOD divided by the cell width.

    The result is zero outside the surface-tropopause range, nonnegative, and
    its trapezoid integral equals the catalogue AOD within the target's
    altitude range. On the catalogue grid it reproduces the catalogue.
    """
    valid_source = (source_altitudes >= surface) & (source_altitudes <= tropopause)
    valid_target = (target >= surface) & (target <= tropopause)
    if not valid_source.any() or not valid_target.any():
        grid = "Catalogue" if not valid_source.any() else "altitudes_m"
        msg = (
            f"{grid} has no level between the surface ({surface:g} m) and "
            f"tropopause ({tropopause:g} m)"
        )
        raise ValueError(msg)
    first, last = np.flatnonzero(valid_source)[[0, -1]]
    source_bounds = _cell_bounds(source_altitudes)
    bounds = source_bounds[first : last + 2].copy()
    bounds[0] = min(max(surface, source_altitudes[0]), source_altitudes[first])
    bounds[-1] = max(min(tropopause, source_altitudes[-1]), source_altitudes[last])
    source_aod = source_extinction * np.diff(source_bounds)
    cumulative = np.concatenate(([0.0], np.cumsum(source_aod[first : last + 1])))
    spline = PchipInterpolator(bounds, cumulative)
    target_bounds = _cell_bounds(target)
    cell_aod = np.maximum(
        np.diff(spline(np.clip(target_bounds, bounds[0], bounds[-1]))), 0.0
    )
    first, last = np.flatnonzero(valid_target)[[0, -1]]
    cell_aod[first] += cell_aod[:first].sum()
    cell_aod[last] += cell_aod[last + 1 :].sum()
    cell_aod[~valid_target] = 0.0
    return cell_aod / np.diff(target_bounds)


def _profile(
    data: xr.Dataset,
    scenario: str,
    month: int,
    wavelength_nm: float,
    altitudes_m: np.ndarray | None,
) -> xr.Dataset:
    selected_month = _validate_month(month)
    try:
        wavelength = float(wavelength_nm)
    except (TypeError, ValueError):
        wavelength = np.nan
    if not np.isfinite(wavelength) or wavelength <= 0:
        msg = "wavelength_nm must be finite and positive"
        raise ValueError(msg)
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
    tropopause = float(case.tropopause_altitude_m)
    surface = float(case.surface_altitude_m)
    prepared = _conservative_remap(
        source_altitudes, source_extinction, target, surface, tropopause
    )
    below = target < surface
    above = target > tropopause
    result = xr.Dataset(
        {
            "extinction_per_m": (
                "altitude_m",
                prepared,
                {"units": "m-1", "wavelength_nm": wavelength},
            ),
            "region": (
                "altitude_m",
                np.where(below, -1, np.where(above, 0, 1)).astype(np.int8),
                {
                    "flag_values": [-1, 0, 1],
                    "flag_meanings": "below_surface above_tropopause troposphere",
                },
            ),
            "tropospheric_aod": xr.DataArray(
                float(trapezoid(prepared, target)),
                attrs={
                    "units": "1",
                    "wavelength_nm": wavelength,
                    "description": (
                        "Trapezoid integral of extinction_per_m over altitude_m. "
                        "Equals the catalogue column AOD when altitude_m spans "
                        "the surface to the tropopause."
                    ),
                },
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


class _MERRA2MixedAerosol(
    ScalarOnlyOpticalDatabaseMixin, OpticalDatabaseGenericScattererRust
):
    """One case's MERRA2 aerosol mixture, parameterized by ``altitude_m``."""

    def _require_wavelengths(self, atmo) -> None:
        wavelengths = self._database.wavelength_nm.values
        requested = np.atleast_1d(atmo.wavelengths_nm)
        if requested.min() < wavelengths.min() or requested.max() > wavelengths.max():
            msg = (
                "MERRA2 tropospheric aerosol optics cover "
                f"{wavelengths.min():g}-{wavelengths.max():g} nm; the atmosphere "
                f"requests {requested.min():g}-{requested.max():g} nm"
            )
            raise ValueError(msg)

    def _warn_about_engine_settings(self, atmo) -> None:
        """Warn about engine settings that misrepresent forward-peaked aerosol.

        Coarse dust and sea salt need hundreds of phase-function moments near
        backscatter. In testing, truncating them to 128 moments, using the
        discrete-ordinates single-scatter source, or running multiple scatter
        without delta-M scaling each gave radiance errors of tens of percent.
        """
        available = atmo.leg_coeff.a1.shape[0]
        stored = self._database.sizes["legendre"]
        if available < stored:
            warnings.warn(
                f"The engine keeps {available} of the aerosol's {stored} phase "
                "function moments. Forward-peaked MERRA2 aerosols (dust, sea "
                "salt) can then give a much too small or negative phase function "
                "and radiance; set config.num_singlescatter_moments to "
                f"{stored}.",
                UserWarning,
                stacklevel=3,
            )
        config = getattr(atmo, "_config", None)
        if config is None:
            return
        multiple_scatter = config.multiple_scatter_source
        if (
            multiple_scatter != sk.MultipleScatterSource.NoSource
            and not config.delta_m_scaling
        ):
            warnings.warn(
                "config.delta_m_scaling is off. Multiple scatter uses only "
                "config.num_streams phase moments, which misrepresent "
                "forward-peaked MERRA2 aerosols (dust, sea salt) without delta-M "
                "scaling; set config.delta_m_scaling = True.",
                UserWarning,
                stacklevel=3,
            )
        if (
            config.single_scatter_source == sk.SingleScatterSource.DiscreteOrdinates
            and multiple_scatter == sk.MultipleScatterSource.DiscreteOrdinates
        ):
            warnings.warn(
                "The discrete-ordinates single-scatter source uses only "
                "config.num_streams phase moments, which can misestimate "
                "radiance from forward-peaked MERRA2 aerosols (dust, sea salt) "
                "by tens of percent; use sasktran2.SingleScatterSource.Exact with "
                f"config.num_singlescatter_moments = {stored}.",
                UserWarning,
                stacklevel=3,
            )

    def atmosphere_quantities(self, atmo, **kwargs) -> OpticalQuantities:
        self._require_wavelengths(atmo)
        self._warn_about_engine_settings(atmo)
        return super().atmosphere_quantities(atmo, **kwargs)

    def optical_derivatives(self, atmo, **kwargs) -> dict:
        self._require_wavelengths(atmo)
        return super().optical_derivatives(atmo, **kwargs)


def _nearest_valid(values: xr.DataArray, valid: np.ndarray) -> xr.DataArray:
    """Replace values at invalid altitudes with those of the nearest valid one."""
    indices = np.flatnonzero(valid)
    positions = np.arange(len(valid))
    nearest = indices[np.abs(positions[:, None] - indices[None, :]).argmin(axis=1)]
    return values.isel(altitude_m=nearest).assign_coords(
        altitude_m=values.altitude_m.values
    )


def _merra2_optics(
    case: xr.Dataset,
    optics_root: str | Path | None,
    max_legendre_moments: int | None,
) -> xr.Dataset:
    """Scalar optical database for one catalogue case.

    Extinction follows the catalogue spectra exactly. Single-scattering albedo
    and the P11 phase moments come from mixing the MERRA2 species optics at
    each level's relative humidity, weighting each species by its mass
    (550 nm extinction divided by its 550 nm mass extinction efficiency).
    """
    wavelengths = case.wavelength_nm.values
    if _REFERENCE_WAVELENGTH_NM not in wavelengths:
        msg = f"Catalogue wavelengths must include {_REFERENCE_WAVELENGTH_NM:g} nm"
        raise ValueError(msg)
    humidity = case.relative_humidity.reset_coords(drop=True)
    extinction = 0.0
    scattering = 0.0
    weighted_moments = []
    for name in case.species.values.tolist():
        species_extinction = case.species_extinction_550nm_per_m.sel(
            species=name
        ).reset_coords(drop=True)
        if not (species_extinction.values > 0).any():
            continue
        database = OpticalDatabaseMERRA2Aerosol(
            name, optics_root=optics_root, max_legendre_moments=max_legendre_moments
        )
        with database.load_ds() as opened:
            table = opened.load()
        table_wavelengths = table.wavelength_nm.values
        if (
            wavelengths.min() < table_wavelengths.min()
            or wavelengths.max() > table_wavelengths.max()
        ):
            msg = (
                f"MERRA2 optics table for {name} covers "
                f"{table_wavelengths.min():g}-{table_wavelengths.max():g} nm, not "
                f"the catalogue's {wavelengths.min():g}-{wavelengths.max():g} nm"
            )
            raise ValueError(msg)
        at_levels = table.interp(
            relative_humidity=humidity.clip(
                table.relative_humidity.min(), table.relative_humidity.max()
            )
        )
        mass = species_extinction / at_levels.xs_total.interp(
            wavelength_nm=_REFERENCE_WAVELENGTH_NM
        )
        at_levels = at_levels.interp(wavelength_nm=wavelengths)
        extinction = extinction + mass * at_levels.xs_total
        species_scattering = mass * at_levels.xs_scattering
        scattering = scattering + species_scattering
        weighted_moments.append(species_scattering * at_levels.lm_a1)
    if not weighted_moments:
        msg = "Catalogue case has no positive species extinction"
        raise ValueError(msg)

    moments = max(item.sizes["legendre"] for item in weighted_moments)
    phase = sum(
        item.pad(legendre=(0, moments - item.sizes["legendre"]), constant_values=0.0)
        for item in weighted_moments
    )
    reference = case.extinction_per_m.sel(wavelength_nm=_REFERENCE_WAVELENGTH_NM)
    valid = reference.values > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        fields = {
            "xs_total": case.extinction_per_m / reference,
            "ssa": scattering / extinction,
            "lm_a1": phase / scattering,
        }
    fields = {
        name: _nearest_valid(value.reset_coords(drop=True), valid)
        for name, value in fields.items()
    }
    lm_a1 = fields["lm_a1"].transpose("altitude_m", "wavelength_nm", "legendre")
    result = xr.Dataset(
        {
            "xs_total": fields["xs_total"].transpose("altitude_m", "wavelength_nm"),
            "xs_scattering": (fields["ssa"] * fields["xs_total"]).transpose(
                "altitude_m", "wavelength_nm"
            ),
            "lm_a1": lm_a1,
            **{
                name: xr.zeros_like(lm_a1)
                for name in ("lm_a2", "lm_a3", "lm_a4", "lm_b1", "lm_b2")
            },
        }
    ).astype(np.float64)
    if not all(np.isfinite(value.values).all() for value in result.data_vars.values()):
        msg = "MERRA2 aerosol optics produced non-finite values for this case"
        raise ValueError(msg)
    return result


def constituent(
    scenario: str,
    *,
    month: int,
    optical_property: OpticalProperty | None = None,
    wavelength_nm: float = 550.0,
    altitudes_m: np.ndarray | None = None,
    version: str = "v1",
    path: str | Path | None = None,
    db_root: str | Path | None = None,
    optics_root: str | Path | None = None,
    max_legendre_moments: int | None = None,
) -> ExtinctionScatterer:
    """Create an extinction-normalized aerosol constituent.

    By default the constituent is MERRA2-consistent and scalar only
    (``nstokes=1``): its extinction follows the catalogue spectra at every
    catalogue wavelength (250 nm to 40 um), and its single-scattering albedo
    and phase function come from the case's MERRA2 species mixture and
    relative humidity, using the MERRA2 aerosol optics tables (see
    :class:`~sasktran2.database.OpticalDatabaseMERRA2Aerosol`). ``optics_root``
    selects a local directory holding those tables; otherwise they are
    downloaded from the SASKTRAN2 standard database.

    Coarse dust and sea salt are strongly forward-peaked and need hundreds of
    phase-function moments near backscatter, so ``max_legendre_moments``
    defaults to ``None``, keeping every table moment (751 for the MERRA2
    tables). Truncating sea salt to 128 moments gave radiance errors of tens
    of percent, or negative radiance, in testing. For accurate radiance,
    configure the engine with:

    - ``config.num_singlescatter_moments`` equal to the number of moments kept
      (751 by default);
    - ``config.delta_m_scaling = True`` when a multiple-scatter source is used;
    - the default ``sasktran2.SingleScatterSource.Exact``. The
      discrete-ordinates single-scatter source uses only
      ``config.num_streams`` moments.

    With 16 streams these settings were within 0.5% of a 64-stream reference.
    A warning is raised for each of these settings that is not met.

    Passing ``optical_property`` overrides the MERRA2 optics: its spectral
    shape, scattering and phase function are then used, normalized to the
    catalogue extinction at ``wavelength_nm``. That may contradict the
    catalogue's own spectra.
    """
    data = load_dataset(version, path=path, db_root=db_root)
    selected = _profile(data, scenario, month, wavelength_nm, altitudes_m)
    altitudes = selected.altitude_m.values
    kwargs = {}
    if optical_property is None:
        case = data.sel(scenario=scenario, month=_validate_month(month))
        optical_property = _MERRA2MixedAerosol(
            db=_merra2_optics(case, optics_root, max_legendre_moments)
        )
        kwargs["altitude_m"] = altitudes
    return sk.constituent.ExtinctionScatterer(
        optical_property,
        altitudes,
        selected.extinction_per_m.values,
        wavelength_nm,
        **kwargs,
    )
