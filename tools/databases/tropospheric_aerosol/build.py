"""Build the compact MERRA2 tropospheric aerosol reference catalogue.

The default input is streamed from GES DISC with HTTP range requests. Earthdata
Login credentials must be available through the user's netrc. ``--data-root``
uses locally available granules instead.
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import hashlib
import importlib
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MethodType

import numpy as np
import sasktran2 as sk
import xarray as xr

_MONTHS = (1, 4, 7, 10)
_BANDS = {
    "sh_midlat": (-55.0, -35.0),
    "tropical": (-20.0, 20.0),
    "nh_midlat": (35.0, 55.0),
}
_TIERS = {
    "low": (0.10, 0.03),
    "typical": (0.50, 0.03),
    "elevated": (0.90, 0.03),
    "extreme": (0.99, 0.005),
}
_COMPONENTS = {
    "SO4": ("optics_SU.v1_3.nc", 0),
    **{f"DU{i:03d}": ("optics_DU.v14_2.nc", i - 1) for i in range(1, 6)},
    **{f"SS{i:03d}": ("optics_SS.v3_3.nc", i - 1) for i in range(1, 6)},
    "BCPHILIC": ("optics_BC.v1_3.nc", 1),
    "BCPHOBIC": ("optics_BC.v1_3.nc", 0),
    "OCPHILIC": ("optics_OC.v1_3.nc", 1),
    "OCPHOBIC": ("optics_OC.v1_3.nc", 0),
}
_AEROSOL_VARIABLES = [
    "AIRDENS",
    "DELP",
    "RH",
    *_COMPONENTS,
]
_METEOROLOGY_VARIABLES = ["H", "PL", "T", "PHIS"]
_TARGET_ALTITUDES_M = np.arange(0.0, 20001.0, 250.0)
_GAS_CONSTANT_DRY_AIR = 287.05
_STANDARD_GRAVITY = 9.80665
_GES_DISC_ROOT = "https://data.gesdisc.earthdata.nasa.gov/data/MERRA2"
_EGM2008_DATABASE_KEY = "geodesy/EGM2008/EGM2008_MERRA2.nc"
_PRODUCTS = {
    "aerosol": ("M2I3NVAER", "inst3_3d_aer_Nv"),
    "meteorology": ("M2I3NVASM", "inst3_3d_asm_Nv"),
    "tropopause": ("M2I1NXASM", "inst1_2d_asm_Nx"),
}
_OPTICS_FILES = tuple(sorted({filename for filename, _ in _COMPONENTS.values()}))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_units(data: xr.Dataset, variable: str, units: str) -> None:
    if variable not in data:
        msg = f"MERRA2 input is missing required variable {variable!r}"
        raise ValueError(msg)
    actual = data[variable].attrs.get("units")
    if actual != units:
        msg = f"MERRA2 {variable} must have units {units!r}, got {actual!r}"
        raise ValueError(msg)


def _load_geoid_undulation() -> xr.DataArray:
    path = sk.database.StandardDatabase().path(_EGM2008_DATABASE_KEY)
    if path is None or not path.is_file():
        msg = f"Unable to load EGM2008 geoid undulation grid: {path}"
        raise FileNotFoundError(msg)
    with xr.open_dataarray(path) as data:
        if not {"lat", "lon"} <= set(data.coords):
            msg = "EGM2008 grid must have lat and lon coordinates"
            raise ValueError(msg)
        return data.load().sortby("lat").sortby("lon")


def _interpolate_geoid_undulation(
    geoid_undulation: xr.DataArray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    return geoid_undulation.interp(
        lat=xr.DataArray(np.asarray(latitude, dtype=float), dims=("lat",)),
        lon=xr.DataArray(np.asarray(longitude, dtype=float), dims=("lon",)),
    ).values


def _geoid_radius(latitude: np.ndarray) -> np.ndarray:
    latitude_rad = np.deg2rad(np.asarray(latitude, dtype=float))
    equatorial_radius_m = 6_378_137.0
    polar_radius_m = equatorial_radius_m * (1 - 1 / 298.257223563)
    return np.sqrt(
        1
        / (
            np.cos(latitude_rad) ** 2 / equatorial_radius_m**2
            + np.sin(latitude_rad) ** 2 / polar_radius_m**2
        )
    )


def _geopotential_to_geometric_height(
    geopotential_height_m: np.ndarray,
    latitude: np.ndarray,
) -> np.ndarray:
    latitude_rad = np.deg2rad(2 * np.asarray(latitude, dtype=float))
    gravity_ratio = (
        1 - 0.002637 * np.cos(latitude_rad) + 0.0000059 * np.cos(latitude_rad) ** 2
    )
    earth_radius_m = _geoid_radius(latitude)
    height = np.asarray(geopotential_height_m, dtype=float)
    return height * earth_radius_m / (gravity_ratio * earth_radius_m - height)


def load_efficiency_profiles(
    optics_dir: Path, wavelength_nm: float = 550.0
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Load species-bin mass-extinction efficiency profiles."""
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
        msg = "wavelength_nm must be finite and positive"
        raise ValueError(msg)
    loaded = {}
    wavelength_grids = []
    for filename in _OPTICS_FILES:
        path = optics_dir / filename
        if not path.is_file():
            msg = f"Missing MERRA2 aerosol optical table: {path}"
            raise FileNotFoundError(msg)
        with xr.open_dataset(path, engine="scipy") as data:
            if data["bext"].attrs.get("units") != "m2 (kg dry mass)-1":
                msg = f"{filename} bext must have units 'm2 (kg dry mass)-1'"
                raise ValueError(msg)
            if data["lambda"].attrs.get("units") != "m":
                msg = f"{filename} lambda coordinate must use metres"
                raise ValueError(msg)
            wavelength_grid = np.asarray(data["lambda"].values, dtype=float) * 1e9
            relative_humidity = np.asarray(data["rh"].values, dtype=float)
            wavelength_grids.append(wavelength_grid)
            profiles = {}
            for species, (source, bin_index) in _COMPONENTS.items():
                if source != filename:
                    continue
                if bin_index >= data.sizes["radius"]:
                    msg = f"{filename} has no radius bin {bin_index} for {species}"
                    raise ValueError(msg)
                values = (
                    data["bext"]
                    .isel(radius=bin_index)
                    .transpose("rh", "lambda")
                    .values.astype(float)
                )
                profiles[species] = (
                    relative_humidity,
                    wavelength_grid,
                    values,
                )
            loaded[filename] = profiles

    low = max(grid.min() for grid in wavelength_grids)
    high = min(grid.max() for grid in wavelength_grids)
    if not low <= wavelength_nm <= high:
        msg = (
            f"{wavelength_nm:g} nm is outside the shared MERRA2 optical-table "
            f"range {low:g}-{high:g} nm"
        )
        raise ValueError(msg)
    wavelengths = np.unique(
        np.concatenate(
            [grid[(grid >= low) & (grid <= high)] for grid in wavelength_grids]
            + [np.array([wavelength_nm])]
        )
    )
    result = {}
    for species, (filename, _) in _COMPONENTS.items():
        rh_grid, source_wavelengths, source_values = loaded[filename][species]
        spectral_values = np.stack(
            [np.interp(wavelengths, source_wavelengths, row) for row in source_values],
            axis=0,
        )
        result[species] = (rh_grid, spectral_values)
    return result, wavelengths


def _efficiency_at_wavelength(
    relative_humidity: np.ndarray,
    profile: tuple[np.ndarray, np.ndarray],
    wavelength_index: int,
) -> np.ndarray:
    rh_grid, spectral_values = profile
    humidity = np.asarray(relative_humidity, dtype=float)
    column = spectral_values[:, wavelength_index]
    return np.interp(np.clip(humidity, rh_grid[0], rh_grid[-1]), rh_grid, column)


def calculate_extinction(
    aerosol: xr.Dataset,
    efficiency_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    wavelengths_nm: np.ndarray,
    wavelength_nm: float = 550.0,
) -> np.ndarray:
    """Calculate MERRA2 total aerosol extinction at one wavelength in m-1."""
    if wavelength_nm < wavelengths_nm[0] or wavelength_nm > wavelengths_nm[-1]:
        msg = f"wavelength_nm must be within {wavelengths_nm[0]:g}-{wavelengths_nm[-1]:g} nm"
        raise ValueError(msg)
    air_density = np.asarray(aerosol["AIRDENS"].values, dtype=float)
    relative_humidity = np.asarray(aerosol["RH"].values, dtype=float)
    if air_density.shape != relative_humidity.shape:
        msg = "MERRA2 AIRDENS and RH must use matching dimensions"
        raise ValueError(msg)
    result = np.zeros_like(air_density, dtype=np.float64)
    for species, (rh_grid, spectral_values) in efficiency_profiles.items():
        if species not in aerosol:
            msg = f"MERRA2 aerosol input is missing required species {species!r}"
            raise ValueError(msg)
        ratio = np.asarray(aerosol[species].values, dtype=float)
        if ratio.shape != result.shape:
            msg = f"MERRA2 {species} dimensions do not match AIRDENS"
            raise ValueError(msg)
        values = spectral_values
        profile = (
            rh_grid,
            values,
        )
        upper = int(np.searchsorted(wavelengths_nm, wavelength_nm, side="right"))
        lower = max(0, upper - 1)
        upper = min(upper, len(wavelengths_nm) - 1)
        if lower == upper:
            efficiency = _efficiency_at_wavelength(relative_humidity, profile, lower)
        else:
            fraction = (wavelength_nm - wavelengths_nm[lower]) / (
                wavelengths_nm[upper] - wavelengths_nm[lower]
            )
            lower_efficiency = _efficiency_at_wavelength(
                relative_humidity, profile, lower
            )
            upper_efficiency = _efficiency_at_wavelength(
                relative_humidity, profile, upper
            )
            efficiency = lower_efficiency * (1 - fraction) + upper_efficiency * fraction
        result += ratio * air_density * efficiency
    return result


def calculate_spectral_extinction(
    aerosol: xr.Dataset,
    efficiency_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Calculate one-column extinction for all wavelengths in m-1."""
    air_density = np.asarray(aerosol["AIRDENS"].values, dtype=float)
    relative_humidity = np.asarray(aerosol["RH"].values, dtype=float)
    result = np.zeros((air_density.size, len(wavelengths_nm)), dtype=np.float64)
    humidity = np.clip(relative_humidity, 0.0, 0.99)
    for species, (rh_grid, values) in efficiency_profiles.items():
        ratio = np.asarray(aerosol[species].values, dtype=float)
        if ratio.shape != air_density.shape:
            msg = f"MERRA2 {species} dimensions do not match AIRDENS"
            raise ValueError(msg)
        efficiency = np.stack(
            [
                np.interp(humidity, rh_grid, values[:, i])
                for i in range(len(wavelengths_nm))
            ],
            axis=-1,
        )
        result += (ratio * air_density).reshape(-1, 1) * efficiency.reshape(
            -1, len(wavelengths_nm)
        )
    return result


def _collection_number(day: date) -> int:
    if day.year <= 1991:
        return 100
    if day.year <= 2000:
        return 200
    if day.year <= 2010:
        return 300
    return 400


def _source_location(product: str, day: date, data_root: Path | None) -> Path | str:
    short_name, filename_suffix = _PRODUCTS[product]
    filename = f"MERRA2_{_collection_number(day)}.{filename_suffix}.{day:%Y%m%d}.nc4"
    if data_root is not None:
        return data_root / short_name / filename
    return f"{_GES_DISC_ROOT}/{short_name}.5.12.4/{day:%Y/%m}/{filename}"


@contextmanager
def _open_granule(location: Path | str):
    if isinstance(location, Path):
        if not location.is_file():
            msg = f"MERRA2 granule not found: {location}"
            raise FileNotFoundError(msg)
        with xr.open_dataset(location, engine="h5netcdf") as data:
            yield data
    else:
        try:
            aiohttp = importlib.import_module("aiohttp")
            fsspec_async = importlib.import_module("fsspec.asyn")
            http_module = importlib.import_module("fsspec.implementations.http")
        except ImportError as exc:
            msg = (
                "Install the MERRA2 build dependencies, including "
                "fsspec[http], to stream GES DISC granules"
            )
            raise ImportError(msg) from exc

        def retryable(error: Exception) -> bool:
            if isinstance(error, aiohttp.ClientResponseError):
                return error.status in {429, 500, 502, 503, 504}
            return isinstance(
                error,
                (aiohttp.ClientConnectionError, asyncio.TimeoutError, OSError),
            )

        def retry_delay(attempt: int) -> float:
            return min(2**attempt, 30) + random.uniform(0, 1)

        class RetryingHTTPFileSystem(http_module.HTTPFileSystem):
            def _open(self, *args, **kwargs):
                for attempt in range(5):
                    try:
                        remote_file = super()._open(*args, **kwargs)
                        break
                    except (
                        aiohttp.ClientResponseError,
                        aiohttp.ClientConnectionError,
                        asyncio.TimeoutError,
                        OSError,
                    ) as exc:
                        if not retryable(exc) or attempt == 4:
                            raise
                        delay = retry_delay(attempt)
                        logging.warning(
                            "Transient GES DISC open failure (%s); retrying in %.1f s",
                            type(exc).__name__,
                            delay,
                        )
                        time.sleep(delay)

                original_fetch = remote_file.async_fetch_range

                async def fetch_with_retry(_file, start, end):
                    attempt = 0
                    while attempt < 5:
                        try:
                            return await original_fetch(start, end)
                        except (
                            aiohttp.ClientResponseError,
                            aiohttp.ClientConnectionError,
                            asyncio.TimeoutError,
                            OSError,
                        ) as exc:
                            if not retryable(exc) or attempt == 4:
                                raise
                            delay = retry_delay(attempt)
                            logging.warning(
                                "Transient GES DISC range failure (%s); "
                                "retrying in %.1f s",
                                type(exc).__name__,
                                delay,
                            )
                            attempt += 1
                            await asyncio.sleep(delay)
                    msg = "HTTP range retry loop exited unexpectedly"
                    raise RuntimeError(msg)

                remote_file.async_fetch_range = MethodType(
                    fetch_with_retry, remote_file
                )
                remote_file.cache.fetcher = fsspec_async.sync_wrapper(
                    remote_file.async_fetch_range, remote_file
                )
                return remote_file

        with (
            RetryingHTTPFileSystem(
                block_size=8 * 1024 * 1024,
                client_kwargs={"trust_env": True},
            ).open(
                location,
                "rb",
            ) as stream,
            xr.open_dataset(stream, engine="h5netcdf") as data,
        ):
            yield data


def _validate_inputs(
    aerosol: xr.Dataset, meteorology: xr.Dataset, tropopause: xr.Dataset
) -> None:
    for name, units in [
        ("AIRDENS", "kg m-3"),
        ("DELP", "Pa"),
        ("RH", "1"),
        *[(species, "kg kg-1") for species in _COMPONENTS],
    ]:
        _required_units(aerosol, name, units)
    for name, units in [
        ("H", "m"),
        ("PL", "Pa"),
        ("T", "K"),
        ("PHIS", "m+2 s-2"),
    ]:
        _required_units(meteorology, name, units)
    _required_units(tropopause, "TROPPB", "Pa")
    for dimension in ("lat", "lon"):
        if not np.array_equal(aerosol[dimension].values, meteorology[dimension].values):
            msg = f"MERRA2 aerosol and meteorology {dimension} coordinates differ"
            raise ValueError(msg)
        if not np.array_equal(aerosol[dimension].values, tropopause[dimension].values):
            msg = f"MERRA2 aerosol and tropopause {dimension} coordinates differ"
            raise ValueError(msg)
    if not np.array_equal(aerosol.time.values, meteorology.time.values):
        msg = "MERRA2 aerosol and meteorology time coordinates differ"
        raise ValueError(msg)
    if not np.array_equal(aerosol["lev"].values, meteorology["lev"].values):
        msg = "MERRA2 aerosol and meteorology model levels differ"
        raise ValueError(msg)


def run_smoke_test(
    day: date,
    optics_root: Path,
    *,
    data_root: Path | None = None,
) -> dict:
    """Read and process a sample column from each MERRA2 product for one day."""
    locations = {
        product: _file_for_product(product, day, data_root) for product in _PRODUCTS
    }
    with (
        _open_granule(locations["aerosol"]) as aerosol,
        _open_granule(locations["meteorology"]) as meteorology,
        _open_granule(locations["tropopause"]) as tropopause,
    ):
        _validate_inputs(aerosol, meteorology, tropopause)
        aerosol_times = aerosol.time.values
        tropopause_time_indices = np.searchsorted(tropopause.time.values, aerosol_times)
        matched = tropopause_time_indices < len(tropopause.time)
        matched[matched] &= (
            tropopause.time.values[tropopause_time_indices[matched]]
            == aerosol_times[matched]
        )
        if not matched.all():
            msg = f"MERRA2 tropopause product is missing aerosol timestamps for {day}"
            raise ValueError(msg)
        latitude_index = int(np.argmin(np.abs(aerosol.lat.values)))
        longitude_index = len(aerosol.lon) // 2
        aerosol_column = (
            aerosol[_AEROSOL_VARIABLES]
            .isel(lat=latitude_index, lon=longitude_index)
            .load()
        )
        meteorology_column = (
            meteorology[_METEOROLOGY_VARIABLES]
            .isel(lat=latitude_index, lon=longitude_index)
            .load()
        )
        tropopause_pressures = tropopause.TROPPB.isel(
            time=tropopause_time_indices,
            lat=latitude_index,
            lon=longitude_index,
        ).values.astype(float)
        latitude = float(aerosol.lat.values[latitude_index])
        longitude = float(aerosol.lon.values[longitude_index])

    efficiencies, wavelengths_nm = load_efficiency_profiles(optics_root)
    extinction = calculate_extinction(aerosol_column, efficiencies, wavelengths_nm)
    pressure = meteorology_column.PL.values.astype(float)
    temperature = meteorology_column.T.values.astype(float)
    delp = aerosol_column.DELP.values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        layer_thickness = (
            _GAS_CONSTANT_DRY_AIR * temperature / _STANDARD_GRAVITY * delp / pressure
        )
    troposphere_mask = pressure >= tropopause_pressures[:, None]
    valid = (
        troposphere_mask
        & np.isfinite(extinction)
        & np.isfinite(layer_thickness)
        & (layer_thickness > 0)
    )
    if not valid.any(axis=1).all():
        msg = f"No valid tropospheric model layers in smoke-test samples for {day}"
        raise ValueError(msg)
    aod = np.sum(np.where(valid, extinction * layer_thickness, 0.0), axis=1)
    if not np.isfinite(aod).all() or np.any(aod <= 0):
        msg = f"Computed smoke-test AOD is not finite and positive: {aod.tolist()}"
        raise ValueError(msg)
    return {
        "status": "ok",
        "date": day.isoformat(),
        "timestamps_processed": [
            np.datetime_as_string(timestamp, unit="s") for timestamp in aerosol_times
        ],
        "latitude_degrees": latitude,
        "longitude_degrees": longitude,
        "model_levels_per_column": int(extinction.shape[-1]),
        "tropospheric_levels_per_timestamp": valid.sum(axis=1).astype(int).tolist(),
        "reference_wavelength_nm": 550.0,
        "tropospheric_aod_550nm_per_timestamp": aod.tolist(),
        "tropospheric_aod_550nm_min": float(aod.min()),
        "tropospheric_aod_550nm_max": float(aod.max()),
        "products": {name: str(location) for name, location in locations.items()},
    }


def _height_at_pressure(
    pressure_pa: np.ndarray, height_m: np.ndarray, tropopause_pressure_pa: float
) -> float:
    valid = np.isfinite(pressure_pa) & np.isfinite(height_m)
    pressure_pa = pressure_pa[valid]
    height_m = height_m[valid]
    order = np.argsort(pressure_pa)
    pressure = pressure_pa[order]
    height = height_m[order]
    if pressure.size < 2 or np.any(np.diff(pressure) <= 0):
        msg = "MERRA2 column pressure and height must be finite and monotonic"
        raise ValueError(msg)
    return float(np.interp(tropopause_pressure_pa, pressure, height))


class _WeightedReservoir:
    def __init__(self, capacity: int, rng: np.random.Generator):
        if capacity < 1000:
            msg = "reservoir_size must be at least 1000 for the P99 loading tier"
            raise ValueError(msg)
        self.capacity = capacity
        self.rng = rng
        self.keys = np.empty(0)
        self.aod = np.empty(0)
        self.profiles = np.empty((0, len(_TARGET_ALTITUDES_M)))
        self.latitude = np.empty(0)
        self.longitude = np.empty(0)
        self.time = np.empty(0, dtype="datetime64[ns]")
        self.time_index = np.empty(0, dtype=np.int16)
        self.latitude_index = np.empty(0, dtype=np.int16)
        self.longitude_index = np.empty(0, dtype=np.int16)
        self.tropopause_altitude = np.empty(0)
        self.surface_altitude = np.empty(0)
        self.valid_candidates = 0

    def add(
        self,
        extinction: np.ndarray,
        height_m: np.ndarray,
        pressure_pa: np.ndarray,
        layer_thickness_m: np.ndarray,
        tropopause_pressure_pa: np.ndarray,
        surface_altitude_m: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        time: np.datetime64,
        time_index: int,
        latitude_indices: np.ndarray,
    ) -> None:
        in_troposphere = pressure_pa >= tropopause_pressure_pa[None, :, :]
        valid_levels = (
            np.isfinite(extinction)
            & np.isfinite(height_m)
            & np.isfinite(pressure_pa)
            & np.isfinite(layer_thickness_m)
        )
        valid_columns = (
            np.isfinite(tropopause_pressure_pa)
            & (tropopause_pressure_pa > 0)
            & np.isfinite(surface_altitude_m)
            & np.all(~in_troposphere | valid_levels, axis=0)
        )
        column_aod = np.sum(
            np.where(
                in_troposphere & valid_levels,
                extinction * layer_thickness_m,
                0.0,
            ),
            axis=0,
        )
        valid_columns &= np.isfinite(column_aod) & (column_aod > 0)
        flat = np.flatnonzero(valid_columns)
        if flat.size == 0:
            return
        nlon = len(longitude)
        local_latitude_indices = flat // nlon
        cell_latitudes = latitude[local_latitude_indices]
        weights = np.cos(np.deg2rad(cell_latitudes))
        random_values = np.maximum(self.rng.random(flat.size), np.finfo(float).tiny)
        keys = np.log(random_values) / weights
        if self.keys.size < self.capacity:
            number_to_keep = min(self.capacity - self.keys.size, keys.size)
            accepted = (
                np.arange(keys.size)
                if number_to_keep == keys.size
                else np.argpartition(keys, -number_to_keep)[-number_to_keep:]
            )
        else:
            accepted = np.flatnonzero(keys > self.keys.min())
        if accepted.size == 0:
            self.valid_candidates += flat.size
            return
        new_keys = keys[accepted]
        new_flat = flat[accepted]
        new_profiles = np.empty((len(accepted), len(_TARGET_ALTITUDES_M)))
        new_tropopause_altitude = np.empty(len(accepted))
        for index, flat_index in enumerate(new_flat):
            ilat, ilon = divmod(int(flat_index), nlon)
            column_pressure = pressure_pa[:, ilat, ilon]
            column_height = height_m[:, ilat, ilon]
            column_extinction = extinction[:, ilat, ilon]
            tropospheric = column_pressure >= tropopause_pressure_pa[ilat, ilon]
            order = np.argsort(column_height)
            sorted_height = column_height[order]
            sorted_extinction = np.where(tropospheric, column_extinction, 0.0)[order]
            if not np.isfinite(sorted_height).all() or np.any(
                np.diff(sorted_height) <= 0
            ):
                new_profiles[index] = np.nan
                new_tropopause_altitude[index] = np.nan
                continue
            new_tropopause_altitude[index] = _height_at_pressure(
                column_pressure,
                column_height,
                tropopause_pressure_pa[ilat, ilon],
            )
            new_profiles[index] = np.interp(
                _TARGET_ALTITUDES_M,
                sorted_height,
                sorted_extinction,
                left=sorted_extinction[0],
                right=0.0,
            )
            new_profiles[
                index, surface_altitude_m[ilat, ilon] > _TARGET_ALTITUDES_M
            ] = 0
            new_profiles[
                index, new_tropopause_altitude[index] < _TARGET_ALTITUDES_M
            ] = 0

        accepted_flat = flat[accepted]
        accepted_lat_index, accepted_lon_index = np.divmod(accepted_flat, nlon)
        keep = np.isfinite(new_profiles).all(axis=1) & np.isfinite(
            new_tropopause_altitude
        )
        new_keys = new_keys[keep]
        new_flat = accepted_flat[keep]
        new_profiles = new_profiles[keep]
        new_tropopause_altitude = new_tropopause_altitude[keep]
        accepted_lat_index = accepted_lat_index[keep]
        accepted_lon_index = accepted_lon_index[keep]
        new_fields = {
            "aod": column_aod.ravel()[new_flat],
            "profiles": new_profiles,
            "latitude": latitude[accepted_lat_index],
            "longitude": longitude[accepted_lon_index],
            "time": np.full(len(new_keys), time, dtype="datetime64[ns]"),
            "time_index": np.full(len(new_keys), time_index, dtype=np.int16),
            "latitude_index": latitude_indices[accepted_lat_index].astype(np.int16),
            "longitude_index": accepted_lon_index.astype(np.int16),
            "tropopause_altitude": new_tropopause_altitude,
            "surface_altitude": surface_altitude_m[
                accepted_lat_index, accepted_lon_index
            ],
        }
        old_fields = {
            "aod": self.aod,
            "profiles": self.profiles,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "time": self.time,
            "time_index": self.time_index,
            "latitude_index": self.latitude_index,
            "longitude_index": self.longitude_index,
            "tropopause_altitude": self.tropopause_altitude,
            "surface_altitude": self.surface_altitude,
        }
        combined_keys = np.concatenate((self.keys, new_keys))
        combined_fields = {
            name: np.concatenate((old_fields[name], new_fields[name]), axis=0)
            for name in old_fields
        }
        keep_count = min(self.capacity, len(combined_keys))
        selected = (
            np.arange(len(combined_keys))
            if keep_count == len(combined_keys)
            else np.argpartition(combined_keys, -keep_count)[-keep_count:]
        )
        self.keys = combined_keys[selected]
        for name, values in combined_fields.items():
            setattr(self, name, values[selected])
        self.valid_candidates += flat.size

    def select_tier(
        self, quantile: float, half_width: float
    ) -> tuple[int, float, int, float, float]:
        if self.aod.size < 1000:
            msg = (
                "At least 1000 valid MERRA2 columns are required per "
                "latitude-band/month"
            )
            raise ValueError(msg)
        lower, upper = np.quantile(
            self.aod,
            [max(0.0, quantile - half_width), min(1.0, quantile + half_width)],
        )
        group = np.flatnonzero((self.aod >= lower) & (self.aod <= upper))
        if group.size < 5:
            msg = (
                f"Only {group.size} sampled columns in percentile window "
                f"{quantile - half_width:.3f}-{quantile + half_width:.3f}"
            )
            raise ValueError(msg)
        group_profiles = self.profiles[group]
        positive = group_profiles[group_profiles > 0]
        if positive.size == 0:
            msg = "Loading tier candidate group has no positive extinction"
            raise ValueError(msg)
        scale = float(np.median(positive))
        transformed = np.log1p(group_profiles / scale)
        median_profile = np.median(transformed, axis=0)
        score = np.mean((transformed - median_profile) ** 2, axis=1)
        selected_index = int(group[np.argmin(score)])
        percentile = float(100 * np.mean(self.aod <= self.aod[selected_index]))
        estimated_count = round(group.size / self.aod.size * self.valid_candidates)
        return (
            selected_index,
            percentile,
            max(estimated_count, 1),
            float(lower),
            float(upper),
        )


def _file_for_product(
    product: str,
    day: date,
    data_root: Path | None,
) -> Path | str:
    return _source_location(product, day, data_root)


def _read_column_spectrum(
    day: date,
    time_index: int,
    latitude_index: int,
    longitude_index: int,
    data_root: Path | None,
    geoid_undulation: xr.DataArray,
    efficiency_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    locations = {name: _file_for_product(name, day, data_root) for name in _PRODUCTS}
    with (
        _open_granule(locations["aerosol"]) as aerosol,
        _open_granule(locations["meteorology"]) as meteorology,
        _open_granule(locations["tropopause"]) as tropopause,
    ):
        aerosol_column = (
            aerosol[_AEROSOL_VARIABLES]
            .isel(time=time_index, lat=latitude_index, lon=longitude_index)
            .load()
        )
        meteorology_column = (
            meteorology[_METEOROLOGY_VARIABLES]
            .isel(time=time_index, lat=latitude_index, lon=longitude_index)
            .load()
        )
        timestamp = aerosol.time.values[time_index]
        tropopause_index = np.flatnonzero(tropopause.time.values == timestamp)
        if tropopause_index.size != 1:
            msg = f"MERRA2 tropopause file has no exact sample for {timestamp}"
            raise ValueError(msg)
        tropopause_pressure = float(
            tropopause.TROPPB.isel(
                time=int(tropopause_index[0]),
                lat=latitude_index,
                lon=longitude_index,
            ).values
        )
        latitude = float(aerosol.lat.values[latitude_index])
        longitude = float(aerosol.lon.values[longitude_index])
        surface_geopotential_height = (
            float(meteorology_column.PHIS.values) / _STANDARD_GRAVITY
        )
    geoid_offset = float(
        _interpolate_geoid_undulation(
            geoid_undulation, np.array([latitude]), np.array([longitude])
        )[0, 0]
    )
    extinction = calculate_spectral_extinction(
        aerosol_column, efficiency_profiles, wavelengths_nm
    )
    pressure = meteorology_column.PL.values.astype(float)
    height = (
        _geopotential_to_geometric_height(
            meteorology_column.H.values.astype(float), latitude
        )
        + geoid_offset
    )
    surface_height = (
        float(_geopotential_to_geometric_height(surface_geopotential_height, latitude))
        + geoid_offset
    )
    above = pressure < tropopause_pressure
    extinction[above, :] = 0
    order = np.argsort(height)
    target = np.empty((len(_TARGET_ALTITUDES_M), len(wavelengths_nm)))
    for index in range(len(wavelengths_nm)):
        target[:, index] = np.interp(
            _TARGET_ALTITUDES_M,
            height[order],
            extinction[order, index],
            left=extinction[order, index][0],
            right=0.0,
        )
    tropopause_height = _height_at_pressure(pressure, height, tropopause_pressure)
    target[surface_height > _TARGET_ALTITUDES_M] = 0
    target[tropopause_height < _TARGET_ALTITUDES_M] = 0
    return target


def _process_month(
    month: int,
    start_year: int,
    end_year: int,
    data_root: Path | None,
    geoid_undulation: xr.DataArray,
    efficiency_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    wavelengths_nm: np.ndarray,
    reservoir_size: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed + month)
    reservoirs = {band: _WeightedReservoir(reservoir_size, rng) for band in _BANDS}
    total_days = sum(
        calendar.monthrange(year, month)[1] for year in range(start_year, end_year + 1)
    )
    processed_days = 0
    for year in range(start_year, end_year + 1):
        day = date(year, month, 1)
        while day.month == month:
            locations = {
                product: _file_for_product(product, day, data_root)
                for product in _PRODUCTS
            }
            if data_root is not None and not all(
                Path(location).is_file() for location in locations.values()
            ):
                missing_products = [
                    name
                    for name, location in locations.items()
                    if not Path(location).is_file()
                ]
                msg = (
                    f"Local MERRA2 inputs are incomplete for {day}: missing "
                    f"{', '.join(missing_products)}"
                )
                raise FileNotFoundError(msg)
            with (
                _open_granule(locations["aerosol"]) as aerosol,
                _open_granule(locations["meteorology"]) as meteorology,
                _open_granule(locations["tropopause"]) as tropopause,
            ):
                _validate_inputs(aerosol, meteorology, tropopause)
                latitudes = aerosol.lat.values
                times = aerosol.time.values
                for time_index, timestamp in enumerate(times):
                    tropopause_index = np.flatnonzero(
                        tropopause.time.values == timestamp
                    )
                    if tropopause_index.size != 1:
                        msg = (
                            f"No exact M2I1NXASM TROPPB sample for {timestamp} in {day}"
                        )
                        raise ValueError(msg)
                    for band, (south, north) in _BANDS.items():
                        latitude_indices = np.flatnonzero(
                            (latitudes >= south) & (latitudes <= north)
                        )
                        lat_slice = slice(
                            int(latitude_indices[0]), int(latitude_indices[-1]) + 1
                        )
                        aer_part = (
                            aerosol[_AEROSOL_VARIABLES]
                            .isel(time=time_index, lat=lat_slice)
                            .load()
                        )
                        met_part = (
                            meteorology[_METEOROLOGY_VARIABLES]
                            .isel(time=time_index, lat=lat_slice)
                            .load()
                        )
                        tp_part = tropopause.TROPPB.isel(
                            time=int(tropopause_index[0]), lat=lat_slice
                        ).load()
                        if not np.array_equal(
                            aer_part.lat.values, met_part.lat.values
                        ) or not np.array_equal(
                            aer_part.lon.values, met_part.lon.values
                        ):
                            msg = "MERRA2 source latitude/longitude grids do not match"
                            raise ValueError(msg)
                        extinction = calculate_extinction(
                            aer_part, efficiency_profiles, wavelengths_nm
                        )
                        model_pressure = met_part.PL.values.astype(float)
                        temperature = met_part.T.values.astype(float)
                        delp = aer_part.DELP.values.astype(float)
                        latitude = aer_part.lat.values.astype(float)
                        longitude = aer_part.lon.values.astype(float)
                        geoid_offset = _interpolate_geoid_undulation(
                            geoid_undulation, latitude, longitude
                        )
                        model_height = (
                            _geopotential_to_geometric_height(
                                met_part.H.values.astype(float),
                                latitude[None, :, None],
                            )
                            + geoid_offset[None, :, :]
                        )
                        surface_altitude = (
                            _geopotential_to_geometric_height(
                                met_part.PHIS.values.astype(float) / _STANDARD_GRAVITY,
                                latitude[:, None],
                            )
                            + geoid_offset
                        )
                        with np.errstate(divide="ignore", invalid="ignore"):
                            layer_thickness = (
                                _GAS_CONSTANT_DRY_AIR
                                * temperature
                                / _STANDARD_GRAVITY
                                * delp
                                / model_pressure
                            )
                        reservoirs[band].add(
                            extinction,
                            model_height,
                            model_pressure,
                            layer_thickness,
                            tp_part.values.astype(float),
                            surface_altitude,
                            aer_part.lat.values.astype(float),
                            aer_part.lon.values.astype(float),
                            timestamp,
                            time_index,
                            latitude_indices,
                        )
            processed_days += 1
            logging.info(
                "MERRA2 month %02d progress: %d/%d days complete (last: %s)",
                month,
                processed_days,
                total_days,
                day.isoformat(),
            )
            day += timedelta(days=1)

    result = []
    for band in _BANDS:
        reservoir = reservoirs[band]
        logging.info(
            "Month %02d %s sampled %d/%d valid columns",
            month,
            band,
            reservoir.aod.size,
            reservoir.valid_candidates,
        )
        for loading, (quantile, half_width) in _TIERS.items():
            index, percentile, count, lower, upper = reservoir.select_tier(
                quantile, half_width
            )
            timestamp = np.datetime_as_string(reservoir.time[index], unit="s")
            selected_day = date.fromisoformat(timestamp[:10])
            spectral_profile = _read_column_spectrum(
                selected_day,
                int(reservoir.time_index[index]),
                int(reservoir.latitude_index[index]),
                int(reservoir.longitude_index[index]),
                data_root,
                geoid_undulation,
                efficiency_profiles,
                wavelengths_nm,
            )
            result.append(
                {
                    "scenario": f"{band}_{loading}",
                    "month": month,
                    "extinction_per_m": spectral_profile,
                    "selection_aod_550nm": float(reservoir.aod[index]),
                    "selection_percentile": percentile,
                    "candidate_count": count,
                    "candidate_window_aod_min": lower,
                    "candidate_window_aod_max": upper,
                    "selected_latitude_degrees": float(reservoir.latitude[index]),
                    "selected_longitude_degrees": float(reservoir.longitude[index]),
                    "selected_time": timestamp,
                    "tropopause_altitude_m": float(
                        reservoir.tropopause_altitude[index]
                    ),
                    "surface_altitude_m": float(reservoir.surface_altitude[index]),
                    "sampled_candidate_count": int(reservoir.aod.size),
                    "valid_candidate_count": int(reservoir.valid_candidates),
                }
            )
    return result


def build_catalogue(
    optics_root: Path,
    output: Path,
    *,
    data_root: Path | None = None,
    start_year: int = 1980,
    end_year: int = datetime.now(timezone.utc).year - 1,
    reservoir_size: int = 50000,
    seed: int = 20261021,
    workers: int = 1,
) -> Path:
    if start_year < 1980 or end_year < start_year:
        msg = "MERRA2 years must satisfy 1980 <= start_year <= end_year"
        raise ValueError(msg)
    if (
        isinstance(workers, (bool, np.bool_))
        or not isinstance(workers, (int, np.integer))
        or workers < 1
    ):
        msg = "workers must be at least 1"
        raise ValueError(msg)
    workers = int(workers)
    efficiencies, wavelengths_nm = load_efficiency_profiles(optics_root)
    geoid_undulation = _load_geoid_undulation()
    month_arguments = [
        (
            month,
            start_year,
            end_year,
            data_root,
            geoid_undulation,
            efficiencies,
            wavelengths_nm,
            reservoir_size,
            seed,
        )
        for month in _MONTHS
    ]
    if workers == 1:
        month_records = []
        for arguments in month_arguments:
            logging.info("Processing MERRA2 month %02d", arguments[0])
            month_records.append(_process_month(*arguments))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_month, *arguments)
                for arguments in month_arguments
            ]
            month_records = [future.result() for future in futures]
    records = [record for group in month_records for record in group]
    scenarios = [f"{band}_{loading}" for band in _BANDS for loading in _TIERS]
    months = list(_MONTHS)
    expected_cases = {(scenario, month) for scenario in scenarios for month in months}
    actual_cases = {(record["scenario"], record["month"]) for record in records}
    if len(records) != len(expected_cases) or actual_cases != expected_cases:
        msg = (
            "Catalogue generation must produce exactly one profile for every "
            "scenario/month pair before writing output"
        )
        raise ValueError(msg)
    altitude_size = len(_TARGET_ALTITUDES_M)
    extinction = np.zeros(
        (len(scenarios), len(months), altitude_size, len(wavelengths_nm)),
        dtype=np.float32,
    )
    metadata_names = [
        "selection_aod_550nm",
        "selection_percentile",
        "candidate_count",
        "candidate_window_aod_min",
        "candidate_window_aod_max",
        "selected_latitude_degrees",
        "selected_longitude_degrees",
        "selected_time",
        "tropopause_altitude_m",
        "surface_altitude_m",
        "sampled_candidate_count",
        "valid_candidate_count",
    ]
    metadata = {
        name: np.empty((len(scenarios), len(months)), dtype=object)
        for name in metadata_names
    }
    for record in records:
        scenario_index = scenarios.index(record["scenario"])
        month_index = months.index(record["month"])
        extinction[scenario_index, month_index] = record["extinction_per_m"]
        for name in metadata_names:
            metadata[name][scenario_index, month_index] = record[name]
    for name in metadata_names:
        if not np.isfinite(
            np.asarray(metadata[name], dtype=float)
            if name != "selected_time"
            else np.ones(metadata[name].shape)
        ).all():
            msg = f"Catalogue build did not select complete metadata for {name}"
            raise ValueError(msg)

    latitude_bands = np.broadcast_to(
        np.array(
            [
                next(band for band in _BANDS if scenario.startswith(f"{band}_"))
                for scenario in scenarios
            ]
        )[:, None],
        (len(scenarios), len(months)),
    )
    loading_levels = np.broadcast_to(
        np.array([scenario.rsplit("_", maxsplit=1)[-1] for scenario in scenarios])[
            :, None
        ],
        (len(scenarios), len(months)),
    )
    dataset = xr.Dataset(
        {
            "extinction_per_m": (
                ("scenario", "month", "altitude_m", "wavelength_nm"),
                extinction,
                {"units": "m-1"},
            ),
            "latitude_band": (("scenario", "month"), latitude_bands),
            "loading": (("scenario", "month"), loading_levels),
            **{
                name: (
                    ("scenario", "month"),
                    np.asarray(values).astype(
                        "datetime64[ns]" if name == "selected_time" else float
                    ),
                )
                for name, values in metadata.items()
            },
        },
        coords={
            "scenario": scenarios,
            "month": months,
            "altitude_m": (
                "altitude_m",
                _TARGET_ALTITUDES_M,
                {"units": "m", "standard_name": "altitude"},
            ),
            "wavelength_nm": (
                "wavelength_nm",
                wavelengths_nm,
                {"units": "nm", "standard_name": "radiation_wavelength"},
            ),
        },
        attrs={
            "catalogue_version": "v1",
            "schema_version": 1,
            "source_products": "M2I3NVAER, M2I3NVASM, M2I1NXASM",
            "source_data_version": "MERRA2 5.12.4",
            "source_years": f"{start_year}-{end_year}",
            "tropopause_field": "M2I1NXASM TROPPB (blended tropopause pressure)",
            "height_coordinate_definition": (
                "MERRA2 geopotential height converted to geometric height and "
                "offset by EGM2008 geoid undulation above the WGS84 ellipsoid"
            ),
            "aod_definition": "Sum(extinction * hydrostatic layer thickness) for PL >= TROPPB",
            "layer_thickness_definition": "287.05 * T / 9.80665 * DELP / PL",
            "reference_wavelength_nm": 550.0,
            "loading_percentile_method": "Area-weighted individual columns; weighted reservoir sample",
            "loading_windows": json.dumps(_TIERS, sort_keys=True),
            "profile_selection": "Actual candidate nearest group median log1p(extinction / median_positive_extinction)",
            "candidate_sampling_seed": seed,
            "candidate_reservoir_size": reservoir_size,
            "profile_altitude_grid": "0-20 km every 250 m; zero below ground and above TROPPB",
            "optical_table_files": json.dumps(_OPTICS_FILES),
            "remote_access": "GES DISC authenticated HTTP range streaming; no full granule downloads",
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    catalogue_path = output / "tropospheric_aerosol_v1.nc"
    dataset.to_netcdf(
        catalogue_path,
        encoding={"extinction_per_m": {"zlib": True, "complevel": 4}},
    )
    optical_digests = {
        filename: _sha256(optics_root / filename) for filename in _OPTICS_FILES
    }
    report = {
        "catalogue_sha256": _sha256(catalogue_path),
        "catalogue_size_bytes": catalogue_path.stat().st_size,
        "source_products": dataset.attrs["source_products"],
        "source_data_version": dataset.attrs["source_data_version"],
        "source_years": dataset.attrs["source_years"],
        "height_coordinate_definition": dataset.attrs["height_coordinate_definition"],
        "optical_tables_sha256": optical_digests,
        "candidate_sampling_seed": seed,
        "candidate_reservoir_size": reservoir_size,
        "months": _MONTHS,
        "latitude_bands": _BANDS,
        "loading_windows": _TIERS,
        "scenarios": records,
    }
    (output / "build_report_v1.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logging.info(
        "Wrote %s (%d bytes), sha256=%s",
        catalogue_path,
        catalogue_path.stat().st_size,
        report["catalogue_sha256"],
    )
    return catalogue_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the compact MERRA2 tropospheric aerosol catalogue."
    )
    parser.add_argument(
        "--optics-root",
        type=Path,
        required=True,
        help="Directory containing the five MERRA2 optical NetCDF tables",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Local root containing M2I1NXASM, M2I3NVAER, and M2I3NVASM "
            "directories; omit to stream from authenticated GES DISC"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/sasktran2/_data/tropospheric_aerosol"),
        help="Output catalogue directory",
    )
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument(
        "--end-year", type=int, default=datetime.now(timezone.utc).year - 1
    )
    parser.add_argument("--reservoir-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of months to process concurrently",
    )
    parser.add_argument(
        "--smoke-date",
        type=date.fromisoformat,
        help="Read and process one sample column for this YYYY-MM-DD date, then exit",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    data_root = None if args.data_root is None else args.data_root.expanduser()
    if args.smoke_date is not None:
        result = run_smoke_test(
            args.smoke_date,
            args.optics_root.expanduser(),
            data_root=data_root,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        report_path = args.output / f"merra2_smoke_{args.smoke_date:%Y%m%d}.json"
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logging.info("Smoke-test result: %s", json.dumps(result, sort_keys=True))
        logging.info("Wrote smoke-test report to %s", report_path)
        return
    build_catalogue(
        args.optics_root.expanduser(),
        args.output,
        data_root=data_root,
        start_year=args.start_year,
        end_year=args.end_year,
        reservoir_size=args.reservoir_size,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
