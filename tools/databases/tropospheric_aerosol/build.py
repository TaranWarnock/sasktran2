"""Build the compact MERRA2 tropospheric aerosol reference catalogue.

By default each day's granules are downloaded from GES DISC to temporary
storage (``--no-stage-daily`` streams them with HTTP range requests instead).
Earthdata Login credentials must be available through the user's netrc.
``--data-root`` uses locally available granules instead.
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import hashlib
import importlib
import json
import logging
import multiprocessing
import os
import random
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from datetime import date, datetime, timezone
from functools import partial
from pathlib import Path
from types import MethodType
from typing import NamedTuple
from urllib.request import urlretrieve

import numpy as np
import requests
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
_WAVELENGTH_DECIMALS_NM = 2
_REFERENCE_WAVELENGTH_NM = 550.0
_GAS_CONSTANT_DRY_AIR = 287.05
_STANDARD_GRAVITY = 9.80665
_GES_DISC_ROOT = "https://data.gesdisc.earthdata.nasa.gov/data/MERRA2"
_EGM2008_DATABASE_KEY = "geodesy/EGM2008/EGM2008_MERRA2.nc"
_EGM2008_DATABASE_URL = (
    "https://arg.usask.ca/sasktranfiles/sasktran2_db/v_latest/"
    f"{_EGM2008_DATABASE_KEY}"
)
_CHECKPOINT_ALGORITHM_VERSION = 2
# GES DISC reprocessed these periods and publishes them only as stream 401.
_STREAM_401_PERIODS = (
    (date(2020, 9, 1), date(2020, 9, 30)),
    (date(2021, 6, 1), date(2021, 9, 30)),
)
_PRODUCTS = {
    "aerosol": ("M2I3NVAER", "inst3_3d_aer_Nv"),
    "meteorology": ("M2I3NVASM", "inst3_3d_asm_Nv"),
    "tropopause": ("M2I1NXASM", "inst1_2d_asm_Nx"),
}
_OPTICS_FILES = tuple(sorted({filename for filename, _ in _COMPONENTS.values()}))
_RESERVOIR_ARRAY_FIELDS = (
    "keys",
    "aod",
    "profiles",
    "latitude",
    "longitude",
    "time",
    "time_index",
    "latitude_index",
    "longitude_index",
    "tropopause_altitude",
    "surface_altitude",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_source_identity(path: Path) -> dict:
    return {
        "kind": "local",
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _remote_source_identity(location: str, info: dict) -> dict:
    etag = info.get("ETag") or info.get("etag")
    last_modified = info.get("Last-Modified") or info.get("last_modified")
    size = info.get("size")
    if not etag and (not last_modified or size is None):
        msg = (
            "GES DISC did not provide an ETag or a Last-Modified header with "
            "known size; "
            f"cannot safely checkpoint source identity for {location}"
        )
        raise ValueError(msg)
    return {
        "kind": "remote",
        "url": location,
        "etag": etag,
        "last_modified": last_modified,
        "size": size,
    }


def _source_identity(location: Path | str) -> dict:
    if isinstance(location, Path):
        if not location.is_file():
            msg = f"MERRA2 granule not found: {location}"
            raise FileNotFoundError(msg)
        return _local_source_identity(location)

    try:
        http_module = importlib.import_module("fsspec.implementations.http")
    except ImportError as exc:
        msg = "Install fsspec[http] to validate remote MERRA2 checkpoint sources"
        raise ImportError(msg) from exc
    filesystem = http_module.HTTPFileSystem(client_kwargs={"trust_env": True})
    fsspec_async = importlib.import_module("fsspec.asyn")
    info = fsspec_async.sync(filesystem.loop, filesystem._info, location)
    return _remote_source_identity(location, info)


def _checkpoint_id(checkpoint_config: dict) -> str:
    return hashlib.sha256(
        json.dumps(checkpoint_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _required_units(data: xr.Dataset, variable: str, units: str) -> None:
    if variable not in data:
        msg = f"MERRA2 input is missing required variable {variable!r}"
        raise ValueError(msg)
    actual = data[variable].attrs.get("units")
    if actual != units:
        msg = f"MERRA2 {variable} must have units {units!r}, got {actual!r}"
        raise ValueError(msg)


def _geoid_cache_path() -> Path:
    # Follows the SASKTRAN2_DATABASE_ROOT override and Linux default of
    # sasktran2.appconfig.database_root() without importing sasktran2, which is
    # not installed in the Slurm build environment. The user config.yml and
    # appdirs macOS/Windows locations are not consulted.
    database_root = os.environ.get("SASKTRAN2_DATABASE_ROOT")
    if database_root:
        return Path(database_root) / _EGM2008_DATABASE_KEY
    data_home = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(data_home) / "sasktran2" / "database" / _EGM2008_DATABASE_KEY


def _read_geoid_undulation(path: Path) -> xr.DataArray:
    with xr.open_dataarray(path) as data:
        if not {"lat", "lon"} <= set(data.coords):
            msg = "EGM2008 grid must have lat and lon coordinates"
            raise ValueError(msg)
        return data.load().sortby("lat").sortby("lon")


def _load_geoid_undulation() -> xr.DataArray:
    path = _geoid_cache_path()
    if path.is_file():
        return _read_geoid_undulation(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        urlretrieve(_EGM2008_DATABASE_URL, temporary_path)
        geoid_undulation = _read_geoid_undulation(temporary_path)
        temporary_path.replace(path)
        return geoid_undulation
    finally:
        temporary_path.unlink(missing_ok=True)


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

    # The tables store wavelength as float32 metres, rounded differently per
    # table; round before the union so one wavelength isn't kept several times.
    rounded_grids = [
        np.round(grid, _WAVELENGTH_DECIMALS_NM) for grid in wavelength_grids
    ]
    low = max(grid.min() for grid in rounded_grids)
    high = min(grid.max() for grid in rounded_grids)
    if not low <= wavelength_nm <= high:
        msg = (
            f"{wavelength_nm:g} nm is outside the shared MERRA2 optical-table "
            f"range {low:g}-{high:g} nm"
        )
        raise ValueError(msg)
    wavelengths = np.unique(
        np.concatenate(
            [grid[(grid >= low) & (grid <= high)] for grid in rounded_grids]
            + [np.array([round(wavelength_nm, _WAVELENGTH_DECIMALS_NM)])]
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


def calculate_species_spectral_extinction(
    aerosol: xr.Dataset,
    efficiency_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """One-column extinction in m-1 per species: (levels, species, wavelengths).

    Species follow the order of ``efficiency_profiles``.
    """
    air_density = np.asarray(aerosol["AIRDENS"].values, dtype=float)
    humidity = np.clip(np.asarray(aerosol["RH"].values, dtype=float), 0.0, 0.99)
    result = np.empty(
        (air_density.size, len(efficiency_profiles), len(wavelengths_nm)),
        dtype=np.float64,
    )
    for index, (species, (rh_grid, values)) in enumerate(efficiency_profiles.items()):
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
        result[:, index, :] = (ratio * air_density).reshape(-1, 1) * efficiency.reshape(
            -1, len(wavelengths_nm)
        )
    return result


def calculate_spectral_extinction(
    aerosol: xr.Dataset,
    efficiency_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Calculate one-column extinction for all wavelengths in m-1."""
    return calculate_species_spectral_extinction(
        aerosol, efficiency_profiles, wavelengths_nm
    ).sum(axis=1)


def _collection_number(day: date) -> int:
    if any(start <= day <= end for start, end in _STREAM_401_PERIODS):
        return 401
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
def _open_granule(location: Path | str, identity_callback=None):
    granule_name = Path(location).name
    logging.info("Opening MERRA2 granule %s", granule_name)
    if isinstance(location, Path):
        if not location.is_file():
            msg = f"MERRA2 granule not found: {location}"
            raise FileNotFoundError(msg)
        if identity_callback is not None:
            identity_callback(_local_source_identity(location))
        with xr.open_dataset(location, engine="h5netcdf") as data:
            logging.info("Opened MERRA2 granule %s", granule_name)
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
            def info(self, path, **kwargs):
                result = fsspec_async.sync(self.loop, self._info, path, **kwargs)
                if identity_callback is not None and path == location:
                    identity_callback(_remote_source_identity(location, result))
                return result

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
                            expired_redirect = (
                                isinstance(exc, aiohttp.ClientResponseError)
                                and exc.status == 403
                            )
                            if (
                                (not retryable(exc) and not expired_redirect)
                                or attempt == 4
                            ):
                                raise
                            if expired_redirect:
                                try:
                                    info = await _file.fs._info(location)
                                except Exception as refresh_error:
                                    raise exc from refresh_error
                                refreshed_url = info.get("url")
                                if not refreshed_url:
                                    raise exc
                                _file.url = refreshed_url
                                logging.warning(
                                    "GES DISC range request returned HTTP 403; "
                                    "refreshed the signed URL"
                                )
                            else:
                                logging.warning(
                                    "Transient GES DISC range failure (%s); "
                                    "retrying",
                                    type(exc).__name__,
                                )
                            delay = retry_delay(attempt)
                            if not expired_redirect:
                                logging.warning("Retrying in %.1f s", delay)
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
            logging.info("Opened MERRA2 granule %s", granule_name)
            yield data


_RETRYABLE_DOWNLOAD_STATUS = frozenset({401, 403, 429, 500, 502, 503, 504})


def _download_granule(location: str, destination: Path) -> dict:
    """Download ``location`` to ``destination``.

    Returns the response's ``size``, ``ETag`` and ``Last-Modified`` in the
    same form as fsspec's ``info()``, so ``_remote_source_identity`` gives the
    same identity as a separate metadata request without the extra round trip.
    """
    temporary_path = destination.with_name(f".{destination.name}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        started = time.monotonic()
        received = 0
        try:
            logging.info("Downloading MERRA2 granule %s", Path(location).name)
            with requests.get(
                location,
                headers={"Accept-Encoding": "identity"},
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                expected_size = int(content_length) if content_length else None
                info = {
                    "size": expected_size,
                    "ETag": response.headers.get("ETag"),
                    "Last-Modified": response.headers.get("Last-Modified"),
                }
                with temporary_path.open("wb") as stream:
                    last_progress = started
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            received += len(chunk)
                            now = time.monotonic()
                            if now - last_progress >= 60:
                                if expected_size is None:
                                    logging.info(
                                        "MERRA2 download progress %s: %d bytes",
                                        Path(location).name,
                                        received,
                                    )
                                else:
                                    logging.info(
                                        "MERRA2 download progress %s: %.1f%% "
                                        "(%d/%d bytes)",
                                        Path(location).name,
                                        100 * received / expected_size,
                                        received,
                                        expected_size,
                                    )
                                last_progress = now
            if expected_size is not None and received != expected_size:
                msg = (
                    f"Incomplete download of {location}: received {received} "
                    f"of {expected_size} bytes"
                )
                raise requests.ConnectionError(msg)
            temporary_path.replace(destination)
            logging.info(
                "Downloaded MERRA2 granule %s (%d bytes, %.1f s)",
                Path(location).name,
                received,
                time.monotonic() - started,
            )
            return info
        except requests.RequestException as exc:
            temporary_path.unlink(missing_ok=True)
            status = (
                exc.response.status_code
                if isinstance(exc, requests.HTTPError) and exc.response is not None
                else None
            )
            # 401/403 can come from an expired Earthdata session or signed
            # redirect; each retry starts a fresh request and authentication.
            retryable = status in _RETRYABLE_DOWNLOAD_STATUS or status is None
            if not retryable or attempt == 4:
                raise
            delay = min(2**attempt, 30) + random.uniform(0, 1)
            logging.warning(
                "Transient MERRA2 download failure for %s (%s: %s); retrying in %.1f s",
                Path(location).name,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise
    msg = "unreachable: the last download attempt returns or raises"
    raise AssertionError(msg)


def _month_executor(workers: int) -> ProcessPoolExecutor:
    """Process pool for month workers, with the parent's logging configuration.

    Threads would serialize on the GIL and on h5py's global HDF5 lock: four
    threads ran only about 1.2x faster than one, four processes about 3.2x.
    ``spawn`` behaves the same on Linux and Windows and does not inherit lock
    state from the parent.
    """
    root = logging.getLogger()
    formats = [
        handler.formatter._fmt
        for handler in root.handlers
        if handler.formatter is not None and handler.formatter._fmt
    ]
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=partial(
            logging.basicConfig,
            level=root.getEffectiveLevel(),
            format=formats[0] if formats else logging.BASIC_FORMAT,
        ),
    )


def _stage_day(
    day: date, directory: Path, checkpointing: bool
) -> tuple[dict, dict[str, Path]]:
    """Download one remote day's granules into ``directory``.

    Failures are logged here, with a traceback, as soon as they happen: the
    month worker only sees them when it reaches this day.
    """
    logging.info("Staging MERRA2 granules for %s in %s", day.isoformat(), directory)
    identities = {}
    staged_locations = {}
    try:
        for product, location in _source_locations(day, None).items():
            staged_path = directory / Path(location).name
            info = _download_granule(location, staged_path)
            if checkpointing:
                # Taken from the download response itself: a separate metadata
                # request costs 5-10 s per granule in the download thread.
                identities[product] = _remote_source_identity(location, info)
            staged_locations[product] = staged_path
    except Exception:
        logging.exception("Failed to stage MERRA2 granules for %s", day.isoformat())
        raise
    return identities, staged_locations


class _DayStager:
    """Stage remote days' granules one day ahead of processing.

    While one day is open, the next day's granules download in a background
    thread into their own temporary directory, so each month worker holds at
    most two staged days at once. Every directory is removed when its day is
    closed or when the stager closes, including after failures.
    """

    def __init__(self, days, checkpointing: bool):
        self._days = list(days)
        self._checkpointing = checkpointing
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = {}

    def _start(self, day: date) -> None:
        if day in self._pending:
            return
        directory = Path(
            tempfile.mkdtemp(
                prefix=f"sasktran2-merra2-{day:%Y%m%d}-",
                dir=os.environ.get("SLURM_TMPDIR") or None,
            )
        )
        future = self._executor.submit(_stage_day, day, directory, self._checkpointing)
        self._pending[day] = (directory, future)

    @contextmanager
    def staged(self, day: date):
        """Yield ``(identities, staged_paths)`` for ``day``."""
        self._start(day)
        position = self._days.index(day)
        if position + 1 < len(self._days):
            self._start(self._days[position + 1])
        directory, future = self._pending.pop(day)
        try:
            yield future.result()
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def close(self) -> None:
        for _, future in self._pending.values():
            future.cancel()
        # Wait for an in-flight download before deleting its directory.
        self._executor.shutdown(wait=True)
        for directory, _ in self._pending.values():
            shutil.rmtree(directory, ignore_errors=True)
        self._pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


@contextmanager
def _open_daily_granules(
    day: date,
    data_root: Path | None,
    *,
    stage_daily: bool,
    checkpointing: bool,
    stager: _DayStager | None = None,
):
    locations = _source_locations(day, data_root)
    missing_products = [
        product
        for product, location in locations.items()
        if data_root is not None and not Path(location).is_file()
    ]
    if missing_products:
        msg = (
            f"Local MERRA2 inputs are incomplete for {day}: missing "
            f"{', '.join(missing_products)}"
        )
        raise FileNotFoundError(msg)

    identities = {}
    if stage_daily and data_root is None:
        # A one-off stager for a single day; _process_month passes a shared
        # one so the next day downloads while this one is processed.
        owner = (
            _DayStager([day], checkpointing) if stager is None else nullcontext(stager)
        )
        with (
            owner as day_stager,
            day_stager.staged(day) as (identities, staged_locations),
            _open_granule(staged_locations["aerosol"]) as aerosol,
            _open_granule(staged_locations["meteorology"]) as meteorology,
            _open_granule(staged_locations["tropopause"]) as tropopause,
        ):
            yield identities, aerosol, meteorology, tropopause
    else:
        callbacks = (
            {
                product: partial(identities.__setitem__, product)
                for product in _PRODUCTS
            }
            if checkpointing
            else {}
        )
        with (
            _open_granule(locations["aerosol"], callbacks.get("aerosol")) as aerosol,
            _open_granule(
                locations["meteorology"], callbacks.get("meteorology")
            ) as meteorology,
            _open_granule(
                locations["tropopause"], callbacks.get("tropopause")
            ) as tropopause,
        ):
            yield identities, aerosol, meteorology, tropopause


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
    locations = _source_locations(day, data_root)
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


def _input_access_description(data_root: Path | None, stage_daily: bool) -> str:
    if data_root is not None:
        return "local MERRA2 granules"
    if stage_daily:
        return (
            "GES DISC authenticated HTTP; full daily granules staged to local "
            "temporary storage and deleted after processing"
        )
    return "GES DISC authenticated HTTP range streaming; no full granule downloads"


def _conservative_grid_profile(
    extinction: np.ndarray,
    height_m: np.ndarray,
    layer_thickness_m: np.ndarray,
    tropospheric: np.ndarray,
    surface_altitude_m: float,
    tropopause_altitude_m: float,
) -> np.ndarray:
    """Map layer extinction (levels, ...) to _TARGET_ALTITUDES_M conserving AOD.

    Each tropospheric layer's AOD (extinction * hydrostatic thickness) is spread
    uniformly between the midpoints of its neighbours' heights, starting at the
    surface. The cumulative AOD is binned into cells around each grid point;
    cells below the surface or above the tropopause fold into the nearest valid
    point (the runtime zeroes those points). Dividing by the trapezoid weights
    makes the trapezoid integral on the grid equal the native tropospheric AOD.
    """
    order = np.argsort(height_m)
    height = height_m[order]
    columns = extinction.reshape(len(height_m), -1)[order]
    layer_aod = np.where(
        tropospheric[order, None], columns * layer_thickness_m[order, None], 0.0
    )
    edges = np.concatenate(
        (
            [min(surface_altitude_m, height[0])],
            0.5 * (height[1:] + height[:-1]),
            [height[-1] + 0.5 * (height[-1] - height[-2])],
        )
    )
    cumulative = np.vstack((np.zeros(layer_aod.shape[1]), np.cumsum(layer_aod, axis=0)))
    grid = _TARGET_ALTITUDES_M
    spacing = grid[1] - grid[0]
    bounds = np.concatenate(([-np.inf], grid[:-1] + 0.5 * spacing, [np.inf]))
    cell_aod = np.diff(
        np.column_stack([np.interp(bounds, edges, column) for column in cumulative.T]),
        axis=0,
    )
    valid = _valid_grid_points(surface_altitude_m, tropopause_altitude_m)
    if not valid.any():
        return np.full((len(grid), *extinction.shape[1:]), np.nan)
    first, last = np.flatnonzero(valid)[[0, -1]]
    cell_aod[first] += cell_aod[:first].sum(axis=0)
    cell_aod[last] += cell_aod[last + 1 :].sum(axis=0)
    cell_aod[~valid] = 0.0
    weights = np.full(len(grid), spacing)
    weights[[0, -1]] = 0.5 * spacing
    return (cell_aod / weights[:, None]).reshape(len(grid), *extinction.shape[1:])


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
            new_profiles[index] = _conservative_grid_profile(
                column_extinction,
                column_height,
                layer_thickness_m[:, ilat, ilon],
                tropospheric,
                surface_altitude_m[ilat, ilon],
                new_tropopause_altitude[index],
            )

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


def _source_locations(day: date, data_root: Path | None) -> dict[str, Path | str]:
    return {
        product: _file_for_product(product, day, data_root) for product in _PRODUCTS
    }


class _ColumnProfiles(NamedTuple):
    """One selected column on _TARGET_ALTITUDES_M.

    Shapes: extinction (altitude, wavelength); species extinction (altitude,
    species in _COMPONENTS order); relative humidity (altitude,).
    """

    extinction_per_m: np.ndarray
    species_extinction_550nm_per_m: np.ndarray
    relative_humidity: np.ndarray


def _valid_grid_points(
    surface_altitude_m: float, tropopause_altitude_m: float
) -> np.ndarray:
    grid = _TARGET_ALTITUDES_M
    return (grid >= surface_altitude_m) & (grid <= tropopause_altitude_m)


def _grid_humidity(
    relative_humidity: np.ndarray,
    height_m: np.ndarray,
    surface_altitude_m: float,
    tropopause_altitude_m: float,
) -> np.ndarray:
    """RH (clipped 0-0.99) at grid points; outside the troposphere, nearest valid."""
    order = np.argsort(height_m)
    humidity = np.interp(
        _TARGET_ALTITUDES_M,
        height_m[order],
        np.clip(relative_humidity, 0.0, 0.99)[order],
    )
    valid = np.flatnonzero(
        _valid_grid_points(surface_altitude_m, tropopause_altitude_m)
    )
    if valid.size:
        humidity[: valid[0]] = humidity[valid[0]]
        humidity[valid[-1] + 1 :] = humidity[valid[-1]]
    return humidity


def _read_column_spectra(
    day: date,
    columns: list[tuple[int, int, int]],
    data_root: Path | None,
    geoid_undulation: xr.DataArray,
    efficiency_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    wavelengths_nm: np.ndarray,
    *,
    stage_daily: bool,
    stager: _DayStager | None = None,
) -> list[_ColumnProfiles]:
    """Read ``(time, latitude, longitude)`` index columns from one day.

    The day's granules are opened once, the same way as in the daily pass:
    remote granules are downloaded first when ``stage_daily`` is set and
    streamed with range requests otherwise.
    """
    with _open_daily_granules(
        day,
        data_root,
        stage_daily=stage_daily,
        checkpointing=False,
        stager=stager,
    ) as (_, aerosol, meteorology, tropopause):
        return [
            _column_spectrum(
                aerosol,
                meteorology,
                tropopause,
                time_index,
                latitude_index,
                longitude_index,
                geoid_undulation,
                efficiency_profiles,
                wavelengths_nm,
            )
            for time_index, latitude_index, longitude_index in columns
        ]


def _column_spectrum(
    aerosol: xr.Dataset,
    meteorology: xr.Dataset,
    tropopause: xr.Dataset,
    time_index: int,
    latitude_index: int,
    longitude_index: int,
    geoid_undulation: xr.DataArray,
    efficiency_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    wavelengths_nm: np.ndarray,
) -> _ColumnProfiles:
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
    species_extinction = calculate_species_spectral_extinction(
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
    layer_thickness = (
        _GAS_CONSTANT_DRY_AIR
        * meteorology_column.T.values.astype(float)
        / _STANDARD_GRAVITY
        * aerosol_column.DELP.values.astype(float)
        / pressure
    )
    tropospheric = pressure >= tropopause_pressure
    tropopause_height = _height_at_pressure(pressure, height, tropopause_pressure)

    def on_grid(values: np.ndarray) -> np.ndarray:
        return _conservative_grid_profile(
            values,
            height,
            layer_thickness,
            tropospheric,
            surface_height,
            tropopause_height,
        )

    return _ColumnProfiles(
        extinction_per_m=on_grid(species_extinction.sum(axis=1)),
        species_extinction_550nm_per_m=on_grid(
            species_extinction[:, :, _reference_wavelength_index(wavelengths_nm)]
        ),
        relative_humidity=_grid_humidity(
            aerosol_column.RH.values.astype(float),
            height,
            surface_height,
            tropopause_height,
        ),
    )


def _reference_wavelength_index(wavelengths_nm: np.ndarray) -> int:
    matches = np.flatnonzero(wavelengths_nm == _REFERENCE_WAVELENGTH_NM)
    if matches.size != 1:
        msg = f"Wavelength grid must contain {_REFERENCE_WAVELENGTH_NM} nm exactly once"
        raise ValueError(msg)
    return int(matches[0])


def _save_month_checkpoint(
    path: Path,
    month: int,
    checkpoint_config: dict,
    last_completed_day: date,
    rng: np.random.Generator,
    reservoirs: dict[str, _WeightedReservoir],
    source_identities: dict[str, dict[str, dict]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "month": month,
        "checkpoint_config": checkpoint_config,
        "last_completed_day": last_completed_day.isoformat(),
        "rng_state": rng.bit_generator.state,
        "source_identities": source_identities,
        "valid_candidates": {
            band: reservoir.valid_candidates for band, reservoir in reservoirs.items()
        },
    }
    arrays = {
        f"{band}_{field}": getattr(reservoir, field)
        for band, reservoir in reservoirs.items()
        for field in _RESERVOIR_ARRAY_FIELDS
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_file, **arrays)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _month_days(month: int, start_year: int, end_year: int) -> list[date]:
    return [
        date(year, month, day)
        for year in range(start_year, end_year + 1)
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ]


def _load_month_checkpoint(
    path: Path,
    month: int,
    checkpoint_config: dict,
    start_year: int,
    end_year: int,
    data_root: Path | None,
    rng: np.random.Generator,
    reservoirs: dict[str, _WeightedReservoir],
) -> tuple[date, dict[str, dict[str, dict]]] | None:
    if not path.is_file():
        return None

    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            if metadata.get("schema_version") != 1:
                msg = f"Unsupported checkpoint schema in {path}"
                raise ValueError(msg)
            if metadata.get("month") != month:
                msg = f"Checkpoint month does not match {month:02d}: {path}"
                raise ValueError(msg)
            if metadata.get("checkpoint_config") != checkpoint_config:
                msg = (
                    f"Checkpoint settings or algorithm version do not match "
                    f"this build: {path}. Use matching inputs or remove the "
                    "stale checkpoint."
                )
                raise ValueError(msg)

            last_completed_day = date.fromisoformat(metadata["last_completed_day"])
            if (
                last_completed_day.month != month
                or not start_year <= last_completed_day.year <= end_year
            ):
                msg = f"Checkpoint date is outside the requested build: {path}"
                raise ValueError(msg)
            source_identities = metadata["source_identities"]
            completed_days = [
                day
                for day in _month_days(month, start_year, end_year)
                if day <= last_completed_day
            ]
            if set(source_identities) != {day.isoformat() for day in completed_days}:
                msg = (
                    f"Checkpoint source days do not match the requested "
                    f"month's days; remove the stale checkpoint: {path}"
                )
                raise ValueError(msg)
            identity_days = len(completed_days)
            logging.info(
                "Validating checkpoint sources for MERRA2 month %02d (%d days)",
                month,
                identity_days,
            )
            for validated_days, day in enumerate(completed_days, start=1):
                expected_identities = source_identities[day.isoformat()]
                actual_identities = {
                    product: _source_identity(location)
                    for product, location in _source_locations(day, data_root).items()
                }
                if actual_identities != expected_identities:
                    msg = (
                        f"MERRA2 source identity changed for {day}; remove "
                        f"the stale checkpoint to rebuild: {path}"
                    )
                    raise ValueError(msg)
                if validated_days % 10 == 0 or day == last_completed_day:
                    logging.info(
                        "Validated checkpoint sources for MERRA2 month %02d: "
                        "%d/%d days (last: %s)",
                        month,
                        validated_days,
                        identity_days,
                        day.isoformat(),
                    )
            for band, reservoir in reservoirs.items():
                fields = {
                    field: checkpoint[f"{band}_{field}"].copy()
                    for field in _RESERVOIR_ARRAY_FIELDS
                }
                sample_count = len(fields["keys"])
                if (
                    sample_count > reservoir.capacity
                    or fields["profiles"].shape
                    != (sample_count, len(_TARGET_ALTITUDES_M))
                    or any(len(values) != sample_count for values in fields.values())
                ):
                    msg = f"Invalid reservoir arrays in checkpoint: {path}"
                    raise ValueError(msg)
                for field, values in fields.items():
                    setattr(reservoir, field, values)
                valid_candidates = metadata["valid_candidates"][band]
                if valid_candidates < sample_count:
                    msg = f"Invalid candidate count in checkpoint: {path}"
                    raise ValueError(msg)
                reservoir.valid_candidates = int(valid_candidates)
            rng.bit_generator.state = metadata["rng_state"]
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        msg = (
            f"Unable to read or validate MERRA2 checkpoint {path}; remove the "
            "stale or corrupt checkpoint and restart"
        )
        raise ValueError(msg) from exc

    return last_completed_day, source_identities


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
    checkpoint_path: Path | None = None,
    checkpoint_config: dict | None = None,
    stage_daily: bool = False,
) -> list[dict]:
    rng = np.random.default_rng(seed + month)
    reservoirs = {band: _WeightedReservoir(reservoir_size, rng) for band in _BANDS}
    days = _month_days(month, start_year, end_year)
    total_days = len(days)
    processed_days = 0
    source_identities = {}
    checkpointing = checkpoint_path is not None
    logging.info(
        "Starting MERRA2 month %02d: %s through %s (%d days)",
        month,
        days[0].isoformat(),
        days[-1].isoformat(),
        total_days,
    )
    if checkpointing:
        if checkpoint_config is None:
            msg = "checkpoint_config is required when checkpointing is enabled"
            raise ValueError(msg)
        logging.info("Checking checkpoint for MERRA2 month %02d", month)
        checkpoint_state = _load_month_checkpoint(
            checkpoint_path,
            month,
            checkpoint_config,
            start_year,
            end_year,
            data_root,
            rng,
            reservoirs,
        )
        if checkpoint_state is not None:
            last_completed_day, source_identities = checkpoint_state
            processed_days = sum(day <= last_completed_day for day in days)
            logging.info(
                "Resuming MERRA2 month %02d after %s (%d/%d days complete)",
                month,
                last_completed_day.isoformat(),
                processed_days,
                total_days,
            )
    stager = (
        _DayStager(days[processed_days:], checkpointing)
        if stage_daily and data_root is None
        else None
    )
    try:
        for day_number, day in enumerate(
            days[processed_days:], start=processed_days + 1
        ):
            logging.info(
                "Starting MERRA2 month %02d day %s (%d/%d)",
                month,
                day.isoformat(),
                day_number,
                total_days,
            )
            with _open_daily_granules(
                day,
                data_root,
                stage_daily=stage_daily,
                checkpointing=checkpointing,
                stager=stager,
            ) as (
                day_source_identities,
                aerosol,
                meteorology,
                tropopause,
            ):
                _validate_inputs(aerosol, meteorology, tropopause)
                latitudes = aerosol.lat.values
                times = aerosol.time.values
                logging.info(
                    "Opened MERRA2 inputs for %s (%d timestamps)",
                    day.isoformat(),
                    len(times),
                )
                for time_index, timestamp in enumerate(times):
                    logging.info(
                        "Processing MERRA2 timestamp %d/%d for %s: %s",
                        time_index + 1,
                        len(times),
                        day.isoformat(),
                        timestamp,
                    )
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
                    logging.info(
                        "Processed MERRA2 timestamp %d/%d for %s: %s",
                        time_index + 1,
                        len(times),
                        day.isoformat(),
                        timestamp,
                    )
            if checkpointing:
                if set(day_source_identities) != set(_PRODUCTS):
                    msg = f"Unable to identify all MERRA2 source granules for {day}"
                    raise ValueError(msg)
                source_identities[day.isoformat()] = day_source_identities
                _save_month_checkpoint(
                    checkpoint_path,
                    month,
                    checkpoint_config,
                    day,
                    rng,
                    reservoirs,
                    source_identities,
                )
            logging.info(
                "MERRA2 month %02d progress: %d/%d days complete (last: %s)",
                month,
                day_number,
                total_days,
                day.isoformat(),
            )
    finally:
        if stager is not None:
            stager.close()

    selections = []
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
            selections.append(
                (band, loading, index, percentile, count, lower, upper, timestamp)
            )

    # Read the selected columns one day at a time, so a day chosen for several
    # scenarios is opened (and downloaded) once.
    selected_days = sorted({date.fromisoformat(item[-1][:10]) for item in selections})
    column_stager = (
        _DayStager(selected_days, checkpointing=False)
        if stage_daily and data_root is None
        else None
    )
    columns = {}
    try:
        for day in selected_days:
            day_selections = [
                item for item in selections if item[-1][:10] == day.isoformat()
            ]
            logging.info(
                "Reading %d selected MERRA2 column(s) for month %02d from %s",
                len(day_selections),
                month,
                day.isoformat(),
            )
            day_columns = _read_column_spectra(
                day,
                [
                    (
                        int(reservoirs[band].time_index[index]),
                        int(reservoirs[band].latitude_index[index]),
                        int(reservoirs[band].longitude_index[index]),
                    )
                    for band, _, index, *_ in day_selections
                ],
                data_root,
                geoid_undulation,
                efficiency_profiles,
                wavelengths_nm,
                stage_daily=stage_daily,
                stager=column_stager,
            )
            for (band, loading, *_), column in zip(
                day_selections, day_columns, strict=True
            ):
                columns[band, loading] = column
    finally:
        if column_stager is not None:
            column_stager.close()

    result = []
    for band, loading, index, percentile, count, lower, upper, timestamp in selections:
        reservoir = reservoirs[band]
        column = columns[band, loading]
        result.append(
            {
                "scenario": f"{band}_{loading}",
                "month": month,
                **column._asdict(),
                "selection_aod_550nm": float(reservoir.aod[index]),
                "selection_percentile": percentile,
                "candidate_count": count,
                "candidate_window_aod_min": lower,
                "candidate_window_aod_max": upper,
                "selected_latitude_degrees": float(reservoir.latitude[index]),
                "selected_longitude_degrees": float(reservoir.longitude[index]),
                "selected_time": timestamp,
                "tropopause_altitude_m": float(reservoir.tropopause_altitude[index]),
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
    stage_daily: bool = True,
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
    logging.info(
        "Preparing MERRA2 catalogue for %d-%d using %d worker(s)",
        start_year,
        end_year,
        workers,
    )
    logging.info("Loading MERRA2 optical tables from %s", optics_root)
    efficiencies, wavelengths_nm = load_efficiency_profiles(optics_root)
    logging.info("Loaded MERRA2 optical tables")
    logging.info("Loading EGM2008 geoid grid")
    geoid_undulation = _load_geoid_undulation()
    logging.info("Loaded EGM2008 geoid grid")
    logging.info("Hashing optical tables and geoid grid for checkpoint validation")
    optical_digests = {
        filename: _sha256(optics_root / filename) for filename in _OPTICS_FILES
    }
    checkpoint_config = {
        "algorithm_version": _CHECKPOINT_ALGORITHM_VERSION,
        "start_year": start_year,
        "end_year": end_year,
        "reservoir_size": reservoir_size,
        "seed": seed,
        "data_root": None if data_root is None else str(data_root.resolve()),
        "optical_tables_sha256": optical_digests,
        "geoid_sha256": _sha256(_geoid_cache_path()),
    }
    checkpoint_id = _checkpoint_id(checkpoint_config)
    checkpoint_dir = output / ".checkpoints" / checkpoint_id
    logging.info("Using checkpoint directory %s", checkpoint_dir)
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
            checkpoint_dir / f"month_{month:02d}.npz",
            checkpoint_config,
            stage_daily,
        )
        for month in _MONTHS
    ]
    if workers == 1:
        month_records = []
        for arguments in month_arguments:
            logging.info("Processing MERRA2 month %02d", arguments[0])
            month_records.append(_process_month(*arguments))
    else:
        logging.info("Starting %d MERRA2 month worker processes", workers)
        with _month_executor(workers) as executor:
            futures = {
                executor.submit(_process_month, *arguments): arguments[0]
                for arguments in month_arguments
            }
            records_by_month = {}
            failures = {}
            # Report each month as it finishes. A failed month does not stop
            # the others, so their checkpoints are complete for a resume.
            for future in as_completed(futures):
                month = futures[future]
                try:
                    records_by_month[month] = future.result()
                except Exception as exc:
                    logging.error(
                        "MERRA2 month %02d failed (%s: %s); the remaining "
                        "months continue so their checkpoints can be resumed",
                        month,
                        type(exc).__name__,
                        exc,
                        exc_info=exc,
                    )
                    failures[month] = exc
                else:
                    logging.info("MERRA2 month %02d complete", month)
        if failures:
            failed = min(failures)
            msg = (
                "MERRA2 month(s) "
                + ", ".join(f"{month:02d}" for month in sorted(failures))
                + " failed; resubmit with the same settings to resume from "
                "checkpoints"
            )
            raise RuntimeError(msg) from failures[failed]
        month_records = [records_by_month[month] for month in _MONTHS]
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
    species_extinction = np.zeros(
        (len(scenarios), len(months), len(_COMPONENTS), altitude_size),
        dtype=np.float32,
    )
    relative_humidity = np.zeros(
        (len(scenarios), len(months), altitude_size), dtype=np.float32
    )
    reference_index = _reference_wavelength_index(wavelengths_nm)
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
        species = np.asarray(record["species_extinction_550nm_per_m"])
        if not np.allclose(
            species.sum(axis=1),
            record["extinction_per_m"][:, reference_index],
            rtol=1e-6,
            atol=0,
        ):
            msg = (
                f"{record['scenario']} month {record['month']}: species extinction "
                "does not sum to the total 550 nm extinction"
            )
            raise ValueError(msg)
        extinction[scenario_index, month_index] = record["extinction_per_m"]
        species_extinction[scenario_index, month_index] = species.T
        relative_humidity[scenario_index, month_index] = record["relative_humidity"]
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
            "species_extinction_550nm_per_m": (
                ("scenario", "month", "species", "altitude_m"),
                species_extinction,
                {
                    "units": "m-1",
                    "wavelength_nm": _REFERENCE_WAVELENGTH_NM,
                    "description": (
                        "550 nm extinction per MERRA2 aerosol species bin; sums to "
                        "extinction_per_m at 550 nm"
                    ),
                },
            ),
            "relative_humidity": (
                ("scenario", "month", "altitude_m"),
                relative_humidity,
                {
                    "units": "1",
                    "description": (
                        "MERRA2 RH clipped to 0-0.99 at grid points; outside the "
                        "troposphere, the nearest tropospheric value"
                    ),
                },
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
            "species": list(_COMPONENTS),
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
            "schema_version": 2,
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
            "profile_altitude_grid": (
                "0-20 km every 250 m; zero below ground and above TROPPB; "
                "values chosen so the trapezoid integral on this grid equals "
                "the native-layer tropospheric AOD"
            ),
            "optical_table_files": json.dumps(_OPTICS_FILES),
            "remote_access": _input_access_description(data_root, stage_daily),
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    catalogue_path = output / "tropospheric_aerosol_v1.nc"
    dataset.to_netcdf(
        catalogue_path,
        encoding={
            name: {"zlib": True, "complevel": 4}
            for name in (
                "extinction_per_m",
                "species_extinction_550nm_per_m",
                "relative_humidity",
            )
        },
    )
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
        # Profiles live in the catalogue; the report keeps only scalar metadata.
        "scenarios": [
            {
                key: value
                for key, value in record.items()
                if key not in _ColumnProfiles._fields
            }
            for record in records
        ],
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
    for month in _MONTHS:
        (checkpoint_dir / f"month_{month:02d}.npz").unlink(missing_ok=True)
    try:
        checkpoint_dir.rmdir()
    except OSError as exc:
        logging.warning("Unable to remove checkpoint directory %s: %s", checkpoint_dir, exc)
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
            "directories; omit to read from authenticated GES DISC"
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
        "--stage-daily",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Download each remote day's granules to temporary storage before "
            "processing (default); --no-stage-daily streams them with HTTP "
            "range reads instead"
        ),
    )
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
    logging.info(
        "MERRA2 builder started (years %d-%d, workers %d, input %s)",
        args.start_year,
        args.end_year,
        args.workers,
        data_root if data_root is not None else "GES DISC",
    )
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
        stage_daily=args.stage_daily,
    )


if __name__ == "__main__":
    main()
