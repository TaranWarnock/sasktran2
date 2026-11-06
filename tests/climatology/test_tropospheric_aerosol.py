from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import sasktran2 as sk
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from sasktran2.climatology import tropospheric_aerosol as aerosol
from sasktran2.optical.base import OpticalProperty, OpticalQuantities

_BUILD_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "databases"
    / "tropospheric_aerosol"
    / "build.py"
)
_BUILD_SPEC = importlib.util.spec_from_file_location(
    "tropospheric_aerosol_build", _BUILD_PATH
)
assert _BUILD_SPEC is not None
assert _BUILD_SPEC.loader is not None
build = importlib.util.module_from_spec(_BUILD_SPEC)
sys.modules[_BUILD_SPEC.name] = build
_BUILD_SPEC.loader.exec_module(build)


@pytest.fixture
def catalogue():
    scenarios = [
        f"{band}_{loading}"
        for band in ("sh_midlat", "tropical", "nh_midlat")
        for loading in ("low", "typical", "elevated", "extreme")
    ]
    months = [1, 4, 7, 10]
    altitude = np.arange(0.0, 20001.0, 1000.0)
    wavelengths = np.array([250.0, 550.0, 1000.0])
    ext = np.empty((12, 4, len(altitude), len(wavelengths)))
    tropopause = np.full((12, 4), 12000.0)
    surface = np.full((12, 4), 1000.0)
    for i in range(12):
        for j in range(4):
            ext[i, j] = (
                (i + 1)
                * (j + 1)
                * 1e-6
                * np.exp(-altitude[:, None] / 6000)
                * (550 / wavelengths)[None, :]
            )
            ext[i, j, altitude > tropopause[i, j], :] = 0
            ext[i, j, altitude < surface[i, j], :] = 0
    band = np.repeat(
        np.array(["sh_midlat"] * 4 + ["tropical"] * 4 + ["nh_midlat"] * 4)[:, None],
        4,
        axis=1,
    )
    loading = np.repeat(
        np.array(["low", "typical", "elevated", "extreme"] * 3)[:, None],
        4,
        axis=1,
    )
    shape = (12, 4)
    species = list(build._COMPONENTS)
    shares = np.arange(1, len(species) + 1, dtype=float)
    shares /= shares.sum()
    species_ext = shares[None, None, :, None] * ext[:, :, None, :, 1]
    return xr.Dataset(
        {
            "extinction_per_m": (
                ("scenario", "month", "altitude_m", "wavelength_nm"),
                ext,
                {"units": "m-1"},
            ),
            "species_extinction_550nm_per_m": (
                ("scenario", "month", "species", "altitude_m"),
                species_ext,
                {"units": "m-1"},
            ),
            "relative_humidity": (
                ("scenario", "month", "altitude_m"),
                np.full((12, 4, len(altitude)), 0.5),
                {"units": "1"},
            ),
            "latitude_band": (("scenario", "month"), band),
            "loading": (("scenario", "month"), loading),
            "selection_aod_550nm": (("scenario", "month"), np.full(shape, 0.1)),
            "selection_percentile": (("scenario", "month"), np.full(shape, 50.0)),
            "candidate_count": (("scenario", "month"), np.full(shape, 1000)),
            "selected_latitude_degrees": (
                ("scenario", "month"),
                np.zeros(shape),
            ),
            "selected_longitude_degrees": (
                ("scenario", "month"),
                np.zeros(shape),
            ),
            "selected_time": (
                ("scenario", "month"),
                np.full(shape, np.datetime64("2025-01-01")),
            ),
            "tropopause_altitude_m": (
                ("scenario", "month"),
                tropopause,
                {"units": "m"},
            ),
            "surface_altitude_m": (
                ("scenario", "month"),
                surface,
                {"units": "m"},
            ),
        },
        coords={
            "scenario": scenarios,
            "month": months,
            "species": species,
            "altitude_m": ("altitude_m", altitude, {"units": "m"}),
            "wavelength_nm": ("wavelength_nm", wavelengths, {"units": "nm"}),
        },
        attrs={"catalogue_version": "v1", "schema_version": 2},
    )


@pytest.fixture
def catalogue_path(catalogue, tmp_path):
    path = tmp_path / "catalogue.nc"
    catalogue.to_netcdf(path)
    return path


def test_load_scenarios_and_profile(catalogue, catalogue_path):
    data = aerosol.load_dataset(path=catalogue_path)
    assert (
        data.attrs["source_sha256"]
        == hashlib.sha256(catalogue_path.read_bytes()).hexdigest()
    )
    listing = aerosol.scenarios(path=catalogue_path)
    assert listing.sizes == {"scenario": 12, "month": 4}
    assert aerosol.scenarios(month=4, path=catalogue_path).sizes == {"scenario": 12}

    result = aerosol.profile(
        "tropical_typical",
        month=4,
        wavelength_nm=550,
        altitudes_m=np.arange(0.0, 21001.0, 500.0),
        path=catalogue_path,
    )
    source = catalogue.sel(scenario="tropical_typical", month=4)
    assert result.sizes["altitude_m"] == 43
    assert float(result.extinction_per_m.sel(altitude_m=0)) == 0
    assert float(result.extinction_per_m.sel(altitude_m=1500)) > 0
    assert float(result.extinction_per_m.sel(altitude_m=12500)) == 0
    assert_allclose(
        result.tropospheric_aod,
        np.trapezoid(
            source.extinction_per_m.sel(wavelength_nm=550),
            source.altitude_m,
        ),
    )
    assert result.attrs["source_tropopause_field"] == "TROPPB"
    assert float(result.attrs["extinction_wavelength_nm"]) == 550


@pytest.mark.parametrize(
    ("scenario", "month", "wavelength", "message"),
    [
        ("tropical", 1, 550, "exact aerosol scenario"),
        ("tropical_typical", 2, 550, "month must be"),
        ("tropical_typical", 1.0, 550, "month must be"),
        ("tropical_typical", 1, 10000, "within"),
    ],
)
def test_invalid_profile_selection(
    catalogue_path, scenario, month, wavelength, message
):
    with pytest.raises(ValueError, match=message):
        aerosol.profile(
            scenario,
            month=month,
            wavelength_nm=wavelength,
            path=catalogue_path,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("schema", "schema/version"),
        ("unit", "units"),
        ("species", "species_extinction_550nm_per_m"),
        ("humidity", "relative_humidity"),
    ],
)
def test_invalid_catalogue(catalogue, tmp_path, change, message):
    if change == "schema":
        catalogue.attrs["schema_version"] = 1
    elif change == "unit":
        catalogue.extinction_per_m.attrs["units"] = "km-1"
    elif change == "species":
        catalogue = catalogue.drop_vars("species_extinction_550nm_per_m")
    else:
        catalogue["relative_humidity"] = catalogue.relative_humidity + 1.0
    path = tmp_path / "invalid.nc"
    catalogue.to_netcdf(path)
    with pytest.raises(ValueError, match=message):
        aerosol.load_dataset(path=path)


class _ConstantOptics(OpticalProperty):
    def cross_sections(self, wavelengths_nm, altitudes_m, **kwargs):
        extinction = np.ones((len(altitudes_m), len(wavelengths_nm)))
        return OpticalQuantities(extinction=extinction, ssa=np.ones_like(extinction))

    def atmosphere_quantities(self, atmo, **kwargs):
        raise NotImplementedError


def test_constituent_uses_selected_reference_extinction(catalogue_path):
    result = aerosol.profile("sh_midlat_low", month=1, path=catalogue_path)
    constituent = aerosol.constituent(
        "sh_midlat_low",
        month=1,
        optical_property=_ConstantOptics(),
        path=catalogue_path,
    )
    assert_allclose(constituent.extinction_per_m, result.extinction_per_m)


@pytest.fixture
def wide_optics_root(tmp_path, monkeypatch):
    # Cover the synthetic catalogue's 250-1000 nm, and keep generated optical
    # databases out of the user's database root.
    monkeypatch.setenv("SASKTRAN2_DATABASE_ROOT", str(tmp_path / "database"))
    root = tmp_path / "optics"
    root.mkdir()
    return _write_optics_tables(
        root,
        dict.fromkeys(build._OPTICS_FILES, np.array([200e-9, 550e-9, 1100e-9])),
    )


def _expected_mixture(catalogue):
    # Synthetic tables: every species has the same bext at a given RH, so the
    # mass weights reduce to the stored species shares.
    shares = catalogue.species_extinction_550nm_per_m.isel(
        scenario=0, month=0, altitude_m=1
    ).values
    shares = shares / shares.sum()
    tables = [build._COMPONENTS[name][0] for name in catalogue.species.values]
    ssa = np.array([_TABLE_SSA_G[table][0] for table in tables])
    g = np.array([_TABLE_SSA_G[table][1] for table in tables])
    mixed_ssa = np.sum(shares * ssa)
    mixed_a1 = np.array([1.0, np.sum(shares * ssa * 3 * g) / mixed_ssa, 0.2])
    return mixed_ssa, mixed_a1


def test_default_constituent_follows_catalogue_spectra_and_merra2_optics(
    catalogue, catalogue_path, wide_optics_root
):
    scenario, month = "tropical_typical", 4
    constituent = aerosol.constituent(
        scenario, month=month, path=catalogue_path, optics_root=wide_optics_root
    )
    case = catalogue.sel(scenario=scenario, month=month)
    reference = case.extinction_per_m.sel(wavelength_nm=550.0).values
    valid = reference > 0
    assert_allclose(constituent.extinction_per_m, reference)

    database = constituent._optical_property._database
    assert database.xs_total.dims == ("altitude_m", "wavelength_nm")
    for wavelength in case.wavelength_nm.values:
        assert_allclose(
            database.xs_total.sel(wavelength_nm=wavelength).values[valid]
            * reference[valid],
            case.extinction_per_m.sel(wavelength_nm=wavelength).values[valid],
            rtol=1e-12,
        )
    mixed_ssa, mixed_a1 = _expected_mixture(catalogue)
    ssa = (database.xs_scattering / database.xs_total).values
    assert (ssa <= 1).all()
    assert_allclose(ssa, mixed_ssa, rtol=1e-12)
    assert_allclose(
        database.lm_a1.isel(altitude_m=1, wavelength_nm=0).values,
        mixed_a1,
        rtol=1e-12,
    )
    for name in ("lm_a2", "lm_a3", "lm_a4", "lm_b1", "lm_b2"):
        assert not database[name].values.any()


def _engine_inputs(nstokes, constituent):
    altitudes_m = np.arange(0.0, 20001.0, 1000.0)
    config = sk.Config()
    config.num_stokes = nstokes
    config.num_streams = 4
    geometry = sk.Geometry1D(
        0.6,
        0.0,
        6372000.0,
        altitudes_m,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.PlaneParallel,
    )
    viewing_geo = sk.ViewingGeometry()
    viewing_geo.add_ray(sk.GroundViewingSolar(0.6, 0.0, 0.8, 200000.0))
    atmo = sk.Atmosphere(geometry, config, wavelengths_nm=np.array([400.0, 600.0]))
    sk.climatology.us76.add_us76_standard_atmosphere(atmo)
    atmo["rayleigh"] = sk.constituent.Rayleigh()
    atmo["tropospheric_aerosol"] = constituent
    return config, geometry, viewing_geo, atmo


def test_default_constituent_runs_scalar_and_rejects_polarized(
    catalogue_path, wide_optics_root
):
    constituent = aerosol.constituent(
        "nh_midlat_extreme",
        month=7,
        path=catalogue_path,
        optics_root=wide_optics_root,
    )
    config, geometry, viewing_geo, atmo = _engine_inputs(1, constituent)
    radiance = sk.Engine(config, geometry, viewing_geo).calculate_radiance(atmo)
    values = radiance["radiance"].to_numpy()
    assert np.isfinite(values).all()
    assert (values > 0).all()

    config, geometry, viewing_geo, atmo = _engine_inputs(3, constituent)
    with pytest.raises(ValueError, match="scalar"):
        sk.Engine(config, geometry, viewing_geo).calculate_radiance(atmo)

    # The synthetic catalogue covers 250-1000 nm.
    config, geometry, viewing_geo, atmo = _engine_inputs(1, constituent)
    atmo = sk.Atmosphere(geometry, config, wavelengths_nm=np.array([600.0, 1500.0]))
    sk.climatology.us76.add_us76_standard_atmosphere(atmo)
    atmo["tropospheric_aerosol"] = constituent
    with pytest.raises(ValueError, match="optics cover"):
        sk.Engine(config, geometry, viewing_geo).calculate_radiance(atmo)


def test_mixed_optics_warn_when_the_engine_truncates_the_phase_function():
    altitude = np.array([0.0, 20000.0])
    wavelength = np.array([300.0, 900.0])
    moments = np.zeros((2, 2, 32))
    moments[..., 0] = 1.0
    moments[..., 1] = 2.1
    database = xr.Dataset(
        {
            "xs_total": (("altitude_m", "wavelength_nm"), np.ones((2, 2))),
            "xs_scattering": (("altitude_m", "wavelength_nm"), np.full((2, 2), 0.9)),
            **{
                name: (("altitude_m", "wavelength_nm", "legendre"), values)
                for name, values in [("lm_a1", moments)]
                + [
                    (name, np.zeros_like(moments))
                    for name in ("lm_a2", "lm_a3", "lm_a4", "lm_b1", "lm_b2")
                ]
            },
        },
        coords={"altitude_m": altitude, "wavelength_nm": wavelength},
    )
    optics = aerosol._MERRA2MixedAerosol(db=database)
    config = sk.Config()  # 16 single-scatter moments by default
    geometry = sk.Geometry1D(
        0.6,
        0.0,
        6372000.0,
        np.arange(0.0, 20001.0, 1000.0),
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.PlaneParallel,
    )
    atmo = sk.Atmosphere(geometry, config, wavelengths_nm=np.array([550.0]))
    with pytest.warns(UserWarning, match="num_singlescatter_moments"):
        optics.atmosphere_quantities(atmo, altitude_m=geometry.altitudes())


def test_default_constituent_requires_optics_covering_the_catalogue(
    catalogue_path, optics_root, monkeypatch, tmp_path
):
    monkeypatch.setenv("SASKTRAN2_DATABASE_ROOT", str(tmp_path / "database"))
    # The default synthetic tables cover only 500-600 nm.
    with pytest.raises(ValueError, match="optics table"):
        aerosol.constituent(
            "tropical_low", month=1, path=catalogue_path, optics_root=optics_root
        )


# Per-table (single-scattering albedo, asymmetry parameter) of the synthetic tables
_TABLE_SSA_G = {
    "optics_BC.v1_3.nc": (0.3, 0.5),
    "optics_DU.v14_2.nc": (0.9, 0.7),
    "optics_OC.v1_3.nc": (0.95, 0.6),
    "optics_SS.v3_3.nc": (1.0, 0.75),
    "optics_SU.v1_3.nc": (1.0, 0.65),
}


def _write_optics_tables(root, wavelengths_by_file=None):
    radius_bins = {
        "optics_BC.v1_3.nc": 2,
        "optics_DU.v14_2.nc": 5,
        "optics_OC.v1_3.nc": 2,
        "optics_SS.v3_3.nc": 5,
        "optics_SU.v1_3.nc": 2,
    }
    humidity = np.array([0.0, 0.5, 0.99])
    for filename, radius_count in radius_bins.items():
        wavelengths = (wavelengths_by_file or {}).get(
            filename, np.array([500e-9, 600e-9])
        )
        values = np.empty((radius_count, len(humidity), len(wavelengths)))
        for ih, rh in enumerate(humidity):
            values[:, ih, :] = 1 + rh * 2
        ssa, g = _TABLE_SSA_G[filename]
        pmom = np.zeros((6, 3, *values.shape))
        pmom[0] = np.array([1.0, 3.0 * g, 0.2])[:, None, None, None]
        xr.Dataset(
            {
                "bext": (
                    ("radius", "rh", "lambda"),
                    values,
                    {"units": "m2 (kg dry mass)-1"},
                ),
                "bsca": (
                    ("radius", "rh", "lambda"),
                    ssa * values,
                    {"units": "m2 (kg dry mass)-1"},
                ),
                "pmom": (("nPol", "nMom", "radius", "rh", "lambda"), pmom),
            },
            coords={
                "radius": np.arange(radius_count),
                "rh": humidity,
                "lambda": ("lambda", wavelengths, {"units": "m"}),
            },
        ).to_netcdf(root / filename, engine="scipy")
    return root


@pytest.fixture
def optics_root(tmp_path):
    return _write_optics_tables(tmp_path)


def test_wavelength_grid_merges_float32_copies_of_the_same_wavelength(tmp_path):
    # The real tables store wavelength as float32 metres, so the same wavelength
    # differs slightly between tables (e.g. 550.0000119 vs 550.0 nm).
    wavelengths = np.array([400e-9, 550e-9, 700e-9])
    _write_optics_tables(
        tmp_path,
        {
            **dict.fromkeys(build._OPTICS_FILES, wavelengths),
            "optics_SU.v1_3.nc": wavelengths.astype(np.float32).astype(float),
        },
    )

    _, grid = build.load_efficiency_profiles(tmp_path)

    assert_array_equal(grid, [400.0, 550.0, 700.0])
    assert np.diff(grid).min() > 0.01


def test_merra_extinction_uses_mass_density_humidity_and_wavelength(
    optics_root,
):
    profiles, wavelengths = build.load_efficiency_profiles(optics_root)
    assert 550 in wavelengths
    ratios = {name: np.full(2, 1e-8) for name in build._COMPONENTS}
    aerosol_data = xr.Dataset(
        {
            "AIRDENS": ("lev", np.full(2, 1.2)),
            "RH": ("lev", np.full(2, 0.5)),
            **{name: ("lev", values) for name, values in ratios.items()},
        }
    )
    extinction = build.calculate_extinction(
        aerosol_data, profiles, wavelengths, wavelength_nm=550
    )
    expected = len(ratios) * 1e-8 * 1.2 * 2
    assert_allclose(extinction, np.full(2, expected))


def test_merra_height_conversion_and_geoid_interpolation():
    latitude = np.array([0.0, 45.0])
    geopotential_height = np.array([10000.0, 10000.0])
    latitude_rad = np.deg2rad(2 * latitude)
    equatorial_radius = 6_378_137.0
    polar_radius = equatorial_radius * (1 - 1 / 298.257223563)
    earth_radius = np.sqrt(
        1
        / (
            np.cos(np.deg2rad(latitude)) ** 2 / equatorial_radius**2
            + np.sin(np.deg2rad(latitude)) ** 2 / polar_radius**2
        )
    )
    gravity_ratio = (
        1 - 0.002637 * np.cos(latitude_rad) + 0.0000059 * np.cos(latitude_rad) ** 2
    )
    expected_height = (
        geopotential_height
        * earth_radius
        / (gravity_ratio * earth_radius - geopotential_height)
    )
    assert_allclose(
        build._geopotential_to_geometric_height(geopotential_height, latitude),
        expected_height,
    )

    geoid = xr.DataArray(
        [[0.0, 10.0], [20.0, 30.0]],
        coords={"lat": [0.0, 10.0], "lon": [0.0, 10.0]},
        dims=("lat", "lon"),
    )
    offsets = build._interpolate_geoid_undulation(
        geoid, np.array([5.0]), np.array([5.0])
    )
    assert_allclose(offsets, [[15.0]])


def test_geoid_grid_is_downloaded_and_cached(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache" / "EGM2008_MERRA2.nc"
    geoid = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        coords={"lat": [-90.0, 90.0], "lon": [0.0, 360.0]},
        dims=("lat", "lon"),
        name="geoid_undulation",
    )
    downloads = []

    def download(url, destination):
        downloads.append((url, Path(destination)))
        geoid.to_netcdf(destination, engine="scipy")

    monkeypatch.setattr(build, "_geoid_cache_path", lambda: cache_path)
    monkeypatch.setattr(build, "urlretrieve", download)

    first = build._load_geoid_undulation()
    second = build._load_geoid_undulation()

    assert len(downloads) == 1
    assert downloads[0][0] == build._EGM2008_DATABASE_URL
    assert cache_path.is_file()
    assert_allclose(first, geoid.sortby("lat").sortby("lon"))
    assert_allclose(second, first)


def test_geoid_cache_path_follows_sasktran2_database_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SASKTRAN2_DATABASE_ROOT", str(tmp_path / "db"))
    assert build._geoid_cache_path() == (tmp_path / "db" / build._EGM2008_DATABASE_KEY)

    monkeypatch.delenv("SASKTRAN2_DATABASE_ROOT")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert build._geoid_cache_path() == (
        tmp_path / "xdg" / "sasktran2" / "database" / build._EGM2008_DATABASE_KEY
    )

    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setattr(build.Path, "home", lambda: tmp_path / "home")
    assert build._geoid_cache_path() == (
        tmp_path
        / "home"
        / ".local"
        / "share"
        / "sasktran2"
        / "database"
        / build._EGM2008_DATABASE_KEY
    )


@pytest.mark.parametrize(
    ("day", "stream"),
    [
        ((1991, 12, 31), 100),
        ((1992, 1, 1), 200),
        ((2001, 1, 1), 300),
        ((2011, 1, 1), 400),
        ((2020, 8, 31), 400),
        ((2020, 9, 1), 401),
        ((2020, 9, 30), 401),
        ((2020, 10, 1), 400),
        ((2021, 5, 31), 400),
        ((2021, 6, 1), 401),
        ((2021, 9, 30), 401),
        ((2021, 10, 1), 400),
        ((2024, 7, 1), 400),
    ],
)
def test_collection_number_uses_reprocessed_401_stream(day, stream):
    assert build._collection_number(build.date(*day)) == stream


def test_month_days_cover_only_the_requested_month_in_each_year():
    days = build._month_days(1, 2023, 2024)
    assert len(days) == 62
    assert days[0] == build.date(2023, 1, 1)
    assert days[30] == build.date(2023, 1, 31)
    assert days[31] == build.date(2024, 1, 1)
    assert all(day.month == 1 for day in days)
    assert len(build._month_days(2, 2023, 2024)) == 28 + 29


def test_multi_year_month_skips_days_outside_the_month(tmp_path, monkeypatch):
    opened_days = []

    class _Stop(Exception):
        pass

    @contextmanager
    def record_day(day, *_args, **_kwargs):
        opened_days.append(day)
        if len(opened_days) > 31:
            raise _Stop
        yield (
            {},
            xr.Dataset(coords={"lat": [0.0], "time": np.array([], "M8[ns]")}),
            None,
            xr.Dataset(coords={"time": np.array([], "M8[ns]")}),
        )

    monkeypatch.setattr(build, "_open_daily_granules", record_day)
    monkeypatch.setattr(build, "_validate_inputs", lambda *_args: None)
    with pytest.raises(_Stop):
        build._process_month(
            1,
            2023,
            2024,
            tmp_path,
            xr.DataArray(
                np.zeros((2, 2)),
                coords={"lat": [-90.0, 90.0], "lon": [0.0, 360.0]},
                dims=("lat", "lon"),
            ),
            {},
            np.array([550.0]),
            1000,
            42,
        )
    assert opened_days[30] == build.date(2023, 1, 31)
    assert opened_days[31] == build.date(2024, 1, 1)


def test_smoke_test_reads_and_processes_one_column(optics_root, monkeypatch):
    timestamps = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T03:00:00"],
        dtype="datetime64[s]",
    )
    coords = {
        "time": timestamps,
        "lev": [1, 2, 3],
        "lat": [0.5],
        "lon": [180.0],
    }
    aerosol = xr.Dataset(
        {
            "AIRDENS": (
                ("time", "lev", "lat", "lon"),
                np.ones((2, 3, 1, 1)),
                {"units": "kg m-3"},
            ),
            "DELP": (
                ("time", "lev", "lat", "lon"),
                np.full((2, 3, 1, 1), 10000.0),
                {"units": "Pa"},
            ),
            "RH": (
                ("time", "lev", "lat", "lon"),
                np.full((2, 3, 1, 1), 0.5),
                {"units": "1"},
            ),
            **{
                species: (
                    ("time", "lev", "lat", "lon"),
                    np.full((2, 3, 1, 1), 1e-8),
                    {"units": "kg kg-1"},
                )
                for species in build._COMPONENTS
            },
        },
        coords=coords,
    )
    meteorology = xr.Dataset(
        {
            "H": (
                ("time", "lev", "lat", "lon"),
                np.broadcast_to(
                    np.array([1000.0, 5000.0, 15000.0]).reshape(1, 3, 1, 1),
                    (2, 3, 1, 1),
                ),
                {"units": "m"},
            ),
            "PL": (
                ("time", "lev", "lat", "lon"),
                np.broadcast_to(
                    np.array([90000.0, 50000.0, 10000.0]).reshape(1, 3, 1, 1),
                    (2, 3, 1, 1),
                ),
                {"units": "Pa"},
            ),
            "T": (
                ("time", "lev", "lat", "lon"),
                np.full((2, 3, 1, 1), 280.0),
                {"units": "K"},
            ),
            "PHIS": (
                ("time", "lat", "lon"),
                np.zeros((2, 1, 1)),
                {"units": "m+2 s-2"},
            ),
        },
        coords=coords,
    )
    tropopause = xr.Dataset(
        {
            "TROPPB": (
                ("time", "lat", "lon"),
                np.full((4, 1, 1), 60000.0),
                {"units": "Pa"},
            )
        },
        coords={
            **coords,
            "time": np.array(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-01T01:00:00",
                    "2024-01-01T02:00:00",
                    "2024-01-01T03:00:00",
                ],
                dtype="datetime64[s]",
            ),
            "lev": [1, 2, 3],
        },
    )
    datasets = {
        "aerosol": aerosol,
        "meteorology": meteorology,
        "tropopause": tropopause,
    }

    @contextmanager
    def open_test_dataset(location):
        yield datasets[location]

    monkeypatch.setattr(build, "_file_for_product", lambda product, *_: product)
    monkeypatch.setattr(build, "_open_granule", open_test_dataset)
    result = build.run_smoke_test(
        build.date(2024, 1, 1), optics_root, data_root=Path("unused")
    )
    assert result["status"] == "ok"
    assert len(result["timestamps_processed"]) == 2
    assert result["model_levels_per_column"] == 3
    assert result["tropospheric_levels_per_timestamp"] == [1, 1]
    assert len(result["tropospheric_aod_550nm_per_timestamp"]) == 2
    assert result["tropospheric_aod_550nm_min"] > 0

    efficiencies, wavelengths_nm = build.load_efficiency_profiles(optics_root)
    column = build._column_spectrum(
        aerosol,
        meteorology,
        tropopause,
        0,
        0,
        0,
        xr.DataArray(
            np.zeros((2, 2)),
            coords={"lat": [-90.0, 90.0], "lon": [0.0, 360.0]},
            dims=("lat", "lon"),
        ),
        efficiencies,
        wavelengths_nm,
    )
    iw550 = int(np.argmin(np.abs(wavelengths_nm - 550.0)))
    grid = build._TARGET_ALTITUDES_M
    assert_allclose(
        np.trapezoid(column.extinction_per_m[:, iw550], grid),
        result["tropospheric_aod_550nm_per_timestamp"][0],
        rtol=1e-6,
    )
    species = column.species_extinction_550nm_per_m
    assert species.shape == (len(grid), len(build._COMPONENTS))
    assert (species >= 0).all()
    assert_allclose(
        species.sum(axis=1), column.extinction_per_m[:, iw550], rtol=1e-12, atol=0
    )
    # The fake granules hold RH = 0.5 everywhere.
    assert_allclose(column.relative_humidity, 0.5)


@pytest.mark.parametrize("stage_daily", [True, False])
def test_selected_columns_open_granules_like_the_daily_pass(monkeypatch, stage_daily):
    opened = []

    @contextmanager
    def open_daily_granules(day, data_root, **kwargs):
        opened.append((day, data_root, kwargs))
        yield {}, "aerosol", "meteorology", "tropopause"

    monkeypatch.setattr(build, "_open_daily_granules", open_daily_granules)
    monkeypatch.setattr(
        build,
        "_column_spectrum",
        lambda _aer, _met, _trop, time_index, lat_index, lon_index, *_args: (
            time_index,
            lat_index,
            lon_index,
        ),
    )
    stager = object()

    columns = build._read_column_spectra(
        build.date(2024, 4, 10),
        [(1, 2, 3), (4, 5, 6)],
        None,
        None,
        {},
        np.array([550.0]),
        stage_daily=stage_daily,
        stager=stager,
    )

    assert columns == [(1, 2, 3), (4, 5, 6)]
    assert opened == [
        (
            build.date(2024, 4, 10),
            None,
            {"stage_daily": stage_daily, "checkpointing": False, "stager": stager},
        )
    ]


def test_input_access_description_matches_how_inputs_were_read(tmp_path):
    assert "range streaming" in build._input_access_description(None, False)
    assert "full daily granules" in build._input_access_description(None, True)
    assert "local" in build._input_access_description(tmp_path, False)
    assert "local" in build._input_access_description(tmp_path, True)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], True), (["--stage-daily"], True), (["--no-stage-daily"], False)],
)
def test_command_line_stages_daily_granules_by_default(
    monkeypatch, tmp_path, flags, expected
):
    calls = []
    monkeypatch.setattr(
        build, "build_catalogue", lambda *args, **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        "sys.argv",
        ["build.py", "--optics-root", str(tmp_path), "--output", str(tmp_path), *flags],
    )
    build.main()
    assert calls[0]["stage_daily"] is expected


@pytest.mark.parametrize("surface", [10.0, 545.0, -10.0])
def test_conservative_grid_profile_preserves_tropospheric_aod(surface):
    # MERRA2 levels run top-down; the helper must not depend on that order.
    offsets = np.array([60.0, 250.0, 700.0, 1500.0, 3000.0, 6000.0, 9500.0, 11500.0])
    heights = (surface + np.append(offsets, 14000.0))[::-1]
    thickness = np.array(
        [120.0, 260.0, 640.0, 960.0, 2040.0, 3960.0, 3040.0, 960.0, 4000.0]
    )[::-1]
    tropospheric = heights < surface + 11800.0
    tropopause_altitude = surface + 11800.0
    rng = np.random.default_rng(3)
    extinction = rng.uniform(1e-6, 1e-4, size=(len(heights), 2))

    profile = build._conservative_grid_profile(
        extinction, heights, thickness, tropospheric, surface, tropopause_altitude
    )

    grid = build._TARGET_ALTITUDES_M
    assert profile.shape == (len(grid), 2)
    assert (profile >= 0).all()
    assert not profile[(grid < surface) | (grid > tropopause_altitude)].any()
    expected = np.sum(
        np.where(tropospheric[:, None], extinction * thickness[:, None], 0.0), axis=0
    )
    assert_allclose(np.trapezoid(profile, grid, axis=0), expected, rtol=1e-12)


def test_remote_granule_retries_transient_failures_and_refreshes_expired_url(
    tmp_path,
):
    pytest.importorskip("aiohttp")
    pytest.importorskip("h5netcdf")
    netcdf_path = tmp_path / "remote.nc"
    xr.Dataset({"sample": ("x", np.arange(4, dtype=np.float32))}).to_netcdf(
        netcdf_path, engine="h5netcdf"
    )
    content = netcdf_path.read_bytes()
    requests = {
        "head": 0,
        "info_get": 0,
        "range_get": 0,
        "signed_head": 0,
        "signed_get": 0,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            if self.path.startswith("/signed"):
                requests["signed_head"] += 1
            else:
                requests["head"] += 1
                if requests["head"] == 1:
                    self.send_error(503)
                    return
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{self.server.server_port}/signed"
                )
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", '"stable-test-etag"')
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/signed"):
                requests["signed_get"] += 1
            else:
                if "Range" not in self.headers:
                    requests["info_get"] += 1
                    self.send_error(503)
                    return
                requests["range_get"] += 1
                if requests["range_get"] == 1:
                    self.send_error(403)
                    return
            start, end = map(
                int, self.headers["Range"].removeprefix("bytes=").split("-")
            )
            end = min(end, len(content) - 1)
            data = content[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source_url = f"http://127.0.0.1:{server.server_port}/remote.nc"
    captured_identity = []
    try:
        with build._open_granule(source_url, captured_identity.append) as dataset:
            result = dataset["sample"].load().values
        assert_allclose(result, np.arange(4, dtype=np.float32))
        assert captured_identity == [
            {
                "kind": "remote",
                "url": source_url,
                "etag": '"stable-test-etag"',
                "last_modified": None,
                "size": len(content),
            }
        ]
        assert requests == {
            "head": 3,
            "info_get": 1,
            "range_get": 1,
            "signed_head": 2,
            "signed_get": 1,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_source_identities_use_stable_local_and_remote_metadata(tmp_path):
    local_path = tmp_path / "granule.nc4"
    local_path.write_bytes(b"first")
    first_identity = build._local_source_identity(local_path)
    local_path.write_bytes(b"other")
    second_identity = build._local_source_identity(local_path)
    assert first_identity["sha256"] != second_identity["sha256"]

    source_url = "https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/file.nc4"
    remote_identity = build._remote_source_identity(
        source_url,
        {
            "url": "https://temporary-cloudfront.example/file.nc4?signature=secret",
            "ETag": '"stable-object-etag"',
            "size": 1234,
        },
    )
    assert remote_identity == {
        "kind": "remote",
        "url": source_url,
        "etag": '"stable-object-etag"',
        "last_modified": None,
        "size": 1234,
    }
    last_modified_identity = build._remote_source_identity(
        source_url,
        {"Last-Modified": "Mon, 02 Nov 2026 10:00:00 GMT", "size": 1234},
    )
    assert last_modified_identity["last_modified"] == "Mon, 02 Nov 2026 10:00:00 GMT"
    with pytest.raises(ValueError, match="Last-Modified header with known size"):
        build._remote_source_identity(
            source_url, {"Last-Modified": "Mon, 02 Nov 2026 10:00:00 GMT"}
        )
    with pytest.raises(ValueError, match="Last-Modified header with known size"):
        build._remote_source_identity(source_url, {"size": 1234})


def test_daily_staging_downloads_remote_granules_and_cleans_temporary_files(
    tmp_path, monkeypatch
):
    netcdf_path = tmp_path / "source.nc4"
    xr.Dataset({"sample": ("x", np.array([3.0], dtype=np.float32))}).to_netcdf(
        netcdf_path, engine="h5netcdf"
    )
    content = netcdf_path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("ETag", '"stable-test-etag"')
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_port}"
    locations = {
        product: f"{root_url}/{product}.nc4" for product in build._PRODUCTS
    }
    monkeypatch.setattr(build, "_source_locations", lambda *_args: locations)
    monkeypatch.setattr(
        build,
        "_source_identity",
        lambda location: build._remote_source_identity(
            location, {"ETag": '"stable-test-etag"', "size": len(content)}
        ),
    )
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    try:
        with build._open_daily_granules(
            build.date(2024, 1, 1),
            None,
            stage_daily=True,
            checkpointing=True,
        ) as (identities, aerosol, meteorology, tropopause):
            assert identities == {
                product: build._source_identity(location)
                for product, location in locations.items()
            }
            assert aerosol["sample"].load().item() == 3.0
            assert meteorology["sample"].load().item() == 3.0
            assert tropopause["sample"].load().item() == 3.0
            assert len(list(tmp_path.glob("sasktran2-merra2-20240101-*"))) == 1
        assert not list(tmp_path.glob("sasktran2-merra2-20240101-*"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_area_weighted_reservoir_keeps_representative_profiles():
    nlon = 1000
    longitude = np.arange(nlon, dtype=float)
    latitude = np.array([45.0])
    levels = 2
    extinction = np.zeros((levels, 1, nlon))
    extinction[0, 0] = np.linspace(1e-7, 1e-5, nlon)
    height = np.broadcast_to(
        np.array([1000.0, 3000.0])[:, None, None], extinction.shape
    )
    pressure = np.broadcast_to(
        np.array([90000.0, 30000.0])[:, None, None], extinction.shape
    )
    layer = np.broadcast_to(np.array([1000.0, 1000.0])[:, None, None], extinction.shape)
    tp_pressure = np.full((1, nlon), 60000.0)
    surface = np.zeros((1, nlon))
    reservoir = build._WeightedReservoir(1000, np.random.default_rng(7))
    reservoir.add(
        extinction,
        height,
        pressure,
        layer,
        tp_pressure,
        surface,
        latitude,
        longitude,
        np.datetime64("2026-01-01T00:00"),
        0,
        np.array([160]),
    )
    assert reservoir.aod.size == 1000
    assert reservoir.valid_candidates == 1000
    assert_allclose(
        np.trapezoid(reservoir.profiles, build._TARGET_ALTITUDES_M, axis=1),
        reservoir.aod,
        rtol=1e-12,
    )
    index, percentile, count, lower, upper = reservoir.select_tier(0.5, 0.03)
    assert 47 <= percentile <= 53
    assert count > 0
    assert lower <= reservoir.aod[index] <= upper


def test_month_checkpoint_restores_reservoir_and_skips_completed_days(
    tmp_path, monkeypatch
):
    month = 1
    config = {
        "algorithm_version": build._CHECKPOINT_ALGORITHM_VERSION,
        "start_year": 2024,
        "end_year": 2024,
        "seed": 42,
    }
    assert build._checkpoint_id(config) != build._checkpoint_id(
        {**config, "algorithm_version": config["algorithm_version"] + 1}
    )
    checkpoint_path = tmp_path / "checkpoints" / "month_01.npz"
    source_identities = {
        build.date(2024, 1, day).isoformat(): {
            product: {
                "kind": "remote",
                "url": str(build._source_location(product, build.date(2024, 1, day), None)),
                "etag": f'"etag-{product}-{day}"',
                "size": 1000,
            }
            for product in build._PRODUCTS
        }
        for day in range(1, 32)
    }
    changed_source = False

    def source_identity(location):
        source = str(location)
        if changed_source and source.endswith("20240102.nc4"):
            return {"kind": "remote", "url": source, "etag": '"changed"', "size": 1000}
        for daily_identities in source_identities.values():
            for identity in daily_identities.values():
                if identity["url"] == source:
                    return identity
        msg = f"Unexpected source location: {location}"
        raise AssertionError(msg)

    monkeypatch.setattr(build, "_source_identity", source_identity)
    rng = np.random.default_rng(43)
    reservoirs = {
        band: build._WeightedReservoir(1000, rng) for band in build._BANDS
    }
    for reservoir in reservoirs.values():
        reservoir.keys = np.linspace(-1.0, -0.01, 1000)
        reservoir.aod = np.linspace(0.01, 1.0, 1000)
        reservoir.profiles = np.ones((1000, len(build._TARGET_ALTITUDES_M)))
        reservoir.latitude = np.full(1000, 0.0)
        reservoir.longitude = np.full(1000, 180.0)
        reservoir.time = np.full(1000, np.datetime64("2024-01-01"))
        reservoir.time_index = np.zeros(1000, dtype=np.int16)
        reservoir.latitude_index = np.zeros(1000, dtype=np.int16)
        reservoir.longitude_index = np.zeros(1000, dtype=np.int16)
        reservoir.tropopause_altitude = np.full(1000, 12000.0)
        reservoir.surface_altitude = np.zeros(1000)
        reservoir.valid_candidates = 1000

    build._save_month_checkpoint(
        checkpoint_path,
        month,
        config,
        build.date(2024, 1, 31),
        rng,
        reservoirs,
        source_identities,
    )
    expected_random = rng.random(3)

    with pytest.raises(ValueError, match="algorithm version do not match"):
        build._load_month_checkpoint(
            checkpoint_path,
            month,
            {**config, "algorithm_version": config["algorithm_version"] + 1},
            2024,
            2024,
            None,
            np.random.default_rng(0),
            {},
        )

    changed_source = True
    with pytest.raises(ValueError, match="source identity changed"):
        build._load_month_checkpoint(
            checkpoint_path,
            month,
            config,
            2024,
            2024,
            None,
            np.random.default_rng(0),
            {},
        )
    changed_source = False

    resumed_rng = np.random.default_rng(0)
    resumed_reservoirs = {
        band: build._WeightedReservoir(1000, resumed_rng) for band in build._BANDS
    }
    last_completed_day, restored_source_identities = build._load_month_checkpoint(
        checkpoint_path,
        month,
        config,
        2024,
        2024,
        None,
        resumed_rng,
        resumed_reservoirs,
    )
    assert last_completed_day == build.date(2024, 1, 31)
    assert restored_source_identities == source_identities
    assert_allclose(resumed_rng.random(3), expected_random)
    for band in build._BANDS:
        assert_allclose(resumed_reservoirs[band].aod, reservoirs[band].aod)
        assert_array_equal(resumed_reservoirs[band].time, reservoirs[band].time)

    def unexpected_source_read(*_args, **_kwargs):
        pytest.fail("A completed month checkpoint should not reread MERRA2 data")

    monkeypatch.setattr(build, "_open_granule", unexpected_source_read)
    column_reads = []

    def read_column_spectra(day, columns, *_args, stage_daily, stager):
        column_reads.append((day, len(columns), stage_daily, stager))
        return [_fake_column(np.array([550.0])) for _ in columns]

    monkeypatch.setattr(build, "_read_column_spectra", read_column_spectra)
    records = build._process_month(
        month,
        2024,
        2024,
        None,
        xr.DataArray(
            np.zeros((2, 2)),
            coords={"lat": [-90.0, 90.0], "lon": [0.0, 360.0]},
            dims=("lat", "lon"),
        ),
        {},
        np.array([550.0]),
        1000,
        42,
        checkpoint_path,
        config,
        stage_daily=True,
    )
    assert len(records) == len(build._BANDS) * len(build._TIERS)
    # Every selection is on 2024-01-01, so that day is read once for all of them.
    assert len(column_reads) == 1
    day, column_count, stage_daily, stager = column_reads[0]
    assert day == build.date(2024, 1, 1)
    assert column_count == len(records)
    assert stage_daily is True
    assert isinstance(stager, build._DayStager)


def test_multi_year_checkpoint_validates_only_days_in_the_month(tmp_path, monkeypatch):
    month = 1
    config = {
        "algorithm_version": build._CHECKPOINT_ALGORITHM_VERSION,
        "start_year": 2023,
        "end_year": 2024,
        "seed": 42,
    }
    completed_days = [
        *(build.date(2023, 1, day) for day in range(1, 32)),
        *(build.date(2024, 1, day) for day in range(1, 3)),
    ]
    source_identities = {
        day.isoformat(): {
            product: {"kind": "remote", "url": str(product), "size": 1}
            for product in build._PRODUCTS
        }
        for day in completed_days
    }
    checked_days = []
    monkeypatch.setattr(
        build,
        "_source_locations",
        lambda day, _data_root: (
            checked_days.append(day)
            or {product: (day, product) for product in build._PRODUCTS}
        ),
    )
    monkeypatch.setattr(
        build,
        "_source_identity",
        lambda location: source_identities[location[0].isoformat()][location[1]],
    )
    path = tmp_path / "month_01.npz"

    def save(identities):
        build._save_month_checkpoint(
            path,
            month,
            config,
            completed_days[-1],
            np.random.default_rng(1),
            {},
            identities,
        )

    def load():
        return build._load_month_checkpoint(
            path, month, config, 2023, 2024, None, np.random.default_rng(0), {}
        )

    save(source_identities)
    last_completed_day, _ = load()
    assert last_completed_day == build.date(2024, 1, 2)
    assert checked_days == completed_days

    # A checkpoint written by the old loop also holds days outside the month.
    save({**source_identities, "2023-02-01": source_identities["2023-01-01"]})
    with pytest.raises(ValueError, match="source days do not match"):
        load()

    missing_day = dict(source_identities)
    del missing_day["2023-01-15"]
    save(missing_day)
    with pytest.raises(ValueError, match="source days do not match"):
        load()


def test_builder_rejects_incomplete_local_source(tmp_path, caplog):
    geoid_undulation = xr.DataArray(
        np.zeros((2, 2)),
        coords={"lat": [-90.0, 90.0], "lon": [0.0, 360.0]},
        dims=("lat", "lon"),
    )
    with (
        caplog.at_level("INFO"),
        pytest.raises(FileNotFoundError, match="Local MERRA2 inputs are incomplete"),
    ):
        build._process_month(
            1,
            2026,
            2026,
            tmp_path,
            geoid_undulation,
            {},
            np.array([550.0]),
            1000,
            42,
        )
    assert "Starting MERRA2 month 01" in caplog.text
    assert "Starting MERRA2 month 01 day 2026-01-01" in caplog.text


def test_builder_requires_a_positive_worker_count(tmp_path):
    with pytest.raises(ValueError, match="workers must be at least 1"):
        build.build_catalogue(tmp_path, tmp_path / "output", workers=0)


def _fake_column(wavelengths_nm, species_scale=1.0):
    grid = build._TARGET_ALTITUDES_M
    extinction_550 = np.where(grid <= 10000.0, 1e-5, 0.0)
    shares = np.arange(1, len(build._COMPONENTS) + 1, dtype=float)
    shares /= shares.sum()
    return build._ColumnProfiles(
        extinction_per_m=np.repeat(
            extinction_550[:, None], len(wavelengths_nm), axis=1
        ),
        species_extinction_550nm_per_m=species_scale
        * extinction_550[:, None]
        * shares[None, :],
        relative_humidity=np.full(len(grid), 0.4),
    )


def _fake_records(month, wavelengths_nm, species_scale=1.0):
    return [
        {
            "scenario": f"{band}_{tier}",
            "month": month,
            **_fake_column(wavelengths_nm, species_scale)._asdict(),
            "selection_aod_550nm": 0.1,
            "selection_percentile": 50.0,
            "candidate_count": 10,
            "candidate_window_aod_min": 0.09,
            "candidate_window_aod_max": 0.11,
            "selected_latitude_degrees": 0.0,
            "selected_longitude_degrees": 0.0,
            "selected_time": f"2024-{month:02d}-01T00:00:00",
            "tropopause_altitude_m": 10000.0,
            "surface_altitude_m": 0.0,
            "sampled_candidate_count": 1000,
            "valid_candidate_count": 10000,
        }
        for band in build._BANDS
        for tier in build._TIERS
    ]


def _patch_catalogue_inputs(monkeypatch, tmp_path, records_for_month):
    geoid_path = tmp_path / "geoid.nc"
    geoid_path.write_bytes(b"geoid")
    monkeypatch.setattr(build, "_geoid_cache_path", lambda: geoid_path)
    monkeypatch.setattr(
        build,
        "_load_geoid_undulation",
        lambda: xr.DataArray(
            np.zeros((2, 2)),
            coords={"lat": [-90.0, 90.0], "lon": [0.0, 360.0]},
            dims=("lat", "lon"),
        ),
    )
    monkeypatch.setattr(
        build, "_process_month", lambda month, *_args: records_for_month(month)
    )


def test_build_catalogue_writes_species_extinction_and_humidity(
    optics_root, tmp_path, monkeypatch
):
    _, wavelengths = build.load_efficiency_profiles(optics_root)
    _patch_catalogue_inputs(
        monkeypatch, tmp_path, lambda month: _fake_records(month, wavelengths)
    )

    path = build.build_catalogue(
        optics_root,
        tmp_path / "output",
        data_root=tmp_path,
        start_year=2024,
        end_year=2024,
    )

    with xr.open_dataset(path) as catalogue:
        assert catalogue.attrs["schema_version"] == 2
        assert catalogue.species.values.tolist() == list(build._COMPONENTS)
        species = catalogue.species_extinction_550nm_per_m
        assert species.dims == ("scenario", "month", "species", "altitude_m")
        assert species.attrs["units"] == "m-1"
        assert_allclose(
            species.sum("species"),
            catalogue.extinction_per_m.sel(wavelength_nm=550.0),
            rtol=1e-6,
        )
        humidity = catalogue.relative_humidity
        assert humidity.dims == ("scenario", "month", "altitude_m")
        assert_allclose(humidity, 0.4, rtol=1e-6)
    report = json.loads((tmp_path / "output" / "build_report_v1.json").read_text())
    for scenario in report["scenarios"]:
        assert not {
            "extinction_per_m",
            "species_extinction_550nm_per_m",
            "relative_humidity",
        } & set(scenario)


def test_build_catalogue_rejects_species_that_do_not_sum_to_total(
    optics_root, tmp_path, monkeypatch
):
    _, wavelengths = build.load_efficiency_profiles(optics_root)
    _patch_catalogue_inputs(
        monkeypatch,
        tmp_path,
        lambda month: _fake_records(month, wavelengths, species_scale=1.01),
    )

    with pytest.raises(ValueError, match="species extinction"):
        build.build_catalogue(
            optics_root,
            tmp_path / "output",
            data_root=tmp_path,
            start_year=2024,
            end_year=2024,
        )
    assert not (tmp_path / "output" / "tropospheric_aerosol_v1.nc").exists()


def test_failed_month_is_reported_when_it_fails_and_others_finish(
    optics_root, tmp_path, monkeypatch, caplog
):
    _, wavelengths = build.load_efficiency_profiles(optics_root)
    finished = []

    def records_for_month(month):
        if month == 4:
            msg = "staging failed"
            raise OSError(msg)
        finished.append(month)
        return _fake_records(month, wavelengths)

    _patch_catalogue_inputs(monkeypatch, tmp_path, records_for_month)
    monkeypatch.setattr(build, "_month_executor", ThreadPoolExecutor)
    caplog.set_level(logging.INFO)

    with pytest.raises(RuntimeError, match=r"month\(s\) 04 failed") as error:
        build.build_catalogue(
            optics_root,
            tmp_path / "output",
            data_root=tmp_path,
            start_year=2024,
            end_year=2024,
            workers=2,
        )

    assert isinstance(error.value.__cause__, OSError)
    assert sorted(finished) == [1, 7, 10]
    assert "MERRA2 month 04 failed (OSError: staging failed)" in caplog.text
    assert "MERRA2 month 07 complete" in caplog.text
    assert not (tmp_path / "output" / "tropospheric_aerosol_v1.nc").exists()


class _FakeDownload:
    def __init__(self, status, body=b"granule"):
        self.status_code = status
        self.body = body
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            msg = f"HTTP {self.status_code}"
            raise build.requests.HTTPError(msg, response=self)

    def iter_content(self, chunk_size):  # noqa: ARG002
        yield self.body


@pytest.mark.parametrize("status", [401, 403, 503])
def test_download_retries_authentication_and_server_errors(
    tmp_path, monkeypatch, status
):
    responses = [_FakeDownload(status), _FakeDownload(200)]
    monkeypatch.setattr(build.requests, "get", lambda *_a, **_k: responses.pop(0))
    monkeypatch.setattr(build.time, "sleep", lambda _seconds: None)
    destination = tmp_path / "granule.nc4"

    build._download_granule("https://example.test/granule.nc4", destination)

    assert destination.read_bytes() == b"granule"
    assert not responses


def test_download_does_not_retry_a_missing_granule(tmp_path, monkeypatch):
    calls = []

    def get(*_args, **_kwargs):
        calls.append(1)
        return _FakeDownload(404)

    monkeypatch.setattr(build.requests, "get", get)
    monkeypatch.setattr(build.time, "sleep", lambda _seconds: None)

    with pytest.raises(build.requests.HTTPError):
        build._download_granule(
            "https://example.test/granule.nc4", tmp_path / "granule.nc4"
        )
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())


def test_staging_failure_is_logged_when_it_happens(tmp_path, monkeypatch, caplog):
    def fail(_location, _destination):
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(build, "_download_granule", fail)
    caplog.set_level(logging.INFO)

    with pytest.raises(OSError, match="disk full"):
        build._stage_day(build.date(2024, 4, 10), tmp_path, checkpointing=False)

    failure = next(
        record for record in caplog.records if "Failed to stage" in record.message
    )
    assert "2024-04-10" in failure.message
    assert failure.exc_info is not None


def test_month_workers_run_in_separate_processes_with_parent_logging():
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        with build._month_executor(2) as executor:
            child_pid = executor.submit(os.getpid).result()
            child_level = executor.submit(root.getEffectiveLevel).result()
    finally:
        root.setLevel(previous_level)
    assert child_pid != os.getpid()
    assert child_level == logging.INFO


def _fake_stage_day(started, fail_on=None):
    def stage(day, directory, _checkpointing):
        started[day] = True
        if day == fail_on:
            msg = f"download failed for {day}"
            raise OSError(msg)
        path = directory / f"{day:%Y%m%d}.nc4"
        path.write_bytes(b"granule")
        return {}, dict.fromkeys(build._PRODUCTS, path)

    return stage


def test_day_stager_downloads_the_next_day_while_the_current_day_is_open(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    started = {}
    monkeypatch.setattr(build, "_stage_day", _fake_stage_day(started))
    days = [build.date(2024, 1, day) for day in (1, 2, 3)]
    stager = build._DayStager(days, checkpointing=False)
    try:
        for index, day in enumerate(days):
            with stager.staged(day) as (_identities, paths):
                assert paths["aerosol"].read_bytes() == b"granule"
                if index + 1 < len(days):
                    # The next day's download starts before this day is done.
                    next_day = days[index + 1]
                    for _ in range(100):
                        if next_day in started:
                            break
                        threading.Event().wait(0.05)
                    assert next_day in started
            assert not paths["aerosol"].exists()
    finally:
        stager.close()
    assert list(started) == days
    assert not list(tmp_path.glob("sasktran2-merra2-*"))


def test_day_stager_cleans_up_after_a_failed_download(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    days = [build.date(2024, 1, day) for day in (1, 2, 3)]
    monkeypatch.setattr(build, "_stage_day", _fake_stage_day({}, fail_on=days[1]))
    stager = build._DayStager(days, checkpointing=False)
    try:
        with stager.staged(days[0]):
            pass
        with pytest.raises(OSError, match="download failed"), stager.staged(days[1]):
            pass
    finally:
        stager.close()
    assert not list(tmp_path.glob("sasktran2-merra2-*"))
