from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
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
    return xr.Dataset(
        {
            "extinction_per_m": (
                ("scenario", "month", "altitude_m", "wavelength_nm"),
                ext,
                {"units": "m-1"},
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
            "altitude_m": ("altitude_m", altitude, {"units": "m"}),
            "wavelength_nm": ("wavelength_nm", wavelengths, {"units": "nm"}),
        },
        attrs={"catalogue_version": "v1", "schema_version": 1},
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
    ("change", "message"), [("schema", "schema/version"), ("unit", "units")]
)
def test_invalid_catalogue(catalogue, tmp_path, change, message):
    if change == "schema":
        catalogue.attrs["schema_version"] = 2
    else:
        catalogue.extinction_per_m.attrs["units"] = "km-1"
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
def optics_root(tmp_path):
    radius_bins = {
        "optics_BC.v1_3.nc": 2,
        "optics_DU.v14_2.nc": 5,
        "optics_OC.v1_3.nc": 2,
        "optics_SS.v3_3.nc": 5,
        "optics_SU.v1_3.nc": 2,
    }
    wavelengths = np.array([500e-9, 600e-9])
    humidity = np.array([0.0, 0.5, 0.99])
    for filename, radius_count in radius_bins.items():
        values = np.empty((radius_count, len(humidity), len(wavelengths)))
        for ih, rh in enumerate(humidity):
            values[:, ih, :] = 1 + rh * 2
        xr.Dataset(
            {
                "bext": (
                    ("radius", "rh", "lambda"),
                    values,
                    {"units": "m2 (kg dry mass)-1"},
                )
            },
            coords={
                "radius": np.arange(radius_count),
                "rh": humidity,
                "lambda": ("lambda", wavelengths, {"units": "m"}),
            },
        ).to_netcdf(tmp_path / filename, engine="scipy")
    return tmp_path


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


def test_remote_granule_retries_transient_http_failures(tmp_path):
    pytest.importorskip("aiohttp")
    pytest.importorskip("h5netcdf")
    netcdf_path = tmp_path / "remote.nc"
    xr.Dataset({"sample": ("x", np.arange(4, dtype=np.float32))}).to_netcdf(
        netcdf_path, engine="h5netcdf"
    )
    content = netcdf_path.read_bytes()
    requests = {"head": 0, "get": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            requests["head"] += 1
            if requests["head"] == 1:
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            requests["get"] += 1
            if requests["get"] == 1:
                self.send_error(503)
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
    try:
        with build._open_granule(
            f"http://127.0.0.1:{server.server_port}/remote.nc"
        ) as dataset:
            result = dataset["sample"].load().values
        assert_allclose(result, np.arange(4, dtype=np.float32))
        assert requests == {"head": 2, "get": 2}
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
    index, percentile, count, lower, upper = reservoir.select_tier(0.5, 0.03)
    assert 47 <= percentile <= 53
    assert count > 0
    assert lower <= reservoir.aod[index] <= upper


def test_builder_rejects_incomplete_local_source(tmp_path):
    geoid_undulation = xr.DataArray(
        np.zeros((2, 2)),
        coords={"lat": [-90.0, 90.0], "lon": [0.0, 360.0]},
        dims=("lat", "lon"),
    )
    with pytest.raises(FileNotFoundError, match="Local MERRA2 inputs are incomplete"):
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


def test_builder_requires_a_positive_worker_count(tmp_path):
    with pytest.raises(ValueError, match="workers must be at least 1"):
        build.build_catalogue(tmp_path, tmp_path / "output", workers=0)
