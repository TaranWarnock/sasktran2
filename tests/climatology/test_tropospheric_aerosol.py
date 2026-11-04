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
    monkeypatch.setattr(
        build,
        "_read_column_spectrum",
        lambda *_args, **_kwargs: np.ones(
            (len(build._TARGET_ALTITUDES_M), 1)
        ),
    )
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
    )
    assert len(records) == len(build._BANDS) * len(build._TIERS)


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
    rng = np.random.default_rng(1)
    build._save_month_checkpoint(
        tmp_path / "month_01.npz",
        month,
        config,
        completed_days[-1],
        rng,
        {},
        source_identities,
    )
    last_completed_day, _ = build._load_month_checkpoint(
        tmp_path / "month_01.npz",
        month,
        config,
        2023,
        2024,
        None,
        np.random.default_rng(0),
        {},
    )
    assert last_completed_day == build.date(2024, 1, 2)
    assert checked_days == completed_days

    # A checkpoint written by the old loop also holds days outside the month.
    stale_identities = {
        **source_identities,
        "2023-02-01": source_identities["2023-01-01"],
    }
    build._save_month_checkpoint(
        tmp_path / "month_01.npz",
        month,
        config,
        completed_days[-1],
        rng,
        {},
        stale_identities,
    )
    with pytest.raises(ValueError, match="days outside the requested month"):
        build._load_month_checkpoint(
            tmp_path / "month_01.npz",
            month,
            config,
            2023,
            2024,
            None,
            np.random.default_rng(0),
            {},
        )


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
