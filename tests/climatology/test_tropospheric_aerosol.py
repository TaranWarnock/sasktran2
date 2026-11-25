from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import os
import sys
import threading
import warnings
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
from sasktran2.database import tropospheric_aerosol as database
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


# Aerosol type used by the library tests that need just one.
_TYPE = "DU"
_CASE_DIMS = ("scenario", "aerosol_type", "month")


def _type_species_shares(species):
    """Each type's species shares of its AOD, (aerosol_type, species)."""
    shares = np.zeros((len(build._AEROSOL_TYPES), len(species)))
    for index, members in enumerate(build._AEROSOL_TYPES.values()):
        positions = [species.index(name) for name in members]
        shares[index, positions] = np.arange(1, len(positions) + 1)
        shares[index] /= shares[index].sum()
    return shares


@pytest.fixture()
def catalogue():
    scenarios = [
        f"{band}_{loading}"
        for band in ("sh_midlat", "tropical", "nh_midlat")
        for loading in ("low", "typical", "elevated", "extreme")
    ]
    aerosol_types = list(build._AEROSOL_TYPES)
    months = [1, 4, 7, 10]
    altitude = np.arange(0.0, 20001.0, 1000.0)
    wavelengths = np.array([250.0, 550.0, 1000.0])
    shape = (len(scenarios), len(aerosol_types), len(months))
    i, t, j = np.meshgrid(*(np.arange(size) for size in shape), indexing="ij")
    scale = (i + 1) * (t + 1) * (j + 1) * 1e-6
    # The whole column, as the builder stores it: zero below the surface, and
    # aerosol above the tropopause too. The tropopause (12500 m) lies on a
    # catalogue cell boundary, so cutting there keeps whole cells.
    ext = (
        scale[..., None, None]
        * np.exp(-altitude / 6000)[:, None]
        * (550 / wavelengths)[None, :]
    )
    surface = np.full(shape, 1000.0)
    tropopause = np.full(shape, 12500.0)
    ext[..., altitude < 1000.0, :] = 0
    species = list(build._COMPONENTS)
    shares = _type_species_shares(species)
    species_ext = shares[None, :, None, :, None] * ext[:, :, :, None, :, 1]
    return xr.Dataset(
        {
            "extinction_per_m": (
                (*_CASE_DIMS, "altitude_m", "wavelength_nm"),
                ext,
                {"units": "m-1"},
            ),
            "species_extinction_550nm_per_m": (
                (*_CASE_DIMS, "species", "altitude_m"),
                species_ext,
                {"units": "m-1"},
            ),
            "relative_humidity": (
                (*_CASE_DIMS, "altitude_m"),
                np.full((*shape, len(altitude)), 0.5),
                {"units": "1"},
            ),
            "latitude_band": (
                ("scenario",),
                [name.rsplit("_", 1)[0] for name in scenarios],
            ),
            "loading": (("scenario",), [name.rsplit("_", 1)[1] for name in scenarios]),
            "selection_aod_550nm": (_CASE_DIMS, np.full(shape, 0.1)),
            "selection_percentile": (_CASE_DIMS, np.full(shape, 50.0)),
            "candidate_count": (_CASE_DIMS, np.full(shape, 1000)),
            "column_total_aod_550nm": (_CASE_DIMS, np.full(shape, 0.2)),
            "selected_latitude_degrees": (_CASE_DIMS, np.zeros(shape)),
            "selected_longitude_degrees": (_CASE_DIMS, np.zeros(shape)),
            "selected_time": (
                _CASE_DIMS,
                np.full(shape, np.datetime64("2025-01-01")),
            ),
            "tropopause_altitude_m": (_CASE_DIMS, tropopause, {"units": "m"}),
            "surface_altitude_m": (_CASE_DIMS, surface, {"units": "m"}),
        },
        coords={
            "scenario": scenarios,
            "aerosol_type": aerosol_types,
            "month": months,
            "species": species,
            "altitude_m": ("altitude_m", altitude, {"units": "m"}),
            "wavelength_nm": ("wavelength_nm", wavelengths, {"units": "nm"}),
        },
        attrs={"catalogue_version": "v1", "schema_version": 3},
    )


@pytest.fixture()
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
    assert listing.sizes == {"scenario": 12, "aerosol_type": 5, "month": 4}
    assert aerosol.scenarios(month=4, path=catalogue_path).sizes == {
        "scenario": 12,
        "aerosol_type": 5,
    }
    dust = aerosol.scenarios(month=4, aerosol_type="dust", path=catalogue_path)
    assert dust.sizes == {"scenario": 12}
    assert dust.aerosol_type.item() == "DU"
    assert dust.latitude_band.sel(scenario="tropical_typical").item() == "tropical"

    result = aerosol.profile(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        wavelength_nm=550,
        altitudes_m=np.arange(0.0, 21001.0, 500.0),
        path=catalogue_path,
    )
    assert result.sizes["altitude_m"] == 43
    assert float(result.extinction_per_m.sel(altitude_m=0)) == 0
    assert float(result.extinction_per_m.sel(altitude_m=1500)) > 0
    assert float(result.extinction_per_m.sel(altitude_m=13000)) == 0
    assert_allclose(result.tropospheric_aod, _catalogue_aod(catalogue))
    assert_allclose(result.aod, result.tropospheric_aod)
    assert result.attrs["aerosol_type"] == "DU"
    assert result.attrs["cut_at_tropopause"] == 1
    assert result.attrs["source_tropopause_field"] == "TROPPB"
    assert float(result.attrs["extinction_wavelength_nm"]) == 550


@pytest.fixture()
def offgrid_catalogue(catalogue):
    # Like the builder's output: the surface falls between catalogue levels,
    # levels below it are zero, and the column continues above the
    # tropopause. The tropopause (11500 m) lies on a cell boundary.
    catalogue = catalogue.copy(deep=True)
    catalogue["surface_altitude_m"][:] = 1300.0
    catalogue["tropopause_altitude_m"][:] = 11500.0
    below = catalogue.altitude_m < 1300.0
    for name in ("extinction_per_m", "species_extinction_550nm_per_m"):
        attrs = catalogue[name].attrs
        catalogue[name] = catalogue[name].where(~below, 0.0).assign_attrs(attrs)
    return catalogue


@pytest.fixture()
def offgrid_catalogue_path(offgrid_catalogue, tmp_path):
    path = tmp_path / "offgrid.nc"
    offgrid_catalogue.to_netcdf(path)
    return path


def _case(catalogue, scenario="tropical_typical", month=4):
    return catalogue.sel(scenario=scenario, aerosol_type=_TYPE, month=month)


def _cut_extinction(case, wavelength_nm=550.0):
    """Catalogue extinction with the levels above the tropopause zeroed.

    The fixtures' tropopause lies on a cell boundary, so this is exactly the
    profile cut at the tropopause on the catalogue grid.
    """
    extinction = case.extinction_per_m.sel(wavelength_nm=wavelength_nm)
    return extinction.where(case.altitude_m <= case.tropopause_altitude_m, 0.0)


def _catalogue_aod(catalogue, wavelength_nm=550.0):
    """The catalogue case's tropospheric AOD."""
    source = _case(catalogue)
    return np.trapezoid(_cut_extinction(source, wavelength_nm), source.altitude_m)


@pytest.mark.parametrize("spacing", [10.0, 37.0, 250.0, 700.0, 2500.0])
def test_profile_conserves_catalogue_aod_on_any_grid(
    offgrid_catalogue, offgrid_catalogue_path, spacing
):
    altitudes = np.arange(-500.0, 25000.0, spacing)
    result = aerosol.profile(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        altitudes_m=altitudes,
        path=offgrid_catalogue_path,
    )
    extinction = result.extinction_per_m.values
    assert_allclose(
        np.trapezoid(extinction, altitudes),
        _catalogue_aod(offgrid_catalogue),
        rtol=1e-12,
    )
    assert (extinction >= 0).all()
    assert not extinction[(altitudes < 1300.0) | (altitudes > 11500.0)].any()


def test_profile_reproduces_catalogue_on_catalogue_grid(
    offgrid_catalogue, offgrid_catalogue_path
):
    source = _case(offgrid_catalogue)
    for wavelength in source.wavelength_nm.values:
        for altitudes in (None, source.altitude_m.values):
            result = aerosol.profile(
                "tropical_typical",
                aerosol_type=_TYPE,
                month=4,
                wavelength_nm=wavelength,
                altitudes_m=altitudes,
                path=offgrid_catalogue_path,
            )
            assert_allclose(
                result.extinction_per_m.values,
                _cut_extinction(source, wavelength).values,
                rtol=1e-12,
                atol=1e-25,
            )
            uncut = aerosol.profile(
                "tropical_typical",
                aerosol_type=_TYPE,
                month=4,
                wavelength_nm=wavelength,
                altitudes_m=altitudes,
                cut_at_tropopause=False,
                path=offgrid_catalogue_path,
            )
            assert_allclose(
                uncut.extinction_per_m.values,
                source.extinction_per_m.sel(wavelength_nm=wavelength).values,
                rtol=1e-12,
                atol=1e-25,
            )


def test_tropospheric_aod_integrates_the_returned_profile(
    offgrid_catalogue, offgrid_catalogue_path
):
    for top in (6000.0, 20000.0):
        altitudes = np.arange(0.0, top + 1.0, 10.0)
        result = aerosol.profile(
            "tropical_typical",
            aerosol_type=_TYPE,
            month=4,
            altitudes_m=altitudes,
            path=offgrid_catalogue_path,
        )
        assert_allclose(
            float(result.tropospheric_aod),
            np.trapezoid(result.extinction_per_m.values, altitudes),
            rtol=1e-12,
        )
    # The 0-6 km grid misses the upper troposphere; the full grid does not.
    assert float(result.tropospheric_aod) == pytest.approx(
        _catalogue_aod(offgrid_catalogue), rel=1e-12
    )


def test_profile_needs_a_tropospheric_level(offgrid_catalogue_path):
    with pytest.raises(ValueError, match="no level between the surface"):
        aerosol.profile(
            "tropical_typical",
            aerosol_type=_TYPE,
            month=4,
            altitudes_m=[0.0, 1000.0, 12000.0],
            path=offgrid_catalogue_path,
        )


def test_region_flags_distinguish_below_surface_and_above_tropopause(
    offgrid_catalogue_path,
):
    result = aerosol.profile(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        altitudes_m=[0.0, 1299.0, 1300.0, 11500.0, 11501.0, 20000.0],
        path=offgrid_catalogue_path,
    )
    assert_array_equal(result.region.values, [-1, -1, 1, 1, 0, 0])
    assert result.region.attrs["flag_values"] == [-1, 0, 1]
    assert (
        result.region.attrs["flag_meanings"]
        == "below_surface above_tropopause troposphere"
    )


def test_smoothing_keeps_aod_and_stays_in_the_troposphere(
    offgrid_catalogue, offgrid_catalogue_path
):
    source = _case(offgrid_catalogue)
    altitudes = source.altitude_m.values
    for wavelength in source.wavelength_nm.values:
        raw = aerosol.profile(
            "tropical_typical",
            aerosol_type=_TYPE,
            month=4,
            wavelength_nm=wavelength,
            path=offgrid_catalogue_path,
        )
        smooth = aerosol.profile(
            "tropical_typical",
            aerosol_type=_TYPE,
            month=4,
            wavelength_nm=wavelength,
            smoothing_fwhm_m=3000.0,
            path=offgrid_catalogue_path,
        )
        extinction = smooth.extinction_per_m.values
        assert_allclose(
            np.trapezoid(extinction, altitudes),
            np.trapezoid(raw.extinction_per_m.values, altitudes),
            rtol=1e-12,
        )
        assert (extinction >= 0).all()
        assert not extinction[(altitudes < 1300.0) | (altitudes > 11500.0)].any()
        assert not np.allclose(extinction, raw.extinction_per_m.values)
        assert smooth.attrs["smoothing_fwhm_m"] == 3000.0
    # Smoothing evens out the profile: the exponential decay flattens.
    inside = (altitudes > 1300.0) & (altitudes < 11500.0)
    assert np.ptp(extinction[inside]) < np.ptp(raw.extinction_per_m.values[inside])


def test_smoothing_keeps_species_summing_to_the_total(offgrid_catalogue):
    case = _case(offgrid_catalogue)
    smooth = aerosol._smooth(case, 3000.0, 11500.0)
    assert_allclose(
        smooth.species_extinction_550nm_per_m.sum("species"),
        smooth.extinction_per_m.sel(wavelength_nm=550.0),
        rtol=1e-12,
        atol=1e-25,
    )
    assert smooth.species_extinction_550nm_per_m.dims == (
        case.species_extinction_550nm_per_m.dims
    )
    assert smooth.extinction_per_m.attrs["units"] == "m-1"


@pytest.mark.parametrize("fwhm", [250.0, 1000.0, 5000.0, 50000.0])
def test_smoothing_leaves_a_constant_profile_unchanged_at_the_boundaries(fwhm):
    altitudes = np.arange(0.0, 20001.0, 250.0)
    valid = (altitudes >= 1250.0) & (altitudes <= 11750.0)
    matrix = aerosol._smoothing_matrix(altitudes, valid, fwhm)
    constant = np.where(valid, 3.0, 0.0)
    assert_allclose(matrix @ constant, constant, rtol=1e-12, atol=1e-15)
    assert_allclose(matrix.sum(axis=0)[valid], 1.0, rtol=1e-12)
    assert not matrix[~valid].any()
    assert not matrix[:, ~valid].any()


def test_smoothing_an_uncut_profile_keeps_the_coarse_grid_unchanged(
    catalogue, tmp_path
):
    # A catalogue grid like the builder's: regular up to a point, coarser above.
    altitude = np.concatenate((np.arange(0.0, 15000.0, 500.0), [17000.0, 20000.0]))
    regridded = catalogue.interp(altitude_m=altitude)
    regridded.altitude_m.attrs["units"] = "m"
    for name in ("extinction_per_m", "species_extinction_550nm_per_m"):
        regridded[name].attrs["units"] = "m-1"
    regridded.relative_humidity.attrs["units"] = "1"
    path = tmp_path / "irregular.nc"
    regridded.to_netcdf(path)
    options = {
        "aerosol_type": _TYPE,
        "month": 4,
        "cut_at_tropopause": False,
        "path": path,
    }
    raw = aerosol.profile("tropical_typical", **options)
    smooth = aerosol.profile("tropical_typical", smoothing_fwhm_m=3000.0, **options)
    assert_allclose(float(smooth.aod), float(raw.aod), rtol=1e-12)
    coarse = smooth.altitude_m.values > 15000.0
    assert_array_equal(
        smooth.extinction_per_m.values[coarse], raw.extinction_per_m.values[coarse]
    )
    assert not np.allclose(smooth.extinction_per_m.values, raw.extinction_per_m.values)


def test_zero_smoothing_returns_the_catalogue(offgrid_catalogue_path):
    raw = aerosol.profile(
        "tropical_typical", aerosol_type=_TYPE, month=4, path=offgrid_catalogue_path
    )
    same = aerosol.profile(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        smoothing_fwhm_m=0,
        path=offgrid_catalogue_path,
    )
    assert_array_equal(same.extinction_per_m.values, raw.extinction_per_m.values)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"smoothing_fwhm_m": -1.0}, "smoothing_fwhm_m"),
        ({"smoothing_fwhm_m": np.nan}, "smoothing_fwhm_m"),
        ({"ground_altitude_m": np.inf}, "ground_altitude_m"),
    ],
)
def test_invalid_smoothing_or_ground(offgrid_catalogue_path, options, message):
    with pytest.raises(ValueError, match=message):
        aerosol.profile(
            "tropical_typical",
            aerosol_type=_TYPE,
            month=4,
            path=offgrid_catalogue_path,
            **options,
        )


def test_ground_altitude_moves_the_profile_to_the_model_ground(
    offgrid_catalogue, offgrid_catalogue_path
):
    raw = aerosol.profile(
        "tropical_typical", aerosol_type=_TYPE, month=4, path=offgrid_catalogue_path
    )
    moved = aerosol.profile(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        ground_altitude_m=0.0,
        path=offgrid_catalogue_path,
    )
    # The catalogue surface is at 1300 m; everything moves down by that much.
    assert_allclose(moved.altitude_m, raw.altitude_m - 1300.0)
    assert_array_equal(moved.extinction_per_m.values, raw.extinction_per_m.values)
    assert float(moved.surface_altitude_m) == 0.0
    assert float(moved.tropopause_altitude_m) == 10200.0
    assert moved.attrs["vertical_shift_m"] == -1300.0
    assert moved.attrs["ground_altitude_m"] == 0.0
    assert raw.attrs["ground_altitude_m"] == "catalogue"

    altitudes = np.arange(0.0, 20001.0, 1000.0)
    on_grid = aerosol.profile(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        altitudes_m=altitudes,
        ground_altitude_m=0.0,
        path=offgrid_catalogue_path,
    )
    assert float(on_grid.extinction_per_m.sel(altitude_m=0.0)) > 0
    assert_allclose(
        float(on_grid.tropospheric_aod), _catalogue_aod(offgrid_catalogue), rtol=1e-12
    )


def _atmosphere(altitudes_m, wavelengths_nm=(400.0, 600.0)):
    config = sk.Config()
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
    atmo = sk.Atmosphere(geometry, config, wavelengths_nm=np.array(wavelengths_nm))
    sk.climatology.us76.add_us76_standard_atmosphere(atmo)
    atmo["rayleigh"] = sk.constituent.Rayleigh()
    return config, geometry, viewing_geo, atmo


def _radiance(constituent, altitudes_m):
    config, geometry, viewing_geo, atmo = _atmosphere(altitudes_m)
    atmo["tropospheric_aerosol"] = constituent
    engine = sk.Engine(config, geometry, viewing_geo)
    return engine.calculate_radiance(atmo)["radiance"].to_numpy()


def test_constituent_follows_the_atmosphere_grid_and_keeps_the_aod(
    offgrid_catalogue, offgrid_catalogue_path, wide_optics_root
):
    constituent = aerosol.constituent(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        path=offgrid_catalogue_path,
        optics_root=wide_optics_root,
    )
    assert isinstance(constituent, aerosol.TroposphericAerosolConstituent)
    assert_array_equal(constituent.altitudes_m, offgrid_catalogue.altitude_m.values)
    for spacing in (1000.0, 2500.0):
        altitudes = np.arange(0.0, 30001.0, spacing)
        assert np.isfinite(_radiance(constituent, altitudes)).all()
        assert_array_equal(constituent.altitudes_m, altitudes)
        assert_allclose(
            np.trapezoid(constituent.extinction_per_m, altitudes),
            _catalogue_aod(offgrid_catalogue),
            rtol=1e-12,
        )


def test_following_the_grid_matches_an_explicit_grid(
    offgrid_catalogue_path, wide_optics_root
):
    altitudes = np.arange(0.0, 30001.0, 1000.0)
    options = {
        "month": 4,
        "path": offgrid_catalogue_path,
        "optics_root": wide_optics_root,
        "smoothing_fwhm_m": 1000.0,
        "ground_altitude_m": 0.0,
    }
    following = _radiance(
        aerosol.constituent("tropical_typical", aerosol_type=_TYPE, **options),
        altitudes,
    )
    explicit = _radiance(
        aerosol.constituent(
            "tropical_typical", aerosol_type=_TYPE, altitudes_m=altitudes, **options
        ),
        altitudes,
    )
    assert np.isfinite(following).all()
    assert_allclose(following, explicit, rtol=1e-12)
    config, geometry, viewing_geo, atmo = _atmosphere(altitudes)
    rayleigh_only = (
        sk.Engine(config, geometry, viewing_geo)
        .calculate_radiance(atmo)["radiance"]
        .to_numpy()
    )
    assert not np.allclose(following, rayleigh_only)


def test_explicit_grid_is_kept_when_added_to_another_atmosphere(
    offgrid_catalogue_path, wide_optics_root
):
    altitudes = np.arange(0.0, 20001.0, 500.0)
    constituent = aerosol.constituent(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        altitudes_m=altitudes,
        path=offgrid_catalogue_path,
        optics_root=wide_optics_root,
    )
    assert np.isfinite(_radiance(constituent, np.arange(0.0, 30001.0, 1000.0))).all()
    assert_array_equal(constituent.altitudes_m, altitudes)


def test_edited_extinction_is_not_silently_remapped(
    offgrid_catalogue_path, wide_optics_root
):
    constituent = aerosol.constituent(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        path=offgrid_catalogue_path,
        optics_root=wide_optics_root,
    )
    constituent.extinction_per_m = 2 * constituent.extinction_per_m
    with pytest.raises(ValueError, match="altitudes_m="):
        _radiance(constituent, np.arange(0.0, 30001.0, 1000.0))


def _count_hashes(monkeypatch):
    calls = []
    original = database._sha256

    def counting_sha256(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(database, "_sha256", counting_sha256)
    monkeypatch.setattr(aerosol, "_sha256", counting_sha256)
    return calls


def test_load_dataset_caches_each_file_until_it_changes(
    catalogue, catalogue_path, monkeypatch
):
    calls = _count_hashes(monkeypatch)
    first = aerosol.load_dataset(path=catalogue_path)
    first["extinction_per_m"][:] = -1.0
    second = aerosol.load_dataset(path=catalogue_path)
    assert len(calls) == 1
    xr.testing.assert_identical(second, aerosol.load_dataset(path=catalogue_path))
    assert (second.extinction_per_m.values >= 0).all()

    changed = catalogue.copy(deep=True)
    changed["extinction_per_m"] = 2 * changed.extinction_per_m
    changed.attrs["note"] = "rebuilt"
    changed.to_netcdf(catalogue_path)
    status = catalogue_path.stat()
    os.utime(catalogue_path, ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))
    reloaded = aerosol.load_dataset(path=catalogue_path)
    assert len(calls) == 2
    assert reloaded.attrs["note"] == "rebuilt"
    assert_allclose(reloaded.extinction_per_m, 2 * second.extinction_per_m)
    assert (
        reloaded.attrs["source_sha256"]
        == hashlib.sha256(catalogue_path.read_bytes()).hexdigest()
    )


def test_verified_catalogue_is_hashed_once_until_it_changes(
    catalogue, tmp_path, monkeypatch
):
    cache = tmp_path / "climatology" / "tropospheric_aerosol" / "v1"
    cache.mkdir(parents=True)
    cached_file = cache / "tropospheric_aerosol_v1.nc"
    catalogue.to_netcdf(cached_file)
    checksum = hashlib.sha256(cached_file.read_bytes()).hexdigest()
    monkeypatch.setattr(database, "_FILES", {"v1": (cached_file.name, checksum)})
    calls = _count_hashes(monkeypatch)

    data = aerosol.load_dataset(db_root=tmp_path)
    assert data.attrs["source_mode"] == "verified_cache"
    assert len(calls) == 2  # verification and the recorded source checksum
    aerosol.scenarios(db_root=tmp_path)
    aerosol.profile("tropical_typical", aerosol_type=_TYPE, month=4, db_root=tmp_path)
    assert len(calls) == 2

    with cached_file.open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(OSError, match="checksum mismatch"):
        aerosol.load_dataset(db_root=tmp_path)


@pytest.mark.parametrize(
    ("scenario", "aerosol_type", "month", "wavelength", "message"),
    [
        ("tropical", "DU", 1, 550, "exact aerosol scenario"),
        ("tropical_typical", "smoke", 1, 550, "aerosol_type must be one of"),
        ("tropical_typical", None, 1, 550, "aerosol_type must be one of"),
        ("tropical_typical", "DU", 2, 550, "month must be"),
        ("tropical_typical", "DU", 1.0, 550, "month must be"),
        ("tropical_typical", "DU", 1, 10000, "within"),
    ],
)
def test_invalid_profile_selection(
    catalogue_path, scenario, aerosol_type, month, wavelength, message
):
    with pytest.raises(ValueError, match=message):
        aerosol.profile(
            scenario,
            aerosol_type=aerosol_type,
            month=month,
            wavelength_nm=wavelength,
            path=catalogue_path,
        )


@pytest.mark.parametrize("aerosol_type", ["DU", "du", "dust", "Dust", " DUST "])
def test_aerosol_type_accepts_codes_and_names(catalogue_path, aerosol_type):
    result = aerosol.profile(
        "tropical_typical", aerosol_type=aerosol_type, month=4, path=catalogue_path
    )
    assert result.attrs["aerosol_type"] == "DU"


def test_every_aerosol_type_name_selects_its_own_case(catalogue, catalogue_path):
    for name, code in sk.database.MERRA2_AEROSOL_TYPE_NAMES.items():
        by_name = aerosol.profile(
            "nh_midlat_low", aerosol_type=name, month=7, path=catalogue_path
        )
        by_code = aerosol.profile(
            "nh_midlat_low", aerosol_type=code, month=7, path=catalogue_path
        )
        assert_array_equal(by_name.extinction_per_m, by_code.extinction_per_m)
        source = catalogue.sel(scenario="nh_midlat_low", aerosol_type=code, month=7)
        assert_allclose(by_code.extinction_per_m, _cut_extinction(source))


def test_uncut_profile_keeps_aerosol_above_the_tropopause(catalogue, catalogue_path):
    altitudes = np.arange(0.0, 20001.0, 250.0)
    cut = aerosol.profile(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        altitudes_m=altitudes,
        path=catalogue_path,
    )
    uncut = aerosol.profile(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        altitudes_m=altitudes,
        cut_at_tropopause=False,
        path=catalogue_path,
    )
    above = altitudes > 12500.0
    assert not cut.extinction_per_m.values[above].any()
    assert (uncut.extinction_per_m.values[above] > 0).all()
    source = _case(catalogue)
    total = np.trapezoid(
        source.extinction_per_m.sel(wavelength_nm=550.0), source.altitude_m
    )
    assert_allclose(float(uncut.aod), total, rtol=1e-12)
    assert_allclose(float(uncut.tropospheric_aod), _catalogue_aod(catalogue))
    assert_allclose(float(cut.aod), float(cut.tropospheric_aod))
    assert_allclose(float(cut.aod), _catalogue_aod(catalogue), rtol=1e-12)
    assert uncut.attrs["cut_at_tropopause"] == 0
    assert "top of the catalogue grid" in uncut.attrs["profile_extension"]
    with pytest.raises(ValueError, match="cut_at_tropopause"):
        aerosol.profile(
            "tropical_typical",
            aerosol_type=_TYPE,
            month=4,
            cut_at_tropopause="no",
            path=catalogue_path,
        )


def test_cut_within_a_cell_keeps_only_the_part_below_the_tropopause(
    catalogue, tmp_path
):
    # Move the tropopause inside a catalogue cell (11500-12500 m): the cut
    # profile keeps part of that cell, so its AOD lies between that of the
    # whole cells below and that with the straddling cell included.
    catalogue = catalogue.copy(deep=True)
    catalogue["tropopause_altitude_m"][:] = 12100.0
    path = tmp_path / "straddle.nc"
    catalogue.to_netcdf(path)
    result = aerosol.profile(
        "tropical_typical",
        aerosol_type=_TYPE,
        month=4,
        altitudes_m=np.arange(0.0, 20001.0, 10.0),
        path=path,
    )
    source = _case(catalogue).extinction_per_m.sel(wavelength_nm=550.0)
    widths = np.diff(aerosol._cell_bounds(source.altitude_m.values))
    cells = source.values * widths
    below = cells[source.altitude_m.values <= 11000.0].sum()
    straddling = cells[source.altitude_m.values == 12000.0].item()
    aod = float(result.tropospheric_aod)
    assert below < aod < below + straddling
    assert aod == pytest.approx(below + 0.6 * straddling, rel=0.05)
    assert not result.extinction_per_m.values[result.altitude_m.values > 12100.0].any()


def test_constituents_of_two_types_add_up(catalogue_path, wide_optics_root):
    altitudes = np.arange(0.0, 30001.0, 1000.0)
    config, geometry, viewing_geo, atmo = _atmosphere(altitudes)
    for name in ("dust", "sulfate"):
        atmo[name] = aerosol.constituent(
            "tropical_typical",
            aerosol_type=name,
            month=4,
            path=catalogue_path,
            optics_root=wide_optics_root,
        )
    radiance = (
        sk.Engine(config, geometry, viewing_geo)
        .calculate_radiance(atmo)["radiance"]
        .to_numpy()
    )
    assert np.isfinite(radiance).all()
    dust, sulfate = (
        aerosol.profile(
            "tropical_typical",
            aerosol_type=name,
            month=4,
            altitudes_m=altitudes,
            path=catalogue_path,
        ).extinction_per_m.values
        for name in ("dust", "sulfate")
    )
    # Sulfate is the first type and dust the second in the synthetic catalogue.
    assert_allclose(dust, 2 * sulfate, rtol=1e-12)
    assert_allclose(atmo["dust"].extinction_per_m, dust)
    assert_allclose(atmo["sulfate"].extinction_per_m, sulfate)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("schema", "schema/version"),
        ("unit", "units"),
        ("species", "species_extinction_550nm_per_m"),
        ("humidity", "relative_humidity"),
        ("schema2", "schema/version"),
        ("types", "aerosol types"),
        ("outside", "outside each case's aerosol type"),
    ],
)
def test_invalid_catalogue(catalogue, tmp_path, change, message):
    if change == "schema":
        catalogue.attrs["schema_version"] = 1
    elif change == "schema2":
        catalogue.attrs["schema_version"] = 2
    elif change == "types":
        catalogue = catalogue.isel(aerosol_type=slice(0, 4))
    elif change == "outside":
        catalogue["species_extinction_550nm_per_m"][:] = 1e-6
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
    result = aerosol.profile(
        "sh_midlat_low", aerosol_type=_TYPE, month=1, path=catalogue_path
    )
    constituent = aerosol.constituent(
        "sh_midlat_low",
        aerosol_type=_TYPE,
        month=1,
        optical_property=_ConstantOptics(),
        path=catalogue_path,
    )
    assert_allclose(constituent.extinction_per_m, result.extinction_per_m)


@pytest.fixture()
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
    shares = (
        catalogue.species_extinction_550nm_per_m.sel(aerosol_type=_TYPE)
        .isel(scenario=0, month=0, altitude_m=1)
        .values
    )
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
        scenario,
        aerosol_type=_TYPE,
        month=month,
        path=catalogue_path,
        optics_root=wide_optics_root,
    )
    case = _case(catalogue, scenario, month)
    reference = case.extinction_per_m.sel(wavelength_nm=550.0).values
    valid = reference > 0
    assert_allclose(constituent.extinction_per_m, _cut_extinction(case).values)

    database = constituent._optical_property._database
    assert database.xs_total.dims == ("altitude_m", "wavelength_nm")
    # The optics cover the levels the cut profile uses (surface 1000 m to the
    # tropopause at 12500 m), plus one level either side.
    assert database.altitude_m.values.tolist() == list(range(0, 14000, 1000))
    tabulated = np.isin(case.altitude_m.values, database.altitude_m.values)
    valid = valid[tabulated]
    reference = reference[tabulated]
    for wavelength in case.wavelength_nm.values:
        assert_allclose(
            database.xs_total.sel(wavelength_nm=wavelength).values[valid]
            * reference[valid],
            case.extinction_per_m.sel(wavelength_nm=wavelength).values[tabulated][
                valid
            ],
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
        aerosol_type=_TYPE,
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


def _mixed_optics_with_32_moments():
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
    return aerosol._MERRA2MixedAerosol(db=database)


def _recommended_config(moments):
    config = sk.Config()
    config.num_singlescatter_moments = moments
    config.delta_m_scaling = True
    config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
    config.num_streams = 4
    return config


def _mixed_optics_quantities(optics, config):
    geometry = sk.Geometry1D(
        0.6,
        0.0,
        6372000.0,
        np.arange(0.0, 20001.0, 1000.0),
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.PlaneParallel,
    )
    atmo = sk.Atmosphere(geometry, config, wavelengths_nm=np.array([550.0]))
    return optics.atmosphere_quantities(atmo, altitude_m=geometry.altitudes())


def test_mixed_optics_do_not_warn_with_the_recommended_engine_settings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _mixed_optics_quantities(
            _mixed_optics_with_32_moments(), _recommended_config(32)
        )


def _truncated(config):
    config.num_singlescatter_moments = 16


def _without_delta_m(config):
    config.delta_m_scaling = False


def _discrete_ordinates_single_scatter(config):
    config.single_scatter_source = sk.SingleScatterSource.DiscreteOrdinates


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_truncated, "keeps 16 of the aerosol's 32"),
        (_without_delta_m, "delta_m_scaling is off"),
        (_discrete_ordinates_single_scatter, "discrete-ordinates single-scatter"),
    ],
)
def test_mixed_optics_warn_about_engine_settings_for_forward_peaked_aerosol(
    change, message
):
    config = _recommended_config(32)
    change(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _mixed_optics_quantities(_mixed_optics_with_32_moments(), config)
    messages = [str(item.message) for item in caught]
    assert len(messages) == 1, messages
    assert message in messages[0]


def test_mixed_optics_without_multiple_scatter_need_no_delta_m():
    config = sk.Config()  # no multiple-scatter source, delta-M off
    config.num_singlescatter_moments = 32
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _mixed_optics_quantities(_mixed_optics_with_32_moments(), config)


def test_default_constituent_keeps_every_table_moment(
    catalogue_path, wide_optics_root, monkeypatch
):
    # The synthetic tables store only three P11 moments, too few for a default
    # cap to show, so check what the optics tables are asked for.
    requested = []
    database = aerosol.OpticalDatabaseMERRA2Aerosol

    def recording_database(*args, **kwargs):
        requested.append(kwargs["max_legendre_moments"])
        return database(*args, **kwargs)

    monkeypatch.setattr(aerosol, "OpticalDatabaseMERRA2Aerosol", recording_database)
    default = aerosol.constituent(
        "tropical_low",
        aerosol_type=_TYPE,
        month=1,
        path=catalogue_path,
        optics_root=wide_optics_root,
    )
    assert requested
    assert set(requested) == {None}
    truncated = aerosol.constituent(
        "tropical_low",
        aerosol_type=_TYPE,
        month=1,
        path=catalogue_path,
        optics_root=wide_optics_root,
        max_legendre_moments=2,
    )
    assert default._optical_property._database.sizes["legendre"] == 3
    assert truncated._optical_property._database.sizes["legendre"] == 2


def test_default_constituent_requires_optics_covering_the_catalogue(
    catalogue_path, optics_root, monkeypatch, tmp_path
):
    monkeypatch.setenv("SASKTRAN2_DATABASE_ROOT", str(tmp_path / "database"))
    # The default synthetic tables cover only 500-600 nm.
    with pytest.raises(ValueError, match="optics table"):
        aerosol.constituent(
            "tropical_low",
            aerosol_type=_TYPE,
            month=1,
            path=catalogue_path,
            optics_root=optics_root,
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


@pytest.fixture()
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


def test_aerosol_types_cover_every_component_once():
    members = [name for species in build._AEROSOL_TYPES.values() for name in species]
    assert sorted(members) == sorted(build._COMPONENTS)
    assert set(build._AEROSOL_TYPE_NAMES) == set(build._AEROSOL_TYPES)


def test_type_extinction_splits_the_total_by_aerosol_type(optics_root):
    profiles, wavelengths = build.load_efficiency_profiles(optics_root)
    ratios = {name: np.full(2, 1e-8) for name in build._COMPONENTS}
    ratios["SO4"] = np.full(2, 3e-8)
    aerosol_data = xr.Dataset(
        {
            "AIRDENS": ("lev", np.full(2, 1.2)),
            "RH": ("lev", np.full(2, 0.5)),
            **{name: ("lev", values) for name, values in ratios.items()},
        }
    )
    by_type = build.calculate_type_extinction(aerosol_data, profiles, wavelengths)
    total = build.calculate_extinction(aerosol_data, profiles, wavelengths)
    assert by_type.shape == (len(build._AEROSOL_TYPES), 2)
    assert_allclose(by_type.sum(axis=0), total, rtol=1e-12)
    # Each row is the extinction of that type's species alone, in type order.
    for row, species in zip(by_type, build._AEROSOL_TYPES.values(), strict=True):
        alone = build.calculate_extinction(
            aerosol_data, {name: profiles[name] for name in species}, wavelengths
        )
        assert_allclose(row, alone, rtol=1e-12)


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


@pytest.mark.parametrize("access", ["local", "staged", "streamed"])
def test_smoke_test_reads_and_processes_one_column(
    optics_root, tmp_path, monkeypatch, access
):
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

    opened, downloaded = [], []

    @contextmanager
    def open_test_dataset(location, *_args):
        opened.append(str(location))
        yield datasets[Path(str(location)).stem]

    def download(location, destination):
        downloaded.append(location)
        destination.write_bytes(b"granule")
        return {}

    if access == "local":
        data_root = tmp_path
        for product in datasets:
            (tmp_path / f"{product}.nc4").write_bytes(b"granule")
        monkeypatch.setattr(
            build, "_file_for_product", lambda product, *_: tmp_path / f"{product}.nc4"
        )
    else:
        data_root = None
        monkeypatch.setattr(
            build,
            "_file_for_product",
            lambda product, *_: f"https://example.test/{product}.nc4",
        )
    monkeypatch.setattr(build, "_open_granule", open_test_dataset)
    monkeypatch.setattr(build, "_download_granule", download)
    result = build.run_smoke_test(
        build.date(2024, 1, 1),
        optics_root,
        data_root=data_root,
        stage_daily=access != "streamed",
    )
    remote = [f"https://example.test/{product}.nc4" for product in datasets]
    if access == "staged":
        # Like the catalogue build, remote granules are downloaded whole first.
        assert downloaded == remote
        assert not any(location.startswith("https://") for location in opened)
    elif access == "streamed":
        assert not downloaded
        assert opened == remote
    else:
        assert not downloaded
    assert result["input_access"] == build._input_access_description(
        data_root, access != "streamed"
    )
    assert result["status"] == "ok"
    assert len(result["timestamps_processed"]) == 2
    assert result["model_levels_per_column"] == 3
    # The 50 kPa level's layer is partly below the 60 kPa tropopause.
    assert result["tropospheric_levels_per_timestamp"] == [2, 2]
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
    species = column.species_extinction_per_m
    assert species.shape == (len(grid), len(build._COMPONENTS), len(wavelengths_nm))
    assert (species >= 0).all()
    extinction_550 = species[:, :, iw550].sum(axis=1)
    # The whole column: every layer, the top one reaching half a spacing above
    # 15 km, well inside the catalogue grid.
    height = build._geopotential_to_geometric_height(
        np.array([1000.0, 5000.0, 15000.0]), 0.5
    )
    edges = np.concatenate(
        ([0.0], 0.5 * (height[1:] + height[:-1]), [1.5 * height[-1] - 0.5 * height[-2]])
    )
    thickness = (
        build._GAS_CONSTANT_DRY_AIR
        * 280.0
        / build._STANDARD_GRAVITY
        * 10000.0
        / np.array([90000.0, 50000.0, 10000.0])
    )
    native = build.calculate_extinction(
        aerosol.isel(time=0, lat=0, lon=0), efficiencies, wavelengths_nm
    )
    layer_aod = native * thickness
    assert edges[-1] < grid[-1]
    expected = layer_aod.sum()
    assert_allclose(np.trapezoid(extinction_550, grid), expected, rtol=1e-6)
    assert extinction_550[(grid > 15000.0) & (grid < 19500.0)].min() > 0
    assert not extinction_550[grid > 21000.0].any()
    # Cutting at the tropopause, as sasktran2 does by default, recovers the
    # tropospheric AOD. Extinction is uniform in the layer holding the
    # tropopause, so the cut within its catalogue cell is exact.
    _, tropopause_height = build._tropospheric_fraction(
        height, np.array([90000.0, 50000.0, 10000.0]), 60000.0, 0.0
    )
    cut = sk.climatology.tropospheric_aerosol._conservative_remap(
        grid, extinction_550, grid, 0.0, float(tropopause_height)
    )
    assert_allclose(
        np.trapezoid(cut, grid),
        result["tropospheric_aod_550nm_per_timestamp"][0],
        rtol=1e-6,
    )
    # The fake granules hold RH = 0.5 everywhere, up to the top of the grid.
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
    assert "range" not in build._input_access_description(None, True)
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
    # Linear in pressure between levels, so the tropopause is inside a layer.
    pressures = 101325.0 * np.exp(-(heights - surface) / 8000.0)
    fraction, tropopause_altitude = build._tropospheric_fraction(
        heights, pressures, 101325.0 * np.exp(-11800.0 / 8000.0), surface
    )
    assert 0 < np.min(fraction[fraction > 0]) < 1
    rng = np.random.default_rng(3)
    extinction = rng.uniform(1e-6, 1e-4, size=(len(heights), 2))

    profile = build._conservative_grid_profile(
        extinction, heights, thickness, fraction, surface, float(tropopause_altitude)
    )

    grid = build._TARGET_ALTITUDES_M
    assert profile.shape == (len(grid), 2)
    assert (profile >= 0).all()
    assert not profile[(grid < surface) | (grid > tropopause_altitude)].any()
    expected = np.sum(extinction * (thickness * fraction)[:, None], axis=0)
    assert_allclose(np.trapezoid(profile, grid, axis=0), expected, rtol=1e-12)


def test_tropospheric_fraction_matches_column_helpers():
    rng = np.random.default_rng(5)
    nlev, ncol = 12, 40
    height = np.sort(rng.uniform(0.0, 20000.0, (nlev, ncol)), axis=0)[::-1]
    pressure = 101325.0 * np.exp(-height / 7500.0)
    tropopause = 101325.0 * np.exp(-rng.uniform(-500.0, 21000.0, ncol) / 7500.0)
    surface = rng.uniform(-50.0, 300.0, ncol)
    fraction, tropopause_height = build._tropospheric_fraction(
        height, pressure, tropopause, surface
    )
    for column in range(ncol):
        assert tropopause_height[column] == pytest.approx(
            build._height_at_pressure(
                pressure[:, column], height[:, column], tropopause[column]
            )
        )
        # Same result for one column at a time and for any level order.
        flipped = build._tropospheric_fraction(
            height[::-1, column],
            pressure[::-1, column],
            tropopause[column],
            surface[column],
        )
        assert_allclose(flipped[0][::-1], fraction[:, column])
        below = height[:, column] < tropopause_height[column]
        # Whole layers below, nothing above, and at most two partial layers
        # around the tropopause.
        assert np.all(fraction[~below, column] < 1)
        assert np.sum((fraction[:, column] > 0) & (fraction[:, column] < 1)) <= 1
    height[3, 0] = np.nan
    height[4, 1] = height[5, 1]
    fraction, tropopause_height = build._tropospheric_fraction(
        height, pressure, tropopause, surface
    )
    assert np.isnan(fraction[:, :2]).all()
    assert np.isnan(tropopause_height[:2]).all()
    assert np.isfinite(fraction[:, 2:]).all()


@pytest.mark.parametrize("nudge", [-1e-9, 0.0, 1e-9])
def test_tropopause_on_a_model_level_gives_no_spike_or_gap(nudge):
    # MERRA2's TROPPB usually equals a model level's pressure. Uniform
    # extinction must map to a uniform profile up to the tropopause whether
    # rounding puts that level just inside or just outside the troposphere.
    heights = np.arange(500.0, 20000.0, 1000.0)
    pressures = 101325.0 * np.exp(-heights / 7500.0)
    thickness = np.full(heights.size, 1000.0)
    extinction = np.full((heights.size, 1), 2e-6)
    tropopause_pressure = pressures[12] * (1 + nudge)  # level at 12.5 km
    fraction, tropopause_altitude = build._tropospheric_fraction(
        heights, pressures, tropopause_pressure, 0.0
    )
    assert float(tropopause_altitude) == pytest.approx(12500.0, abs=1e-3)
    assert fraction[12] == pytest.approx(0.5, abs=1e-6)
    profile = build._conservative_grid_profile(
        extinction, heights, thickness, fraction, 0.0, float(tropopause_altitude)
    )[:, 0]
    grid = build._TARGET_ALTITUDES_M
    inside = grid <= float(tropopause_altitude)
    assert_allclose(profile[inside][:-1], 2e-6, rtol=1e-6)
    # The top point's cell is cut at the tropopause (here at the point
    # itself), so it carries 0.5-1.5 times the local extinction.
    assert 0.5 * 2e-6 * (1 - 1e-6) <= profile[inside][-1] <= 1.5 * 2e-6
    assert not profile[~inside].any()
    assert np.trapezoid(profile, grid) == pytest.approx(12500.0 * 2e-6, rel=1e-6)


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
    locations = {product: f"{root_url}/{product}.nc4" for product in build._PRODUCTS}
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


def test_catalogue_grid_covers_the_merra2_column():
    grid = build._TARGET_ALTITUDES_M
    assert grid[0] == 0.0
    # The MERRA2 model top (0.01 hPa) is near 80 km.
    assert grid[-1] > 80000.0
    assert np.all(np.diff(grid) > 0)
    regular = build._SELECTION_ALTITUDES_M
    assert_array_equal(regular, np.arange(0.0, 20001.0, 250.0))
    assert_array_equal(grid[: len(regular)], regular)
    weights = build._trapezoid_weights(grid)
    assert_allclose(weights.sum(), grid[-1] - grid[0])


def test_whole_column_profile_folds_aod_above_the_grid_into_its_top():
    grid = np.array([0.0, 1000.0, 3000.0, 6000.0])
    heights = np.array([500.0, 2000.0, 5000.0, 9000.0])
    thickness = np.array([1000.0, 2000.0, 3000.0, 5000.0])
    extinction = np.array([4e-6, 2e-6, 1e-6, 5e-7])
    profile = build._conservative_grid_profile(
        extinction[:, None], heights, thickness, np.ones(4), 0.0, np.inf, grid
    )[:, 0]
    assert_allclose(
        np.trapezoid(profile, grid), np.sum(extinction * thickness), rtol=1e-12
    )
    assert (profile > 0).all()


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
    # 70% sulfate, and 30% dust split 2:1 between its first two size bins.
    species = list(build._COMPONENTS)
    species_extinction = np.zeros((len(species), *extinction.shape))
    species_extinction[species.index("SO4")] = 0.7 * extinction
    species_extinction[species.index("DU001")] = 0.2 * extinction
    species_extinction[species.index("DU002")] = 0.1 * extinction
    reservoir = build._WeightedReservoir(1000, np.random.default_rng(7))
    reservoir.add(
        species_extinction,
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
    assert_allclose(reservoir.species_aod.sum(axis=1), reservoir.aod, rtol=1e-12)
    assert_allclose(
        reservoir.type_aod / reservoir.aod[:, None],
        [[0.7, 0.3, 0.0, 0.0, 0.0]] * 1000,
    )
    assert reservoir.type_profiles.shape == (
        1000,
        len(build._AEROSOL_TYPES),
        len(build._SELECTION_ALTITUDES_M),
    )
    assert_allclose(
        np.trapezoid(reservoir.type_profiles, build._SELECTION_ALTITUDES_M, axis=2),
        reservoir.type_aod,
        rtol=1e-6,
    )
    dust_mix = reservoir.species_mix(np.arange(3), "DU")
    expected = np.zeros(len(species))
    expected[[species.index("DU001"), species.index("DU002")]] = [2 / 3, 1 / 3]
    assert_allclose(dust_mix, [expected] * 3)
    for aerosol_type in ("SU", "DU"):
        index, percentile, count, lower, upper = reservoir.select_tier(
            0.5, 0.03, aerosol_type
        )
        type_aod = reservoir.type_aod[:, list(build._AEROSOL_TYPES).index(aerosol_type)]
        assert 47 <= percentile <= 53
        assert count > 0
        assert lower <= type_aod[index] <= upper
    # No sea salt anywhere: there is no column to choose.
    with pytest.raises(ValueError, match="positive SS AOD"):
        reservoir.select_tier(0.5, 0.03, "SS")


def _synthetic_reservoir(rng, size, species_alpha=None):
    """A reservoir whose per-type AOD, profile shape and species mix vary freely.

    Shapes and species mixes don't depend on loading. Returns the reservoir and
    the profile shapes (columns, types, altitudes), each integrating to 1.
    """
    altitudes = build._SELECTION_ALTITUDES_M
    types = len(build._AEROSOL_TYPES)
    type_aod = rng.lognormal(np.log(0.1), 0.7, (size, types))
    scale_height = rng.uniform(800.0, 3000.0, (size, types, 1))
    shapes = np.exp(-altitudes / scale_height) * rng.lognormal(
        0.0, 0.4, (size, types, altitudes.size)
    )
    shapes[..., altitudes > 12000.0] = 0.0
    shapes /= np.trapezoid(shapes, altitudes, axis=2)[..., None]
    species_aod = np.zeros((size, len(build._COMPONENTS)))
    for index, (code, positions) in enumerate(build._TYPE_SPECIES_INDEX.items()):
        alpha = (species_alpha or {}).get(code, np.ones(len(positions)))
        species_aod[:, positions] = type_aod[:, index, None] * rng.dirichlet(
            alpha, size
        )
    reservoir = build._WeightedReservoir(size, rng)
    reservoir.aod = type_aod.sum(axis=1)
    reservoir.species_aod = species_aod
    reservoir.type_profiles = type_aod[..., None] * shapes
    reservoir.valid_candidates = size
    return reservoir, shapes


def test_selection_is_not_biased_toward_low_loading_within_the_window():
    # Profile shapes here don't depend on loading, so the selected column should
    # sit anywhere in its type's percentile window. Scoring unnormalized
    # profiles put selections low: -0.52 of the half-width in the 2024
    # catalogue.
    rng = np.random.default_rng(3)
    positions = []
    for _ in range(10):
        reservoir, _ = _synthetic_reservoir(rng, 5000)
        for aerosol_type in build._AEROSOL_TYPES:
            for quantile, half_width in build._TIERS.values():
                _, percentile, *_ = reservoir.select_tier(
                    quantile, half_width, aerosol_type
                )
                positions.append((percentile / 100 - quantile) / half_width)
    assert abs(np.mean(positions)) < 0.15


def test_selection_prefers_the_window_median_species_mix_and_shape():
    # Species mix and shape vary independently of loading. Shape-only
    # selection would leave the chosen column's mix at a random rank.
    rng = np.random.default_rng(3)
    mix_ranks, shape_ranks, sulfate_shape_ranks = [], [], []
    types = list(build._AEROSOL_TYPES)
    for _ in range(20):
        reservoir, shapes = _synthetic_reservoir(
            rng, 5000, {"DU": [2.0, 1.0, 1.5, 0.3, 0.8]}
        )
        for aerosol_type in ("DU", "SU"):
            t = types.index(aerosol_type)
            type_aod = reservoir.type_aod[:, t]
            for quantile, half_width in build._TIERS.values():
                index, *_ = reservoir.select_tier(quantile, half_width, aerosol_type)
                group = np.flatnonzero(
                    (type_aod >= np.quantile(type_aod, max(0.0, quantile - half_width)))
                    & (
                        type_aod
                        <= np.quantile(type_aod, min(1.0, quantile + half_width))
                    )
                )
                chosen = np.flatnonzero(group == index)[0]
                group_shapes = shapes[group, t]
                transformed = np.log1p(
                    group_shapes / np.median(group_shapes[group_shapes > 0])
                )
                shape_distance = np.mean(
                    (transformed - np.median(transformed, axis=0)) ** 2, axis=1
                )
                shape_rank = np.mean(shape_distance < shape_distance[chosen])
                if aerosol_type == "SU":
                    sulfate_shape_ranks.append(shape_rank)
                    continue
                target = reservoir.median_species_mix(quantile, half_width, "DU")
                assert_allclose(target.sum(), 1.0)
                outside = np.ones(len(build._COMPONENTS), dtype=bool)
                outside[build._TYPE_SPECIES_INDEX["DU"]] = False
                assert not target[outside].any()
                mix = reservoir.species_mix(group, "DU")
                mix_distance = np.sum((np.sqrt(mix) - np.sqrt(target)) ** 2, axis=1)
                mix_ranks.append(np.mean(mix_distance < mix_distance[chosen]))
                shape_ranks.append(shape_rank)
    assert np.median(mix_ranks) < 0.05
    assert np.max(mix_ranks) < 0.25
    assert np.median(shape_ranks) < 0.15
    # Sulfate has a single species, so only its shape counts. Profiles are
    # stored as float32, so allow a near tie.
    assert np.max(sulfate_shape_ranks) < 0.01


# Species shares of every column in the checkpoint test's reservoirs.
_CHECKPOINT_SPECIES_SHARES = np.arange(1.0, len(build._COMPONENTS) + 1.0) / sum(
    range(1, len(build._COMPONENTS) + 1)
)


@pytest.mark.parametrize("stage_daily", [True, False])
def test_month_checkpoint_restores_reservoir_and_skips_completed_days(
    tmp_path, monkeypatch, stage_daily
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
                "url": str(
                    build._source_location(product, build.date(2024, 1, day), None)
                ),
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
    reservoirs = {band: build._WeightedReservoir(1000, rng) for band in build._BANDS}
    for reservoir in reservoirs.values():
        reservoir.keys = np.linspace(-1.0, -0.01, 1000)
        reservoir.aod = np.linspace(0.01, 1.0, 1000)
        reservoir.species_aod = np.outer(reservoir.aod, _CHECKPOINT_SPECIES_SHARES)
        reservoir.type_profiles = np.ones(
            (1000, len(build._AEROSOL_TYPES), len(build._SELECTION_ALTITUDES_M)),
            dtype=np.float32,
        )
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
        assert_allclose(
            resumed_reservoirs[band].species_aod, reservoirs[band].species_aod
        )
        assert_array_equal(
            resumed_reservoirs[band].type_profiles, reservoirs[band].type_profiles
        )

    def unexpected_source_read(*_args, **_kwargs):
        pytest.fail("A completed month checkpoint should not reread MERRA2 data")

    monkeypatch.setattr(build, "_open_granule", unexpected_source_read)
    column_reads = []

    def read_column_spectra(day, columns, *_args, stage_daily, stager=None):
        column_reads.append((day, list(columns), stage_daily, stager))
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
        stage_daily=stage_daily,
    )
    assert len(records) == (
        len(build._BANDS) * len(build._AEROSOL_TYPES) * len(build._TIERS)
    )
    assert {(record["aerosol_type"], record["scenario"]) for record in records} == {
        (aerosol_type, f"{band}_{tier}")
        for aerosol_type in build._AEROSOL_TYPES
        for band in build._BANDS
        for tier in build._TIERS
    }
    for record in records:
        positions = build._TYPE_SPECIES_INDEX[record["aerosol_type"]]
        expected = np.zeros(len(build._COMPONENTS))
        expected[positions] = _CHECKPOINT_SPECIES_SHARES[positions]
        expected /= expected.sum()
        assert_allclose(record["selection_species_aod_fraction"], expected)
        assert_allclose(record["candidate_median_species_aod_fraction"], expected)
        assert_allclose(
            record["selection_aod_550nm"],
            record["column_total_aod_550nm"]
            * _CHECKPOINT_SPECIES_SHARES[positions].sum(),
        )
        # Each case holds only its type's species.
        outside = np.ones(len(build._COMPONENTS), dtype=bool)
        outside[positions] = False
        assert not record["species_extinction_550nm_per_m"][:, outside].any()
        assert_allclose(
            record["species_extinction_550nm_per_m"].sum(axis=1),
            record["extinction_per_m"][:, 0],
        )
    # Every selection is the same column on 2024-01-01: one day, read once,
    # and opened the way the daily pass opens its granules. A staged read
    # shares one stager over the selected days, closed afterwards.
    assert len(column_reads) == 1
    day, columns, read_staged, stager = column_reads[0]
    assert day == build.date(2024, 1, 1)
    assert columns == [(0, 0, 0)]
    assert read_staged is stage_daily
    if stage_daily:
        assert isinstance(stager, build._DayStager)
        assert stager._days == [build.date(2024, 1, 1)]
        assert stager._checkpointing is False
        assert stager._executor._shutdown
    else:
        assert stager is None


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


def _fake_column(wavelengths_nm):
    # The whole column: aerosol above the 10 km tropopause too.
    grid = build._TARGET_ALTITUDES_M
    extinction = np.where(grid <= 10000.0, 1e-5, 1e-7)
    shares = np.arange(1, len(build._COMPONENTS) + 1, dtype=float)
    shares /= shares.sum()
    return build._ColumnProfiles(
        species_extinction_per_m=np.repeat(
            (extinction[:, None] * shares[None, :])[:, :, None],
            len(wavelengths_nm),
            axis=2,
        ),
        relative_humidity=np.full(len(grid), 0.4),
    )


def _fake_records(month, wavelengths_nm, species_scale=1.0):
    column = _fake_column(wavelengths_nm)
    reference = int(np.flatnonzero(np.asarray(wavelengths_nm) == 550.0)[0])
    records = []
    for band in build._BANDS:
        for aerosol_type, positions in build._TYPE_SPECIES_INDEX.items():
            spectra = column.species_extinction_per_m[:, positions, :]
            species_550nm = np.zeros(column.species_extinction_per_m.shape[:2])
            species_550nm[:, positions] = species_scale * spectra[:, :, reference]
            mix = np.zeros(len(build._COMPONENTS))
            mix[positions] = 1 / len(positions)
            records.extend(
                {
                    "scenario": f"{band}_{tier}",
                    "aerosol_type": aerosol_type,
                    "month": month,
                    "extinction_per_m": spectra.sum(axis=1),
                    "species_extinction_550nm_per_m": species_550nm,
                    "relative_humidity": column.relative_humidity,
                    "selection_aod_550nm": 0.1,
                    "selection_percentile": 50.0,
                    "candidate_count": 10,
                    "candidate_window_aod_min": 0.09,
                    "candidate_window_aod_max": 0.11,
                    "column_total_aod_550nm": 0.3,
                    "selected_latitude_degrees": 0.0,
                    "selected_longitude_degrees": 0.0,
                    "selected_time": f"2024-{month:02d}-01T00:00:00",
                    "tropopause_altitude_m": 10000.0,
                    "surface_altitude_m": 0.0,
                    "sampled_candidate_count": 1000,
                    "valid_candidate_count": 10000,
                    "selection_species_aod_fraction": mix.tolist(),
                    "candidate_median_species_aod_fraction": mix.tolist(),
                }
                for tier in build._TIERS
            )
    return records


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
        assert catalogue.attrs["schema_version"] == 3
        assert catalogue.species.values.tolist() == list(build._COMPONENTS)
        assert catalogue.aerosol_type.values.tolist() == list(build._AEROSOL_TYPES)
        assert catalogue.extinction_per_m.dims == (
            *_CASE_DIMS,
            "altitude_m",
            "wavelength_nm",
        )
        species = catalogue.species_extinction_550nm_per_m
        assert species.dims == (*_CASE_DIMS, "species", "altitude_m")
        assert species.attrs["units"] == "m-1"
        assert_allclose(
            species.sum("species"),
            catalogue.extinction_per_m.sel(wavelength_nm=550.0),
            rtol=1e-6,
        )
        for code, members in build._AEROSOL_TYPES.items():
            others = [name for name in build._COMPONENTS if name not in members]
            assert not species.sel(aerosol_type=code, species=others).values.any()
        # Profiles continue above the tropopause.
        above = catalogue.altitude_m > 10000.0
        assert (catalogue.extinction_per_m.where(above, drop=True) > 0).all()
        humidity = catalogue.relative_humidity
        assert humidity.dims == (*_CASE_DIMS, "altitude_m")
        assert_allclose(humidity, 0.4, rtol=1e-6)
        assert catalogue.latitude_band.dims == ("scenario",)
        assert catalogue.loading.dims == ("scenario",)
        assert catalogue.selection_aod_550nm.dims == _CASE_DIMS
        assert catalogue.selection_species_aod_fraction.dims == (*_CASE_DIMS, "species")
    report = json.loads((tmp_path / "output" / "build_report_v1.json").read_text())
    assert len(report["scenarios"]) == 12 * len(build._AEROSOL_TYPES) * 4
    for scenario in report["scenarios"]:
        assert scenario["aerosol_type"] in build._AEROSOL_TYPES
        assert not {
            "extinction_per_m",
            "species_extinction_550nm_per_m",
            "relative_humidity",
        } & set(scenario)


def test_builder_catalogue_loads_and_builds_the_default_constituent(
    tmp_path, monkeypatch
):
    # The library tests use a hand-written catalogue; this checks the builder's
    # own output against the library, with tables storing wavelength as float32
    # metres like the real ones (40 um reads back as 39999.999 nm).
    monkeypatch.setenv("SASKTRAN2_DATABASE_ROOT", str(tmp_path / "database"))
    optics = tmp_path / "optics"
    optics.mkdir()
    wavelengths_m = np.array([250e-9, 550e-9, 40e-6])
    _write_optics_tables(
        optics,
        {
            **dict.fromkeys(build._OPTICS_FILES, wavelengths_m.astype(np.float32)),
            "optics_SS.v3_3.nc": wavelengths_m,
        },
    )
    _, wavelengths = build.load_efficiency_profiles(optics)
    _patch_catalogue_inputs(
        monkeypatch, tmp_path, lambda month: _fake_records(month, wavelengths)
    )
    path = build.build_catalogue(
        optics,
        tmp_path / "output",
        data_root=tmp_path,
        start_year=2024,
        end_year=2024,
    )

    catalogue = aerosol.load_dataset(path=path)
    assert catalogue.wavelength_nm.values.tolist() == [250.0, 550.0, 40000.0]
    for name, units in build._METADATA_UNITS.items():
        assert catalogue[name].attrs["units"] == units
    assert catalogue.aerosol_type.values.tolist() == list(build._AEROSOL_TYPES)
    dust_mix = np.zeros(len(build._COMPONENTS))
    dust_mix[build._TYPE_SPECIES_INDEX["DU"]] = 0.2
    for name in (
        "selection_species_aod_fraction",
        "candidate_median_species_aod_fraction",
    ):
        assert_allclose(
            catalogue[name].sel(
                scenario="tropical_extreme", aerosol_type="DU", month=7
            ),
            dust_mix,
        )
    for name in sk.database.MERRA2_AEROSOL_TYPE_NAMES:
        constituent = aerosol.constituent(
            "tropical_extreme",
            aerosol_type=name,
            month=7,
            path=path,
            optics_root=optics,
        )
        assert isinstance(constituent, sk.constituent.ExtinctionScatterer)
        uncut = aerosol.profile(
            "tropical_extreme",
            aerosol_type=name,
            month=7,
            cut_at_tropopause=False,
            path=path,
        )
        assert float(uncut.aod) > float(uncut.tropospheric_aod)


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


def test_build_catalogue_rejects_species_outside_the_aerosol_type(
    optics_root, tmp_path, monkeypatch
):
    _, wavelengths = build.load_efficiency_profiles(optics_root)

    def records_for_month(month):
        records = _fake_records(month, wavelengths)
        dust = next(record for record in records if record["aerosol_type"] == "DU")
        # Move some dust extinction onto sulfate; the total is unchanged.
        species = dust["species_extinction_550nm_per_m"]
        moved = 0.5 * species[:, build._TYPE_SPECIES_INDEX["DU"][0]]
        species[:, build._TYPE_SPECIES_INDEX["DU"][0]] -= moved
        species[:, build._TYPE_SPECIES_INDEX["SU"][0]] += moved
        return records

    _patch_catalogue_inputs(monkeypatch, tmp_path, records_for_month)
    with pytest.raises(ValueError, match="outside the aerosol type"):
        build.build_catalogue(
            optics_root,
            tmp_path / "output",
            data_root=tmp_path,
            start_year=2024,
            end_year=2024,
        )


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


_SIGNED_URL = (
    "https://data.example.test/login?code=SECRET-CODE&state=%2Fdata%2Fgranule.nc4"
)


class _FakeSignedFailure(_FakeDownload):
    def raise_for_status(self):
        msg = f"{self.status_code} Server Error: Bad Gateway for url: {_SIGNED_URL}"
        raise build.requests.HTTPError(msg, response=self)


def test_url_queries_are_redacted():
    text = (
        f"error for url: {_SIGNED_URL}; see https://example.test/a/b.nc4#frag "
        "and https://example.test/plain.nc4."
    )
    redacted = build._redact_url_queries(text)
    assert "SECRET-CODE" not in redacted
    assert "frag" not in redacted
    assert "https://data.example.test/login?<redacted>" in redacted
    assert "https://example.test/a/b.nc4?<redacted>" in redacted
    assert "https://example.test/plain.nc4." in redacted


def test_download_logs_and_raises_without_url_queries(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(build.time, "sleep", lambda _seconds: None)
    destination = tmp_path / "granule.nc4"
    responses = [_FakeSignedFailure(502), _FakeDownload(200)]
    monkeypatch.setattr(build.requests, "get", lambda *_a, **_k: responses.pop(0))
    with caplog.at_level(logging.WARNING):
        build._download_granule("https://example.test/granule.nc4", destination)
    assert "Transient MERRA2 download failure" in caplog.text
    assert "SECRET-CODE" not in caplog.text
    assert "login?<redacted>" in caplog.text

    monkeypatch.setattr(
        build.requests, "get", lambda *_a, **_k: _FakeSignedFailure(404)
    )
    with pytest.raises(build.requests.HTTPError) as error:
        build._download_granule("https://example.test/granule.nc4", destination)
    assert "SECRET-CODE" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
    assert error.value.response.status_code == 404


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


def test_staged_identity_comes_from_the_download_response(tmp_path, monkeypatch):
    def get(url, **_kwargs):
        response = _FakeDownload(200, body=url.encode())
        response.headers |= {
            "ETag": f'"{Path(url).name}"',
            "Last-Modified": "Mon, 01 Apr 2024 00:00:00 GMT",
        }
        return response

    def no_metadata_request(_location):
        msg = "staging must not make a separate metadata request"
        raise AssertionError(msg)

    monkeypatch.setattr(build.requests, "get", get)
    monkeypatch.setattr(build, "_source_identity", no_metadata_request)
    day = build.date(2024, 4, 10)

    identities, staged = build._stage_day(day, tmp_path, checkpointing=True)

    for product, location in build._source_locations(day, None).items():
        assert staged[product].read_bytes() == location.encode()
        # Same form as _source_identity's fsspec metadata, so resumed runs
        # validate these checkpoints against a metadata request.
        assert identities[product] == build._remote_source_identity(
            location,
            {
                "size": len(location.encode()),
                "ETag": f'"{Path(location).name}"',
                "Last-Modified": "Mon, 01 Apr 2024 00:00:00 GMT",
            },
        )


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


def test_builder_logging_redacts_url_queries_in_messages_and_tracebacks():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []
    stream = io.StringIO()
    try:
        build._configure_logging(logging.INFO, "%(levelname)s %(message)s")
        for handler in root.handlers:
            handler.setStream(stream)
        try:
            msg = f"Bad Gateway for url: {_SIGNED_URL}"
            raise OSError(msg)
        except OSError as exc:
            logging.error("range failed (%s)", exc, exc_info=exc)
    finally:
        root.handlers, root.level = saved_handlers, saved_level
    text = stream.getvalue()
    assert "SECRET-CODE" not in text
    assert text.count("login?<redacted>") == 2  # message and traceback


def test_month_task_installs_url_redaction_on_worker_logging(monkeypatch):
    # A spawned month worker gets a plain basicConfig handler; the month task
    # itself switches it to the redacting formatter before doing any work.
    root = logging.getLogger()
    handler = logging.StreamHandler(io.StringIO())
    handler.setFormatter(logging.Formatter("%(message)s"))
    monkeypatch.setattr(root, "handlers", [handler])
    monkeypatch.setattr(build, "_month_days", _stop_after_redaction)
    with pytest.raises(_RedactionInstalled):
        build._process_month(1, 2024, 2024, None, None, {}, None, 1000, 0)
    assert isinstance(handler.formatter, build._RedactingFormatter)
    assert handler.formatter._fmt == "%(message)s"


class _RedactionInstalled(Exception):
    pass


def _stop_after_redaction(*_args):
    raise _RedactionInstalled


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
