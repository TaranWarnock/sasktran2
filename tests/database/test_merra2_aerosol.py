from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import sasktran2 as sk
import xarray as xr
from numpy.testing import assert_allclose
from sasktran2.database import merra2_aerosol

_BUILD_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "databases"
    / "tropospheric_aerosol"
    / "build.py"
)

_N_MOM = 8
_RH = np.array([0.0, 0.5, 0.9])
_LAMBDA_M = np.array([3.0e-7, 5.0e-7, 1.0e-6])
_RADIUS = np.array([1.0e-7, 5.0e-7])
_G = 0.6


def _load_build_module():
    spec = importlib.util.spec_from_file_location(
        "tropospheric_aerosol_build_merra2_optics", _BUILD_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_table(path: Path) -> Path:
    """Write a small MERRA2-style optics table and return its path."""
    shape = (len(_RADIUS), len(_RH), len(_LAMBDA_M))
    radius_factor = np.array([1.0, 2.0])[:, None, None]
    rh_factor = (1.0 + _RH)[None, :, None]
    lambda_factor = (1.0 / (_LAMBDA_M * 1e6))[None, None, :]
    bext = 1000.0 * radius_factor * rh_factor * lambda_factor
    bsca = 0.9 * bext

    # Henyey-Greenstein P11 moments in sasktran2's convention, (2l+1) g**l
    moments = np.arange(_N_MOM)
    p11 = (2 * moments + 1) * _G**moments
    # Non-P11 elements are deliberately non-zero so the test detects any leak
    # of them into the Greek coefficients
    pmom = np.empty((6, _N_MOM, *shape))
    for pol in range(6):
        pmom[pol] = (pol + 1) * p11[:, None, None, None]

    ds = xr.Dataset(
        {
            "bext": (("radius", "rh", "lambda"), bext.astype(np.float32)),
            "bsca": (("radius", "rh", "lambda"), bsca.astype(np.float32)),
            "g": (
                ("radius", "rh", "lambda"),
                np.full(shape, _G, dtype=np.float32),
            ),
            "pmom": (
                ("nPol", "nMom", "radius", "rh", "lambda"),
                pmom.astype(np.float32),
            ),
        },
        coords={
            "radius": _RADIUS.astype(np.float32),
            "rh": _RH.astype(np.float32),
            "lambda": _LAMBDA_M.astype(np.float32),
        },
    )
    ds["bext"].attrs["units"] = "m2 (kg dry mass)-1"
    ds["bsca"].attrs["units"] = "m2 (kg dry mass)-1"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def optics_root(tmp_path) -> Path:
    root = tmp_path / "optics"
    root.mkdir()
    _synthetic_table(root / "optics_SU.v1_3.nc")
    return root


@pytest.fixture
def db_root(tmp_path) -> Path:
    root = tmp_path / "cache"
    root.mkdir()
    return root


def test_species_map_matches_catalogue_builder():
    build = _load_build_module()
    assert build._COMPONENTS == merra2_aerosol.MERRA2_AEROSOL_SPECIES


def test_wavelength_rounding_matches_catalogue_builder():
    build = _load_build_module()
    assert build._WAVELENGTH_DECIMALS_NM == merra2_aerosol._WAVELENGTH_DECIMALS_NM


def test_table_end_wavelength_matches_the_catalogue_grid(tmp_path, db_root):
    # float32(4e-5) m is 39999.999 nm; unrounded, the catalogue's 40000 nm
    # fell outside the table and constituent() refused the catalogue.
    root = tmp_path / "optics"
    root.mkdir()
    table = _synthetic_table(root / "optics_SU.v1_3.nc")
    with xr.open_dataset(table) as source:
        wide = source.load().assign_coords(
            {"lambda": np.array([2.5e-7, 5.5e-7, 4.0e-5], dtype=np.float32)}
        )
    table.unlink()
    wide.to_netcdf(table)
    assert np.float64(np.float32(4.0e-5)) * 1e9 != 40000.0

    ds = merra2_aerosol.OpticalDatabaseMERRA2Aerosol(
        "SO4", optics_root=root, db_root=db_root
    ).load_ds()

    np.testing.assert_array_equal(
        ds["wavelength_nm"].to_numpy(), [250.0, 550.0, 40000.0]
    )


def test_exported_from_database_package():
    assert (
        sk.database.OpticalDatabaseMERRA2Aerosol
        is merra2_aerosol.OpticalDatabaseMERRA2Aerosol
    )
    assert sk.database.merra2_optics_table is merra2_aerosol.merra2_optics_table


def test_generate_maps_table_to_scalar_database(optics_root, db_root):
    db = merra2_aerosol.OpticalDatabaseMERRA2Aerosol(
        "SO4", optics_root=optics_root, db_root=db_root
    )
    ds = db.load_ds()
    source = xr.open_dataset(optics_root / "optics_SU.v1_3.nc").isel(radius=0)

    # float32 metres rounded to the builder's 0.01 nm grid
    np.testing.assert_array_equal(
        ds["wavelength_nm"].to_numpy(), [300.0, 500.0, 1000.0]
    )
    assert_allclose(ds["relative_humidity"].to_numpy(), _RH, rtol=1e-6)
    for name in ("xs_total", "xs_scattering", "lm_a1"):
        assert ds[name].dtype == np.float64

    expected_ext = source["bext"].transpose("rh", "lambda").to_numpy()
    expected_sca = source["bsca"].transpose("rh", "lambda").to_numpy()
    assert_allclose(
        ds["xs_total"].transpose("relative_humidity", "wavelength_nm").to_numpy(),
        expected_ext,
    )
    assert_allclose(
        ds["xs_scattering"].transpose("relative_humidity", "wavelength_nm").to_numpy(),
        expected_sca,
    )

    lm_a1 = ds["lm_a1"].transpose("legendre", "relative_humidity", "wavelength_nm")
    expected_a1 = (
        source["pmom"].isel(nPol=0).transpose("nMom", "rh", "lambda").to_numpy()
    )
    assert lm_a1.shape[0] == _N_MOM
    assert_allclose(lm_a1.to_numpy(), expected_a1)
    assert_allclose(lm_a1.isel(legendre=0), 1.0)
    assert_allclose(lm_a1.isel(legendre=1), 3 * _G, rtol=1e-6)

    for name in ("lm_a2", "lm_a3", "lm_a4", "lm_b1", "lm_b2"):
        assert ds[name].dims == ds["lm_a1"].dims
        assert np.all(ds[name].to_numpy() == 0.0)


def test_radius_bin_and_truncation(optics_root, db_root):
    db = merra2_aerosol.OpticalDatabaseMERRA2Aerosol(
        input_file="optics_SU.v1_3.nc",
        radius_bin=1,
        optics_root=optics_root,
        db_root=db_root,
        max_legendre_moments=4,
    )
    ds = db.load_ds()
    assert ds.sizes["legendre"] == 4
    assert_allclose(
        ds["xs_total"].sel(
            relative_humidity=0.0, wavelength_nm=500.0, method="nearest"
        ),
        2.0 * 1000.0 / 0.5,
        rtol=1e-6,
    )
    assert db.path().parent == db_root / "cross_sections" / "merra2"
    assert "optics_SU.v1_3" in db.path().name
    assert "bin1" in db.path().name
    assert "mom4" in db.path().name


def test_rejects_ambiguous_or_unknown_inputs(optics_root, db_root):
    with pytest.raises(ValueError, match="species"):
        merra2_aerosol.OpticalDatabaseMERRA2Aerosol(
            "NOTASPECIES", optics_root=optics_root, db_root=db_root
        )
    with pytest.raises(ValueError, match="radius_bin"):
        merra2_aerosol.OpticalDatabaseMERRA2Aerosol(
            input_file="optics_SU.v1_3.nc",
            optics_root=optics_root,
            db_root=db_root,
        )
    with pytest.raises(ValueError, match="not both"):
        merra2_aerosol.OpticalDatabaseMERRA2Aerosol(
            "SO4",
            input_file="optics_SU.v1_3.nc",
            radius_bin=0,
            optics_root=optics_root,
            db_root=db_root,
        )


def _atmosphere(nstokes: int, db, altitudes_m) -> sk.Atmosphere:
    config = sk.Config()
    config.num_stokes = nstokes
    config.num_streams = 4
    config.single_scatter_source = sk.SingleScatterSource.DiscreteOrdinates
    config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
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
    atmo["aerosol"] = sk.constituent.NumberDensityScatterer(
        db,
        altitudes_m,
        1e-9 * np.exp(-altitudes_m / 2000.0),
        relative_humidity=np.full_like(altitudes_m, 0.5),
    )
    return config, geometry, viewing_geo, atmo


def test_nstokes_guard_rejects_polarized_atmosphere(optics_root, db_root):
    db = merra2_aerosol.OpticalDatabaseMERRA2Aerosol(
        "SO4", optics_root=optics_root, db_root=db_root
    )
    altitudes_m = np.arange(0.0, 20001.0, 1000.0)
    config, geometry, viewing_geo, atmo = _atmosphere(3, db, altitudes_m)
    rh = np.full_like(altitudes_m, 0.5)

    with pytest.raises(ValueError, match="scalar"):
        db.atmosphere_quantities(atmo, relative_humidity=rh)
    with pytest.raises(ValueError, match="scalar"):
        db.optical_derivatives(atmo, relative_humidity=rh)
    with pytest.raises(ValueError, match="scalar"):
        sk.Engine(config, geometry, viewing_geo).calculate_radiance(atmo)


def test_scalar_engine_run(optics_root, db_root):
    db = merra2_aerosol.OpticalDatabaseMERRA2Aerosol(
        "SO4", optics_root=optics_root, db_root=db_root
    )
    altitudes_m = np.arange(0.0, 20001.0, 1000.0)
    config, geometry, viewing_geo, atmo = _atmosphere(1, db, altitudes_m)

    quantities = db.atmosphere_quantities(
        atmo, relative_humidity=np.full_like(altitudes_m, 0.5)
    )
    assert np.all(np.asarray(quantities.extinction) > 0)

    radiance = sk.Engine(config, geometry, viewing_geo).calculate_radiance(atmo)
    values = radiance["radiance"].to_numpy()
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)


def test_guard_applies_to_any_rust_scatterer(optics_root, db_root):
    class GuardedScatterer(
        merra2_aerosol.ScalarOnlyOpticalDatabaseMixin,
        sk.optical.database.OpticalDatabaseGenericScattererRust,
    ):
        pass

    source = merra2_aerosol.OpticalDatabaseMERRA2Aerosol(
        "SO4", optics_root=optics_root, db_root=db_root
    )
    guarded = GuardedScatterer(db=source.load_ds().load())
    altitudes_m = np.arange(0.0, 20001.0, 1000.0)
    _, _, _, atmo = _atmosphere(3, guarded, altitudes_m)
    with pytest.raises(ValueError, match="scalar"):
        guarded.atmosphere_quantities(
            atmo, relative_humidity=np.full_like(altitudes_m, 0.5)
        )


def test_optics_table_uses_explicit_root(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        msg = "network access"
        raise AssertionError(msg)

    monkeypatch.setattr(merra2_aerosol, "StandardDatabase", fail)
    assert (
        merra2_aerosol.merra2_optics_table("optics_SU.v1_3.nc", optics_root=tmp_path)
        == tmp_path / "optics_SU.v1_3.nc"
    )


def _fake_standard_database(served: Path, keys: list):
    class FakeStandardDatabase:
        def path(self, key, **kwargs):
            keys.append(key)
            return served

    return FakeStandardDatabase


def test_optics_table_downloads_and_verifies_checksum(tmp_path, monkeypatch):
    served = _synthetic_table(tmp_path / "optics_SU.v1_3.nc")
    keys = []
    monkeypatch.setattr(
        merra2_aerosol, "StandardDatabase", _fake_standard_database(served, keys)
    )
    monkeypatch.setitem(
        merra2_aerosol._OPTICS_SHA256,
        "optics_SU.v1_3.nc",
        merra2_aerosol._sha256(served),
    )

    assert merra2_aerosol.merra2_optics_table("optics_SU.v1_3.nc") == served
    assert keys == ["merra2/aerosol_optics/optics_SU.v1_3.nc"]


def test_optics_table_rejects_checksum_mismatch(tmp_path, monkeypatch):
    served = _synthetic_table(tmp_path / "optics_SU.v1_3.nc")
    monkeypatch.setattr(
        merra2_aerosol, "StandardDatabase", _fake_standard_database(served, [])
    )
    monkeypatch.setitem(merra2_aerosol._OPTICS_SHA256, "optics_SU.v1_3.nc", "0" * 64)

    with pytest.raises(OSError, match="checksum"):
        merra2_aerosol.merra2_optics_table("optics_SU.v1_3.nc")


def test_optics_table_requires_pinned_checksum(tmp_path, monkeypatch):
    served = _synthetic_table(tmp_path / "optics_SU.v1_3.nc")
    monkeypatch.setattr(
        merra2_aerosol, "StandardDatabase", _fake_standard_database(served, [])
    )
    monkeypatch.setitem(merra2_aerosol._OPTICS_SHA256, "optics_SU.v1_3.nc", None)

    with pytest.raises(FileNotFoundError, match="checksum"):
        merra2_aerosol.merra2_optics_table("optics_SU.v1_3.nc")


def test_optics_table_rejects_unknown_filename():
    with pytest.raises(ValueError, match="optics_XX"):
        merra2_aerosol.merra2_optics_table("optics_XX.nc")


def test_every_species_table_has_checksum_entry():
    filenames = {name for name, _ in merra2_aerosol.MERRA2_AEROSOL_SPECIES.values()}
    assert filenames == set(merra2_aerosol._OPTICS_SHA256)
