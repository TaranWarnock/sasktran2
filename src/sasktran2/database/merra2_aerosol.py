"""Scalar optical properties from the MERRA2 (GOCART) aerosol optics tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import xarray as xr

from sasktran2.atmosphere import Atmosphere
from sasktran2.optical.base import OpticalQuantities
from sasktran2.optical.database import OpticalDatabaseGenericScattererRust

from .base import CachedDatabase
from .web import StandardDatabase

# MERRA2 species -> (optics table, radius bin); must match the catalogue
# builder in tools/databases/tropospheric_aerosol/build.py
MERRA2_AEROSOL_SPECIES = {
    "SO4": ("optics_SU.v1_3.nc", 0),
    **{f"DU{i:03d}": ("optics_DU.v14_2.nc", i - 1) for i in range(1, 6)},
    **{f"SS{i:03d}": ("optics_SS.v3_3.nc", i - 1) for i in range(1, 6)},
    "BCPHILIC": ("optics_BC.v1_3.nc", 1),
    "BCPHOBIC": ("optics_BC.v1_3.nc", 0),
    "OCPHILIC": ("optics_OC.v1_3.nc", 1),
    "OCPHOBIC": ("optics_OC.v1_3.nc", 0),
}

_STANDARD_DATABASE_PREFIX = "merra2/aerosol_optics/"

# TODO(maintainer): upload the tables to
# sasktran2_db/v_latest/merra2/aerosol_optics/ on arg.usask.ca and replace each
# None with the SHA-256 of the uploaded file. Until then, downloads are refused
# and the tables must be supplied locally through ``optics_root``.
_OPTICS_SHA256: dict[str, str | None] = {
    "optics_BC.v1_3.nc": None,
    "optics_DU.v14_2.nc": None,
    "optics_OC.v1_3.nc": None,
    "optics_SS.v3_3.nc": None,
    "optics_SU.v1_3.nc": None,
}

# Order of the scattering matrix elements along the MERRA2 ``nPol`` dimension
_P11 = 0
_GREEK_ZERO = ("lm_a2", "lm_a3", "lm_a4", "lm_b1", "lm_b2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merra2_optics_table(filename: str, optics_root: str | Path | None = None) -> Path:
    """
    Locate a MERRA2 aerosol optics table.

    Parameters
    ----------
    filename : str
        Table file name, e.g. ``"optics_SU.v1_3.nc"``
    optics_root : str | Path, optional
        Local directory holding the tables. When omitted the table is fetched
        from the SASKTRAN2 standard database and verified against its pinned
        SHA-256.

    Returns
    -------
    Path
        Path to the table
    """
    if optics_root is not None:
        return Path(optics_root) / filename
    if filename not in _OPTICS_SHA256:
        msg = (
            f"Unknown MERRA2 aerosol optics table {filename!r}; "
            f"available: {sorted(_OPTICS_SHA256)}"
        )
        raise ValueError(msg)
    checksum = _OPTICS_SHA256[filename]
    if checksum is None:
        msg = (
            f"No pinned checksum for MERRA2 aerosol optics table {filename!r}; "
            "pass optics_root to use a local copy"
        )
        raise FileNotFoundError(msg)
    path = Path(StandardDatabase().path(_STANDARD_DATABASE_PREFIX + filename))
    if _sha256(path) != checksum:
        msg = (
            f"MERRA2 aerosol optics table checksum mismatch for {path}; "
            "remove the damaged cache entry to restore it"
        )
        raise OSError(msg)
    return path


class ScalarOnlyOpticalDatabaseMixin:
    """
    Restrict an optical database to scalar (NSTOKES=1) calculations.

    Place before the database class in the bases, e.g.
    ``class X(ScalarOnlyOpticalDatabaseMixin, OpticalDatabaseGenericScattererRust)``.
    """

    @staticmethod
    def _require_scalar(atmo: Atmosphere):
        if atmo.nstokes != 1:
            msg = (
                "This optical database supports only scalar calculations "
                f"(nstokes=1); the atmosphere has nstokes={atmo.nstokes}"
            )
            raise ValueError(msg)

    def atmosphere_quantities(self, atmo: Atmosphere, **kwargs) -> OpticalQuantities:
        self._require_scalar(atmo)
        return super().atmosphere_quantities(atmo, **kwargs)

    def optical_derivatives(self, atmo: Atmosphere, **kwargs) -> dict:
        self._require_scalar(atmo)
        return super().optical_derivatives(atmo, **kwargs)

    def _into_rust_object(self):
        # Constituents call the Rust database directly when given one, which
        # would bypass the guard. Without it they fall back to the Python
        # methods above, which still delegate to the same Rust database.
        return None


class OpticalDatabaseMERRA2Aerosol(
    ScalarOnlyOpticalDatabaseMixin, CachedDatabase, OpticalDatabaseGenericScattererRust
):
    def __init__(
        self,
        species: str | None = None,
        *,
        input_file: str | Path | None = None,
        radius_bin: int | None = None,
        optics_root: str | Path | None = None,
        max_legendre_moments: int | None = None,
        db_root: Path | None = None,
    ) -> None:
        """
        Scalar scatterer database built from a MERRA2 aerosol optics table.

        The database has a ``relative_humidity`` dimension (fraction, 0-1) that
        must be supplied when creating constituents. ``xs_total`` and
        ``xs_scattering`` hold mass extinction and scattering efficiencies in
        m2 kg-1 (dry mass), so constituent profiles are mass densities in kg m-3.

        Only the P11 phase function moments are used; the polarized Greek
        coefficients are zero and calculations with nstokes != 1 raise.

        Parameters
        ----------
        species : str, optional
            MERRA2 species name, e.g. ``"SO4"`` or ``"DU003"``; see
            ``MERRA2_AEROSOL_SPECIES``
        input_file : str | Path, optional
            Table file name or path, used with ``radius_bin`` instead of
            ``species``. A bare name is resolved with ``merra2_optics_table``.
        radius_bin : int, optional
            Radius bin of ``input_file``
        optics_root : str | Path, optional
            Local directory holding the optics tables
        max_legendre_moments : int, optional
            Truncate the phase function to this many moments
        db_root : Path, optional
            Root directory of the cache, by default the sasktran2 database root
        """
        if species is not None:
            if input_file is not None or radius_bin is not None:
                msg = "Specify either species or (input_file, radius_bin), not both"
                raise ValueError(msg)
            if species not in MERRA2_AEROSOL_SPECIES:
                msg = (
                    f"Unknown MERRA2 aerosol species {species!r}; "
                    f"available: {list(MERRA2_AEROSOL_SPECIES)}"
                )
                raise ValueError(msg)
            input_file, radius_bin = MERRA2_AEROSOL_SPECIES[species]
        elif input_file is None or radius_bin is None:
            msg = "Specify a species or both input_file and radius_bin"
            raise ValueError(msg)
        if max_legendre_moments is not None and max_legendre_moments < 1:
            msg = "max_legendre_moments must be positive"
            raise ValueError(msg)

        input_file = Path(input_file)
        self._table_name = input_file.name
        # A bare file name is resolved through merra2_optics_table at generation
        self._input_file = input_file if input_file.parent != Path() else None
        self._optics_root = optics_root
        self._radius_bin = int(radius_bin)
        self._max_legendre_moments = max_legendre_moments

        moments = "all" if max_legendre_moments is None else max_legendre_moments
        identifier = f"{input_file.stem}_bin{self._radius_bin}_mom{moments}"

        CachedDatabase.__init__(
            self, db_root=db_root, rel_path=Path("cross_sections") / "merra2"
        )
        self._data_file = self._db_root / f"{identifier}.nc"

        OpticalDatabaseGenericScattererRust.__init__(self, self.path())

    def _source_table(self) -> Path:
        if self._input_file is not None:
            return self._input_file
        return merra2_optics_table(self._table_name, self._optics_root)

    def generate(self):
        with xr.open_dataset(self._source_table()) as table:
            ds = table.isel(radius=self._radius_bin).load()

        p11 = ds["pmom"].isel(nPol=_P11)
        if self._max_legendre_moments is not None:
            p11 = p11.isel(nMom=slice(0, self._max_legendre_moments))

        out = xr.Dataset(
            {
                "xs_total": ds["bext"],
                "xs_scattering": ds["bsca"],
                "lm_a1": p11,
            }
        )
        out = out.drop_vars([c for c in out.coords if c not in ("lambda", "rh")])
        out = out.rename({"lambda": "wavelength_nm", "rh": "relative_humidity"})
        out = out.rename_dims({"nMom": "legendre"}).astype(np.float64)
        out = out.assign_coords(
            wavelength_nm=ds["lambda"].to_numpy().astype(np.float64) * 1e9,
            relative_humidity=ds["rh"].to_numpy().astype(np.float64),
        )
        for name in _GREEK_ZERO:
            out[name] = xr.zeros_like(out["lm_a1"])

        out["wavelength_nm"].attrs["units"] = "nm"
        out["relative_humidity"].attrs["units"] = "1"
        out["xs_total"].attrs["units"] = "m2 kg-1"
        out["xs_scattering"].attrs["units"] = "m2 kg-1"
        out.attrs["source_table"] = self._table_name
        out.attrs["radius_bin"] = self._radius_bin

        temporary = self._data_file.with_suffix(".nc.tmp")
        out.to_netcdf(temporary)
        temporary.replace(self._data_file)

    def clear(self):
        if self._data_file.exists():
            self._data_file.unlink()

    def path(self, key: str | None = None, **kwargs) -> Path:  # noqa: ARG002
        if not self._data_file.exists():
            self.generate()
        return self._data_file

    def load_ds(self, key: str | None = None, **kwargs) -> xr.Dataset:
        return xr.open_dataset(self.path(key, **kwargs))
