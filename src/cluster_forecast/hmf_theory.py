from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pyccl as ccl

from src.cluster_forecast.helpers import get_scale_factor_from_redshift


class MassFunctionBase(ABC):
    """Abstract base for halo mass functions.

    Subclasses must implement :meth:`mass_function` to return the differential
    number density ``dn/dlnM`` as a function of mass and redshift.

    Conventions:
        * Input mass ``mass`` is in Msun/h.
        * Output is ``dn/dlnM`` in Mpc^-3.
    """

    @abstractmethod
    def mass_function(self, mass: Iterable[float] | np.ndarray, redshift: float) -> np.ndarray:
        """Evaluate ``dn/dlnM`` at the given mass and redshift.

        Args:
            mass (array-like): Masses (Msun/h).
            redshift (float): Redshift.

        Returns:
            np.ndarray: ``dn/dlnM`` (Mpc^-3) for each input mass.
        """
        raise NotImplementedError


class Jenkins(MassFunctionBase):
    """Jenkins (2001) friends-of-friends mass function (FoF, 'matter').

    This uses PyCCL's Jenkins01 implementation with a FoF mass definition.

    Args:
        cosmo (ccl.Cosmology): Cosmology instance.
    """

    def __init__(self, cosmo: ccl.Cosmology):
        self.cosmo = cosmo
        mass_def = ccl.halos.massdef.MassDef("fof", "matter")
        self._mf = ccl.halos.MassFuncJenkins01(mass_def=mass_def)

    def mass_function(self, mass: Iterable[float] | np.ndarray, redshift: float) -> np.ndarray:
        """See base class."""
        mass = np.asarray(mass, dtype=float)
        a = get_scale_factor_from_redshift(redshift)
        # PyCCL returns dn/dlog10M (per dex). Convert to dn/dlnM (per natural log).
        dn_dlog10m = self._mf(cosmo=self.cosmo, M=mass, a=a)
        return dn_dlog10m / np.log(10.0)


class Tinker08(MassFunctionBase):
    """Tinker et al. (2008) mass function with 200m mass definition.

    Args:
        cosmo (ccl.Cosmology): Cosmology instance.
    """

    def __init__(self, cosmo: ccl.Cosmology):
        self.cosmo = cosmo
        self._mf = ccl.halos.hmfunc.tinker08.MassFuncTinker08(
            mass_def="200m", mass_def_strict=False
        )

    def mass_function(self, mass: Iterable[float] | np.ndarray, redshift: float) -> np.ndarray:
        """See base class."""
        mass = np.asarray(mass, dtype=float)
        a = get_scale_factor_from_redshift(redshift)
        dn_dlog10m = self._mf(cosmo=self.cosmo, M=mass, a=a)
        return dn_dlog10m / np.log(10.0)


class Tinker10(MassFunctionBase):
    """Tinker et al. (2010) mass function with 200m mass definition.

    Args:
        cosmo (ccl.Cosmology): Cosmology instance.
        norm_all_z (bool): If True, enforce normalization at all redshifts
            (matches PyCCL's ``MassFuncTinker10(norm_all_z=...)``).
    """

    def __init__(self, cosmo: ccl.Cosmology, norm_all_z: bool = False):
        self.cosmo = cosmo
        self._mf = ccl.halos.hmfunc.tinker10.MassFuncTinker10(
            mass_def="200m", mass_def_strict=False, norm_all_z=bool(norm_all_z)
        )

    def mass_function(self, mass: Iterable[float] | np.ndarray, redshift: float) -> np.ndarray:
        """See base class."""
        mass = np.asarray(mass, dtype=float)
        a = get_scale_factor_from_redshift(redshift)
        dn_dlog10m = self._mf(cosmo=self.cosmo, M=mass, a=a)
        return dn_dlog10m / np.log(10.0)


def get_mass_function(name: str, cosmo: ccl.Cosmology, **kwargs) -> MassFunctionBase:
    """Factory: build a halo mass function by name.

    Supported names:
        * "jenkins" → :class:`Jenkins`
        * "tinker08" → :class:`Tinker08`
        * "tinker10" → :class:`Tinker10`

    Args:
        name (str): Identifier of the mass function.
        cosmo (ccl.Cosmology): Cosmology instance.
        **kwargs: Passed to the specific mass-function constructor
            (e.g., ``norm_all_z`` for Tinker10).

    Returns:
        MassFunctionBase: Configured mass-function object.

    Raises:
        ValueError: If the name is not recognized.
    """
    key = name.strip().lower()
    if key == "jenkins":
        return Jenkins(cosmo)
    if key in {"tinker08"}:
        return Tinker08(cosmo)
    if key in {"tinker10"}:
        return Tinker10(cosmo, **kwargs)
    raise ValueError(f"unknown mass-function name: {name!r}")
