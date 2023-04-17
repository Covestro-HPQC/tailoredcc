# Proprietary and Confidential
# Covestro Deutschland AG, 2023

from dataclasses import dataclass

from . import equations_adcc, equations_oe


@dataclass
class CCSDDispatch:
    ccsd_energy: callable
    ccsd_energy_correlation: callable
    singles_residual: callable
    doubles_residual: callable


DISPATCH = {
    "oe": CCSDDispatch(
        equations_oe.ccsd_energy_oe,
        equations_oe.ccsd_energy_correlation_oe,
        equations_oe.singles_residual_oe,
        equations_oe.doubles_residual_oe,
    ),
    "adcc": CCSDDispatch(
        equations_adcc.ccsd_energy_adcc,
        equations_adcc.ccsd_energy_correlation_adcc,
        equations_adcc.singles_residual_adcc,
        equations_adcc.doubles_residual_adcc,
    ),
    "libcc": CCSDDispatch(
        equations_adcc.ccsd_energy_adcc,
        equations_adcc.ccsd_energy_correlation_adcc,
        equations_adcc.singles_residual_libcc,
        equations_adcc.doubles_residual_libcc,
    ),
}


oe = DISPATCH["oe"]
adcc = DISPATCH["adcc"]
libcc = DISPATCH["libcc"]
