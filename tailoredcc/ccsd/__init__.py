# Proprietary and Confidential
# Covestro Deutschland AG, 2023

from dataclasses import dataclass
from typing import Callable

from . import equations_oe


@dataclass
class CCSDDispatch:
    ccsd_energy: Callable
    ccsd_energy_correlation: Callable
    singles_residual: Callable
    doubles_residual: Callable


DISPATCH = {
    "oe": CCSDDispatch(
        equations_oe.ccsd_energy_oe,
        equations_oe.ccsd_energy_correlation_oe,
        equations_oe.singles_residual_oe,
        equations_oe.doubles_residual_oe,
    ),
}


oe = DISPATCH["oe"]
