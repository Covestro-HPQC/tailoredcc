# Proprietary and Confidential
# Covestro Deutschland AG, 2023
from ._version import __version__
from .tailoredcc import (
    _solve_tccsd_oe,
    ec_cc_from_ci,
    tccsd,
    tccsd_from_ci,
    tccsd_from_fqe,
    tccsd_from_vqe,
)

__all__ = [
    "ec_cc_from_ci",
    "tccsd_from_ci",
    "tccsd",
    "_solve_tccsd_oe",
    "tccsd_from_vqe",
    "tccsd_from_fqe",
    "__version__",
]
