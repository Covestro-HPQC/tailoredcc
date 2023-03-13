# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import pytest
import numpy as np

from pyscf.fci import cistring
from pyscf.ci.cisd import tn_addrs_signs

from tailoredcc.amplitudes import detstrings_singles, detstrings_doubles


@pytest.mark.parametrize(
    "nocc, ncas",
    [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 4),
        (4, 4),
        (2, 6),
        (4, 6),
        (7, 10),
        (10, 25),
    ],
)
def test_determinant_string_generation(nocc, ncas):
    nvirt = ncas - nocc
    t1addrs, _ = tn_addrs_signs(ncas, nocc, 1)
    t2addrs, _ = tn_addrs_signs(ncas, nocc, 2)

    if nocc == 0 or ncas == 1:
        assert len(t1addrs) == 0
        assert len(t2addrs) == 0

    if len(t1addrs):
        detstrings_ref = [bin(ds) for ds in cistring.addrs2str(ncas, nocc, t1addrs.ravel())]
        detstrings, detstrings_np = detstrings_singles(nocc, nvirt)
        assert detstrings == detstrings_ref
        assert np.sum(detstrings_np) == nocc * len(detstrings)

    if len(t2addrs):
        detstrings_ref = [bin(ds) for ds in cistring.addrs2str(ncas, nocc, t2addrs.ravel())]
        detstrings, detstrings_np = detstrings_doubles(nocc, nvirt)
        assert detstrings == detstrings_ref
        assert np.sum(detstrings_np) == nocc * len(detstrings)
