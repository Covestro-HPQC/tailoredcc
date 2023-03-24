# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import numpy as np
import pytest

from tailoredcc.utils import spinorb_from_spatial


@pytest.mark.parametrize("nsp", [5, 10, 20, 30])
def test_spinorb_from_spatial(nsp):
    np.random.seed(42)
    oei = np.random.randn(nsp, nsp)
    oei = 0.5 * (oei + oei.T)
    tei = np.random.randn(nsp, nsp, nsp, nsp)
    oeis, teis = spinorb_from_spatial(oei, tei)

    from openfermion.chem.molecular_data import spinorb_from_spatial as spinorb_of

    ref_oei, ref_tei = spinorb_of(oei, tei)

    # NOTE: OF has a zero cutoff of 1e-8
    np.testing.assert_allclose(oeis, ref_oei, atol=1e-8, rtol=0)
    np.testing.assert_allclose(teis, ref_tei, atol=1e-8, rtol=0)
