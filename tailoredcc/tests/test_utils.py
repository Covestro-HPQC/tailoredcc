# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import numpy as np
import pytest

from tailoredcc.utils import (
    spin_blocks_interleaved_to_sequential,
    spin_blocks_sequential_to_interleaved,
    spinorb_from_spatial,
)


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


@pytest.mark.parametrize("ndim", range(1, 6))
@pytest.mark.parametrize(
    "nalpha, nbeta",
    [
        (1, 1),
        (5, 5),
        (10, 10),
    ],
)
def test_spin_blocks_interleaved_to_sequential_even_dim(ndim, nalpha, nbeta):
    N = nalpha + nbeta
    a = np.arange(N**ndim).reshape(ndim * (N,))
    aseq = spin_blocks_interleaved_to_sequential(a)
    alpha_il = slice(0, None, 2)
    beta_il = slice(1, None, 2)

    alpha_seq = slice(0, nalpha, 1)
    beta_seq = slice(nalpha, None, 1)

    lookup = {
        "alpha": (alpha_il, alpha_seq),
        "beta": (beta_il, beta_seq),
    }
    from itertools import product

    for comb in product(["alpha", "beta"], repeat=ndim):
        slices = [lookup[c] for c in comb]
        il = []
        seq = []
        for s in slices:
            il.append(s[0])
            seq.append(s[1])
        interleaved = a[tuple(il)]
        sequential = aseq[tuple(seq)]
        np.testing.assert_equal(interleaved, sequential)

    n_per_dim = [{"alpha": nalpha, "beta": nbeta} for _ in range(ndim)]
    np.testing.assert_equal(a, spin_blocks_sequential_to_interleaved(aseq, n_per_dim))


def test_spin_blocks_interleaved_to_sequential_uneven_dim():
    nocc = 20
    nocca = nocc // 2
    noccb = nocca

    nvirt = 40
    nvirta = nvirt // 2
    nvirtb = nvirta

    N = nocc**2 * nvirt**2
    a = np.arange(N).reshape((nocc, nocc, nvirt, nvirt))
    aseq = spin_blocks_interleaved_to_sequential(a)
    alpha_il = slice(0, None, 2)
    beta_il = slice(1, None, 2)
    occalpha = slice(0, nocca, 1)
    occbeta = slice(nocca, None, 1)
    virtalpha = slice(0, nvirta, 1)
    virtbeta = slice(nvirta, None, 1)

    lookup = {
        "alpha": (alpha_il, occalpha, virtalpha),
        "beta": (beta_il, occbeta, virtbeta),
    }
    from itertools import product

    for comb in product(["alpha", "beta"], repeat=4):
        slices = [lookup[c] for c in comb]
        il = []
        seq = []
        for ix, s in enumerate(slices):
            il.append(s[0])
            if ix < 2:
                seq.append(s[1])
            else:
                seq.append(s[2])
        interleaved = a[tuple(il)]
        sequential = aseq[tuple(seq)]
        np.testing.assert_equal(interleaved, sequential)

    n_per_dim = [
        {"alpha": nocca, "beta": noccb},
        {"alpha": nocca, "beta": noccb},
        {"alpha": nvirta, "beta": nvirtb},
        {"alpha": nvirta, "beta": nvirtb},
    ]
    np.testing.assert_equal(a, spin_blocks_sequential_to_interleaved(aseq, n_per_dim))
