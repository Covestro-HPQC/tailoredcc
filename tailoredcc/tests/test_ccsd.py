# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import adcc
import numpy as np
import pytest
from adcc import block as b
from pyscf import cc, gto, scf

from tailoredcc import ccsd
from tailoredcc.tailoredcc import solve_tccsd
from tailoredcc.utils import (
    spin_blocks_interleaved_to_sequential,
    spin_blocks_sequential_to_interleaved,
)


@pytest.fixture(scope="module")
def scfres_mp():
    mol = gto.M(
        atom="""
C         25.72300       29.68500       26.90100
C         25.51000       29.88900       25.49000
C         26.07700       30.59500       27.82800
C         25.16300       28.92900       24.60400
H         25.62000       28.64500       27.18200
H         25.33300       30.87700       25.08600
H         26.26500       30.17800       28.85200
H         26.32100       31.69500       27.63100
H         24.81000       29.04000       23.56900
H         25.10400       27.88700       24.94600
    """,
        basis="6-31g",
    )
    scfres = scf.RHF(mol).run()
    hf = adcc.ReferenceState(scfres)
    mp = adcc.LazyMp(hf)
    return scfres, mp


@pytest.fixture(scope="module")
def ccsd_pyscf(scfres_mp):
    scfres, _ = scfres_mp
    ccsd = cc.CCSD(scfres)
    ccsd.verbose = 4
    ccsd.conv_tol = 1e-11
    ccsd.kernel()
    return ccsd


def test_residual_crossref(scfres_mp):
    scfres, mp = scfres_mp
    hf = mp.reference_state

    # set the Fock ov block randomly to check the eqs
    x = hf.fov.zeros_like()
    x.set_random()
    fov_orig = hf.fov.copy()
    fvo_orig = hf.fvo.copy()
    hf.fov = x
    hf.fvo = x.T

    t = adcc.AmplitudeVector(ov=mp.mp2_diffdm.ov, oovv=mp.t2oo)

    sres_adcc = ccsd.adcc.singles_residual(mp, t).to_ndarray()
    sres_libcc = ccsd.libcc.singles_residual(mp, t).to_ndarray()

    dres_adcc = ccsd.adcc.doubles_residual(mp, t).to_ndarray()
    dres_libcc = ccsd.libcc.doubles_residual(mp, t).to_ndarray()

    np.testing.assert_allclose(sres_libcc, sres_adcc, atol=1e-12, rtol=0)
    np.testing.assert_allclose(dres_libcc, dres_adcc, atol=1e-12, rtol=0)

    mol = scfres.mol
    nocc = sum(mol.nelec)
    o = slice(None, nocc)
    v = slice(nocc, None)

    nalpha_fun = hf.mospaces.n_orbs_alpha
    nbeta_fun = hf.mospaces.n_orbs_beta

    def n_per_dim(space):
        ret = []
        sp = [*space]
        if any(x in space for x in [b.o, b.v]):
            n = 2
            sp = [space[i : i + n] for i in range(0, len(space), n)]
        ret = [{"alpha": nalpha_fun(s), "beta": nbeta_fun(s)} for s in sp]
        return ret

    fock = spin_blocks_sequential_to_interleaved(hf.fock("ff").to_ndarray(), n_per_dim("ff"))

    fock[o, v] = spin_blocks_sequential_to_interleaved(hf.fov.to_ndarray(), n_per_dim(b.ov))
    fock[v, o] = spin_blocks_sequential_to_interleaved(hf.fvo.to_ndarray(), n_per_dim(b.vo))

    eri_phys_asymm = spin_blocks_sequential_to_interleaved(
        hf.eri("ffff").to_ndarray(), n_per_dim("ffff")
    )
    t1 = spin_blocks_sequential_to_interleaved(t.ov.to_ndarray(), n_per_dim(b.ov))
    t2 = spin_blocks_sequential_to_interleaved(t.oovv.to_ndarray(), n_per_dim(b.oovv))
    sres_np = ccsd.oe.singles_residual(t1.T, t2.transpose(2, 3, 0, 1), fock, eri_phys_asymm, o, v).T
    dres_np = ccsd.oe.doubles_residual(t1.T, t2.transpose(2, 3, 0, 1), fock, eri_phys_asymm, o, v)
    sres_np = spin_blocks_interleaved_to_sequential(sres_np)
    dres_np = spin_blocks_interleaved_to_sequential(dres_np).transpose(2, 3, 0, 1)
    np.testing.assert_allclose(sres_adcc, sres_np, atol=1e-12, rtol=0)
    np.testing.assert_allclose(dres_adcc, dres_np, atol=1e-12, rtol=0)

    hf.fov = fov_orig
    hf.fvo = fvo_orig


@pytest.mark.parametrize("backend", ["adcc", "libcc"])
def test_crossref_ccsd_energy(backend, scfres_mp, ccsd_pyscf):
    _, mp = scfres_mp
    adcc.set_n_threads(2)
    t = solve_tccsd(mp, stopping_eps=1e-11, backend=backend)
    e = getattr(ccsd, backend).ccsd_energy_correlation(mp, t)
    np.testing.assert_allclose(e, ccsd_pyscf.e_corr, atol=1e-8, rtol=0)
