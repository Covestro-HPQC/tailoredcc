# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import numpy as np
import pytest
import scipy
from pyscf import gto, mcscf, scf

from tailoredcc import ec_cc_from_ci, tccsd_from_ci


@pytest.fixture(scope="module")
def scf_ci():
    mol = gto.Mole()
    mol.build(
        verbose=4,
        # atom="N, 0., 0., 0. ; N,  0., 0., 1.4",
        # atom="H, 0., 0., 0. ; H,  0., 0., 1.0",
        # atom="He, 0., 0., 0. ; He,  0., 0., 1.0",
        atom="Li, 0., 0., 0. ; Li,  0., 0., 1.0",
        # basis="minao",
        basis="sto-3g",
        # basis="3-21g",
        # basis="6-31g",
        # basis="cc-pvdz",
        # symmetry = True,
    )
    m = scf.RHF(mol)
    m.kernel()

    ncas = mol.nao_nr()
    nelec = mol.nelec

    print(f"CAS({nelec}, {ncas})")
    mc = mcscf.CASCI(m, ncas, nelec)
    mc.fcisolver.conv_tol = 1e-12
    mc.canonicalization = False
    mc.kernel()

    ncas = 4
    nelec = 4
    print(f"CAS({nelec}, {ncas})")
    mc2 = mcscf.CASCI(m, ncas, nelec)
    mc2.canonicalization = False
    mc2.kernel()

    return m, mc, mc2


@pytest.mark.parametrize("backend", ["pyscf", "oe"])
def test_cas_energy_crossref(backend, scf_ci):
    from tailoredcc import tccsd_from_ci

    m, mc, mc2 = scf_ci
    tcc = tccsd_from_ci(mc, backend=backend)
    np.testing.assert_allclose(tcc.e_cas, mc.e_tot - m.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(tcc.e_tot, mc.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(tcc.e_tot, -14.41978908212513, atol=1e-9, rtol=0)

    tcc = tccsd_from_ci(mc2, backend=backend)
    np.testing.assert_allclose(tcc.e_cas, mc2.e_tot - m.e_tot, atol=1e-9, rtol=0)


def test_crossref_random_orbitals():
    mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="cc-pvdz", verbose=4)
    scfres = scf.RHF(mol)
    scfres.kernel()
    mo_coeff = scfres.mo_coeff.copy()

    ncas, nelecas = (6, 6)

    np.random.seed(42)
    nmo = mo_coeff.shape[1]

    mo_coeff_orig = mo_coeff.copy()
    for _ in range(5):
        Xpq = 1e-2 * np.random.randn(nmo, nmo)
        Xpq = 0.5 * (Xpq - Xpq.T)
        Upq = scipy.linalg.expm(Xpq)
        # randomly rotate orbitals
        mo_coeff = mo_coeff_orig @ Upq

        # check that mo_coeff are 'different enough' from HF orbitals
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(np.abs(scfres.mo_coeff), np.abs(mo_coeff), atol=1e-2, rtol=0)

        mc = mcscf.CASCI(scfres, ncas, nelecas).run(mo_coeff)
        dm1 = scfres.make_rdm1(mo_coeff)
        e_re = scfres.energy_tot(dm1)

        tccp = tccsd_from_ci(mc, backend="pyscf", maxiter=1, mo_coeff=mo_coeff)
        np.testing.assert_allclose(tccp.e_hf, e_re, atol=1e-12, rtol=0)
        np.testing.assert_allclose(tccp.e_cas, mc.e_tot - tccp.e_hf, atol=5e-12, rtol=0)

        tcco = tccsd_from_ci(mc, backend="oe", maxiter=1, mo_coeff=mo_coeff)
        np.testing.assert_allclose(tcco.e_hf, e_re, atol=1e-12, rtol=0)
        np.testing.assert_allclose(tcco.e_cas, mc.e_tot - tcco.e_hf, atol=5e-12, rtol=0)

        # cross-ref test both backends
        tccp = tccsd_from_ci(mc, backend="pyscf", maxiter=100, mo_coeff=mo_coeff, conv_tol=1e-14)
        tcco = tccsd_from_ci(mc, backend="oe", maxiter=100, mo_coeff=mo_coeff, conv_tol=1e-14)
        np.testing.assert_allclose(tccp.e_tot, tcco.e_tot, atol=1e-10, rtol=0)

    # from pyscf import mp
    # mp2 = mp.UMP2(scfres).run()
    # _, mo_coeff = mcscf.addons.make_natural_orbitals(mp2)

    # from pyscf.mcscf import avas
    # ncas, nelecas, mo_coeff = avas.avas(scfres, ['O 2s', 'H 1s', 'O 2p'])
    # mc = mcscf.CASSCF(scfres, nelecas=nelecas, ncas=ncas)
    # mc = mcscf.CASCI(scfres, nelecas=nelecas, ncas=ncas).run()
    # mc.canonicalization = False
    # mc.natorb = False
    # mc.conv_tol = 1e-8
    # mc.conv_tol_grad = 1e-5
    # mc.fcisolver.conv_tol = 1e-12
    # mc.kernel()
    # civec = mc.ci.copy()
    # mo_coeff = mc.mo_coeff

    # re-run CASCI with final orbitals, I thought this
    # was already done in pyscf?!
    # mc2 = mcscf.CASCI(scfres, nelecas=nelecas, ncas=ncas)
    # mc2.fcisolver.conv_tol = 1e-12
    # mc2.kernel(mo_coeff)
    # np.testing.assert_allclose(mc.e_tot, mc2.e_tot, atol=1e-10, rtol=0)
    # mc = mc2


def test_tccsd_with_triples_correction(scf_ci):
    m, mc, mc2 = scf_ci
    tcc = tccsd_from_ci(mc, backend="pyscf", triples_correction=True)
    np.testing.assert_allclose(tcc.e_cas, mc.e_tot - m.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(tcc.e_tot, -14.41978908212513, atol=1e-9, rtol=0)
    # no external amplitudes, so no triples correction
    assert tcc.e_triples == 0.0

    tcc = tccsd_from_ci(mc2, backend="pyscf", triples_correction=True)
    np.testing.assert_allclose(tcc.e_cas, mc2.e_tot - m.e_tot, atol=1e-9, rtol=0)
    assert np.abs(tcc.e_triples) > 1e-8

    tcc_oe = tccsd_from_ci(mc2, backend="oe", triples_correction=True)
    np.testing.assert_allclose(
        tcc.e_tot + tcc.e_triples, tcc_oe.e_tot + tcc_oe.e_triples, atol=1e-8, rtol=0
    )


def test_ec_cc_against_fci():
    mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="3-21g")
    scfres = scf.RHF(mol)
    scfres.kernel()

    mc = mcscf.CASCI(scfres, nelecas=mol.nelec, ncas=mol.nao_nr())
    mc.fcisolver.conv_tol = 1e-12
    mc.kernel()

    ret = ec_cc_from_ci(mc, conv_tol=1e-14, guess_t1_t2_from_ci=False)
    print(ret.e_tot, ret.e_tot - mc.e_tot)
    np.testing.assert_allclose(ret.e_tot, mc.e_tot, atol=1e-9, rtol=0)

    # check that the code runs with a normal CASCI
    mc2 = mcscf.CASCI(scfres, nelecas=8, ncas=8)
    mc2.fcisolver.conv_tol = 1e-12
    mc2.canonicalization = False
    mc2.kernel()
    ret = ec_cc_from_ci(mc2, conv_tol=1e-14, guess_t1_t2_from_ci=True, static_t3_t4=True)
    print(ret.e_tot, ret.e_tot - mc.e_tot)


def test_from_fqe_wfn():
    mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="3-21g")
    scfres = scf.RHF(mol).run()

    ncas = 6
    nelec = (3, 3)
    mc = mcscf.CASCI(scfres, nelecas=nelec, ncas=ncas)
    mc.canonicalization = False
    mc.kernel()

    from tailoredcc.tailoredcc import ec_cc_from_fqe, tccsd_from_fqe
    from tailoredcc.utils import pyscf_to_fqe_wf

    wfn = pyscf_to_fqe_wf(mc.ci, norbs=ncas, nelec=nelec)

    ret_fqe = tccsd_from_fqe(scfres, wfn)
    ret_ci = tccsd_from_ci(mc)

    np.testing.assert_allclose(ret_fqe.e_tot, ret_ci.e_tot, atol=1e-9, rtol=0)

    # ec-CC
    ret_fqe = ec_cc_from_fqe(scfres, wfn, static_t3_t4=True)
    ret_ci = ec_cc_from_ci(mc, static_t3_t4=True)

    np.testing.assert_allclose(ret_fqe.e_tot, ret_ci.e_tot, atol=1e-9, rtol=0)
