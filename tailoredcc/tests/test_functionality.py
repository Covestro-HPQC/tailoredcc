# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import numpy as np
import pytest
from pyscf import ao2mo, gto, mcscf, scf

from tailoredcc import tccsd_from_ci, tccsd_from_vqe


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
    mc.kernel()

    ncas = 4
    nelec = 4
    print(f"CAS({nelec}, {ncas})")
    mc2 = mcscf.CASCI(m, ncas, nelec)
    mc2.kernel()

    return m, mc, mc2


@pytest.mark.parametrize("backend", ["pyscf", "libcc", "adcc", "oe"])
def test_cas_energy_crossref(backend, scf_ci):
    from tailoredcc import tccsd_from_ci

    m, mc, mc2 = scf_ci
    tcc = tccsd_from_ci(mc, backend=backend)
    np.testing.assert_allclose(tcc.e_cas, mc.e_tot - m.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(tcc.e_tot, -14.41978908212513, atol=1e-9, rtol=0)

    tcc = tccsd_from_ci(mc2, backend=backend)
    np.testing.assert_allclose(tcc.e_cas, mc2.e_tot - m.e_tot, atol=1e-9, rtol=0)

    # TODO: not 100% correct...
    # print(f"CASSCF({nelec}, {ncas})")
    # mc = mcscf.CASSCF(m, ncas, nelec)
    # mc.conv_tol = 1e-10
    # mc.conv_tol_grad = 1e-7
    # mc.kernel()
    # tcc = tccsd_from_ci(mc)
    # np.testing.assert_allclose(
    #     tcc.e_cas + tcc.e_hf, mc.e_tot, atol=1e-9, rtol=0
    # )


@pytest.fixture(scope="module")
def scf_casci_vqe():
    import covvqetools as cov
    from covvqetools.instant_vqes import QNPVQE

    conv_tol_e = 1e-12
    conv_tol_g = 1e-8
    nact = 4
    nalpha, nbeta = (2, 2)
    depth = 16
    maxiter_vqe = 5000

    nocca = nalpha
    noccb = nbeta

    assert nocca == noccb
    basis = "6-31g"
    mol = gto.M(
        atom="""
    N 0 0 0
    N 0 0 3.5
    """,
        unit="bohr",
        basis=basis,
        verbose=4,
    )
    scfres = scf.RHF(mol)
    scfres.conv_tol = conv_tol_e
    scfres.conv_tol_grad = conv_tol_g
    scfres.kernel()

    # run pyscf CASCI and generate integrals to start with
    mci = mcscf.CASCI(scfres, nact, (nalpha, nbeta))
    mci.kernel()
    h1, ecore = mci.get_h1eff()
    h2 = ao2mo.restore(1, mci.ao2mo(), nact)
    two_body_integrals = h2.transpose(0, 2, 3, 1)

    # run QNPVQE with fixed integrals
    vqe = QNPVQE(
        depth=depth,
        nact=nact,
        nalpha=nalpha,
        nbeta=nbeta,
        # NOTE: trick to get nocc correct even though we don't have nuclear charges
        nocc=mci.ncore,
        nchar=-sum(mol.atom_charges()),
        ###
        measurement_method=cov.CASBOX,
        core_energy=ecore,
        one_body_integrals=h1,
        two_body_integrals=two_body_integrals,
    )
    np.random.seed(42)
    vqe.params += np.random.randn(*vqe.params.shape) * 1e-1
    opt = cov.LBFGSB(atol=None, gtol=1e-10, ftol=None)
    vqe.vqe_optimize(opt=opt, maxiter=maxiter_vqe)
    # opt_params = np.array(vqe.params)
    # print(opt_params.tolist())
    return scfres, mci, vqe


@pytest.mark.parametrize("backend", ["pyscf", "libcc", "adcc", "oe"])
def test_tccsd_from_vqe_crossref(backend, scf_casci_vqe):
    scfres, mci, vqe = scf_casci_vqe
    energy_vqe = vqe.vqe_energy(vqe.params)
    np.testing.assert_allclose(energy_vqe, mci.e_tot, atol=1e-9, rtol=0)
    ret_ci = tccsd_from_ci(mci, backend=backend)
    ret_vqe = tccsd_from_vqe(scfres, vqe, backend=backend)

    np.testing.assert_allclose(ret_ci.e_cas, ret_vqe.e_cas, atol=1e-9, rtol=0)
    np.testing.assert_allclose(ret_ci.e_tot, ret_vqe.e_tot, atol=1e-9, rtol=0)


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
