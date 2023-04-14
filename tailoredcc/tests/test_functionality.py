# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import numpy as np
import pytest
from pyscf import gto, mcscf, scf


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
    mc.kernel()

    ncas = 4
    nelec = 4
    print(f"CAS({nelec}, {ncas})")
    mc2 = mcscf.CASCI(m, ncas, nelec)
    mc2.kernel()

    return m, mc, mc2


@pytest.mark.parametrize("backend", ["adcc", "opt_einsum"])
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
