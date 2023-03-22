# Proprietary and Confidential
# Covestro Deutschland AG, 2023


def test_functionality():
    # TODO: refactor test...
    import numpy as np
    from pyscf import gto, mcscf, scf

    from tailoredcc import tccsd_from_ci

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
    tcc = tccsd_from_ci(mc)
    np.testing.assert_allclose(tcc.e_cas, mc.e_tot - m.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(tcc.e_tot, -14.41978908212513, atol=1e-9, rtol=0)

    ncas = 4
    nelec = 4
    print(f"CAS({nelec}, {ncas})")
    mc = mcscf.CASCI(m, ncas, nelec)
    mc.kernel()
    tcc = tccsd_from_ci(mc)
    np.testing.assert_allclose(tcc.e_cas, mc.e_tot - m.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(tcc.e_tot, -14.43271127357399, atol=1e-9, rtol=0)

    print(f"CASSCF({nelec}, {ncas})")
    mc = mcscf.CASSCF(m, ncas, nelec)
    mc.conv_tol = 1e-10
    mc.conv_tol_grad = 1e-7
    mc.kernel()
    tcc = tccsd_from_ci(mc)
    np.testing.assert_allclose(
        tcc.e_cas + tcc.e_hf - mol.energy_nuc(), mc.e_tot - mol.energy_nuc(), atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(tcc.e_tot, -14.437612867176817, atol=5e-8, rtol=0)
