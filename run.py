from pyscf import gto, scf, mcscf
from tailoredcc import tccsd_from_ci
import numpy as np

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

# ncas = 2
# nelec = 2

# ncas = 4
# nelec = 4

print(f"CAS({nelec}, {ncas})")
mc = mcscf.CASCI(m, ncas, nelec)
mc.kernel()
etccsd = tccsd_from_ci(mc)
np.testing.assert_allclose(etccsd, -14.41978908212513, atol=1e-9, rtol=0)

ncas = 4
nelec = 4
print(f"CAS({nelec}, {ncas})")
mc = mcscf.CASCI(m, ncas, nelec)
mc.kernel()
etccsd = tccsd_from_ci(mc)
np.testing.assert_allclose(etccsd, -14.43271127357399, atol=1e-9, rtol=0)