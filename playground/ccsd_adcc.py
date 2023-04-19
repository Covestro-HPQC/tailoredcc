import adcc
import numpy as np
from pyscf import ao2mo, cc, gto, lib, mcscf, scf

# from tailoredcc.ccsd import ccsd_energy_correlation_adcc
from tailoredcc import tccsd_from_ci
from tailoredcc.solve_tcc import solve_tccsd

nthreads = 8
adcc.set_n_threads(nthreads)
lib.num_threads(nthreads)

conv_tol_e = 1e-12
conv_tol_g = 1e-8

# xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/n2_real.xyz"
xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/p-benzyne.xyz"
# nact = 4
# nalpha, nbeta = (2, 2)

# nact = 6
# nalpha, nbeta = (3, 3)

nact = 8
nalpha, nbeta = (5, 5)

ncas = nact
nvirta = ncas - nalpha
nvirtb = ncas - nbeta
nocca = nalpha
noccb = nbeta

assert nocca == noccb
# basis = "cc-pvdz"
# basis = "sto-3g"
basis = "6-31g"

mol = gto.M(atom=str(xyzfile), basis=basis, verbose=4)
# mol = gto.M(atom="""
# N 0 0 0
# N 0 0 3.5
# """, unit="bohr", basis=basis, verbose=4)
# mol = gto.M(atom="""
# N 0 0 0
# N 0 0 1.4
# """, basis=basis, verbose=4)
scfres = scf.RHF(mol)
scfres.conv_tol = conv_tol_e
scfres.conv_tol_grad = conv_tol_g
scfres.kernel()

hf = adcc.ReferenceState(scfres)
mp = adcc.LazyMp(hf)

# matrix = adcc.AdcMatrix("adc2", mp)
# diag = matrix.diagonal()
# print(diag.pphh.describe_symmetry())

print(hf.energy_scf - mol.energy_nuc())
t = solve_tccsd(mp, diis_size=7, backend="libcc")
t = solve_tccsd(mp, diis_size=7, backend="adcc")
# ecorr = ccsd_energy_correlation_adcc(mp, t)
# print(ecorr)
# ccsd = cc.CCSD(scfres).run()

# run pyscf CASCI and generate integrals to start with
mci = mcscf.CASCI(scfres, nact, (nalpha, nbeta))
mci.kernel()
ret_ci = tccsd_from_ci(mci, backend="libcc")
ret_ci = tccsd_from_ci(mci, backend="adcc")
ret_ci = tccsd_from_ci(mci, backend="oe")
# occslice = slice(2 * mci.ncore, 2 * mci.ncore + nocca + noccb)
# virtslice = slice(0, nvirta + nvirtb)

# t = solve_tccsd_adcc(mp, diis_size=7, occslice=occslice, virtslice=virtslice)
