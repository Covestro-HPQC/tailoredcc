import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.ci.cisd import tn_addrs_signs

from tailoredcc.amplitudes import (
    amplitudes_to_spinorb,
    detstrings_doubles,
    detstrings_singles,
)
from tailoredcc.tailoredcc import tccsd, tccsd_from_ci, tccsd_from_vqe

conv_tol_e = 1e-12
conv_tol_g = 1e-8

xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/n2_real.xyz"
# xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/p-benzyne.xyz"
# nact = 4
# nalpha, nbeta = (2, 2)

nact = 6
nalpha, nbeta = (3, 3)

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
scfres = scf.RHF(mol)
scfres.conv_tol = conv_tol_e
scfres.conv_tol_grad = conv_tol_g
scfres.kernel()

# run pyscf CASCI and generate integrals to start with
mci = mcscf.CASCI(scfres, nact, (nalpha, nbeta))
mci.kernel()
ret_ci = tccsd_from_ci(mci)
