import numpy as np
from ase.build import bulk
from pyscf import cc, lib, mcscf
from pyscf.pbc import gto, scf, tools
from pyscf.pbc.tools import pyscf_ase

lib.num_threads(16)

A2B = 1.889725989

# R = 2.880

# ase_atom = bulk('C', 'diamond', a=R)

# atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)

# cell = gto.Cell(
#     atom=atom,
#     basis="gth-dzvp",
#     pseudo="gth-pade",
#     a=ase_atom.cell,
#     verbose=4,
#     max_memory=32000,
# )

R = 0.9
alat0 = 3.6 * R
lat = alat0
cell = gto.Cell()
cell.verbose = 0
cell.a = (np.ones((3, 3)) - np.eye(3)) * lat / 2.0
cell.atom = (("C", 0, 0, 0), ("C", np.array([0.25, 0.25, 0.25]) * lat))
cell.basis = "gth-dzvp"
cell.pseudo = "gth-pade"
cell.mesh = [25, 25, 25]
cell.verbose = 4
cell.build()

scfres = scf.RHF(cell)
# scfres.conv_tol = 1e-8
# scfres.kernel()
scfres.run(exxdiv=None)

madelung = tools.madelung(cell, [0.0, 0.0, 0.0])
nalpha, nbeta = cell.nelec
e_mad = -(nalpha + nbeta) / 2.0 * madelung

# mc = mcscf.CASCI(scfres, ncas=8, nelecas=8)
# h1e, ecore = mc.get_h1eff()
# mc.kernel()
# print(mc.e_tot + e_mad)

ccsd = cc.CCSD(scfres)
ccsd.kernel()

e_t = ccsd.ccsd_t()

e_ccsdt = ccsd.e_tot + e_t + e_mad

print(f"R = {R:.4f}, CCSD(T) energy = {e_ccsdt:.12f}")

print("diff", e_ccsdt - -9.546464)

# R(A) Exact      CCSD(T)    Quantum trial     AFQMC       QC-AFQMC
# 2.880 -9.545911 -9.546464   -9.121081          -9.5415(1) -9.54582(5)
# 3.240 -10.229155 -10.230100 -8.625292         -10.2241(3) -10.23051(7)
# 3.600 -10.560477 -10.562229 -10.277938        -10.5525(2) -10.55861(8)
# 3.960 -10.700421 -10.703884 -10.368882        -10.6869(2) -10.6949(1)
# 4.320 -10.744089 -10.751103 -10.222206        -10.7177(3) -10.73701(9)
