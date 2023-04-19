# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import time

import adcc
import numpy as np
from pyscf import cc, gto, lib, mcscf, scf
from pyscf.cc.addons import spatial2spin, spin2spatial

from tailoredcc.amplitudes import (
    ci_to_cluster_amplitudes,
    extract_ci_singles_doubles_amplitudes_spinorb,
)
from tailoredcc.tailoredcc import tccsd_from_ci

# import os
# os.environ["OMP_NUM_THREADS"] = "1"

nthreads = 16
lib.num_threads(nthreads)
adcc.set_n_threads(nthreads)

# C         25.72300       29.68500       26.90100
# C         25.51000       29.88900       25.49000
# C         26.07700       30.59500       27.82800
# C         25.16300       28.92900       24.60400
# H         25.62000       28.64500       27.18200
# H         25.33300       30.87700       25.08600
# H         26.26500       30.17800       28.85200
# H         26.32100       31.69500       27.63100
# H         24.81000       29.04000       23.56900
# H         25.10400       27.88700       24.94600

mol = gto.M(
    atom="""
  H      1.2194     -0.1652      2.1600
  C      0.6825     -0.0924      1.2087
  C     -0.7075     -0.0352      1.1973
  H     -1.2644     -0.0630      2.1393
  C     -1.3898      0.0572     -0.0114
  H     -2.4836      0.1021     -0.0204
  C     -0.6824      0.0925     -1.2088
  H     -1.2194      0.1652     -2.1599
  C      0.7075      0.0352     -1.1973
  H      1.2641      0.0628     -2.1395
  C      1.3899     -0.0572      0.0114
  H      2.4836     -0.1022      0.0205
    """,
    # basis="sto-3g",
    # basis="6-31g",
    # basis="cc-pvdz",
    # basis="aug-cc-pvdz",
    basis="cc-pvtz",
    max_memory=64000,
    verbose=4,
)
scfres = scf.RHF(mol).run()

ccsd = cc.CCSD(scfres)  # .density_fit(auxbasis="aug-cc-pvdz-ri")
ccsd.verbose = 4
ccsd.conv_tol = 1e-8
update_amps = ccsd.update_amps

mc = mcscf.CASCI(scfres, nelecas=(3, 3), ncas=6)
_ = mc.kernel()
nocca, noccb = mc.nelecas
assert nocca == noccb
nvirta = mc.ncas - nocca
nvirtb = mc.ncas - noccb
assert nvirta == nvirtb

c_ia, c_ijab = extract_ci_singles_doubles_amplitudes_spinorb(mc)
t1cas, t2cas = ci_to_cluster_amplitudes(c_ia, c_ijab)
occslice = slice(2 * mc.ncore, 2 * mc.ncore + nocca + noccb)
virtslice = slice(0, nvirta + nvirtb)


def update_freeze_amplitudes_cas(t1, t2, eris):
    t1new, t2new = update_amps(t1, t2, eris)
    print("T_cas amplitudes")
    t1spin = spatial2spin(t1new)
    t2spin = spatial2spin(t2new)

    t1spin[occslice, virtslice] = t1cas
    t2spin[occslice, occslice, virtslice, virtslice] = t2cas

    if isinstance(t1new, np.ndarray) and t1new.ndim == 2:
        nocca, nvirta = t1.shape
        orbspin = np.zeros((nocca + nvirta) * 2, dtype=int)
        orbspin[1::2] = 1
        t1spatial = spin2spatial(t1spin, orbspin)[0]  # take t1a
        t2spatial = spin2spatial(t2spin, orbspin)[1]  # take t2ab
    elif isinstance(t1new, tuple) and len(t1new) == 2:
        nocca, nvirta = t1new[0].shape
        orbspin = np.zeros((nocca + nvirta) * 2, dtype=int)
        orbspin[1::2] = 1
        t1spatial = spin2spatial(t1spin, orbspin)
        t2spatial = spin2spatial(t2spin, orbspin)
    else:
        raise NotImplementedError("Unknown amplitude types.")

    t1new = t1spatial
    t2new = t2spatial

    return t1new, t2new


ccsd.update_amps = update_freeze_amplitudes_cas

start = time.time()
energy = ccsd.kernel()[0]
print(time.time() - start)

t1spin = spatial2spin(ccsd.t1)
t2spin = spatial2spin(ccsd.t2)
np.testing.assert_allclose(t1spin[occslice, virtslice], t1cas, atol=1e-13, rtol=0)
np.testing.assert_allclose(
    t2spin[occslice, occslice, virtslice, virtslice], t2cas, atol=1e-13, rtol=0
)

start = time.time()
tcc = tccsd_from_ci(mc)
print(time.time() - start)
print(tcc.e_tot - ccsd.e_tot)
