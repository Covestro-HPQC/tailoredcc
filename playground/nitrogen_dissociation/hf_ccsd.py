#!/usr/bin/env python
import numpy as np
import pandas as pd
import pyscf
import pyscf.cc
import pyscf.ci
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
from pyscf import gto, lib, mrpt, scf

lib.num_threads(16)
np.random.seed(42)

data = []
data_vqe = []
params = {}
t1 = t2 = None

# mf = gto.M().apply(scf.RHF).as_scanner()
# mf.conv_tol = 1e-8
# mf.max_cycle = 500
# mf.verbose = 4

mos = None
mo_occ = None

for ii, d in enumerate(np.arange(0.8, 2.9, 0.1)):
    mol = pyscf.gto.Mole()
    mol.atom = "N 0 0 0; N 0 0 %f" % d
    mol.basis = "cc-pvdz"
    mol.verbose = 4
    mol.build()

    # Hartree Fock
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-8
    mf.max_cycle = 500
    mf.chkfile = f"mos/scf_{ii}.chk"

    dm1 = None
    if mos is not None:
        dm1 = mf.make_rdm1(mos, mo_occ)
    mf.kernel(dm1)
    assert mf.converged
    mos = mf.mo_coeff.copy()
    mo_occ = mf.mo_occ.copy()
    np.savez(f"mos/mo_{ii}", mos=mos, d=d, e_tot=mf.e_tot, mo_occ=mf.mo_occ)

    # CCSD
    cc = pyscf.cc.CCSD(mf)
    cc.max_cycle = 1000
    cc.iterative_damping = 0.8
    # cc.level_shift = 0.5
    cc.kernel(t1=t1, t2=t2)
    t1, t2 = cc.t1, cc.t2
    assert cc.converged
    e_ccsd = cc.e_tot if cc.converged else np.nan

    # distance, method, energy
    data.extend(
        [
            # [d, "HF", mf.e_tot],
            [d, e_ccsd],
        ]
    )

df1 = pd.DataFrame(data=data, columns=["d", "method", "energy"])
# df2 = pd.DataFrame(data=data_vqe, columns=["d", "method", "energy", "depth"])

# df = pd.concat([df1, df2], ignore_index=True)
# df.to_hdf("n2_dissociation.h5", key="df")
df1.to_hdf("n2_dissociation_ccsd.h5", key="df")
