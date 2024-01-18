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

from tailoredcc import tccsd_from_ci

lib.num_threads(24)
np.random.seed(42)

data = []
data_vqe = []
params = {}
t1 = t2 = None
t1ec = t2ec = None


for ii, d in enumerate(np.arange(0.8, 2.9, 0.1)):
    print(ii, d)
    mol, scf_dict = scf.chkfile.load_scf(f"mos/scf_{ii}.chk")
    mf = scf.RHF(mol)
    mf.__dict__.update(scf_dict)
    dx = np.load(f"mos/mo_{ii}.npz")
    mos = dx["mos"]
    mo_occ = dx["mo_occ"]
    assert d == dx["d"]
    print(mos.shape[1])

    np.testing.assert_allclose(mf.e_tot, dx["e_tot"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(mf.mo_coeff, mos, atol=1e-14, rtol=0)

    ncas = 6
    nalpha, nbeta = 3, 3

    cas = pyscf.mcscf.CASCI(mf, nelecas=(nalpha, nbeta), ncas=ncas)
    cas.canonicalization = False
    cas.kernel()
    # print(cas.ci.size)
    # exit(0)
    assert cas.converged
    np.testing.assert_allclose(mf.mo_coeff, cas.mo_coeff, atol=1e-14, rtol=0)

    e_pt2 = mrpt.NEVPT(cas).kernel()

    tcc = tccsd_from_ci(cas, backend="pyscf", maxiter=200)
    assert tcc.converged

    # CCSD
    cc = pyscf.cc.CCSD(mf)
    cc.max_cycle = 1000
    cc.iterative_damping = 0.8
    # cc.level_shift = 0.5
    cc.kernel(t1=t1, t2=t2)
    t1, t2 = cc.t1, cc.t2
    assert cc.converged

    t1diag = cc.get_t1_diagnostic()

    # ec_cc = ec_cc_from_ci(
    #     cas, t1_guess=t1ec, t2_guess=t2ec, maxiter=200,
    #     guess_t1_t2_from_ci=True,
    #     # diis_size=30,
    #     iterative_damping=0.5,
    #     conv_tol=1e-7, level_shift=0.5, static_t3_t4=True)
    # t1ec = ec_cc.t1
    # t2ec = ec_cc.t2

    # distance, method, energy
    # data.extend([
    #     [d, "HF", mf.e_tot],
    #     # [d, "CCSD", e_ccsd],
    #     [d, "CASCI", cas.e_tot],
    #     [d, "NEVPT2", cas.e_tot + e_pt2],
    #     [d, "TCCSD", tcc.e_tot],
    #     # [d, "ec-CC", float(ec_cc.e_tot)],
    # ])
    data.append(
        [
            d,
            mf.e_tot,
            cc.e_tot,
            cas.e_tot,
            cas.e_tot + e_pt2,
            tcc.e_tot,
            tcc.e_cas,
            tcc.e_corr,
            t1diag,
        ]
    )
    np.testing.assert_allclose(cas.e_tot - mf.e_tot, tcc.e_cas, atol=1e-10, rtol=0)

    np.testing.assert_allclose(mf.mo_coeff, cas.mo_coeff, atol=1e-14, rtol=0)


# df = pd.DataFrame(data=data, columns=["d", "method", "energy"])
df = pd.DataFrame(
    data=data,
    columns=[
        "d",
        "HF",
        "CCSD",
        "CASCI",
        "NEVPT2",
        "TCCSD",
        "tcc_e_cas_exact",
        "tcc_e_corr_exact",
        "t1diag",
    ],
)
df.to_hdf("n2_dissociation_tcc.h5", key="df")
