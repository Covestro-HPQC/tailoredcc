import pandas as pd
from pyscf import fci, gto, mcscf, scf

from tailoredcc.tailoredcc import tccsd_from_ci

data = []
for rnn in [1.0, 1.125, 1.25, 1.5, 1.75, 2.0, 2.25]:
    ncas = 6
    nelecas = 6
    mol = gto.M(
        atom=f"""
    N 0 0 0
    N 0 0 {rnn}
    """,
        basis="ccpvtz",
        verbose=0,
    )
    scfres = scf.RHF(mol)
    scfres.conv_tol = 1e-12
    scfres.conv_tol_grad = 1e-8
    scfres.kernel()

    # with CASSCF orbitals
    import h5py

    bl = float(rnn)
    file_path = "/fs/home/cvsik/Projects/tailored_cc/qcqmc-recirq/n2_ccpvtz"
    f = h5py.File(f"{file_path}/afqmc/scf_{bl}.chk", "r")
    mo = f["/mcscf/mo_coeff"][()]
    f.close()

    mci = mcscf.CASSCF(scfres, ncas, nelecas)
    mci.verbose = 4
    # mci = mcscf.CASCI(scfres, num_orb_cas, num_elec_cas)
    fci.addons.fix_spin_(mci.fcisolver, ss=0)  # Triplet, ss = S*(S+1)
    mci.kernel(mo_coeff=mo)
    print("c0 = ", mci.ci[0, 0])
    mo = mci.mo_coeff

    tcc_casscf = tccsd_from_ci(mci, backend="pyscf", mo_coeff=mo, triples_correction=True)
    assert tcc_casscf.converged

    # with HF orbitals
    mci = mcscf.CASCI(scfres, ncas, nelecas)
    mci.verbose = 3
    fci.addons.fix_spin_(mci.fcisolver, ss=0)  # Triplet, ss = S*(S+1)
    mci.kernel(mo_coeff=mo)
    print("c0 = ", mci.ci[0, 0])

    tcc_hf = tccsd_from_ci(mci, backend="pyscf", mo_coeff=mo, triples_correction=True)
    assert tcc_hf.converged
    data.append([rnn, tcc_casscf.e_tot, tcc_casscf.e_triples, tcc_hf.e_tot, tcc_hf.e_triples])

df = pd.DataFrame(
    data=data, columns=["rnn", "energy_tcc_casscf", "triples_casscf", "energy_tcc_hf", "triples_hf"]
)
df.to_hdf("n2_horror_exp.h5", key="df")
