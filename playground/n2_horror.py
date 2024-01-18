import pandas as pd
from pyscf import fci, gto, mcscf, scf

from tailoredcc.tailoredcc import ec_cc_from_ci, tccsd_from_ci

data = []
for rnn in [1.0, 1.125, 1.25, 1.5, 1.75, 2.0, 2.25]:
    ncas = 6
    nelecas = 6
    mol = gto.M(
        atom=f"""
    N 0 0 0
    N 0 0 {rnn}
    """,
        basis="ccpvdz",
        verbose=0,
    )
    scfres = scf.RHF(mol)
    scfres.conv_tol = 1e-12
    scfres.conv_tol_grad = 1e-8
    scfres.kernel()

    # with CASSCF orbitals
    mci = mcscf.CASSCF(scfres, ncas, nelecas)
    mci.verbose = 3
    # mci = mcscf.CASCI(scfres, num_orb_cas, num_elec_cas)
    fci.addons.fix_spin_(mci.fcisolver, ss=0)  # Triplet, ss = S*(S+1)
    mci.kernel()
    print("c0 = ", mci.ci[0, 0])
    mo = mci.mo_coeff

    tcc_casscf = tccsd_from_ci(mci, backend="pyscf", mo_coeff=mo, triples_correction=True)
    assert tcc_casscf.converged

    ec_cas = ec_cc_from_ci(
        mci, mo_coeff=mo, diis_size=15, diis_start_cycle=5, maxiter=50, conv_tol=1e-5
    )

    # with HF orbitals
    mci = mcscf.CASCI(scfres, ncas, nelecas)
    mci.verbose = 3
    fci.addons.fix_spin_(mci.fcisolver, ss=0)  # Triplet, ss = S*(S+1)
    mci.kernel()
    print("c0 = ", mci.ci[0, 0])

    tcc_hf = tccsd_from_ci(mci, backend="pyscf", triples_correction=True)
    assert tcc_hf.converged

    ec_hf = ec_cc_from_ci(
        mci, mo_coeff=mo, diis_size=15, diis_start_cycle=5, maxiter=50, conv_tol=1e-5
    )
    data.append(
        [
            rnn,
            tcc_casscf.e_tot,
            tcc_casscf.e_triples,
            tcc_hf.e_tot,
            tcc_hf.e_triples,
            ec_cas.e_tot,
            ec_hf.e_tot,
        ]
    )

df = pd.DataFrame(
    data=data,
    columns=[
        "rnn",
        "energy_tcc_casscf",
        "triples_casscf",
        "energy_tcc_hf",
        "triples_hf",
        "energy_ec_cas",
        "energy_ec_hf",
    ],
)
df.to_hdf("n2_horror.h5", key="df")
