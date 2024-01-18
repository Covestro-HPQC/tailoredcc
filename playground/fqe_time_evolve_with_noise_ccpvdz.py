from typing import Union

import fqe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from pyscf import ao2mo, cc, gto, mcscf, mrpt, scf
from pyscf.fci import addons
from tqdm import tqdm

from tailoredcc.amplitudes import (
    add_gaussian_noise,
    amplitudes_to_spinorb,
    assert_spinorb_antisymmetric,
    check_amplitudes_spinorb,
    ci_to_cluster_amplitudes,
    compute_parity,
    detstrings_doubles,
    detstrings_singles,
    extract_ci_amplitudes,
    number_overlaps_eccc,
    prepare_cas_slices,
    remove_index_restriction_doubles,
    spatial_to_spinorb,
    spinorb_to_spatial,
)
from tailoredcc.tailoredcc import (
    ec_cc,
    ec_cc_from_ci,
    ec_cc_from_fqe,
    tccsd_from_ci,
    tccsd_from_fqe,
)
from tailoredcc.utils import fqe_to_fake_ci

if __name__ == "__main__":
    mol = gto.Mole()
    mol.basis = "cc-pvdz"
    mol.atom = """H 0 0 0
    H 0 0 1.23
    H 0 1.23 0
    H 0 1.23 1.23"""
    mol.verbose = 3
    mol.spin = 0
    mol.build()
    mol.symmetry = False
    mf = scf.RHF(mol)
    mf.chkfile = "scf.chk"
    ehf = mf.kernel()
    mo1 = mf.stability(verbose=0)[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability(verbose=0)
    mf_e_tot = mf.e_tot

    Na, Nb = 2, 2
    sz = Na - Nb
    norb = 4
    print(Na + Nb, sz, norb)

    myci = mcscf.CASCI(mf_or_mol=mf, ncas=norb, nelecas=(Na, Nb))
    myci.canonicalization = False
    myci.fcisolver.nroots = 5
    myci.verbose = 3
    addons.fix_spin_(myci.fcisolver, ss=0)
    myci.kernel()
    ept2 = mrpt.NEVPT(myci, root=0).kernel()
    print(f"CASCI Energy {myci.e_tot}")
    casci_e_tot = myci.e_tot[0]
    myci.ci = myci.ci[0]

    fci = mcscf.CASCI(mf, ncas=mol.nao_nr(), nelecas=mol.nelec, ncore=0)
    fci.fcisolver.nroots = 5
    fci.verbose = 3
    addons.fix_spin_(fci.fcisolver, ss=0)
    fci.kernel()
    print(f"FCI Energy {fci.e_tot}")
    fci.e_tot = fci.e_tot[0]
    fci_e_tot = fci.e_tot

    mycc = cc.CCSD(mf)
    mycc.kernel()
    ccsd_e_tot = mycc.e_tot

    oei, ecore = myci.get_h1eff(mo_coeff=mf.mo_coeff)
    tei = myci.get_h2eff(mo_coeff=mf.mo_coeff)
    tei = ao2mo.restore(1, tei, norb)
    of_eris = np.transpose(tei, (0, 2, 3, 1))

    fqe_ham = RestrictedHamiltonian((oei, np.einsum("ijlk", -0.5 * of_eris)), e_0=ecore)
    fqe_wf = fqe.Wavefunction([[Na + Nb, sz, norb]])
    fqe_wf.set_wfn(strategy="hartree-fock")
    fqe_wf.print_wfn()

    print(f"FQE energy = {fqe_wf.expectationValue(fqe_ham).real} vs pyscf = {mf_e_tot}")

    fqe_energies = []
    nsteps = 200
    for _ in range(nsteps):
        fqe_wf = fqe_wf.time_evolve(-1j * 0.085, fqe_ham)
        fqe_wf.normalize()
        # fqe_wf.print_wfn()
        fqe_energy = fqe_wf.expectationValue(fqe_ham).real
        fqe_energies.append(fqe_energy)
        print(
            f"FQE energy = {fqe_energy} vs pyscf = {casci_e_tot} delta = {fqe_energy - casci_e_tot}"
        )

    mc_fqe = fqe_to_fake_ci(fqe_wf, mf)
    ci_amps = extract_ci_amplitudes(mc_fqe, exci=4)
    ci_spinorb = amplitudes_to_spinorb(ci_amps, exci=4)
    nocca, noccb = mc_fqe.nelecas
    nvirta = mc_fqe.ncas - nocca
    nvirtb = mc_fqe.ncas - noccb
    ncore = mc_fqe.ncore
    nvir = mf.mo_coeff.shape[1] - ncore - mc_fqe.ncas
    occslice, virtslice = prepare_cas_slices(
        nocca, noccb, nvirta, nvirtb, ncore, nvir, backend="pyscf"
    )
    ec_exact = ec_cc(mf, *ci_spinorb, occslice, virtslice)
    eccc = ec_cc_from_ci(myci)

    tcc_fqe = tccsd_from_fqe(mf, fqe_wf)
    tcc_exact = tccsd_from_ci(myci)
    # tcc_fqe = tccsd_from_ci(myci)
    print(f"ec_CC energy {ec_exact.e_tot}")
    print(f"after {nsteps} time steps")
    print(f"ec_CC(CASCI) energy error vs FCI {eccc.e_tot - fci.e_tot}")
    print(f"ec_CC(FQE) energy error vs FCI {ec_exact.e_tot - fci.e_tot}")
    print(f"TCCSD(CASCI) energy error vs FCI {tcc_exact.e_tot - fci.e_tot}")
    print(f"TCCSD(FQE) energy error vs FCI {tcc_fqe.e_tot - fci.e_tot}")
    print(f"FQE energy error vs FCI {fqe_energy - fci.e_tot}")
    print(f"CCSD energy error vs FCI {mycc.e_tot - fci.e_tot}")
    print(f"CASCI error vs FCI {myci.e_tot[0] - fci.e_tot}")
    print(f"NEVPT2 error vs FCI {myci.e_tot[0] + ept2 - fci.e_tot}")
    exit(0)

    npoints = 10
    cycles = 50
    # stds = np.logspace(-10, -1, num=npoints, endpoint=True)
    stds = np.logspace(-4, -1, num=npoints, endpoint=True)

    np.random.seed(42)
    data = []
    for std in stds:
        for _ in tqdm(range(cycles)):
            ci_amps_noisy = add_gaussian_noise(ci_amps, std=std)
            ci_spinorb = amplitudes_to_spinorb(ci_amps_noisy, exci=4)
            ret = ec_cc(
                mf,
                *ci_spinorb,
                occslice,
                virtslice,
                guess_t1_t2_from_ci=True,
                verbose=0,
                maxiter=150,
            )
            data.append([std, ret.e_tot])

    import pandas as pd

    df = pd.DataFrame(data=data, columns=["std", "e_tot"])
    df["ec_cc_error"] = np.abs(df.e_tot - ec_exact.e_tot)
    df["fci_error"] = np.abs(df.e_tot - fci.e_tot)

    sns.set_theme(context="talk", palette="colorblind", font_scale=1.2, style="ticks")
    fig, (ax, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 9)
    sns.lineplot(data=df, x="std", y="ec_cc_error", marker="o", ax=ax)
    # sns.scatterplot(data=df, x="std", y="ec_cc_error", marker="o")
    ax.set_yscale("log")
    ax.set_xscale("log")

    sns.scatterplot(data=df, x="std", y="fci_error", marker="o", ax=ax2)
    ax2.axhline(
        np.abs(ec_exact.e_tot - fci.e_tot),
        linestyle="--",
        color=sns.color_palette("colorblind")[1],
        label="without noise",
    )
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.legend()

    ax2.set_ylim((1e-4, 1))

    plt.tight_layout()
    plt.savefig("h4_ec_ccpvdz_noise.png")
