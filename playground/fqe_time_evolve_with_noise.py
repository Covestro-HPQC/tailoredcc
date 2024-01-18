from typing import Union

import fqe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from pyscf import cc, fci, gto, mcscf, scf
from pyscf.fci.cistring import make_strings, num_strings
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
from tailoredcc.ci_to_cc import ci_to_cc
from tailoredcc.clusterdec import dump_clusterdec, run_clusterdec
from tailoredcc.tailoredcc import ec_cc, ec_cc_from_fqe
from tailoredcc.utils import fqe_to_fake_ci

if __name__ == "__main__":
    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.atom = """H 0 0 0
    H 0 0 1.23
    H 0 1.23 0
    H 0 1.23 1.23"""
    mol.verbose = 0
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

    Na, Nb = mol.nelec
    sz = Na - Nb
    norb = mf.mo_coeff.shape[1]
    print(Na + Nb, sz, norb)

    myci = mcscf.CASCI(mf_or_mol=mf, ncas=norb, nelecas=(Na, Nb), ncore=0)
    myci.kernel()
    print(f"FCI Energy {myci.e_tot}")
    fci_e_tot = myci.e_tot

    from pyscf.fci.cistring import gen_occslst

    strs = gen_occslst(range(myci.ncas), myci.nelecas[0])
    excilevels = np.array([np.sum(x > myci.nelecas[0] - 1) for x in strs])
    excilevels = excilevels[:, None] + excilevels
    excilevels = np.asarray(excilevels, dtype=int)
    np.set_printoptions(linewidth=np.inf)

    ci_coeffs = myci.ci.flatten()
    sort_idx = np.argsort(np.abs(ci_coeffs))[::-1]
    ci_coeffs_sorted = np.abs(ci_coeffs[sort_idx])

    sns.set_theme(context="talk", palette="colorblind", font_scale=1.2, style="ticks")
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(18, 9)

    df = pd.DataFrame(
        columns=["index", "abs_coeff", "exci_level"],
        data=np.vstack(
            [np.arange(ci_coeffs.size), ci_coeffs_sorted, excilevels.flatten()[sort_idx]]
        ).T,
    )
    sns.scatterplot(
        x="index", y="abs_coeff", hue="exci_level", ax=ax, data=df, palette="Set2", s=150
    )

    # scatter = ax.scatter(x=np.arange(ci_coeffs.size), y=ci_coeffs_sorted, c=c)
    # ax.scatter(x=np.arange(ci_coeffs.size), y=np.sort(np.abs(ci_coeffs + np.random.normal(loc=0.0, scale=1e-2, size=ci_coeffs.size))))
    ax.set_yscale("log")
    plt.savefig("h4_sto3g_civec.png")
    # exit(0)

    mycc = cc.CCSD(mf)
    mycc.kernel()
    ccsd_e_tot = mycc.e_tot

    # nuclear energy
    ecore = mf.energy_nuc()

    # get mo coefficient matrix
    C = mf.mo_coeff

    # get two-electron integrals
    tei = mol.intor("int2e")

    # transform two-electron integrals to mo basis - chemistry order
    tei = np.einsum("uj,vi,wl,xk,uvwx", C, C, C, C, tei)
    of_eris = np.transpose(tei, (0, 2, 3, 1))

    # get core hamiltonian
    kinetic = mol.intor("int1e_kin")
    potential = mol.intor("int1e_nuc")
    oei = kinetic + potential

    # transform core hamiltonian to mo basis
    oei = np.einsum("uj,vi,uv", C, C, oei)

    fqe_ham = RestrictedHamiltonian((oei, np.einsum("ijlk", -0.5 * of_eris)), e_0=ecore)
    fqe_wf = fqe.Wavefunction([[Na + Nb, sz, norb]])
    fqe_wf.set_wfn(strategy="hartree-fock")
    fqe_wf.print_wfn()

    print(f"FQE energy = {fqe_wf.expectationValue(fqe_ham).real} vs pyscf = {mf_e_tot}")

    fqe_energies = []
    nsteps = 10
    # for _ in range(nsteps):
    #     fqe_wf = fqe_wf.time_evolve(-1j * 0.085, fqe_ham)
    #     fqe_wf.normalize()
    #     # fqe_wf.print_wfn()
    #     fqe_energy = fqe_wf.expectationValue(fqe_ham).real
    #     fqe_energies.append(fqe_energy)
    #     print(f"FQE energy = {fqe_energy} vs pyscf = {fci_e_tot} delta = {fqe_energy - fci_e_tot}")

    # mc_fqe = fqe_to_fake_ci(fqe_wf, mf)
    mc_fqe = myci
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
    ec_exact = ec_cc(mf, *ci_spinorb, occslice, virtslice, conv_tol=1e-12)
    print(f"ec_CC energy {ec_exact.e_tot}", fci_e_tot - ec_exact.e_tot)
    print(f"after {nsteps} time steps")

    ec_cc_from_fqe(mf, fqe_wf)

    npoints = 20
    cycles = 100
    # stds = np.logspace(-10, -1, num=npoints, endpoint=True)
    stds = np.logspace(-4, -1, num=npoints, endpoint=True)

    np.random.seed(42)
    data = []
    vars = []
    for std in stds:
        for _ in tqdm(range(cycles)):
            ci_amps_noisy = add_gaussian_noise(ci_amps, std=std)
            # noisy_ci = mc_fqe.ci + np.random.normal(size=mc_fqe.ci.shape, loc=0.0, scale=std)
            # var = np.var(mc_fqe.ci - noisy_ci)
            # vars.append([std, var])
            ci_spinorb = amplitudes_to_spinorb(ci_amps_noisy, exci=4)
            ret = ec_cc(
                mf,
                *ci_spinorb,
                occslice,
                virtslice,
                guess_t1_t2_from_ci=True,
                verbose=0,
                maxiter=150,
                conv_tol=1e-12,
            )
            data.append([std, ret.e_tot])

    # dfv = pd.DataFrame(data=vars, columns=['std', 'var'])
    # print(dfv.groupby('std').mean().transform('sqrt'))
    # exit(0)

    df = pd.DataFrame(data=data, columns=["std", "e_tot"])
    df["ec_cc_error"] = np.abs(df.e_tot - ec_exact.e_tot)
    df["fci_error"] = np.abs(df.e_tot - myci.e_tot)

    sns.set_theme(context="talk", palette="colorblind", font_scale=1.2, style="ticks")
    fig, (ax, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 9)
    sns.lineplot(data=df, x="std", y="ec_cc_error", marker="o", ax=ax)
    # sns.scatterplot(data=df, x="std", y="ec_cc_error", marker="o")
    ax.set_yscale("log")
    ax.set_xscale("log")

    sns.scatterplot(data=df, x="std", y="fci_error", marker="o", ax=ax2)
    ax2.axhline(
        np.abs(ec_exact.e_tot - myci.e_tot),
        linestyle="--",
        color=sns.color_palette("colorblind")[1],
        label="without noise",
    )
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.legend()

    # ax2.set_ylim((1e-4, 1))

    plt.tight_layout()
    plt.savefig("h4_ec_noise.png")
