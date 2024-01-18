from typing import Union

import fqe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from pyscf import cc, fci, gto, mcscf, scf
from pyscf.fci.cistring import make_strings, num_strings

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
from tailoredcc.utils import fqe_to_fake_ci, fqe_to_pyscf

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
    np.save("h4_sto3g_civec", myci.ci)
    # exit(0)
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
        columns=["index", "coeffs_ci", "exci_level"],
        data=np.vstack(
            [np.arange(ci_coeffs.size), ci_coeffs_sorted, excilevels.flatten()[sort_idx]]
        ).T,
    )
    # sns.scatterplot(x="index", y="abs_coeff", hue="exci_level", ax=ax, data=df, palette="Set2", s=150)

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
    ec_cc_energies = []
    nsteps = 500
    # nsteps = 50
    # from celluloid import Camera
    # cam = Camera(fig)
    data = []
    for ii in range(nsteps):
        print(ii)
        # fqe_wf.print_wfn()
        fqe_energy = fqe_wf.expectationValue(fqe_ham).real
        fqe_energies.append(fqe_energy)
        print(f"FQE energy = {fqe_energy} vs pyscf = {fci_e_tot} delta = {fqe_energy - fci_e_tot}")

        # # get amplitudes
        out = ec_cc_from_fqe(mf, fqe_wf, conv_tol=1e-15, verbose=0, zero_companion_threshold=1e-6)
        print(f"ec_CC energy {out.e_corr + out.e_hf}")
        ec_cc_energies.append(out.e_tot)

        fqe_civec = fqe_to_pyscf(fqe_wf, myci.nelecas)
        # df['coeffs_fqe'] = np.abs(fqe_civec.flatten()[sort_idx])

        fqe_civec /= np.sign(fqe_civec[0, 0]) * np.sign(myci.ci[0, 0])
        diff = myci.ci - fqe_civec
        rmse = np.sqrt(np.var(diff))

        data.append([ii, fqe_energy, out.e_tot, rmse, fqe_civec])

        # dfm = pd.melt(df, id_vars=['index', 'exci_level'], value_vars=['coeffs_ci', 'coeffs_fqe'], value_name="coeff", var_name="method")
        # sns.scatterplot(x="index", y="coeff", hue="exci_level", style="method", ax=ax, data=dfm, palette="Set2", s=150, legend=None)
        # ax.set_yscale("log")
        # ax.set_title(f"t = {ii}")
        # ax.text(0, 1.5, f"t = {ii}")
        # ax.set_ylim((1e-16, 1.4))
        # if ii % 10 == 0:
        # cam.snap()
        print()

        # do the time evolution here to have HF as 0th state
        fqe_wf = fqe_wf.time_evolve(-1j * 0.085, fqe_ham)
        fqe_wf.normalize()

    # animation = cam.animate()
    # animation.save("time_evolution.mp4")
    # # plt.savefig("ci_coeffs_h4_compare.png", dpi=300)
    # exit(0)

    df = pd.DataFrame(data=data, columns=["step", "energy_fqe", "energy_ec", "rmse", "civec"])
    df["energy_fci"] = myci.e_tot
    df.to_hdf("fqe_time_evolve_h4.h5", key="df")

    npoints = 20
    cycles = 50
    # stds = np.logspace(-10, -1, num=npoints, endpoint=True)
    stds = np.logspace(-4, -1, num=npoints, endpoint=True)

    np.random.seed(42)
    from tqdm import tqdm

    mc_fqe = fqe_to_fake_ci(fqe_wf, mf)
    ci_amps = extract_ci_amplitudes(mc_fqe, exci=4)
    nocca, noccb = mc_fqe.nelecas
    nvirta = mc_fqe.ncas - nocca
    nvirtb = mc_fqe.ncas - noccb
    ncore = mc_fqe.ncore
    nvir = mf.mo_coeff.shape[1] - ncore - mc_fqe.ncas
    occslice, virtslice = prepare_cas_slices(
        nocca, noccb, nvirta, nvirtb, ncore, nvir, backend="pyscf"
    )

    data_noise = []
    for std in stds:
        for nc in tqdm(range(cycles)):
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
                zero_companion_threshold=1e-6,
            )
            data_noise.append([std, nc, ret.e_tot, ret.converged, False])
            ci_spinorb_orig = ci_spinorb.copy()

            # thresholding by variances
            for k, ov in ci_amps_noisy.items():
                mask = np.abs(ov) <= 2.0 * std
                # print(k, np.sum(mask))
                ov[mask] = 0.0
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
                zero_companion_threshold=1e-6,
            )
            data_noise.append([std, nc, ret.e_tot, ret.converged, True])
    df = pd.DataFrame(
        data=data_noise, columns=["std", "cycle", "energy_ec", "converged", "thresholded"]
    )
    df.to_hdf("fqe_time_evolve_noise_h4.h5", key="df")

    # fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2)
    # fig.set_size_inches(18, 9)

    # steps = np.arange(nsteps)
    # ax.axhline(y = mf_e_tot, color = 'g', linestyle = '-', label='RHF')
    # ax.plot(steps, fqe_energies, label=r'$e^{-\beta H}|\phi_{0}\rangle$')
    # ax.axhline(y = fci_e_tot, color = 'k', linestyle = '-', label='FCI')
    # ax.plot(steps, ec_cc_energies, label='ec-CC')
    # ax.axhline(y = ccsd_e_tot, color = 'r', linestyle = '--', label='CCSD')
    # ax.legend(fontsize=14, frameon=False)
    # ax.set_xlabel(r"Imaginary time steps $\delta \beta = 0.085$", fontsize=14)
    # ax.set_ylabel("Energy [Ha]", fontsize=14)
    # ax.tick_params(which='both', labelsize=14, direction='in')

    # ax2.plot(steps, np.abs(fqe_energies - fci_e_tot), label=r"error $e^{-\beta H}|\phi_{0}\rangle$")
    # ax2.plot(steps, np.abs(ec_cc_energies - fci_e_tot), label="error ec-CC")
    # ax2.axhline(np.abs(ccsd_e_tot - fci_e_tot), label="error CCSD", color="r", linestyle="--")
    # ax2.set_yscale("log")
    # ax2.legend()

    # plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    # plt.savefig("ec_cc_vs_FQE.png", format='PNG', dpi=300)
