import fqe
import numpy as np
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from pyscf import ao2mo, cc, fci, gto, mcscf, scf
from pyscf.fci.cistring import make_strings, num_strings

from tailoredcc.tailoredcc import (
    ec_cc_from_ci,
    ec_cc_from_fqe,
    tccsd_from_ci,
    tccsd_from_fqe,
)

if __name__ == "__main__":
    mol = gto.Mole()
    # mol.basis = 'cc-pvdz'
    mol.basis = "cc-pvdz"
    mol.atom = """H 0 0 0
    H 0 0 1.23
    H 0 1.23 0
    H 0 1.23 1.23"""
    mol.verbose = 0
    mol.spin = 0
    mol.build()
    mol.symmetry = False
    mf = scf.RHF(mol)
    # mf.chkfile = 'scf.chk'
    ehf = mf.kernel()
    mo1 = mf.stability(verbose=0)[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability(verbose=0)
    mf_e_tot = mf.e_tot

    Na, Nb = (2, 2)
    sz = Na - Nb
    norb = 4
    print(Na + Nb, sz, norb)
    print(mf.mo_coeff.shape)

    myci = mcscf.CASCI(mf, ncas=norb, nelecas=(Na, Nb))
    myci.fcisolver.nroots = 5
    myci.verbose = 3
    fci.addons.fix_spin_(myci.fcisolver, ss=0)

    myci.canonicalization = False
    myci.kernel()
    print(f"CASCI Energy {myci.e_tot}")
    casci_e_tot = myci.e_tot[0]
    myci.ci = myci.ci[0]

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

    energy_hf_fqe = fqe_wf.expectationValue(fqe_ham).real
    print(f"FQE energy = {energy_hf_fqe} vs pyscf = {mf_e_tot}")
    np.testing.assert_allclose(mf_e_tot, energy_hf_fqe, atol=1e-12, rtol=0)

    ec_exact_state = ec_cc_from_ci(myci)
    tcc_exact_state = tccsd_from_ci(myci, backend="pyscf")

    fqe_energies = []
    ec_cc_energies = []
    tcc_energies = []
    nsteps = 10

    energy_aci = -2.097185699411

    for _ in range(nsteps):
        fqe_wf = fqe_wf.time_evolve(-1j * 0.085, fqe_ham)
        fqe_wf.normalize()
        fqe_energy = fqe_wf.expectationValue(fqe_ham).real
        fqe_energies.append(fqe_energy)
        print(
            f"FQE energy = {fqe_energy} vs pyscf = {casci_e_tot} delta = {fqe_energy - casci_e_tot}"
        )

        out = ec_cc_from_fqe(mf, fqe_wf)
        print(f"ec_CC energy {out.e_tot}")
        ec_cc_energies.append(out.e_tot)

        tcc = tccsd_from_fqe(mf, fqe_wf)
        tcc_energies.append(tcc.e_tot)

        print()

    import matplotlib.pyplot as plt

    # fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # fig.set_size_inches(18, 9)
    fig.set_size_inches(16, 9)

    steps = np.arange(nsteps)
    # ax.axhline(y = mf_e_tot, color = 'g', linestyle = '-', label='RHF')
    ax.plot(steps, fqe_energies, label=r"$e^{-\beta H}|\phi_{0}\rangle$")
    ax.axhline(y=casci_e_tot, color="k", linestyle="-", label="CASCI")
    ax.plot(steps, ec_cc_energies, label="ec-CC")
    ax.plot(steps, tcc_energies, label="TCCSD")
    ax.axhline(y=ccsd_e_tot, color="r", linestyle="--", label="CCSD")

    ax.axhline(y=energy_aci, color="gold", linestyle="-", label="ACI")
    ax.axhline(y=ec_exact_state.e_tot, color="deeppink", linestyle="--", label="ec-CC (CASCI)")
    ax.axhline(y=tcc_exact_state.e_tot, color="steelblue", linestyle="--", label="TCCSD (CASCI)")

    ax.legend(fontsize=14, frameon=False)
    ax.set_xlabel(r"Imaginary time steps $\delta \beta = 0.085$", fontsize=14)
    ax.set_ylabel("Energy [Ha]", fontsize=14)
    ax.tick_params(which="both", labelsize=14, direction="in")

    # ax2.plot(steps, np.abs(fqe_energies - casci_e_tot), label=r"error $e^{-\beta H}|\phi_{0}\rangle$")
    # ax2.plot(steps, np.abs(ec_cc_energies - casci_e_tot), label="error ec-CC")
    # ax2.axhline(np.abs(ccsd_e_tot - casci_e_tot), label="error CCSD", color="r", linestyle="--")
    # ax2.set_yscale("log")
    # ax2.legend()

    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("ec_cc_vs_FQE_ccpvdz.png", format="PNG", dpi=300)
