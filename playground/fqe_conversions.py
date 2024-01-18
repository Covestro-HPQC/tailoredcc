"""Converting FQE/pyscf/covvqetools sparse CI representations"""
import covvqetools as cov
import fqe
import numpy as np
import openfermion as of
import pennylane as qml
from covvqetools.pyscf import extract_state_dict
from openfermion.chem.molecular_data import spinorb_from_spatial
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.fci.cistring import make_strings, num_strings


def integrals_to_coefficients(one_body_integrals, two_body_integrals):
    return spinorb_from_spatial(one_body_integrals, two_body_integrals / 2)


# from https://github.com/quantumlib/OpenFermion-FQE/issues/98
def pyscf_to_fqe_wf(pyscf_cimat, pyscf_mf=None, norbs=None, nelec=None):
    if pyscf_mf is None:
        assert norbs is not None
        assert nelec is not None
    else:
        mol = pyscf_mf.mol
        nelec = mol.nelec
        norbs = pyscf_mf.mo_coeff.shape[1]

    norb_list = tuple(list(range(norbs)))
    n_alpha_strings = make_strings(norb_list, nelec[0])
    n_beta_strings = make_strings(norb_list, nelec[1])

    sz = nelec[0] - nelec[1]
    nel_total = sum(nelec)

    fqe_wf_ci = fqe.Wavefunction([[nel_total, sz, norbs]])
    fqe_data_ci = fqe_wf_ci.sector((nel_total, sz))
    fqe_graph_ci = fqe_data_ci.get_fcigraph()
    fqe_orderd_coeff = np.zeros((fqe_graph_ci.lena(), fqe_graph_ci.lenb()))
    for paidx, pyscf_alpha_idx in enumerate(n_alpha_strings):
        for pbidx, pyscf_beta_idx in enumerate(n_beta_strings):
            fqe_orderd_coeff[
                fqe_graph_ci.index_alpha(pyscf_alpha_idx), fqe_graph_ci.index_beta(pyscf_beta_idx)
            ] = pyscf_cimat[paidx, pbidx]

    fqe_data_ci.coeff = fqe_orderd_coeff
    return fqe_wf_ci


def fqe_to_pyscf(wfn, nelec: tuple):
    norbs = wfn.norb()
    nalpha, nbeta = nelec
    sz = nalpha - nbeta

    num_alpha = num_strings(norbs, nalpha)
    num_beta = num_strings(norbs, nbeta)

    fqe_ci = wfn.sector((sum(nelec), sz))
    fqe_graph = fqe_ci.get_fcigraph()
    assert fqe_graph.lena() == num_alpha
    assert fqe_graph.lenb() == num_beta

    norb_list = tuple(list(range(norbs)))
    alpha_strings = make_strings(norb_list, nelec[0])
    beta_strings = make_strings(norb_list, nelec[1])
    ret = np.zeros((num_alpha, num_beta))
    for paidx, pyscf_alpha_idx in enumerate(alpha_strings):
        for pbidx, pyscf_beta_idx in enumerate(beta_strings):
            ret[paidx, pbidx] = fqe_ci.coeff[
                fqe_graph.index_alpha(pyscf_alpha_idx), fqe_graph.index_beta(pyscf_beta_idx)
            ]
    return ret


def state_dict_to_fqe_wfn(state_dict, nact, nalpha, nbeta):
    nel_total = nalpha + nbeta
    sz = nalpha - nbeta
    fqe_wf_ci = fqe.Wavefunction([[nel_total, sz, nact]])
    fqe_data_ci = fqe_wf_ci.sector((nel_total, sz))
    fqe_graph_ci = fqe_data_ci.get_fcigraph()
    fqe_orderd_coeff = np.zeros((fqe_graph_ci.lena(), fqe_graph_ci.lenb()))
    for detstr, coeff in state_dict.items():
        astr = detstr[::2][::-1]
        bstr = detstr[1::2][::-1]
        assert len(detstr) == 2 * nact
        assert astr.count("1") == nalpha
        assert bstr.count("1") == nbeta
        abit = int(astr, 2)
        bbit = int(bstr, 2)
        fqe_orderd_coeff[fqe_graph_ci.index_alpha(abit), fqe_graph_ci.index_beta(bbit)] = coeff
    fqe_data_ci.coeff = fqe_orderd_coeff
    return fqe_wf_ci


def convert_state_jw_seq_to_int(state_dict):
    ret = {}
    for detstr, coeff in state_dict.items():
        astr = detstr[::2]
        bstr = detstr[1::2]
        phase = 1.0
        for ai, aa in enumerate(astr):
            if aa == "1":
                beta_left = bstr[:ai].count("1")
                phase *= (-1) ** beta_left
        ret[detstr] = phase * coeff
    return ret


def convert_state_qubit_int_to_seq(state_dict):
    return {detstr[::2] + detstr[1::2] for detstr, coeff in state_dict.items()}


if __name__ == "__main__":
    mol = gto.M(
        atom="""
    N 0 0 0
    N 0 0 3.5
    """,
        unit="bohr",
        basis="sto-3g",
        verbose=4,
    )
    scfres = scf.RHF(mol)
    scfres.kernel()

    # mci = mcscf.CASCI(scfres, 6, (3, 3))
    mci = mcscf.CASCI(scfres, 4, (2, 2))
    h1, ecore = mci.get_h1eff()
    h2 = ao2mo.restore(1, mci.ao2mo(), mci.ncas).transpose(0, 2, 3, 1)
    h1s, h2s = integrals_to_coefficients(h1, h2)
    mci.kernel()

    wfn: fqe.Wavefunction = pyscf_to_fqe_wf(mci.ci, norbs=mci.ncas, nelec=mci.nelecas)
    wfn.print_wfn()

    civec = fqe_to_pyscf(wfn, mci.nelecas)

    # check CI vector
    np.testing.assert_allclose(mci.ci, civec, atol=1e-14, rtol=0)

    # convert pyscf's CI vector to state dictionary (used for cov.Superposition)
    state_dict = extract_state_dict(
        mci.fcisolver,
        mci.ci,
        nact=mci.ncas,
        nalpha=mci.nelecas[0],
        nbeta=mci.nelecas[1],
        amplitude_cutoff=1e-14,
    )

    fqe_from_cov = state_dict_to_fqe_wfn(
        state_dict, mci.ncas, nalpha=mci.nelecas[0], nbeta=mci.nelecas[1]
    )
    ci2 = fqe_to_pyscf(fqe_from_cov, mci.nelecas)
    # check CI vector
    np.testing.assert_allclose(mci.ci, ci2, atol=1e-14, rtol=0)

    # check energies
    iop = of.InteractionOperator(0.0, h1s, h2s)
    fop = of.get_fermion_operator(iop)
    ham = fqe.build_hamiltonian(fop, norb=mci.ncas)
    e_ci = wfn.expectationValue(ham).real
    np.testing.assert_allclose(e_ci, mci.e_cas, atol=1e-14, rtol=0)

    e_ci2 = fqe_from_cov.expectationValue(ham).real
    np.testing.assert_allclose(e_ci2, mci.e_cas, atol=1e-14, rtol=0)

    # qop = of.jordan_wigner(iop)
    qop = cov.operators.hamiltonian(
        core_energy=0,
        one_body_integrals=h1,
        two_body_integrals=h2,
        convention="of",
        jw_ordering="seq",
        # jw_ordering="int",
        # qubit_ordering="seq",  # unused kwarg
        # qubit_ordering="int",
    )
    dev = qml.device("default.qubit", wires=2 * mci.ncas)
    ms = cov.Hermitian(qop, dev=dev)

    # state_dict = convert_state_jw_seq_to_int(state_dict)
    def ansatz(*_, wires=None):
        cov.Superposition(state_dict, wires=wires)

    e = ms.expval(None, ansatz=ansatz)
    print(e - mci.e_cas)
