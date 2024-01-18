"""Converting FQE/pyscf sparse CI representations"""
import fqe
import numpy as np
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.fci.cistring import make_strings, num_strings


# from https://github.com/quantumlib/OpenFermion-FQE/issues/98
def pyscf_to_fqe_wf(pyscf_cimat, pyscf_mf=None, norbs=None, nelec=None):
    if pyscf_mf is None:
        assert norbs is not None
        assert nelec is not None
    else:
        mol = pyscf_mf.mol
        nelec = mol.nelec
        norbs = pyscf_mf.mo_coeff.shape[1]

    norb_list = tuple(range(norbs))
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


if __name__ == "__main__":
    mol = gto.M(
        atom="""
    N 0 0 0
    N 0 0 3.5
    """,
        unit="bohr",
        basis="sto-3g",
    )
    scfres = scf.RHF(mol)
    scfres.kernel()

    mci = mcscf.CASCI(scfres, 6, (3, 3))
    h1, ecore = mci.get_h1eff()
    h2 = ao2mo.restore(1, mci.ao2mo(), mci.ncas).transpose(0, 2, 3, 1)
    h1s, h2s = spinorb_from_spatial(h1, 0.5 * h2)
    mci.kernel()

    wfn: fqe.Wavefunction = pyscf_to_fqe_wf(mci.ci, norbs=mci.ncas, nelec=mci.nelecas)

    civec = fqe_to_pyscf(wfn, mci.nelecas)

    # check CI vector
    np.testing.assert_allclose(mci.ci, civec, atol=1e-14, rtol=0)

    # check energies
    iop = of.InteractionOperator(0.0, h1s, h2s)
    fop = of.get_fermion_operator(iop)
    ham = fqe.build_hamiltonian(fop, norb=mci.ncas)
    e_ci = wfn.expectationValue(ham).real
    np.testing.assert_allclose(e_ci, mci.e_cas, atol=1e-14, rtol=0)
