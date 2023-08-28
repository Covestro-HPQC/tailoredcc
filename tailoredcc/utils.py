# Proprietary and Confidential
# Covestro Deutschland AG, 2023
import numpy as np


def spinorb_from_spatial(oei, tei):
    # TODO: docs
    nso = 2 * oei.shape[0]
    soei = np.zeros(2 * (nso,))
    soei[::2, ::2] = oei  # (a|a)
    soei[1::2, 1::2] = oei  # (b|b)

    eri_of = np.zeros(4 * (nso,))
    eri_of[::2, ::2, ::2, ::2] = tei  # <aa|aa>
    eri_of[1::2, 1::2, 1::2, 1::2] = tei  # <bb|bb>
    eri_of[::2, 1::2, 1::2, ::2] = tei  # <ab|ba>
    eri_of[1::2, ::2, ::2, 1::2] = tei  # <ba|ab>
    return soei, eri_of


def spin_blocks_interleaved_to_sequential(tensor):
    ret = np.zeros_like(tensor)
    ndim = tensor.ndim
    alpha_il = slice(0, None, 2)
    beta_il = slice(1, None, 2)
    lookup = {
        "alpha": alpha_il,
        "beta": beta_il,
    }
    from itertools import product

    for comb in product(["alpha", "beta"], repeat=ndim):
        take = tuple(lookup[c] for c in comb)
        view = tensor[take]
        slices = []
        for i, k in enumerate(comb):
            start = tensor.shape[i] - view.shape[i] if k == "beta" else 0
            stop = view.shape[i] if k == "alpha" else None
            slices.append(slice(start, stop))
        ret[tuple(slices)] = view
    return ret


def spin_blocks_sequential_to_interleaved(tensor, n_per_dim):
    ret = np.zeros_like(tensor)
    ndim = tensor.ndim
    assert len(n_per_dim) == ndim
    alpha_il = slice(0, None, 2)
    beta_il = slice(1, None, 2)
    lookup = {
        "alpha": alpha_il,
        "beta": beta_il,
    }

    from itertools import product

    for comb in product(["alpha", "beta"], repeat=ndim):
        take = tuple(lookup[c] for c in comb)
        slices = []
        for i, k in enumerate(comb):
            offset = 0 if k == "alpha" else n_per_dim[i]["alpha"]
            start = offset
            stop = offset + n_per_dim[i][k]
            slices.append(slice(start, stop))
        ret[take] = tensor[tuple(slices)]
    return ret


# from https://github.com/quantumlib/OpenFermion-FQE/issues/98
def pyscf_to_fqe_wf(pyscf_cimat, pyscf_mf=None, norbs=None, nelec=None):
    import fqe
    from pyscf.fci.cistring import make_strings

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
    from pyscf.fci.cistring import make_strings, num_strings

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


def fqe_to_fake_ci(wfn, scfres, sz=0):
    """Create a 'fake' pyscf CASCI object from FQE wavefunction.
    Currently only implemented for Sz=0"""
    sectors = list(wfn.sectors())
    szs = [sec[1] for sec in sectors]
    if sz not in szs:
        raise ValueError(f"Sz = {sz} not found in FQE wfn sectors.")
    sz0 = szs.index(sz)
    fqe_data = wfn.sector(sectors[sz0])
    nelec = fqe_data.nalpha(), fqe_data.nbeta()
    civec = fqe_to_pyscf(wfn, nelec=nelec)

    ncorelec = scfres.mol.nelectron - sum(nelec)
    assert ncorelec % 2 == 0
    assert ncorelec >= 0

    class FakeCI:
        nelecas = nelec
        ncas = wfn.norb()
        ncore = ncorelec // 2
        mo_coeff = scfres.mo_coeff
        ci = civec
        _scf = scfres

    mc = FakeCI()
    return mc
