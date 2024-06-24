# Copyright 2024 Covestro Deutschland AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import warnings

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
def pyscf_to_fqe_wf(pyscf_cimat, norbs, nelec):
    import fqe
    from pyscf.fci.cistring import make_strings

    norb_list = tuple(range(norbs))
    n_alpha_strings = make_strings(norb_list, nelec[0])
    n_beta_strings = make_strings(norb_list, nelec[1])

    sz = nelec[0] - nelec[1]
    nel_total = sum(nelec)

    fqe_wf_ci = fqe.Wavefunction([[nel_total, sz, norbs]])
    fqe_data_ci = fqe_wf_ci.sector((nel_total, sz))
    fqe_graph_ci = fqe_data_ci.get_fcigraph()
    fqe_orderd_coeff = np.zeros((fqe_graph_ci.lena(), fqe_graph_ci.lenb()), dtype=pyscf_cimat.dtype)
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

    ret = np.zeros((num_alpha, num_beta), dtype=float)
    fqe_civec = fqe_ci.coeff.copy()

    if np.max(np.abs(fqe_civec.imag)) > 1e-13:
        warnings.warn("Complex FCI vector found in FQE.")
        max_idx = np.argmax(np.abs(fqe_civec))
        max_val = fqe_civec.flatten()[max_idx]
        global_phase = np.exp(1j * np.angle(max_val))
        print("Maximum abs value", max_val)
        print("Global phase", global_phase)
        fqe_civec /= global_phase
        fqe_civec = fqe_civec.real
        fqe_civec /= np.linalg.norm(fqe_civec)
    else:
        fqe_civec = fqe_civec.real
    for paidx, pyscf_alpha_idx in enumerate(alpha_strings):
        for pbidx, pyscf_beta_idx in enumerate(beta_strings):
            ret[paidx, pbidx] = fqe_civec[
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
