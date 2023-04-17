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
