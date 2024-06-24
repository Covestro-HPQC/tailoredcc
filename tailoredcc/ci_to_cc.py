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


from itertools import permutations

import numpy as np
from jax import config

from tailoredcc.amplitudes import compute_parity

config.update("jax_enable_x64", True)
import jax
from jax.numpy import einsum

# def einsum(*args, **kwargs):
#     from opt_einsum import contract, contract_path
#     kwargs["optimize"] = True
#     path_info = contract_path(*args)
#     print(path_info[1])
#     print()
#     return contract(*args, **kwargs)


def get_term_permutation(contr, virt, occ):
    """Find the permutation of the input string that brings both
    virtual and occupied indices into alphabetical order.

    Parameters
    ----------
    contr : str
        Contraction string, e.g., 'ai,bcjk'
    virt : str
        virtual indices, e.g., 'abc'
    occ : str
        occupied indices, e.g., 'ijk'

    Returns
    -------
    tuple
        Sign of the total permutation and total
        permutation as np.ndarray.
    """
    contrv = [v for v in contr if v in virt]
    contro = [o for o in contr if o in occ]
    orderv = [contrv.index(v) for v in virt]
    ordero = [contro.index(o) for o in occ]

    s1 = compute_parity(orderv)
    s2 = compute_parity(ordero)

    perm = np.concatenate([np.array(orderv), np.array(ordero) + len(virt)])
    return s1 * s2, perm


def get_unique_contractions(contraction, virt, occ):
    """For a given contraction string (current version only works for outer products),
    get all the non-redundant permutations of the contraction that obey the anti-symmetry
    relations of the resulting tensor.

    Parameters
    ----------
    contraction : str
        comma-separated contraction string for the outer product
    virt : str
        virtual indices, e.g., 'abc'
    occ : str
        occupied indices, e.g., 'ijk'

    Returns
    -------
    dict
        Dictionary with the permutations and the corresponding sign
    """
    cache = {}
    split = contraction.split(",")
    amps_sizes = [len(x) // 2 for x in split]
    for virtp in list(permutations(virt)):
        for occp in list(permutations(occ)):
            keys = []
            offset = 0
            perm_total = ""
            for ii, _ in enumerate(split):
                sz = amps_sizes[ii]
                cstr = virtp[offset : offset + sz] + occp[offset : offset + sz]
                offset += sz
                sorted_set = sorted(set(cstr))
                keys.append("".join(sorted_set))
                perm_total += "".join(cstr)
            key = ",".join(sorted(keys))
            if key not in cache:
                kk = ",".join(keys)
                cache[key] = kk, get_term_permutation(kk, virt, occ)
    return cache


@jax.jit
def ci_to_cc(r1, r2, r3, r4):
    t1 = r1.copy()
    t2 = r2.copy()
    contracted_intermediate = einsum("aj,bi->abij", t1, t1)
    t2 += +1.0 * contracted_intermediate - 1.0 * contracted_intermediate.transpose(0, 1, 3, 2)

    t3 = r3.copy()
    term1 = -1.0 * einsum("ai,bcjk->abcijk", t1, t2)
    term2 = -1.0 * einsum("ai,bj,ck->abcijk", t1, t1, t1)
    for _, it in get_unique_contractions("ai,bcjk", "abc", "ijk").items():
        _, (sign, perm) = it
        t3 += sign * term1.transpose(perm)
    for _, it in get_unique_contractions("ai,bj,ck", "abc", "ijk").items():
        _, (sign, perm) = it
        t3 += sign * term2.transpose(perm)

    t4 = r4.copy()
    t1t3 = -1.0 * einsum("ai,bcdjkl->abcdijkl", t1, t3)
    t2t2 = -1.0 * einsum("abij,cdkl->abcdijkl", t2, t2)
    t1t1t2 = -1.0 * einsum("ai,bj,cdkl->abcdijkl", t1, t1, t2)
    t1_4 = -1.0 * einsum("ai,bj,ck,dl->abcdijkl", t1, t1, t1, t1)

    for _, it in get_unique_contractions("ai,bcdjkl", "abcd", "ijkl").items():
        _, (sign, perm) = it
        t4 += sign * t1t3.transpose(perm)
    for _, it in get_unique_contractions("abij,cdkl", "abcd", "ijkl").items():
        _, (sign, perm) = it
        t4 += sign * t2t2.transpose(perm)
    for _, it in get_unique_contractions("ai,bj,cdkl", "abcd", "ijkl").items():
        _, (sign, perm) = it
        t4 += sign * t1t1t2.transpose(perm)
    for _, it in get_unique_contractions("ai,bj,ck,dl", "abcd", "ijkl").items():
        _, (sign, perm) = it
        t4 += sign * t1_4.transpose(perm)

    return t1, t2, t3, t4
