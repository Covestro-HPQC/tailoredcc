import numpy as np
from pyscf.ci.cisd import tn_addrs_signs
from pyscf.fci import cistring
from scipy.special import comb

ncas = 20
nocc = 15


t1addrs, _ = tn_addrs_signs(ncas, nocc, 1)
t2addrs, _ = tn_addrs_signs(ncas, nocc, 2)

full_list = cistring.gen_strings4orblist(range(ncas), nocc).tolist()

if len(t1addrs):
    detstrings_ref = [bin(ds) for ds in cistring.addrs2str(ncas, nocc, t1addrs.ravel())]


def str2addr(norb, nelec, string):
    addr = 0
    nelec_left = nelec
    norb_left = norb - 1
    for norb_left in range(norb_left, -1, -1):
        if nelec_left == 0 or norb_left < nelec_left:
            break
        elif 1 << norb_left & string:
            addr += comb(norb_left, nelec_left, exact=True)
            nelec_left -= 1
    return addr


for det, addr_ref in zip(detstrings_ref, t1addrs.ravel()):
    # print(det)
    addr = str2addr(ncas, nocc, int(det, 2))
    # print(addr, addr_ref)
    assert addr == addr_ref


for det, addr_ref in zip(detstrings_ref, t1addrs.ravel()):
    # print(det)
    addr = full_list.index(int(det, 2))
    # print(addr, addr_ref)
    assert addr == addr_ref
