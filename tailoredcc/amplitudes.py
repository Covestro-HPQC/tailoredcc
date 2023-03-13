# Proprietary and Confidential
# Covestro Deutschland AG, 2023
import numpy as np

from pyscf.ci.cisd import tn_addrs_signs
from pyscf.cc.addons import spatial2spin


def extract_ci_singles_doubles_amplitudes_spinorb(mc):
    # TODO: docs
    # NOTE: adapted from https://github.com/pyscf/pyscf/blob/master/examples/ci/20-from_fci.py
    # NOTE: of uses alpha_1, beta_1, alpha_2, beta_2, ... MO ordering
    # (aka interleaved qubit ordering),
    ncas = mc.ncas
    nocca, noccb = mc.nelecas
    if nocca != noccb:
        raise NotImplementedError(
            "Amplitude conversion only implemented " "for closed-shell active space."
        )
    nvirta = mc.ncas - nocca
    nvirtb = mc.ncas - noccb
    assert nvirta == nvirtb

    # get the singly/doubly excited det addresses
    # and the corresponding sign changes (true vacuum -> Fermi vacuum)
    t1addrs, t1signs = tn_addrs_signs(ncas, nocca, 1)
    t2addrs, t2signs = tn_addrs_signs(ncas, nocca, 2)

    fcivec = mc.ci
    # CIS includes two types of amplitudes: alpha -> alpha and beta -> beta
    cis_a = t1signs * fcivec[t1addrs, 0]
    cis_b = t1signs * fcivec[0, t1addrs]
    cis_a = cis_a.reshape(nocca, nvirta)
    cis_b = cis_b.reshape(noccb, nvirtb)

    # CID has:
    #    alpha,alpha -> alpha,alpha
    #    alpha,beta  -> alpha,beta
    #    beta ,beta  -> beta ,beta
    # For alpha,alpha -> alpha,alpha excitations, the redundant coefficients are
    # excluded. The number of coefficients is nocc*(nocc-1)//2 * nvir*(nvir-1)//2,
    # which corresponds to C2_{ijab}, i > j and a > b.
    cid_aa = fcivec[t2addrs, 0] * t2signs
    cid_bb = fcivec[0, t2addrs] * t2signs

    # alpha/beta -> alpha/beta
    cid_ab = np.zeros((nocca, noccb, nvirta, nvirtb))
    if len(t1addrs):
        cid_ab = np.einsum("ij,i,j->ij", fcivec[t1addrs[:, None], t1addrs], t1signs, t1signs)
        cid_ab = cid_ab.reshape(nocca, nvirta, noccb, nvirtb)
        # order is now occa, occb, virta, virtb
        cid_ab = cid_ab.transpose(0, 2, 1, 3)

    c0 = fcivec[0, 0]
    print(f"c0 = {c0:.8f}")
    print(f"|c0|^2 = {c0**2:.8f}")
    return amplitudes_to_spinorb(c0, cis_a, cis_b, cid_aa, cid_ab, cid_bb)


def remove_index_restriction_doubles(cid_aa, nocc, nvirt):
    # TODO: docs
    assert cid_aa.ndim == 1
    assert cid_aa.size == nocc * (nocc - 1) // 2 * nvirt * (nvirt - 1) // 2
    cid_aa_full = np.zeros((nocc, nocc, nvirt, nvirt))
    idx = 0
    for j in range(nocc):
        for i in range(j):
            for b in range(nvirt):
                for a in range(b):
                    coeff = cid_aa[idx]
                    cid_aa_full[i, j, a, b] = coeff
                    cid_aa_full[j, i, a, b] = -1.0 * coeff
                    cid_aa_full[i, j, b, a] = -1.0 * coeff
                    cid_aa_full[j, i, b, a] = coeff
                    idx += 1
    return cid_aa_full


def amplitudes_to_spinorb(c0, cis_a, cis_b, cid_aa, cid_ab, cid_bb):
    # TODO: docs
    nocca, nvirta = cis_a.shape
    noccb, nvirtb = cis_b.shape
    assert cid_ab.shape == (nocca, noccb, nvirta, nvirtb)
    # arrays containing all coefficients without index restrictions
    cid_aa_full = remove_index_restriction_doubles(cid_aa, nocca, nvirta)
    cid_bb_full = remove_index_restriction_doubles(cid_bb, noccb, nvirtb)
    c_ia = spatial2spin((cis_a, cis_b))
    c_ijab = spatial2spin((cid_aa_full, cid_ab, cid_bb_full))

    assert_spinorb_antisymmetric(c_ijab)

    # normalize coefficients
    c_ia /= c0
    c_ijab /= c0
    return c_ia, c_ijab


def ci_to_cluster_amplitudes(c_ia, c_ijab):
    # TODO: docs
    assert_spinorb_antisymmetric(c_ijab)
    t1 = c_ia.copy()
    t2 = c_ijab - np.einsum("ia,jb->ijab", t1, t1) + np.einsum("ib,ja->ijab", t1, t1)
    return t1, t2


def detstrings_singles(nocc, nvirt):
    # TODO: docs
    ncas = nocc + nvirt
    detstrings = []
    detstrings_np = []
    for i in range(nocc):
        for a in range(nvirt):
            string = np.zeros(ncas, dtype=int)
            string[:nocc] = 1
            string[i] = 0
            string[nocc + a] = 1
            detstrings_np.append(string)
            without_zeros = np.trim_zeros(string[::-1], "f")
            cstr = "".join([str(x) for x in without_zeros])
            cstr = "0b" + cstr
            detstrings.append(cstr)
    return detstrings, detstrings_np


def detstrings_doubles(nocc, nvirt):
    # TODO: docs
    ncas = nocc + nvirt
    detstrings = []
    detstrings_np = []
    for j in range(nocc):
        for i in range(j):
            for b in range(nvirt):
                for a in range(b):
                    string = np.zeros(ncas, dtype=int)
                    string[:nocc] = 1
                    string[j] = 0
                    string[i] = 0
                    string[nocc + b] = 1
                    string[nocc + a] = 1
                    detstrings_np.append(string)
                    without_zeros = np.trim_zeros(string[::-1], "f")
                    cstr = "".join([str(x) for x in without_zeros])
                    cstr = "0b" + cstr
                    detstrings.append(cstr)
    return detstrings, detstrings_np


def assert_spinorb_antisymmetric(t2):
    if t2.ndim != 4:
        raise ValueError(f"Tensor must have 4 dimensions, got {t2.ndim}.")
    perm_sign = [
        ((0, 1, 2, 3), 1.0),
        ((1, 0, 2, 3), -1.0),
        ((1, 0, 3, 2), 1.0),
        ((0, 1, 3, 2), -1.0),
    ]
    for p, s in perm_sign:
        np.testing.assert_allclose(
            t2,
            s * t2.transpose(*p),
            err_msg=f"Tensor does not have correct antisymmetry. Permutation {p} failed.",
        )
