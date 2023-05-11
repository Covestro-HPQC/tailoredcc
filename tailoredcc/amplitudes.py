# Proprietary and Confidential
# Covestro Deutschland AG, 2023
from typing import Tuple, Union

import covvqetools as cov
import numpy as np
import numpy.typing as npt
from pyscf import mcscf
from pyscf.cc.addons import spatial2spin, spin2spatial
from pyscf.ci.cisd import tn_addrs_signs


def extract_ci_singles_doubles_amplitudes_spinorb(mc: mcscf.casci.CASCI):
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

    if isinstance(mc.ci, (list, tuple)):
        raise NotImplementedError("Cannot handle CI with multiple roots.")
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
    if np.abs(c0) < 1e-8:
        raise ValueError("Coefficient of ref. determinant is too close to zero.")
    return amplitudes_to_spinorb(c0, cis_a, cis_b, cid_aa, cid_ab, cid_bb)


def extract_vqe_singles_doubles_amplitudes_spinorb(vqe: cov.vqe.ActiveSpaceChemistryVQE):
    if not hasattr(vqe, "compute_vqe_basis_state_overlaps"):
        raise NotImplementedError(
            "Given VQE cannot compute overlap with " "computational basis states."
        )
    # TODO: docs
    # NOTE: of uses alpha_1, beta_1, alpha_2, beta_2, ... MO ordering
    # (aka interleaved qubit ordering),
    ncas = vqe.nact
    nocca, noccb = vqe.nalpha, vqe.nbeta
    if nocca != noccb:
        raise NotImplementedError(
            "Amplitude conversion only implemented " "for closed-shell active space."
        )
    nvirta = ncas - nocca
    nvirtb = ncas - noccb
    assert nvirta == nvirtb

    # get the singly/doubly excited det addresses
    # and the corresponding sign changes (true vacuum -> Fermi vacuum)
    _, t1signs = tn_addrs_signs(ncas, nocca, 1)
    _, t2signs = tn_addrs_signs(ncas, nocca, 2)

    hfdet = np.zeros(ncas, dtype=int)
    hfdet[:nocca] = 1
    _, detsa = detstrings_singles(nocca, nvirta)
    _, detsaa = detstrings_doubles(nocca, nvirta)

    cis_a = (
        vqe.compute_vqe_basis_state_overlaps(interleave_strings(detsa, hfdet), vqe.params) * t1signs
    )
    cis_b = (
        vqe.compute_vqe_basis_state_overlaps(interleave_strings(hfdet, detsa), vqe.params) * t1signs
    )
    cis_a = cis_a.reshape(nocca, nvirta)
    cis_b = cis_b.reshape(noccb, nvirtb)

    cid_ab = vqe.compute_vqe_basis_state_overlaps(
        interleave_strings(detsa, detsa), vqe.params
    ).reshape(nocca * nvirta, noccb * nvirtb)
    cid_ab = np.einsum("ij,i,j->ij", cid_ab, t1signs, t1signs)
    cid_ab = cid_ab.reshape(nocca, nvirta, noccb, nvirtb).transpose(0, 2, 1, 3)

    cid_aa = (
        vqe.compute_vqe_basis_state_overlaps(interleave_strings(detsaa, hfdet), vqe.params)
        * t2signs
    )
    cid_bb = (
        vqe.compute_vqe_basis_state_overlaps(interleave_strings(hfdet, detsaa), vqe.params)
        * t2signs
    )

    c0 = vqe.compute_vqe_basis_state_overlaps(interleave_strings(hfdet, hfdet), vqe.params)[0]
    # TODO: replace with proper logging
    print(f"c0 = {c0:.8f}")
    print(f"|c0|^2 = {c0**2:.8f}")
    if np.abs(c0) < 1e-8:
        raise ValueError("Coefficient of reference determinant is too close to zero.")
    return amplitudes_to_spinorb(c0, cis_a, cis_b, cid_aa, cid_ab, cid_bb)


def remove_index_restriction_doubles(cid_aa: npt.NDArray, nocc: int, nvirt: int):
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


def amplitudes_to_spinorb(
    c0: float,
    cis_a: npt.NDArray,
    cis_b: npt.NDArray,
    cid_aa: npt.NDArray,
    cid_ab: npt.NDArray,
    cid_bb: npt.NDArray,
):
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


def ci_to_cluster_amplitudes(c_ia: npt.NDArray, c_ijab: npt.NDArray):
    # TODO: docs
    assert_spinorb_antisymmetric(c_ijab)
    t1 = c_ia.copy()
    t2 = c_ijab - np.einsum("ia,jb->ijab", t1, t1) + np.einsum("ib,ja->ijab", t1, t1)
    return t1, t2


def detstrings_singles(nocc: int, nvirt: int):
    # TODO: docs
    if nocc <= 0:
        raise ValueError("Cannot build determinant with 0 occupied orbitals.")
    if nvirt <= 0:
        raise ValueError("Cannot build determinant with 0 virtual orbitals.")
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


def detstrings_doubles(nocc: int, nvirt: int):
    # TODO: docs
    if nocc <= 0:
        raise ValueError("Cannot build determinant with 0 occupied orbitals.")
    if nvirt <= 0:
        raise ValueError("Cannot build determinant with 0 virtual orbitals.")
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


def assert_spinorb_antisymmetric(t2: npt.NDArray):
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
            atol=1e-13,
            rtol=0,
        )


def interleave_strings(alphas: Union[list, npt.NDArray], betas: Union[list, npt.NDArray]):
    if not isinstance(alphas[0], (list, np.ndarray)):
        alphas = [alphas]
    if not isinstance(betas[0], (list, np.ndarray)):
        betas = [betas]
    ncas = len(alphas[0])
    ret = []
    for alpha in alphas:
        for beta in betas:
            cdet = np.zeros(2 * ncas, dtype=int)
            cdet[::2] = alpha
            cdet[1::2] = beta
            interleaved_string = "".join(np.char.mod("%d", cdet))
            ret.append(interleaved_string)
    return ret


def set_cas_amplitudes_spatial_from_spinorb(
    t1: npt.NDArray,
    t2: npt.NDArray,
    t1cas: npt.NDArray,
    t2cas: npt.NDArray,
    t1slice: Tuple[slice, slice],
    t2slice: Tuple[slice, slice, slice, slice],
    zero_input=False,
):
    # TODO: docs
    t1spin = spatial2spin(t1)
    t2spin = spatial2spin(t2)

    if zero_input:
        # zero out the input amplitudes and only write t1cas/t2cas to the full array
        t1spin[...] = 0.0
        t2spin[...] = 0.0
    # write CAS amplitudes to slice
    t1spin[t1slice] = t1cas
    t2spin[t2slice] = t2cas

    if isinstance(t1, np.ndarray) and t1.ndim == 2:
        nocca, nvirta = t1.shape
        orbspin = np.zeros((nocca + nvirta) * 2, dtype=int)
        orbspin[1::2] = 1
        t1a, _ = spin2spatial(t1spin, orbspin)
        _, t2ab, _ = spin2spatial(t2spin, orbspin)
        t1spatial = t1a
        t2spatial = t2ab
    elif isinstance(t1, tuple) and len(t1) == 2:
        nocca, nvirta = t1[0].shape
        orbspin = np.zeros((nocca + nvirta) * 2, dtype=int)
        orbspin[1::2] = 1
        t1spatial = spin2spatial(t1spin, orbspin)
        t2spatial = spin2spatial(t2spin, orbspin)
    else:
        raise NotImplementedError("Unknown amplitude types.")
    return t1spatial, t2spatial


def prepare_cas_slices(
    nocca: int, noccb: int, nvirta: int, nvirtb: int, ncore: int, nvir: int, backend: str
):
    if backend in ["adcc", "libcc"]:
        occaslice = np.arange(ncore, ncore + nocca, 1)
        occbslice = np.arange(2 * ncore + nocca, 2 * ncore + nocca + noccb)
        virtaslice = np.arange(0, nvirta, 1)
        virtbslice = np.arange(nvirta + nvir, nvirta + nvir + nvirtb, 1)
        occslice = np.concatenate((occaslice, occbslice), axis=0)
        virtslice = np.concatenate((virtaslice, virtbslice), axis=0)
    elif backend in ["oe", "pyscf"]:
        occslice = slice(2 * ncore, 2 * ncore + nocca + noccb)
        virtslice = slice(0, nvirta + nvirtb)
    else:
        raise NotImplementedError(f"No CAS slices implemented for backend {backend}.")
    return occslice, virtslice
