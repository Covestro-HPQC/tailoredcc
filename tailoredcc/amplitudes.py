# Proprietary and Confidential
# Covestro Deutschland AG, 2023
import operator
from collections import defaultdict
from itertools import combinations_with_replacement, permutations, product
from math import factorial
from typing import Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
from pyscf import mcscf
from pyscf.cc.addons import spatial2spin, spin2spatial
from pyscf.ci.cisd import tn_addrs_signs
from pyscf.fci import cistring


def extract_ci_amplitudes(mc: mcscf.casci.CASCI, exci=2) -> Dict[str, npt.NDArray]:
    """Extract CI amplidutes from a PySCF CASCI object up to a certain excitation level.
    The signs of the CI amplitudes are converted from true vacuum to Fermi vacuum convention.

    Parameters
    ----------
    mc : mcscf.casci.CASCI
        CASCI object with a converged CI vector
    exci : int, optional
        maximum excitation level, by default 2

    Returns
    -------
    Dict[str, npt.NDArray]
        Dictionary with spin block labels
        (alpha -> `"a"`, alpha/beta -> `"ab"`)
        as keys and CI amplitudes as values.
        The axes of the amplitudes are sorted by occupied/virtual and
        alpha/beta, i.e., the `"ab"` block axes are occ-alpha, occ-beta,
        virt-alpha, virt-beta.
    """
    # NOTE: adapted from https://github.com/pyscf/pyscf/blob/master/examples/ci/20-from_fci.py
    ncas = mc.ncas
    nocca, noccb = mc.nelecas
    if nocca != noccb:
        raise NotImplementedError(
            "Amplitude conversion only implemented for closed-shell active space."
        )
    nvirta = mc.ncas - nocca
    nvirtb = mc.ncas - noccb
    assert nvirta == nvirtb

    if isinstance(mc.ci, (list, tuple)):
        raise NotImplementedError("Cannot handle CI with multiple roots.")
    fcivec = mc.ci

    ret = {}
    c0 = fcivec[0, 0]
    ret["0"] = c0

    if exci >= 1:
        t1addrs, t1signs = tn_addrs_signs(ncas, nocca, 1)
        cis_a = t1signs * fcivec[t1addrs, 0]
        cis_b = t1signs * fcivec[0, t1addrs]
        cis_a = cis_a.reshape(nocca, nvirta)
        cis_b = cis_b.reshape(noccb, nvirtb)
        ret["a"] = cis_a
        ret["b"] = cis_b

    if exci >= 2:
        t2addrs, t2signs = tn_addrs_signs(ncas, nocca, 2)
        cid_aa = fcivec[t2addrs, 0] * t2signs
        cid_bb = fcivec[0, t2addrs] * t2signs

        # alpha/beta -> alpha/beta
        cid_ab = np.zeros((nocca, noccb, nvirta, nvirtb))
        if len(t1addrs):
            cid_ab = np.einsum("ij,i,j->ij", fcivec[t1addrs[:, None], t1addrs], t1signs, t1signs)
            cid_ab = cid_ab.reshape(nocca, nvirta, noccb, nvirtb)
            # order is now occa, occb, virta, virtb
            cid_ab = cid_ab.transpose(0, 2, 1, 3)

        ret["aa"] = cid_aa
        ret["bb"] = cid_bb
        ret["ab"] = cid_ab

    if exci >= 3:
        t3addrs, t3signs = tn_addrs_signs(ncas, nocca, 3)
        cit_aaa = fcivec[t3addrs, 0] * t3signs
        cit_bbb = fcivec[0, t3addrs] * t3signs
        cit_aab = np.einsum("ij,i,j->ij", fcivec[t2addrs[:, None], t1addrs], t2signs, t1signs)
        cit_abb = np.einsum("ij,i,j->ij", fcivec[t1addrs[:, None], t2addrs], t1signs, t2signs)
        ret["aaa"] = cit_aaa
        ret["bbb"] = cit_bbb
        ret["aab"] = cit_aab
        ret["abb"] = cit_abb

    if exci >= 4:
        t4addrs, t4signs = tn_addrs_signs(ncas, nocca, 4)
        ciq_aaaa = fcivec[t4addrs, 0] * t4signs
        ciq_bbbb = fcivec[0, t4addrs] * t4signs
        if len(t3addrs) and len(t1addrs):
            ciq_aaab = np.einsum("ij,i,j->ij", fcivec[t3addrs[:, None], t1addrs], t3signs, t1signs)
            ciq_abbb = np.einsum("ij,i,j->ij", fcivec[t1addrs[:, None], t3addrs], t1signs, t3signs)
        else:
            ciq_aaab = np.zeros((0, noccb * nvirtb))
            ciq_abbb = np.zeros((nocca * nvirta, 0))

        if len(t2addrs):
            ciq_aabb = np.einsum("ij,i,j->ij", fcivec[t2addrs[:, None], t2addrs], t2signs, t2signs)
        else:
            raise ValueError()
            ciq_aabb = np.zeros((nocca, nocca, noccb, noccb, nvirta, nvirta, nvirtb, nvirtb))
        ret["aaaa"] = ciq_aaaa
        ret["bbbb"] = ciq_bbbb
        ret["aaab"] = ciq_aaab
        ret["aabb"] = ciq_aabb
        ret["abbb"] = ciq_abbb

    return ret


def extract_from_dict(
    amplitudes: Dict[str, npt.NDArray], ncas: int, nocca: int, noccb: int, exci: int = 2
) -> Dict[str, npt.NDArray]:
    """Maps CI amplitudes contained in a dictionary to the correct shape
    and changes their sign convention from true vacuum to Fermi vacuum.

    Parameters
    ----------
    amplitudes : Dict[str, npt.NDArray]
        dictionary with spin block labels (alpha -> `"a"`, alpha/beta -> `"ab"`) as keys
        and CI amplitudes as values. The values need to be flat arrays, and the values
        are expected to be sorted according to the bit strings
        provided by :py:func:`tailoredcc.amplitudes.determinant_strings`.
    ncas : int
        number of active orbitals
    nocca : int
        number of active alpha electrons
    noccb : int
        number of active beta electrons
    exci : int, optional
        maximum excitation level, by default 2

    Returns
    -------
    Dict[str, npt.NDArray]
        dictionary with spin block labels
        (alpha -> `"a"`, alpha/beta -> `"ab"`)
        as keys and CI amplitudes as values.
        The axes of the amplitudes are sorted by occupied/virtual and
        alpha/beta, i.e., the `"ab"` block axes are occ-alpha, occ-beta,
        virt-alpha, virt-beta.
    """
    if exci > 2:
        raise NotImplementedError("Dictionary input only implemented up to double excitations.")
    assert nocca == noccb
    nvirta = ncas - nocca
    nvirtb = ncas - noccb
    # sign changes (true vacuum -> Fermi vacuum)
    _, t1signs = tn_addrs_signs(ncas, nocca, 1)
    _, t2signs = tn_addrs_signs(ncas, nocca, 2)

    cis_a = amplitudes["a"] * t1signs
    cis_b = amplitudes["b"] * t1signs
    cis_a = cis_a.reshape(nocca, nvirta)
    cis_b = cis_b.reshape(noccb, nvirtb)

    cid_ab = amplitudes["ab"].reshape(nocca * nvirta, noccb * nvirtb)
    cid_ab = np.einsum("ij,i,j->ij", cid_ab, t1signs, t1signs)
    cid_ab = cid_ab.reshape(nocca, nvirta, noccb, nvirtb).transpose(0, 2, 1, 3)

    cid_aa = amplitudes["aa"] * t2signs
    cid_bb = amplitudes["bb"] * t2signs

    c0 = amplitudes["0"]
    ret = {"0": c0, "a": cis_a, "b": cis_b, "aa": cid_aa, "bb": cid_bb, "ab": cid_ab}
    return ret


def remove_index_restriction_doubles(cid_aa: npt.NDArray, nocc: int, nvirt: int):
    # TODO: docs
    assert cid_aa.ndim == 1
    assert cid_aa.size == nocc * (nocc - 1) // 2 * nvirt * (nvirt - 1) // 2
    cid_aa_full = np.zeros((nocc, nocc, nvirt, nvirt), dtype=cid_aa.dtype)
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


def remove_index_restriction_triples(tensor: npt.NDArray, nocc: int, nvirt: int):
    exci = 3
    assert tensor.size == number_nonredundant_amplitudes(
        nocc + nvirt, nocc, exci
    ), f"{tensor.size},{number_nonredundant_amplitudes(nocc + nvirt, nocc, exci)}"
    assert tensor.ndim == 1
    idx = 0
    ret = np.zeros(exci * (nocc,) + exci * (nvirt,))
    perms_signs = get_permutations_signs(exci)
    for k in range(nocc):
        for j in range(k):
            for i in range(j):
                for c in range(nvirt):
                    for b in range(c):
                        for a in range(b):
                            coeff = tensor[idx]
                            indextuple = (i, j, k, a, b, c)
                            for pp, s in perms_signs:
                                current_perm = operator.itemgetter(*pp)(indextuple)
                                ret[current_perm] = s * coeff
                            idx += 1
    assert idx == tensor.size
    return ret


def remove_index_restriction_quadruples(tensor: npt.NDArray, nocc: int, nvirt: int):
    exci = 4
    assert tensor.size == number_nonredundant_amplitudes(nocc + nvirt, nocc, exci)
    assert tensor.ndim == 1
    idx = 0
    ret = np.zeros(exci * (nocc,) + exci * (nvirt,))
    perms_signs = get_permutations_signs(exci)
    for l in range(nocc):
        for k in range(l):
            for j in range(k):
                for i in range(j):
                    for d in range(nvirt):
                        for c in range(d):
                            for b in range(c):
                                for a in range(b):
                                    coeff = tensor[idx]
                                    indextuple = (i, j, k, l, a, b, c, d)
                                    for pp, s in perms_signs:
                                        current_perm = operator.itemgetter(*pp)(indextuple)
                                        ret[current_perm] = s * coeff
                                    idx += 1
    assert idx == tensor.size
    return ret


def amplitudes_to_spinorb(amplitudes: dict, exci: int = 2):
    """Convert a dictionary with spatial
    amplitudes (canonical blocks) to spin
    orbital amplitudes and renormalize them
    to the ref. determinant.

    Parameters
    ----------
    amplitudes : dict
        Dictionary with excitation amplitudes labeled
        by spin block, i.e., the key 'a' denotes a single
        excitation in the alpha block, 'ab' a double excitation
        in alpha/beta, and the key '0' holds the coefficient of the
        reference determinant.
    exci : int, optional
        Maximum excitation level in the input amplitudes, by default 2

    Returns
    -------
    list
        List with spin-orbital amplitudes ordered by excitation level

    Raises
    ------
    ValueError
        If the coefficient of the reference determinant is too close to zero.
    """
    c0 = amplitudes["0"]
    if np.abs(c0) < 1e-8:
        raise ValueError("Coefficient of ref. determinant is too close to zero.")

    ret = []
    if exci >= 1:
        a = amplitudes["a"]
        b = amplitudes["b"]
        nocca, nvirta = a.shape
        noccb, nvirtb = b.shape
        c_ia = spatial_to_spinorb({"a": a, "b": b}, 1, nocca, noccb, nvirta, nvirtb)
        c_ia /= c0
        ret.append(c_ia)

    if exci >= 2:
        aa = amplitudes["aa"]
        bb = amplitudes["bb"]
        ab = amplitudes["ab"]
        assert ab.shape == (nocca, noccb, nvirta, nvirtb)
        # arrays containing all coefficients without index restrictions
        aa_full = remove_index_restriction_doubles(aa, nocca, nvirta)
        bb_full = remove_index_restriction_doubles(bb, noccb, nvirtb)

        c_ijab = spatial_to_spinorb(
            {"aa": aa_full, "ab": ab, "bb": bb_full},
            2,
            nocca,
            noccb,
            nvirta,
            nvirtb,
        )
        c_ijab /= c0
        ret.append(c_ijab)

    if exci >= 3:
        aab = amplitudes["aab"]
        abb = amplitudes["abb"]

        aaa = remove_index_restriction_triples(amplitudes["aaa"], nocca, nvirta)
        bbb = remove_index_restriction_triples(amplitudes["bbb"], noccb, nvirtb)

        aab_full = np.zeros((nocca, nocca, nvirta, nvirta, noccb * nvirtb))
        abb_full = np.zeros((noccb, noccb, nvirtb, nvirtb, nocca * nvirta))
        for row in range(aab.shape[1]):
            aab_full[:, :, :, :, row] = remove_index_restriction_doubles(aab[:, row], nocca, nvirta)

        for row in range(abb.shape[0]):
            abb_full[:, :, :, :, row] = remove_index_restriction_doubles(abb[row, :], noccb, nvirtb)

        # 0 1 | 2 3 | 4(beta) 5(beta) -> 0 1 4 | 2 3 5
        aab_full = aab_full.reshape(nocca, nocca, nvirta, nvirta, noccb, nvirtb).transpose(
            0, 1, 4, 2, 3, 5
        )
        # 0 1 | 2 3 | 4(alpha) 5(alpha) -> 4 0 1 | 5 2 3
        abb_full = abb_full.reshape(noccb, noccb, nvirtb, nvirtb, nocca, nvirta).transpose(
            4, 0, 1, 5, 2, 3
        )

        c_ijkabc = spatial_to_spinorb(
            {"aaa": aaa, "bbb": bbb, "aab": aab_full, "abb": abb_full},
            3,
            nocca,
            noccb,
            nvirta,
            nvirtb,
        )
        c_ijkabc /= c0
        ret.append(c_ijkabc)

    if exci >= 4:
        aaaa = remove_index_restriction_quadruples(amplitudes["aaaa"], nocca, nvirta)
        bbbb = remove_index_restriction_quadruples(amplitudes["bbbb"], noccb, nvirtb)
        abbb = amplitudes["abbb"]
        aaab = amplitudes["aaab"]
        aabb = amplitudes["aabb"]

        aaab_full = np.zeros((nocca, nocca, nocca, nvirta, nvirta, nvirta, noccb * nvirtb))
        for row in range(aaab.shape[1]):
            aaab_full[..., row] = remove_index_restriction_triples(aaab[:, row], nocca, nvirta)
        aaab_full = aaab_full.reshape(
            nocca, nocca, nocca, nvirta, nvirta, nvirta, noccb, nvirtb
        ).transpose(0, 1, 2, 6, 3, 4, 5, 7)

        abbb_full = np.zeros((noccb, noccb, noccb, nvirtb, nvirtb, nvirtb, nocca * nvirta))
        for row in range(abbb.shape[0]):
            abbb_full[..., row] = remove_index_restriction_triples(abbb[row, :], noccb, nvirtb)
        abbb_full = abbb_full.reshape(
            noccb, noccb, noccb, nvirtb, nvirtb, nvirtb, nocca, nvirta
        ).transpose(6, 0, 1, 2, 7, 3, 4, 5)

        # aabb has 2x index restrictions of doubles
        # 1. remove the index restriction in the aa part
        num_nonred_bb = number_nonredundant_amplitudes(noccb + nvirtb, noccb, 2)
        aabb_tmp1 = np.zeros((nocca, nocca, nvirta, nvirta, num_nonred_bb))
        for row in range(aabb.shape[1]):
            aabb_tmp1[..., row] = remove_index_restriction_doubles(aabb[:, row], nocca, nvirta)
        aabb_tmp1 = aabb_tmp1.reshape(nocca * nocca * nvirta * nvirta, num_nonred_bb)

        # 2. remove the index restriction in the bb part
        aabb_full = np.zeros((nocca * nocca * nvirta * nvirta, noccb, noccb, nvirtb, nvirtb))
        for row in range(aabb_tmp1.shape[0]):
            aabb_full[row, ...] = remove_index_restriction_doubles(aabb_tmp1[row, :], noccb, nvirtb)
        aabb_full = aabb_full.reshape(
            nocca, nocca, nvirta, nvirta, noccb, noccb, nvirtb, nvirtb
        ).transpose(0, 1, 4, 5, 2, 3, 6, 7)
        c4 = spatial_to_spinorb(
            {"aaaa": aaaa, "bbbb": bbbb, "aaab": aaab_full, "abbb": abbb_full, "aabb": aabb_full},
            4,
            nocca,
            noccb,
            nvirta,
            nvirtb,
        )
        c4 /= c0
        ret.append(c4)

    if exci >= 5:
        raise NotImplementedError()

    return ret


def check_amplitudes_spinorb(tt, exci=2, check_spin_forbidden_blocks=True):
    # for p1 in permutations(range(exci)):
    #     sign1 = compute_parity(p1)
    #     for p2 in permutations(range(exci)):
    #         sign2 = compute_parity(p2)
    #         perm_total = np.concatenate([np.array(p1), exci + np.array(p2)])
    #         np.testing.assert_allclose(
    #             tt, (sign1 * sign2 * tt.transpose(perm_total)),
    #             err_msg=f"{perm_total}", atol=1e-8
    #         )
    #         isok = np.allclose(tt, (sign1 * sign2 * tt.transpose(perm_total)), atol=1e-8)
    #         if not isok:
    #             print(perm_total, 'failed')
    #         else:
    #             print(perm_total, 'worked')

    # TODO: need to think about whether this is sufficient or not...
    for p1 in permutations(range(exci)):
        sign1 = compute_parity(p1)
        p2 = list(range(exci))
        perm_total = np.concatenate([np.array(p1), exci + np.array(p2)])
        try:
            np.testing.assert_allclose(
                tt, sign1 * tt.transpose(perm_total), err_msg=f"{perm_total}, hole", atol=1e-8
            )
        except AssertionError:
            print(perm_total, "failed")
        else:
            print(perm_total, "OK")
        perm_total = np.concatenate([np.array(p2), exci + np.array(p1)])
        try:
            np.testing.assert_allclose(
                tt, sign1 * tt.transpose(perm_total), err_msg=f"{perm_total}, particle", atol=1e-8
            )
        except AssertionError:
            print(perm_total, "failed")
        else:
            print(perm_total, "OK")

    if check_spin_forbidden_blocks:
        slices = {"a": slice(0, None, 2), "b": slice(1, None, 2)}
        spinblocks = list(product("ab", repeat=exci))
        for comb in product(list(spinblocks), repeat=2):
            ospin, vspin = comb
            ostr = "".join(ospin)
            vstr = "".join(vspin)
            if ospin.count("a") != vspin.count("a") or ospin.count("b") != vspin.count("b"):
                viewslice = tuple(slices[ii] for ii in ostr + vstr)
                view = tt[viewslice]
                np.testing.assert_allclose(view, np.zeros_like(view), atol=1e-14, rtol=0)


def ci_to_cluster_amplitudes(c_ia: npt.NDArray, c_ijab: npt.NDArray):
    t1 = c_ia.copy()
    t2 = c_ijab - np.einsum("ia,jb->ijab", t1, t1) + np.einsum("ib,ja->ijab", t1, t1)
    return t1, t2


def determinant_strings(ncas, nocc, level=4):
    from itertools import product

    addrs_level = []
    for l in range(level + 1):
        addrs = cistring.addrs2str(ncas, nocc, tn_addrs_signs(ncas, nocc, l)[0])
        addrs = np.array([bin(ad) for ad in addrs.ravel()])
        addrs_level.append(addrs)

    ret = {}
    for la, addrsa in enumerate(addrs_level):
        for lb, addrsb in enumerate(addrs_level):
            if la + lb <= level:
                label = "a" * la + "b" * lb if la + lb > 0 else "0"
                strs_ab = product(addrsa, addrsb)
                ret[label] = list(strs_ab)
    return ret


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
    if backend in ["oe", "pyscf"]:
        occslice = slice(2 * ncore, 2 * ncore + nocca + noccb)
        virtslice = slice(0, nvirta + nvirtb)
    else:
        raise NotImplementedError(f"No CAS slices implemented for backend {backend}.")
    return occslice, virtslice


def number_nonredundant_amplitudes(nmo, nocc, exci):
    nvirt = nmo - nocc
    ret = 1
    for e in range(exci):
        ret *= (nocc - e) * (nvirt - e)
    ret //= (factorial(exci)) ** 2
    return ret


def number_overlaps_tccsd(nmo: int, nalpha: int, nbeta: int) -> int:
    """Number of amplitudes/overlaps that need
    to be extracted from a state for TCCSD, i.e.,
    these include the ref. det., all alpha/beta single
    excitations, and double excitations.

    Parameters
    ----------
    nmo : int
        number of spatial active orbitals
    nalpha : int
        number of active alpha electrons
    nbeta : int
        number of active beta electrons

    Returns
    -------
    int
        number of required amplitudes
    """
    nvirta = nmo - nalpha
    nvirtb = nmo - nbeta
    singles = number_nonredundant_amplitudes(nmo, nalpha, 1) + number_nonredundant_amplitudes(
        nmo, nbeta, 1
    )
    doubles = number_nonredundant_amplitudes(nmo, nalpha, 2) + number_nonredundant_amplitudes(
        nmo, nbeta, 2
    )
    ret = 1 + singles + doubles + (nalpha * nbeta * nvirta * nvirtb)
    return ret


def number_overlaps_eccc(nmo: int, nalpha: int, nbeta: int) -> int:
    """Number of amplitudes/overlaps that need
    to be extracted from a state for ec-CC, i.e.,
    these include the ref. det., and all non-redundant
    excitation amplitudes up to quadruple excitations.

    Parameters
    ----------
    nmo : int
        number of spatial active orbitals
    nalpha : int
        number of active alpha electrons
    nbeta : int
        number of active beta electrons

    Returns
    -------
    int
        number of required amplitudes
    """
    nvirta = nmo - nalpha
    nvirtb = nmo - nbeta
    ret = number_overlaps_tccsd(nmo, nalpha, nbeta)
    triples = number_nonredundant_amplitudes(nmo, nalpha, 3) + number_nonredundant_amplitudes(
        nmo, nbeta, 3
    )  # aaa + bbb
    triples += number_nonredundant_amplitudes(nmo, nalpha, 2) * (nbeta * nvirtb)  # aab
    triples += number_nonredundant_amplitudes(nmo, nalpha, 2) * (nalpha * nvirta)  # abb
    quadruples = number_nonredundant_amplitudes(nmo, nalpha, 4) + number_nonredundant_amplitudes(
        nmo, nbeta, 4
    )  # aaaa + bbbb
    quadruples += number_nonredundant_amplitudes(nmo, nalpha, 3) * (nbeta * nvirtb)  # aaab
    quadruples += number_nonredundant_amplitudes(nmo, nbeta, 3) * (nalpha * nvirta)  # abbb
    quadruples += number_nonredundant_amplitudes(nmo, nalpha, 2) * number_nonredundant_amplitudes(
        nmo, nbeta, 2
    )  # aabb
    ret += triples + quadruples
    return ret


def compute_parity(perm):
    inversions = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def get_permutations_signs(exci):
    perms_signs = []
    for p1 in permutations(range(exci)):
        sign1 = compute_parity(p1)
        for p2 in permutations(range(exci)):
            sign2 = compute_parity(p2)
            perm_total = np.concatenate([np.array(p1), exci + np.array(p2)])
            perms_signs.append((perm_total, sign1 * sign2))
    return perms_signs


def get_canonical_block(block, canonical_blocks):
    canonical_block = "".join(sorted(block))
    assert tuple(canonical_block) in canonical_blocks
    perm = list(permutations(range(len(block))))
    if len(perm) == 1:
        return canonical_block, perm[0]
    # NOTE: need the permutation of the canonical block to the non-canonical one,
    # so that we know how to transpose the canonical block and write it to the output tensor
    block_tuple = tuple(block)
    for p in perm:
        if operator.itemgetter(*p)(canonical_block) == block_tuple:
            return canonical_block, list(p)
    raise ValueError(f"No canonical block transposition found for '{block}'.")


def spinorb_to_spatial(tensor, exci_level, nocc_a, nocc_b, nvirt_a, nvirt_b):
    nocc = nocc_a + nocc_b
    nvirt = nvirt_a + nvirt_b
    shape = exci_level * (nocc,) + exci_level * (nvirt,)
    assert tensor.shape == shape

    spatial_strings = list(combinations_with_replacement("ab", exci_level))
    slices = {"a": slice(0, None, 2), "b": slice(1, None, 2)}
    ret = {}
    for canblock in spatial_strings:
        blockstr = "".join(canblock)
        both_blocks = canblock + canblock
        tmpslices = tuple(slices[b] for b in both_blocks)
        ret[blockstr] = tensor[tmpslices]
    return ret


def spatial_to_spinorb(tensors_dict, exci_level, nocc_a, nocc_b, nvirt_a, nvirt_b):
    assert nocc_a == nocc_b
    assert nvirt_a == nvirt_b
    nocc = nocc_a + nocc_b
    nvirt = nvirt_a + nvirt_b

    orbspin = np.zeros(nocc + nvirt, dtype=int)
    orbspin[1::2] = 1

    occa = np.where(orbspin[:nocc] == 0)[0]
    occb = np.where(orbspin[:nocc] == 1)[0]
    virta = np.where(orbspin[nocc:] == 0)[0]
    virtb = np.where(orbspin[nocc:] == 1)[0]

    block_to_indices = {
        "a": (occa, virta),
        "b": (occb, virtb),
    }

    # all possible alpha/beta combinations (also non-canonical ones)
    spinblocks = list(product("ab", repeat=exci_level))
    spatial_strings = list(combinations_with_replacement("ab", exci_level))
    slices_spinorb = defaultdict(dict)
    for block in spinblocks:
        blockstr = "".join(block)
        slices_spinorb["occ"][blockstr] = [block_to_indices[b][0] for b in block]
        slices_spinorb["virt"][blockstr] = [block_to_indices[b][1] for b in block]

    ooo = slices_spinorb["occ"]
    vvv = slices_spinorb["virt"]

    t_out = np.zeros(
        exci_level * (nocc,) + exci_level * (nvirt,), dtype=list(tensors_dict.values())[0].dtype
    )
    for comb in product(list(spinblocks), repeat=2):
        ospin, vspin = comb
        ostr = "".join(ospin)
        vstr = "".join(vspin)
        if ospin.count("a") != vspin.count("a") or ospin.count("b") != vspin.count("b"):
            continue

        ocan, operm = get_canonical_block(ostr, spatial_strings)
        vcan, vperm = get_canonical_block(vstr, spatial_strings)

        t_get = tensors_dict.get(ocan, tensors_dict.get(vcan, None))
        if t_get is None:
            raise ValueError(f"No tensor found: {comb}")

        total_perm = np.concatenate([np.array(operm), np.array(vperm) + len(ospin)])
        sign = compute_parity(operm) * compute_parity(vperm)
        # print(f"Current block: {label}")
        # if ocan != ostr or vcan != vstr:
        #     print(f"Canonical form:", ocan, vcan, operm, vperm)
        #     print("=> Permutation", total_perm, sign)
        t_out[np.ix_(*ooo[ostr], *vvv[vstr])] = sign * t_get.transpose(*total_perm)

    return t_out
