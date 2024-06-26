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


import tempfile
from collections import defaultdict
from itertools import permutations, product
from shutil import which

import numpy as np
import pytest
from pyscf.fci import cistring
from tqdm import tqdm

from tailoredcc.amplitudes import (
    amplitudes_to_spinorb,
    assert_spinorb_antisymmetric,
    check_amplitudes_spinorb,
    ci_to_cluster_amplitudes,
    compute_parity,
    extract_ci_amplitudes,
    number_overlaps_eccc,
    remove_index_restriction_doubles,
    spatial_to_spinorb,
    spinorb_to_spatial,
)
from tailoredcc.ci_to_cc import ci_to_cc
from tailoredcc.clusterdec import dump_clusterdec, run_clusterdec


@pytest.fixture(scope="module")
def scf_ci():
    from pyscf import gto, mcscf, scf

    mol = gto.Mole()
    mol.build(
        verbose=4,
        # atom="N, 0., 0., 0. ; N,  0., 0., 1.4",
        # atom="H, 0., 0., 0. ; H,  0., 0., 1.0",
        # atom="He, 0., 0., 0. ; He,  0., 0., 1.0",
        atom="Li, 0., 0., 0. ; Li,  0., 0., 1.0",
        # basis="minao",
        basis="sto-3g",
        # basis="3-21g",
        # basis="6-31g",
        # basis="cc-pvdz",
        # symmetry = True,
    )
    m = scf.RHF(mol)
    m.kernel()

    ncas = mol.nao_nr()
    nelec = mol.nelec

    print(f"CAS({nelec}, {ncas})")
    mc = mcscf.CASCI(m, ncas, nelec)
    mc.fcisolver.conv_tol = 1e-12
    mc.canonicalization = False
    mc.kernel()

    ncas = 4
    nelec = 4
    print(f"CAS({nelec}, {ncas})")
    mc2 = mcscf.CASCI(m, ncas, nelec)
    mc2.canonicalization = False
    mc2.kernel()

    return m, mc, mc2


def test_determinant_strings(scf_ci):
    from tailoredcc.amplitudes import (
        determinant_strings,
        extract_ci_amplitudes,
        extract_from_dict,
    )

    scfres, mc, mc2 = scf_ci
    ncas, nelec = mc2.ncas, sum(mc2.nelecas)

    dets = determinant_strings(ncas, nelec // 2, level=2)
    civec = mc2.ci

    ret = defaultdict(list)
    for exci, str_tuples in dets.items():
        for str_tuple in str_tuples:
            addr = tuple(cistring.str2addr(ncas, nelec // 2, st) for st in str_tuple)
            ret[exci].append(civec[addr])
        ret[exci] = np.asarray(ret[exci])

    ret = extract_from_dict(ret, ncas, nelec // 2, nelec // 2, exci=2)

    amp_dict_ref = extract_ci_amplitudes(mc2)
    assert ret.keys() == amp_dict_ref.keys()

    for exci in ret:
        ref = amp_dict_ref[exci]
        r = ret[exci]
        print(exci, np.allclose(abs(ref), abs(r), atol=1e-14, rtol=0))
        np.testing.assert_allclose(ref, r, atol=1e-14, rtol=0)


@pytest.mark.parametrize(
    "nocc, nvirt",
    [
        (2, 2),
        (2, 4),
        (6, 7),
        (7, 10),
    ],
)
def test_remove_index_restriction_doubles(nocc, nvirt):
    np.random.seed(42)
    sz = nocc * (nocc - 1) // 2 * nvirt * (nvirt - 1) // 2
    cid = np.random.randn(sz)
    cid_full = remove_index_restriction_doubles(cid, nocc, nvirt)
    assert_spinorb_antisymmetric(cid_full)
    idx = 0
    occ = np.tril_indices(nocc, -1)
    virt = np.tril_indices(nvirt, -1)
    for j, i in zip(*occ):
        for b, a in zip(*virt):
            assert cid[idx] == cid_full[i, j, a, b]
            assert cid[idx] == cid_full[j, i, b, a]
            idx += 1
    assert idx == cid.size


@pytest.mark.parametrize(
    "nocc, nvirt",
    [
        (2, 2),
        (2, 4),
        (6, 7),
        (7, 10),
        (20, 40),
    ],
)
def test_amplitudes_to_spinorb(nocc, nvirt):
    np.random.seed(42)
    c0 = 1.0
    dsz = nocc * (nocc - 1) // 2 * nvirt * (nvirt - 1) // 2

    cis_a = np.random.randn(nocc, nvirt)
    cis_b = np.random.randn(nocc, nvirt)
    cid_aa = np.random.randn(dsz)
    cid_bb = np.random.randn(dsz)
    cid_ab = np.random.randn(nocc, nocc, nvirt, nvirt)
    amps = {"0": c0, "a": cis_a, "b": cis_b, "aa": cid_aa, "ab": cid_ab, "bb": cid_bb}
    c_ia, c_ijab = amplitudes_to_spinorb(amps, exci=2)

    cid_aa_full = remove_index_restriction_doubles(cid_aa, nocc, nvirt)
    cid_bb_full = remove_index_restriction_doubles(cid_bb, nocc, nvirt)

    np.testing.assert_allclose(c_ia[::2, ::2], cis_a)
    np.testing.assert_allclose(c_ia[1::2, 1::2], cis_b)
    np.testing.assert_allclose(c_ijab[::2, ::2, ::2, ::2], cid_aa_full)
    np.testing.assert_allclose(c_ijab[1::2, 1::2, 1::2, 1::2], cid_bb_full)
    np.testing.assert_allclose(c_ijab[::2, 1::2, ::2, 1::2], cid_ab)
    np.testing.assert_allclose(c_ijab[1::2, ::2, 1::2, ::2], cid_ab.transpose(1, 0, 3, 2))
    np.testing.assert_allclose(c_ijab[::2, 1::2, 1::2, ::2], -1.0 * cid_ab.transpose(0, 1, 3, 2))


def test_amplitude_extraction_and_norms():
    from pyscf import gto, mcscf, scf

    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom="N, 0., 0., 0. ; N,  0., 0., 1.4",
        basis="321g",
    )
    m = scf.RHF(mol)
    m.kernel()

    # NOTE: need enough virtuals for a/b-only
    # quadruple excitations
    ncas = 8
    nelec = (4, 4)
    print(f"CAS({nelec}, {ncas})")
    mc = mcscf.CASCI(m, ncas, nelec)
    mc.kernel()

    np.random.seed(42)
    ci = mc.ci
    ci_rnd = ci + np.random.randn(*ci.shape) * 1e-2
    mc.ci = ci_rnd

    ci_amps = extract_ci_amplitudes(mc, exci=4)
    assert all(amp.size > 0 for amp in ci_amps.values())
    c1, c2, c3, c4 = amplitudes_to_spinorb(ci_amps, exci=4)

    # test that the number of overlaps equals the analytic formula
    assert number_overlaps_eccc(ncas, *nelec) == sum(
        np.atleast_1d(c).size for _, c in ci_amps.items()
    )

    # print("Amplitudes extracted.")
    # check_amplitudes_spinorb(c2, 2)
    # check_amplitudes_spinorb(c3, 3)
    # check_amplitudes_spinorb(c4, 4)

    # NOTE: legacy conversion :)
    t_ia, t_ijab = ci_to_cluster_amplitudes(c1, c2)

    c1 = c1.T
    c2 = c2.transpose(2, 3, 0, 1)
    c3 = c3.transpose(3, 4, 5, 0, 1, 2)
    c4 = c4.transpose(4, 5, 6, 7, 0, 1, 2, 3)
    t1, t2, t3, t4 = ci_to_cc(c1, c2, c3, c4)

    print("Checking T amp symmetry.")

    print("T2")
    check_amplitudes_spinorb(t2, 2)
    print("T3")
    check_amplitudes_spinorb(t3, 3)
    # print("T4")
    # check_amplitudes_spinorb(t4, 4)

    t1 = t1.T
    t2 = t2.transpose(2, 3, 0, 1)
    t3 = t3.transpose(3, 4, 5, 0, 1, 2)
    t4 = t4.transpose(4, 5, 6, 7, 0, 1, 2, 3)

    np.testing.assert_allclose(t_ia, t1, atol=1e-12)
    np.testing.assert_allclose(t_ijab, t2, atol=1e-12)

    from math import factorial

    # compare the amplitude norms with ClusterDec
    c1norm = np.vdot(c1, c1)
    c2norm = np.vdot(c2, c2) / factorial(2) ** 2
    c3norm = np.vdot(c3, c3) / factorial(3) ** 2
    c4norm = np.vdot(c4, c4) / factorial(4) ** 2

    t1norm = np.vdot(t1, t1)
    t2norm = np.vdot(t2, t2) / factorial(2) ** 2
    t3norm = np.vdot(t3, t3) / factorial(3) ** 2
    t4norm = np.vdot(t4, t4) / factorial(4) ** 2

    with tempfile.NamedTemporaryFile() as fp:
        dump_clusterdec(mc, fname=fp.name)
        if which("clusterdec_bit.x") is None:
            pytest.skip("clusterdec_bit.x executable not in PATH.")
        (c1sq, c2sq, c3sq, c4sq), (t1sq, t2sq, t3sq, t4sq) = run_clusterdec(fp.name)
        print("|C|^2")
        print(c1sq, c2sq, c3sq, c4sq)
        print(c1norm, c2norm, c3norm, c4norm)

        print("|T|^2")
        print(t1sq, t2sq, t3sq, t4sq)
        print(t1norm, t2norm, t3norm, t4norm)

        # NOTE: print-out by clusterdec is heavily truncated in some cases
        np.testing.assert_allclose(c1norm, c1sq, atol=1e-7, rtol=0)
        np.testing.assert_allclose(c2norm, c2sq, atol=1e-7, rtol=0)
        np.testing.assert_allclose(c3norm, c3sq, atol=1e-7, rtol=0)
        np.testing.assert_allclose(c4norm, c4sq, atol=1e-6, rtol=0)

        np.testing.assert_allclose(t1norm, t1sq, atol=1e-7, rtol=0)
        np.testing.assert_allclose(t2norm, t2sq, atol=1e-7, rtol=0)
        np.testing.assert_allclose(t3norm, t3sq, atol=1e-7, rtol=0)
        np.testing.assert_allclose(t4norm, t4sq, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "alphas, betas, expected",
    [
        ([[1, 1, 0, 0]], [[0, 0, 1, 1]], ["10100101"]),
        ([1, 1, 0, 0], [0, 0, 1, 1], ["10100101"]),
        ([[1, 0, 1], [0, 1, 0]], [[1, 0, 0]], ["110010", "011000"]),
    ],
)
def test_interleave_strings(alphas, betas, expected):
    from tailoredcc.amplitudes import interleave_strings

    ret = interleave_strings(alphas, betas)
    assert len(ret) == len(expected)
    for r, ref in zip(ret, expected):
        assert r == ref


def test_general_spinorb_to_spatial():
    nocc_a = 4
    nocc_b = 4
    nvirt_a = 5
    nvirt_b = 5

    nocc = nocc_a + nocc_b
    nvirt = nvirt_a + nvirt_b
    exci = 3
    np.random.seed(42)
    tt = np.random.randn(*(exci * (nocc,) + exci * (nvirt,)))
    slices = {"a": slice(0, None, 2), "b": slice(1, None, 2)}

    perms = list(permutations(range(exci)))
    parities = {p: compute_parity(p) for p in perms}
    for _ in range(3):
        for p1 in tqdm(perms):
            sign1 = parities[p1]
            for p2 in perms:
                sign2 = parities[p2]
                perm_total = np.concatenate([np.array(p1), exci + np.array(p2)])
                tt = 0.5 * (tt + sign1 * sign2 * tt.transpose(perm_total))

    for p1 in permutations(range(exci)):
        sign1 = compute_parity(p1)
        for p2 in permutations(range(exci)):
            sign2 = compute_parity(p2)
            perm_total = np.concatenate([np.array(p1), exci + np.array(p2)])
            np.testing.assert_allclose(
                tt.ravel(),
                (sign1 * sign2 * tt.transpose(perm_total)).flatten(),
                err_msg=f"{perm_total}",
                atol=1e-8,
            )

    spinblocks = list(product("ab", repeat=exci))
    for comb in product(list(spinblocks), repeat=2):
        ospin, vspin = comb
        ostr = "".join(ospin)
        vstr = "".join(vspin)
        if ospin.count("a") != vspin.count("a") or ospin.count("b") != vspin.count("b"):
            tt[tuple(slices[ii] for ii in ostr + vstr)] = 0.0
    t_spatial = spinorb_to_spatial(tt, exci, nocc_a, nocc_b, nvirt_a, nvirt_b)
    t_out = spatial_to_spinorb(t_spatial, exci, nocc_a, nocc_b, nvirt_a, nvirt_b)
    np.testing.assert_allclose(tt, t_out, atol=1e-10, rtol=0)
