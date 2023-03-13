# Proprietary and Confidential
# Covestro Deutschland AG, 2023

from os import remove
import pytest
import numpy as np

from pyscf.fci import cistring
from pyscf.ci.cisd import tn_addrs_signs

from tailoredcc.amplitudes import (
    ci_to_cluster_amplitudes,
    extract_ci_singles_doubles_amplitudes_spinorb,
    detstrings_singles,
    detstrings_doubles,
    remove_index_restriction_doubles,
    assert_spinorb_antisymmetric,
    amplitudes_to_spinorb,
)
from tailoredcc.clusterdec import dump_clusterdec


@pytest.mark.parametrize(
    "nocc, ncas",
    [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 4),
        (4, 4),
        (2, 6),
        (4, 6),
        (7, 10),
        (10, 25),
    ],
)
def test_determinant_string_generation(nocc, ncas):
    nvirt = ncas - nocc
    t1addrs, _ = tn_addrs_signs(ncas, nocc, 1)
    t2addrs, _ = tn_addrs_signs(ncas, nocc, 2)

    if nocc == 0 or ncas == 1:
        assert len(t1addrs) == 0
        assert len(t2addrs) == 0

    if len(t1addrs):
        detstrings_ref = [bin(ds) for ds in cistring.addrs2str(ncas, nocc, t1addrs.ravel())]
        detstrings, detstrings_np = detstrings_singles(nocc, nvirt)
        assert detstrings == detstrings_ref
        assert np.sum(detstrings_np) == nocc * len(detstrings)

    if len(t2addrs):
        detstrings_ref = [bin(ds) for ds in cistring.addrs2str(ncas, nocc, t2addrs.ravel())]
        detstrings, detstrings_np = detstrings_doubles(nocc, nvirt)
        assert detstrings == detstrings_ref
        assert np.sum(detstrings_np) == nocc * len(detstrings)


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
    c_ia, c_ijab = amplitudes_to_spinorb(c0, cis_a, cis_b, cid_aa, cid_ab, cid_bb)

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
    from pyscf import gto, scf, mcscf

    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom="N, 0., 0., 0. ; N,  0., 0., 1.4",
        basis="sto-3g",
    )
    m = scf.RHF(mol)
    m.kernel()

    ncas = mol.nao_nr()
    nelec = mol.nelec
    print(f"CAS({nelec}, {ncas})")
    mc = mcscf.CASCI(m, ncas, nelec)
    mc.kernel()
    c_ia, c_ijab = extract_ci_singles_doubles_amplitudes_spinorb(mc)
    t_ia, t_ijab = ci_to_cluster_amplitudes(c_ia, c_ijab)

    # compare the amplitude norms with ClusterDec
    c1norm = np.vdot(c_ia, c_ia)
    c2norm = 0.25 * np.vdot(c_ijab, c_ijab)
    t1norm = np.vdot(t_ia, t_ia)
    t2norm = 0.25 * np.vdot(t_ijab, t_ijab)

    import tempfile
    import subprocess
    import os
    from pathlib import Path
    from shutil import which

    with tempfile.NamedTemporaryFile() as fp:
        dump_clusterdec(mc, fname=fp.name)
        if which("clusterdec_bit.x") is None:
            pytest.skip("clusterdec_bit.x executable not in PATH.")
        cwd = os.getcwd()
        exepath = Path(which("clusterdec_bit.x")).parent
        os.chdir(exepath.resolve())
        out = subprocess.run(["clusterdec_bit.x", fp.name], capture_output=True)
        os.chdir(cwd)
        lines = [ll.strip() for ll in out.stdout.decode("ascii").split("\n")]
        for idx, l in enumerate(lines):
            if "|C_n|        |T_n|  |T_n|/|C_n|" in l:
                c1sq = float(lines[idx + 1].split(" ")[1])
                c2sq = float(lines[idx + 2].split(" ")[1])
                t1sq = float(lines[idx + 1].split(" ")[2])
                t2sq = float(lines[idx + 2].split(" ")[2])
        np.testing.assert_allclose(c1norm, c1sq, atol=1e-7, rtol=0)
        np.testing.assert_allclose(c2norm, c2sq, atol=1e-7, rtol=0)
        np.testing.assert_allclose(t1norm, t1sq, atol=1e-7, rtol=0)
        np.testing.assert_allclose(t2norm, t2sq, atol=1e-7, rtol=0)
