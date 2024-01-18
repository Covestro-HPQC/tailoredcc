from functools import cache

import numpy as np
import pandas as pd
import pytest
from pyscf import gto, lib, mcscf, scf
from tqdm import tqdm

from tailoredcc.amplitudes import (
    add_gaussian_noise,
    amplitudes_to_spinorb,
    extract_ci_amplitudes,
    prepare_cas_slices,
)
from tailoredcc.tailoredcc import tccsd_pyscf

lib.num_threads(32)


xyzdict = {
    "n2": """
    N 0.000000    0.000000    0.000000
    N 0.000000    0.000000    1.090000
    """,
    "formaldehyde": """
    H      1.0686     -0.1411      1.0408
    C      0.5979      0.0151      0.0688
    H      1.2687      0.2002     -0.7717
    O     -0.5960     -0.0151     -0.0686
    """,
    "acetaldehyde": """
    H      1.8870     -0.2640      0.6645
    C      1.2203     -0.2396     -0.2074
    H      1.4534      0.6668     -0.7811
    H      1.4662     -1.1053     -0.8363
    C     -0.2092     -0.3158      0.2358
    H     -0.5002     -1.2065      0.8165
    O     -1.0303      0.5372     -0.0127
    """,
    "h2o": """
    O          0.000000000000     0.000000000000    -0.068516219310 
    H          0.000000000000    -0.790689573744     0.543701060724 
    H          0.000000000000     0.790689573744     0.543701060724
    """,
    "nco": """
    N         -0.41161       -0.40290       -0.00215
    C          0.73469       -0.06107        0.00037
    O          1.88728        0.15226        0.00056
    C         -1.73039        0.17153        0.00024
    H         -1.80171        1.00306       -0.70167
    H         -1.99099        0.52719        0.99788
    H         -2.45003       -0.59071       -0.28932
    """,
    "benzene": """
    C 0.000000 1.396792 0.000000
    C 0.000000 -1.396792 0.000000
    C 1.209657 0.698396 0.000000
    C -1.209657 -0.698396 0.000000
    C -1.209657 0.698396 0.000000
    C 1.209657 -0.698396 0.000000
    H 0.000000 2.484212 0.000000
    H 2.151390 1.242106 0.000000
    H -2.151390 -1.242106 0.000000
    H -2.151390 1.242106 0.000000
    H 2.151390 -1.242106 0.000000
    H 0.000000 -2.484212 0.000000
    """,
    "f2": """
    F 0 0 0
    F 0 0 1.412
    """,
    "cl2": """
    Cl 0 0 0
    Cl 0 0 1.988
    """,
    "furan": """
    O      1.1014      0.0267      0.3041
    C      0.6009      0.1157     -0.9767
    H      1.3427      0.2213     -1.7620
    C     -0.7699      0.0473     -0.9565
    H     -1.4514      0.0888     -1.8007
    C     -1.1485     -0.0937      0.4268
    H     -2.1639     -0.1763      0.8025
    C      0.0194     -0.1007      1.1480
    H      0.2580     -0.1826      2.2037
    """,
    # NOTE: 'strong' static correlation
    "n2_stretched": """
    N 0 0 0
    N 0 0 2.5
    """,
    "f2_stretched": """
    F 0 0 0
    F 0 0 2.4
    """,
    "cl2_stretched": """
    Cl 0 0 0
    Cl 0 0 4.0
    """,
    "p-benzyne": """
    C               -0.739600003233    -1.195300005225     0.000000000000
    C                0.739600003233    -1.195300005225     0.000000000000
    C                1.362000005953     0.000000000000     0.000000000000
    C                0.739600003233     1.195300005225     0.000000000000
    C               -0.739600003233     1.195300005225     0.000000000000
    C               -1.362000005953     0.000000000000     0.000000000000
    H                1.199900005245    -2.182400009539     0.000000000000
    H               -1.199900005245     2.182400009539     0.000000000000
    H                1.199900005245     2.182400009539     0.000000000000
    H               -1.199900005245    -2.182400009539     0.000000000000
    """,
}


@cache
def scf_factory(molname, basis):
    mol = gto.M(atom=xyzdict[molname], basis=basis, verbose=4, max_memory=96000)
    scfres = scf.RHF(mol)
    scfres.conv_tol = 1e-12
    scfres.conv_tol_grad = 1e-8
    scfres.kernel()
    return scfres, mol


@pytest.mark.parametrize(
    "basis",
    [
        "sto-3g",
        "6-31g",
        "6-31g**",
        "cc-pvdz",
        "aug-cc-pvdz",
        # "cc-pvtz",
        # "cc-pvqz",
    ],
)
@pytest.mark.parametrize(
    "molname",
    [
        # "n2",
        # "h2o",
        # "nco",
        # "formaldehyde",
        # "acetaldehyde",
        # "benzene",
        # "f2",
        # "cl2",
        # "furan",
        "f2_stretched",
        "cl2_stretched",
        "n2_stretched",
        "p-benzyne",
    ],
)
@pytest.mark.parametrize(
    "nact, nalpha, nbeta",
    [
        # (2, 1, 1),
        # (4, 2, 2),
        # (6, 2, 2),
        (6, 3, 3),
        (8, 2, 2),
        (8, 3, 3),
        (8, 4, 4),
        (10, 5, 5),
        (10, 6, 6),
        (12, 6, 6),
        (14, 5, 5),
        (16, 3, 3),
        (16, 4, 4),
        (16, 5, 5),
        (16, 6, 6),
        (16, 8, 8),  # new
    ],
)
def test_gaussian_noise(basis, molname, nact, nalpha, nbeta, results_bag):
    ncas = nact
    nvirta = ncas - nalpha
    nvirtb = ncas - nbeta
    nocca = nalpha
    noccb = nbeta

    assert nocca == noccb

    scfres, mol = scf_factory(molname, basis)
    if mol.nao_nr() < nact:
        pytest.skip(f"Not enough basis functions for active space {nact}")
    if sum(mol.nelec) < nalpha + nbeta:
        pytest.skip(f"Not enough electrons for active space {nalpha},{nbeta}")

    mc = mcscf.CASCI(scfres, nact, (nalpha, nbeta))
    mc.kernel()

    # prerequisites
    nocca, noccb = mc.nelecas
    assert nocca == noccb
    nvirta = mc.ncas - nocca
    nvirtb = mc.ncas - noccb
    assert nvirta == nvirtb
    assert isinstance(mc.ncore, (int, np.int64))
    ncore = mc.ncore
    ncas = mc.ncas
    nvir = mc.mo_coeff.shape[1] - ncore - ncas

    ci_amps = extract_ci_amplitudes(mc, exci=2)
    noverlaps = 0
    for k, v in ci_amps.items():
        noverlaps += v.size
    c_ia, c_ijab = amplitudes_to_spinorb(ci_amps)
    occslice, virtslice = prepare_cas_slices(
        nocca, noccb, nvirta, nvirtb, ncore, nvir, backend="pyscf"
    )
    ret_exact = tccsd_pyscf(mc._scf, c_ia, c_ijab, occslice, virtslice, verbose=4)

    # np.random.seed(42)
    np.random.seed(4200)
    cycles = 30
    std = 1e-3

    data = []
    for _ in tqdm(range(cycles)):
        ci_amps_noisy = add_gaussian_noise(ci_amps, std=std)
        c_ia, c_ijab = amplitudes_to_spinorb(ci_amps_noisy)
        ret = tccsd_pyscf(mc._scf, c_ia, c_ijab, occslice, virtslice, verbose=0)
        assert ret.converged
        data.append([std, ret.e_tot, ret_exact.e_tot])

    df = pd.DataFrame(data=data, columns=["std", "e_tot", "e_exact"])
    df["tcc_error"] = np.abs(df.e_tot - ret_exact.e_tot)

    nelec = sum(mol.nelec)
    nao = mol.nao_nr()
    nmo = scfres.mo_coeff.shape[1] * 2

    results_bag.df = df
    results_bag.nact = nact
    results_bag.nalpha = nalpha
    results_bag.nbeta = nbeta
    results_bag.molname = molname
    results_bag.basis = basis
    results_bag.nelec = nelec
    results_bag.nao = nao
    results_bag.nmo = nmo
    results_bag.noverlaps_exp = noverlaps


def test_synthesis(module_results_df):
    print(module_results_df)
    module_results_df.to_hdf("noise_data_general.h5", key="df")
