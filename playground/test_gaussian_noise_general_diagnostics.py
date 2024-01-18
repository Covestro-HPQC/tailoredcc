from functools import cache

import numpy as np
import pandas as pd
import pytest
from pyscf import cc, gto, lib, mcscf, scf
from tqdm import tqdm

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
    ],
)
@pytest.mark.parametrize(
    "molname",
    [
        "n2",
        "h2o",
        "nco",
        "formaldehyde",
        "acetaldehyde",
        "benzene",
        "f2",
        "cl2",
        "furan",
        "f2_stretched",
        "cl2_stretched",
        "n2_stretched",
        "p-benzyne",
    ],
)
def test_gaussian_noise(basis, molname, results_bag):
    scfres, mol = scf_factory(molname, basis)

    ccsd = cc.CCSD(scfres)
    ccsd.conv_tol = 1e-6
    ccsd.max_cycle = 200
    ccsd.iterative_damping = 0.75
    ccsd.level_shift = 0.5
    ccsd.kernel()
    assert ccsd.converged

    t1diag = ccsd.get_t1_diagnostic()
    d1diag = ccsd.get_d1_diagnostic()
    t2diag = np.max(np.abs(ccsd.t2))
    maxt1 = np.max(np.abs(ccsd.t1))

    results_bag.molname = molname
    results_bag.basis = basis
    results_bag.t1diag = t1diag
    results_bag.d1diag = d1diag
    results_bag.maxt2 = t2diag
    results_bag.maxt1 = maxt1


def test_synthesis(module_results_df):
    print(module_results_df)
    module_results_df.to_hdf("diagnostic_data_general.h5", key="df")
