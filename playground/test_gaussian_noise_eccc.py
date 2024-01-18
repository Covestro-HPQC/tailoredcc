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

# from tailoredcc.tailoredcc import tccsd_pyscf
from tailoredcc.tailoredcc import ec_cc

lib.num_threads(16)


@pytest.fixture(scope="module")
def scfres():
    conv_tol_e = 1e-12
    conv_tol_g = 1e-8

    # this was used for the "normal benchmark":
    xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/n2_real.xyz"

    basis = "cc-pvdz"
    # basis = "sto-3g"
    # basis = "6-31g"
    mol = gto.M(atom=str(xyzfile), basis=basis, verbose=4, max_memory=32000)

    # formaldehyde
    # mol = gto.M(
    #     atom="""
    #     H      1.0686     -0.1411      1.0408
    #     C      0.5979      0.0151      0.0688
    #     H      1.2687      0.2002     -0.7717
    #     O     -0.5960     -0.0151     -0.0686
    #     """, basis=basis, verbose=4, max_memory=32000
    # )

    # acetaldehyde
    # mol = gto.M(
    #     atom="""
    #     H      1.8870     -0.2640      0.6645
    #     C      1.2203     -0.2396     -0.2074
    #     H      1.4534      0.6668     -0.7811
    #     H      1.4662     -1.1053     -0.8363
    #     C     -0.2092     -0.3158      0.2358
    #     H     -0.5002     -1.2065      0.8165
    #     O     -1.0303      0.5372     -0.0127
    #     """, basis=basis, verbose=4, max_memory=32000
    # )

    # stretched N2
    # mol = gto.M(atom="""
    # N 0 0 0
    # N 0 0 3.5
    # """, unit="bohr", basis=basis, verbose=4, max_memory=32000)

    scfres = scf.RHF(mol)
    scfres.conv_tol = conv_tol_e
    scfres.conv_tol_grad = conv_tol_g
    scfres.kernel()
    return scfres


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
        # (14, 5, 5),
        # (16, 3, 3),
        # (16, 4, 4),
        # (16, 5, 5),
        # (16, 6, 6),
    ],
)
def test_gaussian_noise(scfres, nact, nalpha, nbeta, results_bag):
    ncas = nact
    nvirta = ncas - nalpha
    nvirtb = ncas - nbeta
    nocca = nalpha
    noccb = nbeta

    assert nocca == noccb

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

    ci_amps = extract_ci_amplitudes(mc, exci=4)
    ci_spinorb = amplitudes_to_spinorb(ci_amps, exci=4)
    occslice, virtslice = prepare_cas_slices(
        nocca, noccb, nvirta, nvirtb, ncore, nvir, backend="pyscf"
    )
    ret_exact = ec_cc(mc._scf, *ci_spinorb, occslice, virtslice)

    np.random.seed(42)
    cycles = 100
    npoints = 10

    # cycles = 2
    # npoints = 5
    stds = np.logspace(-10, -1, num=npoints, endpoint=True)

    data = []
    for std in stds:
        for _ in tqdm(range(cycles)):
            ci_amps_noisy = add_gaussian_noise(ci_amps, std=std)
            ci_spinorb = amplitudes_to_spinorb(ci_amps_noisy, exci=4)
            # ret = tccsd_pyscf(mc._scf, c_ia, c_ijab, occslice, virtslice, verbose=0)
            ret = ec_cc(
                mc._scf, *ci_spinorb, occslice, virtslice, guess_t1_t2_from_ci=True, verbose=0
            )
            # assert ret.converged
            data.append([std, ret.e_tot])

    df = pd.DataFrame(data=data, columns=["std", "e_tot"])
    df["tcc_error"] = np.abs(df.e_tot - ret_exact.e_tot)

    results_bag.nact = nact
    results_bag.nalpha = nalpha
    results_bag.nbeta = nbeta
    results_bag.df = df


def test_synthesis(module_results_df):
    print(module_results_df)
    module_results_df.to_hdf("noise_data_eccc.h5", key="df")
