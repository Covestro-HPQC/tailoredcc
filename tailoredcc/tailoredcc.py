# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import time
from dataclasses import dataclass

import numpy as np
from openfermionpyscf._run_pyscf import compute_integrals
from pyscf import scf

from .amplitudes import (
    assert_spinorb_antisymmetric,
    ci_to_cluster_amplitudes,
    extract_ci_singles_doubles_amplitudes_spinorb,
    extract_vqe_singles_doubles_amplitudes_spinorb,
)
from .ccsd_equations import ccsd_energy, doubles_residual, singles_residual
from .utils import spin_blocks_interleaved_to_sequential, spinorb_from_spatial


def solve_tccsd(
    t1,
    t2,
    fock,
    g,
    o,
    v,
    e_ai,
    e_abij,
    occslice,
    virtslice,
    max_iter=100,
    stopping_eps=1.0e-8,
    diis_size=7,
    diis_start_cycle=4,
):
    t1slice = (virtslice, occslice)
    t2slice = (virtslice, virtslice, occslice, occslice)

    # initialize diis if diis_size is not None
    # else normal iterate
    if diis_size is not None:
        from .diis import DIIS

        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_dim = t1.size
        old_vec = np.hstack((t1.flatten(), t2.flatten()))

    old_energy = ccsd_energy(t1, t2, fock, g, o, v)
    print(f"\tInitial CCSD energy: {old_energy}")
    for idx in range(max_iter):
        start = time.time()
        singles_res = singles_residual(t1, t2, fock, g, o, v)
        doubles_res = doubles_residual(t1, t2, fock, g, o, v)

        # set the CAS-only residual to zero
        singles_res[t1slice] = 0.0
        doubles_res[t2slice] = 0.0

        new_singles = t1 + singles_res * e_ai
        new_doubles = t2 + doubles_res * e_abij

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack((new_singles.flatten(), new_doubles.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate, error_vec)
            new_singles = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
            new_doubles = new_vectorized_iterate[t1_dim:].reshape(t2.shape)
            old_vec = new_vectorized_iterate

        current_energy = ccsd_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps:
            print(f"\tConverged in iteration {idx}.")
            return new_singles, new_doubles
        else:
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy
            # print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx, old_energy, delta_e))
            print(
                "\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}\t{: 5.3f}s".format(
                    idx, old_energy, delta_e, time.time() - start
                )
            )
    else:
        print("Did not converge.")
        return new_singles, new_doubles


def zero_slices(singles_res, doubles_res, occslice, virtslice):
    singles = singles_res.to_ndarray()
    doubles = doubles_res.to_ndarray()
    t1slice = np.ix_(occslice, virtslice)
    t2slice = np.ix_(occslice, occslice, virtslice, virtslice)

    singles[t1slice] = 0.0
    doubles[t2slice] = 0.0

    singles_res.set_from_ndarray(singles, 1e-12)
    doubles_res.set_from_ndarray(doubles, 1e-12)


def symmetrize_generic_doubles(invec, mospaces):
    scratch = invec.antisymmetrise([(2, 3)])
    scratch = scratch.symmetrise([(0, 1), (2, 3)])

    # import libadcc
    # symm = libadcc.make_symmetry_eri(mospaces, "o1o1v1v1")
    # x = libadcc.Tensor(symm)
    # x.set_from_ndarray(scratch.to_ndarray(), 1e-14)
    # print(x.describe_symmetry())
    return scratch


def solve_tccsd_adcc(
    mp,
    occslice=None,
    virtslice=None,
    tguess=None,
    max_iter=100,
    stopping_eps=1.0e-8,
    diis_size=7,
    diis_start_cycle=4,
):
    freeze_amplitude_slices = False
    if occslice is not None and virtslice is not None:
        freeze_amplitude_slices = True

    import adcc
    from adcc.functions import direct_sum

    from .ccsd import ccsd_energy_adcc, doubles_residual_adcc, singles_residual_adcc

    hf = mp.reference_state
    e_ia = direct_sum("+i-a->ia", hf.foo.diagonal(), hf.fvv.diagonal())
    e_ijab = (
        direct_sum(
            "+i-a+j-b->ijab",
            hf.foo.diagonal(),
            hf.fvv.diagonal(),
            hf.foo.diagonal(),
            hf.fvv.diagonal(),
        )
        .symmetrise((0, 1))
        .symmetrise((2, 3))
    )
    # print(e_ijab.describe_symmetry())
    # print(mp.t2oo.describe_symmetry())

    # make t1/t2 with zeros
    if tguess is None:
        t = adcc.AmplitudeVector(ov=e_ia.zeros_like(), oovv=mp.t2oo.zeros_like())
    else:
        t = tguess

    # initialize diis if diis_size is not None
    # else normal iterate
    if diis_size is not None:
        from .diis import DIIS

        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        old_vec = t

    old_energy = ccsd_energy_adcc(mp, t)
    print(f"\tInitial CCSD energy: {old_energy}")
    for idx in range(max_iter):
        start = time.time()
        singles_res = singles_residual_adcc(mp, t)
        doubles_res = doubles_residual_adcc(mp, t)
        # TODO: the spin symmetry gets broken in doubles residual

        # set the CAS-only residual to zero
        if freeze_amplitude_slices:
            zero_slices(singles_res, doubles_res, occslice, virtslice)

        new_singles = t.ov + singles_res / e_ia
        new_doubles = t.oovv + doubles_res / e_ijab
        new_t = adcc.AmplitudeVector(ov=new_singles, oovv=new_doubles)
        # print(new_t.oovv.describe_symmetry())

        # diis update
        if diis_size is not None:
            vectorized_iterate = new_t
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(
                vectorized_iterate, error_vec
            ).evaluate()
            new_t = new_vectorized_iterate
            old_vec = new_vectorized_iterate

        current_energy = ccsd_energy_adcc(mp, new_t)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps:
            print(f"\tConverged in iteration {idx}.")
            return new_t
        else:
            t = new_t
            old_energy = current_energy
            print(
                "\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}\t{: 5.3f}s".format(
                    idx, old_energy, delta_e, time.time() - start
                )
            )
    else:
        print("Did not converge.")
        return new_t


def tccsd_from_ci(mc, backend="adcc"):
    # TODO: docs
    nocca, noccb = mc.nelecas
    assert nocca == noccb
    nvirta = mc.ncas - nocca
    nvirtb = mc.ncas - noccb
    assert nvirta == nvirtb

    c_ia, c_ijab = extract_ci_singles_doubles_amplitudes_spinorb(mc)

    if "CASSCF" in str(type(mc)):
        raise NotImplementedError()
        # mc._scf.mo_coeff = mc.mo_coeff
        # mc._scf.mo_energy = mc.mo_energy

    assert isinstance(mc.ncore, int)

    if backend == "adcc":
        occaslice = np.arange(mc.ncore, mc.ncore + nocca, 1)
        occbslice = np.arange(2 * mc.ncore + nocca, 2 * mc.ncore + nocca + noccb)
        virtaslice = np.arange(0, nvirta, 1)
        ncore = mc.ncore
        ncas = mc.ncas
        nvir = mc.mo_coeff.shape[1] - ncore - ncas
        virtbslice = np.arange(nvirta + nvir, nvirta + nvir + nvirtb, 1)
        occslice = np.concatenate((occaslice, occbslice), axis=0)
        virtslice = np.concatenate((virtaslice, virtbslice), axis=0)
        return tccsd(mc._scf, c_ia, c_ijab, occslice, virtslice)
    elif backend == "opt_einsum":
        occslice = slice(2 * mc.ncore, 2 * mc.ncore + nocca + noccb)
        virtslice = slice(0, nvirta + nvirtb)
        return tccsd_opt_einsum(mc._scf, c_ia, c_ijab, occslice, virtslice)
    else:
        raise NotImplementedError()


def tccsd_from_vqe(scfres, vqe):
    # TODO: docs, type hints
    nocca, noccb = vqe.nalpha, vqe.nbeta
    assert nocca == noccb
    ncas = vqe.nact
    nvirta = ncas - nocca
    nvirtb = ncas - noccb
    assert nvirta == nvirtb

    c_ia, c_ijab = extract_vqe_singles_doubles_amplitudes_spinorb(vqe)

    occslice = slice(2 * vqe.nocc, 2 * vqe.nocc + nocca + noccb)
    virtslice = slice(0, nvirta + nvirtb)
    return tccsd(scfres, c_ia, c_ijab, occslice, virtslice)


@dataclass
class TCC:
    scfres: scf.HF
    t1: np.ndarray
    t2: np.ndarray
    e_hf: float
    e_cas: float
    e_corr: float

    @property
    def e_tot(self):
        return self.e_hf + self.e_corr


def tccsd(scfres, c_ia, c_ijab, occslice, virtslice):
    # 1. convert to T amplitudes
    t1, t2 = ci_to_cluster_amplitudes(c_ia, c_ijab)
    t1 = spin_blocks_interleaved_to_sequential(t1)
    t2 = spin_blocks_interleaved_to_sequential(t2)
    print("=> Amplitudes converted.")
    assert_spinorb_antisymmetric(t2)

    mol = scfres.mol
    nocc = sum(mol.nelec)
    nvirt = 2 * mol.nao_nr() - nocc

    import adcc

    hf = adcc.ReferenceState(scfres)
    hf_energy = hf.energy_scf
    mp = adcc.LazyMp(hf)
    print("=> Prerequisites built.")

    # 2. map T amplitudes to full MO space
    t1_mo = np.zeros((nocc, nvirt))
    t2_mo = np.zeros((nocc, nocc, nvirt, nvirt))

    t1slice = np.ix_(occslice, virtslice)
    t2slice = np.ix_(occslice, occslice, virtslice, virtslice)
    t1_mo[t1slice] = t1
    t2_mo[t2slice] = t2
    print("=> T amplitudes mapped to full MO space.")

    assert_spinorb_antisymmetric(t2_mo)

    from .ccsd import ccsd_energy_correlation_adcc

    tguess = adcc.AmplitudeVector(ov=hf.fov.zeros_like(), oovv=mp.t2oo.zeros_like())
    tguess.ov.set_from_ndarray(t1_mo, 1e-12)
    tguess.oovv.set_from_ndarray(t2_mo, 1e-12)

    e_cas = ccsd_energy_correlation_adcc(mp, tguess)
    print(f"CCSD correlation energy from CI amplitudes {e_cas:>12}")

    # solve tccsd amplitude equations
    t = solve_tccsd_adcc(mp, occslice, virtslice, tguess, diis_size=8)
    t1f = t.ov
    t2f = t.oovv
    # test that the T_CAS amplitudes are still intact
    np.testing.assert_allclose(t1, t1f.to_ndarray()[t1slice], atol=1e-12, rtol=0)
    np.testing.assert_allclose(t2, t2f.to_ndarray()[t2slice], atol=1e-12, rtol=0)

    # compute correlation/total TCCSD energy
    e_tcc = ccsd_energy_correlation_adcc(mp, t)

    ret = TCC(scfres, t1f, t2f, hf_energy, e_cas, e_tcc)
    print(
        f"E(TCCSD)= {ret.e_tot:.10f}",
        f"E_corr = {e_tcc:.10f}",
        # f"E_ext = {e_ext:.10f}",
        f"E_cas = {e_cas:.10f}",
    )
    return ret


def tccsd_opt_einsum(scfres, c_ia, c_ijab, occslice, virtslice):
    mol = scfres.mol
    # 1. convert to T amplitudes
    t1, t2 = ci_to_cluster_amplitudes(c_ia, c_ijab)
    print("=> Amplitudes converted.")

    assert_spinorb_antisymmetric(t2)
    # 2. build CCSD prerequisites
    oei, eri_of_spatial = compute_integrals(mol, scfres)
    soei, eri_of = spinorb_from_spatial(oei, eri_of_spatial)
    eri_of_asymm = eri_of - eri_of.transpose(0, 1, 3, 2)
    nocc = sum(mol.nelec)
    nvirt = 2 * mol.nao_nr() - nocc
    # to Physicists' notation <12|1'2'> (OpenFermion stores <12|2'1'>)
    eri_phys_asymm = eri_of_asymm.transpose(0, 1, 3, 2)

    # orbital energy differences
    eps = np.kron(scfres.mo_energy, np.ones(2))
    n = np.newaxis
    o = slice(None, nocc)
    v = slice(nocc, None)
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    # (canonical) Fock operator
    fock = soei + np.einsum("piqi->pq", eri_phys_asymm[:, o, :, o])
    hf_energy = 0.5 * np.einsum("ii", (fock + soei)[o, o])
    print("=> Prerequisites built.")

    # 3. map T amplitudes to full MO space
    t1_mo = np.zeros((nocc, nvirt))
    t2_mo = np.zeros((nocc, nocc, nvirt, nvirt))
    t1_mo[occslice, virtslice] = t1
    t2_mo[occslice, occslice, virtslice, virtslice] = t2
    print("=> T amplitudes mapped to full MO space.")

    assert_spinorb_antisymmetric(t2_mo)

    from .ccsd_equations import ccsd_energy_correlation

    e_cas = ccsd_energy_correlation(
        t1_mo.T, t2_mo.transpose(2, 3, 0, 1), fock, eri_phys_asymm, o, v
    )
    print(f"CCSD correlation energy from CI amplitudes {e_cas:>12}")

    # solve tccsd amplitude equations
    t1f, t2f = solve_tccsd(
        t1_mo.T,
        t2_mo.transpose(2, 3, 0, 1),
        fock,
        eri_phys_asymm,
        o,
        v,
        e_ai,
        e_abij,
        occslice,
        virtslice,
        diis_size=8,
    )
    # test that the T_CAS amplitudes are still intact
    t1slice = (virtslice, occslice)
    t2slice = (virtslice, virtslice, occslice, occslice)
    np.testing.assert_allclose(t1.T, t1f[t1slice], atol=1e-14, rtol=0)
    np.testing.assert_allclose(t2.transpose(2, 3, 0, 1), t2f[t2slice], atol=1e-14, rtol=0)

    # compute correlation/total TCCSD energy
    e_tcc = ccsd_energy_correlation(t1f, t2f, fock, eri_phys_asymm, o, v)  # - hf_energy

    ret = TCC(scfres, t1f, t2f, hf_energy + mol.energy_nuc(), e_cas, e_tcc)
    print(
        f"E(TCCSD)= {ret.e_tot:.10f}",
        f"E_corr = {e_tcc:.10f}",
        # f"E_ext = {e_ext:.10f}",
        f"E_cas = {e_cas:.10f}",
    )
    return ret
