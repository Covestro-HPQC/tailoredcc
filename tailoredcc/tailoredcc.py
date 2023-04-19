# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import warnings
from dataclasses import dataclass

import numpy as np
from pyscf import scf
from pyscf.cc.addons import spatial2spin

from .amplitudes import (
    assert_spinorb_antisymmetric,
    ci_to_cluster_amplitudes,
    extract_ci_singles_doubles_amplitudes_spinorb,
    extract_vqe_singles_doubles_amplitudes_spinorb,
    set_cas_amplitudes_spatial_from_spinorb,
)
from .solve_tcc import _solve_tccsd_oe, solve_tccsd
from .utils import spin_blocks_interleaved_to_sequential, spinorb_from_spatial


def tccsd_from_ci(mc, backend="libcc"):
    # TODO: docs
    nocca, noccb = mc.nelecas
    assert nocca == noccb
    nvirta = mc.ncas - nocca
    nvirtb = mc.ncas - noccb
    assert nvirta == nvirtb

    c_ia, c_ijab = extract_ci_singles_doubles_amplitudes_spinorb(mc)

    if "CASSCF" in str(type(mc)):
        raise NotImplementedError("TCC with CAS orbitals not yet implemented.")
        # mc._scf.mo_coeff = mc.mo_coeff
        # mc._scf.mo_energy = mc.mo_energy

    assert isinstance(mc.ncore, int)

    if backend in ["adcc", "libcc"]:
        occaslice = np.arange(mc.ncore, mc.ncore + nocca, 1)
        occbslice = np.arange(2 * mc.ncore + nocca, 2 * mc.ncore + nocca + noccb)
        virtaslice = np.arange(0, nvirta, 1)
        ncore = mc.ncore
        ncas = mc.ncas
        nvir = mc.mo_coeff.shape[1] - ncore - ncas
        virtbslice = np.arange(nvirta + nvir, nvirta + nvir + nvirtb, 1)
        occslice = np.concatenate((occaslice, occbslice), axis=0)
        virtslice = np.concatenate((virtaslice, virtbslice), axis=0)
        return tccsd(mc._scf, c_ia, c_ijab, occslice, virtslice, backend=backend)
    elif backend == "oe":
        occslice = slice(2 * mc.ncore, 2 * mc.ncore + nocca + noccb)
        virtslice = slice(0, nvirta + nvirtb)
        return _tccsd_opt_einsum(mc._scf, c_ia, c_ijab, occslice, virtslice)
    elif backend == "pyscf":
        occslice = slice(2 * mc.ncore, 2 * mc.ncore + nocca + noccb)
        virtslice = slice(0, nvirta + nvirtb)
        return tccsd_pyscf(mc._scf, c_ia, c_ijab, occslice, virtslice)
    else:
        raise NotImplementedError()


def tccsd_from_vqe(scfres, vqe, backend="oe"):
    # TODO: docs, type hints
    nocca, noccb = vqe.nalpha, vqe.nbeta
    assert nocca == noccb
    ncas = vqe.nact
    nvirta = ncas - nocca
    nvirtb = ncas - noccb
    assert nvirta == nvirtb

    if backend != "oe":
        raise NotImplementedError(f"TCC from VQE not implemented for backend {backend}.")

    c_ia, c_ijab = extract_vqe_singles_doubles_amplitudes_spinorb(vqe)

    occslice = slice(2 * vqe.nocc, 2 * vqe.nocc + nocca + noccb)
    virtslice = slice(0, nvirta + nvirtb)
    return tccsd(scfres, c_ia, c_ijab, occslice, virtslice, backend=backend)


def tccsd_pyscf(scfres, c_ia, c_ijab, occslice, virtslice):
    # TODO: docs, hints
    # 1. convert to T amplitudes
    t1cas, t2cas = ci_to_cluster_amplitudes(c_ia, c_ijab)

    # 2. set up pyscf CCSD object
    from pyscf import cc

    # TODO: density fitting?
    # TODO: provide MO coefficients (e.g., from CASSCF)
    # TODO: UCCSD
    ccsd = cc.CCSD(scfres)
    update_amps = ccsd.update_amps

    t1slice = (occslice, virtslice)
    t2slice = (occslice, occslice, virtslice, virtslice)
    _, t1guess, t2guess = ccsd.init_amps()
    t1guess, t2guess = set_cas_amplitudes_spatial_from_spinorb(
        t1guess, t2guess, t1cas, t2cas, t1slice, t2slice
    )

    # 3. patch update_amps function to freeze CAS amplitudes
    def update_freeze_amplitudes_cas(t1, t2, eris):
        t1new, t2new = update_amps(t1, t2, eris)
        t1new, t2new = set_cas_amplitudes_spatial_from_spinorb(
            t1new, t2new, t1cas, t2cas, t1slice, t2slice
        )
        return t1new, t2new

    ccsd.update_amps = update_freeze_amplitudes_cas
    ccsd.kernel(t1guess, t2guess)

    # 4. check that the CAS amplitudes are still intact
    # NOTE: might break for FCI, i.e., when all external amplitudes are 0
    t1spin = spatial2spin(ccsd.t1)
    t2spin = spatial2spin(ccsd.t2)
    np.testing.assert_allclose(t1spin[t1slice], t1cas, atol=1e-8, rtol=0)
    np.testing.assert_allclose(t2spin[t2slice], t2cas, atol=1e-8, rtol=0)

    t1cas_spatial, t2cas_spatial = set_cas_amplitudes_spatial_from_spinorb(
        ccsd.t1, ccsd.t2, t1cas, t2cas, t1slice, t2slice, zero_input=True
    )
    e_cas = ccsd.energy(t1cas_spatial, t2cas_spatial)
    ccsd.e_cas = e_cas

    return ccsd


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


def tccsd(scfres, c_ia, c_ijab, occslice, virtslice, backend="libcc"):
    # 1. convert to T amplitudes
    t1, t2 = ci_to_cluster_amplitudes(c_ia, c_ijab)
    t1 = spin_blocks_interleaved_to_sequential(t1)
    t2 = spin_blocks_interleaved_to_sequential(t2)
    print("=> Amplitudes converted.")
    # assert_spinorb_antisymmetric(t2)

    mol = scfres.mol
    nocc = sum(mol.nelec)
    nvirt = 2 * mol.nao_nr() - nocc

    import adcc

    from .ccsd import DISPATCH

    cc = DISPATCH[backend]

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

    # assert_spinorb_antisymmetric(t2_mo)

    tguess = adcc.AmplitudeVector(ov=hf.fov.zeros_like(), oovv=mp.t2oo.zeros_like())
    tguess.ov.set_from_ndarray(t1_mo, 1e-12)
    tguess.oovv.set_from_ndarray(t2_mo, 1e-12)

    e_cas = cc.ccsd_energy_correlation(mp, tguess)
    print(f"CCSD correlation energy from CI amplitudes {e_cas:>12}")

    # solve tccsd amplitude equations
    t = solve_tccsd(mp, occslice, virtslice, tguess, backend=backend, diis_size=8)
    t1f = t.ov
    t2f = t.oovv
    # test that the T_CAS amplitudes are still intact
    np.testing.assert_allclose(t1, t1f.to_ndarray()[t1slice], atol=1e-12, rtol=0)
    np.testing.assert_allclose(t2, t2f.to_ndarray()[t2slice], atol=1e-12, rtol=0)

    # compute correlation/total TCCSD energy
    e_tcc = cc.ccsd_energy_correlation(mp, t)

    ret = TCC(scfres, t1f, t2f, hf_energy, e_cas, e_tcc)
    print(
        f"E(TCCSD)= {ret.e_tot:.10f}",
        f"E_corr = {e_tcc:.10f}",
        # f"E_ext = {e_ext:.10f}",
        f"E_cas = {e_cas:.10f}",
    )
    return ret


def _tccsd_opt_einsum(scfres, c_ia, c_ijab, occslice, virtslice):
    warnings.warn(
        "This implementation based on opt_einsum is just a reference"
        " code and should not be used for production calculations."
    )
    mol = scfres.mol
    # 1. convert to T amplitudes
    t1, t2 = ci_to_cluster_amplitudes(c_ia, c_ijab)
    print("=> Amplitudes converted.")

    assert_spinorb_antisymmetric(t2)
    # 2. build CCSD prerequisites
    from openfermionpyscf._run_pyscf import compute_integrals

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

    from .ccsd import oe as cc

    e_cas = cc.ccsd_energy_correlation(
        t1_mo.T, t2_mo.transpose(2, 3, 0, 1), fock, eri_phys_asymm, o, v
    )
    print(f"CCSD correlation energy from CI amplitudes {e_cas:>12}")

    # solve tccsd amplitude equations
    t1f, t2f = _solve_tccsd_oe(
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
    e_tcc = cc.ccsd_energy_correlation(t1f, t2f, fock, eri_phys_asymm, o, v)  # - hf_energy

    ret = TCC(scfres, t1f, t2f, hf_energy + mol.energy_nuc(), e_cas, e_tcc)
    print(
        f"E(TCCSD)= {ret.e_tot:.10f}",
        f"E_corr = {e_tcc:.10f}",
        # f"E_ext = {e_ext:.10f}",
        f"E_cas = {e_cas:.10f}",
    )
    return ret
