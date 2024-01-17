# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict

import covvqetools as cov
import numpy as np
import numpy.typing as npt
from pyscf import mcscf, scf
from pyscf.cc.addons import spatial2spin

from tailoredcc.ci_to_cc import ci_to_cc

from .amplitudes import (
    amplitudes_to_spinorb,
    assert_spinorb_antisymmetric,
    ci_to_cluster_amplitudes,
    extract_ci_amplitudes,
    extract_vqe_singles_doubles_amplitudes,
    prepare_cas_slices,
    set_cas_amplitudes_spatial_from_spinorb,
)
from .solve_tcc import solve_tccsd_oe
from .utils import spinorb_from_spatial


def tccsd_from_ci(mc: mcscf.casci.CASCI, backend="pyscf", **kwargs) -> Any:
    # TODO: docs
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
    c_ia, c_ijab = amplitudes_to_spinorb(ci_amps)

    occslice, virtslice = prepare_cas_slices(nocca, noccb, nvirta, nvirtb, ncore, nvir, backend)
    return _tccsd_map[backend](mc._scf, c_ia, c_ijab, occslice, virtslice, **kwargs)


def tccsd_from_fqe(
    scfres: scf.hf.SCF,
    wfn,
    backend="pyscf",
    **kwargs,
):
    from .utils import fqe_to_fake_ci

    mc = fqe_to_fake_ci(wfn, scfres, sz=0)
    return tccsd_from_ci(mc, backend=backend, **kwargs)


def tccsd_from_vqe(
    scfres: scf.hf.SCF,
    vqe: cov.vqe.ActiveSpaceChemistryVQE,
    backend="pyscf",
    **kwargs,
):
    # TODO: docs, type hints
    nocca, noccb = vqe.nalpha, vqe.nbeta
    assert nocca == noccb
    ncas = vqe.nact
    nvirta = ncas - nocca
    nvirtb = ncas - noccb
    assert nvirta == nvirtb
    ncore = vqe.nocc
    if ncore == 0 and (nocca + noccb) != sum(scfres.mol.nelec):
        raise ValueError("The active space needs to contain all electrons if ncore=0.")
    ncas = vqe.nact
    nvir = scfres.mo_coeff.shape[1] - ncore - ncas

    ci_amps = extract_vqe_singles_doubles_amplitudes(vqe)
    c_ia, c_ijab = amplitudes_to_spinorb(ci_amps)

    c_ia = c_ia.real
    c_ijab = c_ijab.real

    occslice, virtslice = prepare_cas_slices(nocca, noccb, nvirta, nvirtb, ncore, nvir, backend)
    return _tccsd_map[backend](scfres, c_ia, c_ijab, occslice, virtslice, **kwargs)


def tccsd_pyscf(
    scfres: scf.hf.SCF,
    c_ia: npt.NDArray,
    c_ijab: npt.NDArray,
    occslice: slice,
    virtslice: slice,
    **kwargs,
):
    mo_coeff = kwargs.pop("mo_coeff", None)
    if mo_coeff is None:
        mo_coeff = scfres.mo_coeff
    else:
        print("Using provided mo_coeff")

    # TODO: docs, hints
    # 1. convert to T amplitudes
    t1cas, t2cas = ci_to_cluster_amplitudes(c_ia, c_ijab)

    # 2. set up pyscf CCSD object
    from pyscf import cc

    # TODO: density fitting?
    # TODO: UCCSD
    mo_coeff = (mo_coeff, mo_coeff)
    ccsd = cc.UCCSD(scfres, mo_coeff=mo_coeff)
    # ccsd = cc.ccsd.CCSD(scfres, mo_coeff=mo_coeff)
    ccsd.max_cycle = kwargs.get("maxiter", ccsd.max_cycle)
    ccsd.verbose = kwargs.get("verbose", 4)
    ccsd.conv_tol = kwargs.get("conv_tol", 1e-7)
    ccsd.iterative_damping = kwargs.get("iterative_damping", 1.0)
    ccsd.level_shift = kwargs.get("level_shift", 0.0)
    # ccsd.conv_tol_normt = 1e-8
    update_amps = ccsd.update_amps

    t1slice = (occslice, virtslice)
    t2slice = (occslice, occslice, virtslice, virtslice)
    _, t1guess, t2guess = ccsd.init_amps()

    t1 = kwargs.pop("t1", None)
    t2 = kwargs.pop("t2", None)
    if t1 is not None:
        print("t1 guess")
        t1guess = t1
        # del t1
    if t2 is not None:
        print("t2 guess")
        t2guess = t2
        # del t2

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
    # np.testing.assert_allclose(t1spin[t1slice], t1cas, atol=1e-8, rtol=0)
    # np.testing.assert_allclose(t2spin[t2slice], t2cas, atol=1e-8, rtol=0)

    t1cas_spatial, t2cas_spatial = set_cas_amplitudes_spatial_from_spinorb(
        ccsd.t1, ccsd.t2, t1cas, t2cas, t1slice, t2slice, zero_input=True
    )
    e_cas = ccsd.energy(t1cas_spatial, t2cas_spatial)
    ccsd.e_cas = e_cas

    if kwargs.get("triples_correction", False):
        # TCCSD(T) triples correction only considers external
        # amplitudes (i.e., completely active amplitudes are excluded)
        # according to 10.1016/j.cplett.2010.11.058, p. 168
        t1cas_zero = t1cas.copy()
        t2cas_zero = t2cas.copy()
        t1cas_zero[...] = 0.0
        t2cas_zero[...] = 0.0
        t1ext_spatial, t2ext_spatial = set_cas_amplitudes_spatial_from_spinorb(
            ccsd.t1, ccsd.t2, t1cas_zero, t2cas_zero, t1slice, t2slice, zero_input=False
        )
        # check that all active amplitudes are zero
        t1spin = spatial2spin(t1ext_spatial)
        t2spin = spatial2spin(t2ext_spatial)
        assert np.all(t1spin[t1slice] == 0.0)
        assert np.all(t2spin[t2slice] == 0.0)
        e_triples = ccsd.ccsd_t(t1=t1ext_spatial, t2=t2ext_spatial)
        ccsd.e_triples = e_triples

    return ccsd


@dataclass
class TCC:
    scfres: scf.hf.SCF
    t1: npt.NDArray
    t2: npt.NDArray
    e_hf: float
    e_cas: float
    e_corr: float
    converged: bool = True
    e_triples: float = 0.0

    @property
    def e_tot(self):
        return self.e_hf + self.e_corr


def compute_integrals_custom(pyscf_molecule, pyscf_scf, mo_coeff):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    from functools import reduce

    from pyscf import ao2mo

    # Get one electrons integrals.
    n_orbitals = mo_coeff.shape[1]
    one_electron_compressed = reduce(np.dot, (mo_coeff.T, pyscf_scf.get_hcore(), mo_coeff))
    one_electron_integrals = one_electron_compressed.reshape(n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule, mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, two_electron_compressed, n_orbitals  # no permutation symmetry
    )
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(two_electron_integrals.transpose(0, 2, 3, 1), order="C")

    # Return.
    return one_electron_integrals, two_electron_integrals


def tccsd_opt_einsum(
    scfres: scf.hf.SCF,
    c_ia: npt.NDArray,
    c_ijab: npt.NDArray,
    occslice: slice,
    virtslice: slice,
    **kwargs,
):
    warnings.warn(
        "This implementation based on opt_einsum is just a reference"
        " code and should not be used for production calculations."
    )
    mo_coeff = kwargs.pop("mo_coeff", None)
    triples_correction = kwargs.pop("triples_correction", None)
    if mo_coeff is None:
        mo_coeff = scfres.mo_coeff
    else:
        print("Using provided mo_coeff")

    mol = scfres.mol
    # 1. convert to T amplitudes
    t1, t2 = ci_to_cluster_amplitudes(c_ia, c_ijab)
    print("=> Amplitudes converted.")

    assert_spinorb_antisymmetric(t2)
    # 2. build CCSD prerequisites
    oei = kwargs.pop("oei", None)
    eri_of_spatial = kwargs.pop("eri_of", None)
    if oei is None or eri_of_spatial is None:
        oei, eri_of_spatial = compute_integrals_custom(mol, scfres, mo_coeff)

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
    print(0.5 * np.einsum("ii", (soei)[o, o]))
    print(hf_energy)
    print("=> Prerequisites built.")
    # exit(0)

    # 3. map T amplitudes to full MO space
    t1_mo = np.zeros((nocc, nvirt), dtype=t1.dtype)
    t2_mo = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)
    t1_mo[occslice, virtslice] = t1
    t2_mo[occslice, occslice, virtslice, virtslice] = t2
    print("=> T amplitudes mapped to full MO space.")

    np.set_printoptions(linewidth=np.inf)
    print("fov max", np.max(np.abs(fock[o, v])))
    print("fov max", np.max(np.abs(fock[v, o])))
    print("t1 max", np.max(np.abs(t1_mo)))
    print("transpose", np.max(np.abs(fock[o, v] - fock[v, o].T)))

    print("t1 fov energy", np.einsum("ia,ai", fock[o, v], t1_mo.T))
    # print("t1 fov energy", np.vdot(t1_mo, fock[o, v]))

    assert_spinorb_antisymmetric(t2_mo)

    from .ccsd import equations_oe as cc

    e_cas = cc.ccsd_energy_correlation(
        t1_mo.T, t2_mo.transpose(2, 3, 0, 1), fock, eri_phys_asymm, o, v
    )
    print(f"CCSD correlation energy from CI amplitudes {e_cas:>12}")

    # solve tccsd amplitude equations
    t1f, t2f = solve_tccsd_oe(
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
        **kwargs,
    )
    # test that the T_CAS amplitudes are still intact
    t1slice = (virtslice, occslice)
    t2slice = (virtslice, virtslice, occslice, occslice)
    np.testing.assert_allclose(t1.T, t1f[t1slice], atol=1e-14, rtol=0)
    np.testing.assert_allclose(t2.transpose(2, 3, 0, 1), t2f[t2slice], atol=1e-14, rtol=0)

    # compute correlation/total TCCSD energy
    e_tcc = cc.ccsd_energy_correlation(t1f, t2f, fock, eri_phys_asymm, o, v)

    t_en = 0.0
    if triples_correction:
        t1ext = t1f.copy()
        t2ext = t2f.copy()
        t1ext[t1slice] = 0.0
        t2ext[t2slice] = 0.0
        e_abcijk = 1 / (
            -eps[v, n, n, n, n, n]
            - eps[n, v, n, n, n, n]
            - eps[n, n, v, n, n, n]
            + eps[n, n, n, o, n, n]
            + eps[n, n, n, n, o, n]
            + eps[n, n, n, n, n, o]
        )
        from .ccsd.equations_oe import t_energy, triples_residual

        t3f = triples_residual(t1ext, t2ext, fock, eri_phys_asymm, o, v) * e_abcijk
        l1 = t1ext.transpose(1, 0)
        l2 = t2ext.transpose(2, 3, 0, 1)
        t_en = t_energy(l1, l2, t3f, fock, eri_phys_asymm, o, v)

    ret = TCC(scfres, t1f, t2f, float(hf_energy) + mol.energy_nuc(), float(e_cas), float(e_tcc))
    ret.e_triples = float(t_en)
    return ret


def ec_cc_from_ci(mc: mcscf.casci.CASCI, **kwargs):
    # TODO: docs
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
    ci_amps_spinorb = amplitudes_to_spinorb(ci_amps, exci=4)

    # if not np.allclose(mc._scf.mo_coeff, mc.mo_coeff, atol=1e-8, rtol=0):
    #     raise NotImplementedError("ec-CC with orbitals other than HF "
    #                               "orbitals not yet implemented.")

    occslice, virtslice = prepare_cas_slices(nocca, noccb, nvirta, nvirtb, ncore, nvir, "oe")
    return ec_cc(mc._scf, *ci_amps_spinorb, occslice, virtslice, **kwargs)


def ec_cc_from_fqe(
    scfres: scf.hf.SCF,
    wfn,
    **kwargs,
):
    from .utils import fqe_to_fake_ci

    mc = fqe_to_fake_ci(wfn, scfres, sz=0)
    return ec_cc_from_ci(mc, **kwargs)


def ec_cc(
    scfres: scf.hf.SCF,
    c1: npt.NDArray,
    c2: npt.NDArray,
    c3: npt.NDArray,
    c4: npt.NDArray,
    occslice: slice,
    virtslice: slice,
    guess_t1_t2_from_ci: bool = True,
    t1_guess: npt.NDArray = None,
    t2_guess: npt.NDArray = None,
    zero_companion_threshold: float = 1e-8,
    static_t3_t4: bool = False,
    level_shift: float = 0.0,
    **kwargs,
):
    """Run externally corrected CC on top
    of a PySCF HF reference and input spin-orbital
    CI coefficients

    Parameters
    ----------
    scfres : scf.hf.SCF
        SCF result
    c1 : npt.NDArray
        C1 coefficients
    c2 : npt.NDArray
        C2 coefficients
    c3 : npt.NDArray
        C3 coefficients
    c4 : npt.NDArray
        C4 coefficients
    occslice : slice
        slice for occupied CAS orbitals
        for mapping active orbitals to full
        MO space
    virtslice : slice
        slice for virtual CAS orbitals
        for mapping active orbitals to full
        MO space
    guess_t1_t2_from_ci : bool, optional
        Use CAS T1 and T2 amplitudes as initial guess, by default True
    t1_guess : npt.NDArray, optional
        User-specified guess for T1, by default None
    t2_guess : npt.NDArray, optional
        User-specified guess for T2, by default None
    zero_companion_threshold : float, optional
        Set T3 and T4 amplitudes without a corresponding C3 and C4
        companion to zero, set to None to disable, by default 1e-8
    static_t3_t4 : bool, optional
        Only compute T3 and T4 contributions once, by default False
    level_shift: float, optional
        Shift on virtual orbitals to stabilize iterations

    Returns
    -------
    TCC
        Container with results from the ec-CC computation
    """
    mo_coeff = kwargs.pop("mo_coeff", None)
    if mo_coeff is None:
        mo_coeff = scfres.mo_coeff
    else:
        print("Using provided mo_coeff")

    mol = scfres.mol
    # 1. convert to T amplitudes
    t1, t2, t3, t4 = [np.array(T) for T in ci_to_cc(c1, c2, c3, c4)]
    print("=> Amplitudes converted.")

    # if requested, set the elements of T that don't
    # have a 'companion' C amplitude to zero (ec-CC-II)
    if zero_companion_threshold is not None:
        t3zero = np.abs(t3) < zero_companion_threshold
        t4zero = np.abs(t4) < zero_companion_threshold
        t3nz_sum = np.sum(~t3zero)
        t4nz_sum = np.sum(~t4zero)
        print("Before:", t3nz_sum, t4nz_sum)

        c3zero = np.abs(c3) < zero_companion_threshold
        c4zero = np.abs(c4) < zero_companion_threshold
        print(f"Using ec-CC-II with threshold {zero_companion_threshold:.2e}")
        t3[c3zero] = 0.0
        t4[c4zero] = 0.0

        t3zero = np.abs(t3) < zero_companion_threshold
        t4zero = np.abs(t4) < zero_companion_threshold
        t3nz_sum = np.sum(~t3zero)
        t4nz_sum = np.sum(~t4zero)
        print("After:", t3nz_sum, t4nz_sum)

    # 2. build CC prerequisites
    oei = kwargs.pop("oei", None)
    eri_of_spatial = kwargs.pop("eri_of", None)
    if oei is None or eri_of_spatial is None:
        oei, eri_of_spatial = compute_integrals_custom(mol, scfres, mo_coeff)

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
    eps[v] += level_shift
    print(f"Level shift = {level_shift:.3f}")
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    # (canonical) Fock operator
    fock = soei + np.einsum("piqi->pq", eri_phys_asymm[:, o, :, o])
    hf_energy = 0.5 * np.einsum("ii", (fock + soei)[o, o])
    print(hf_energy)
    # print("=> Prerequisites built.")

    # 3. compute the constant contributions to singles/doubles
    # residual from t3, t4, t3t1 terms
    from .solve_ec_cc import (  # static_t3_t4_contractions,
        solve_ec_cc,
        static_t3_t4_contractions_subspace,
    )

    t1 = t1.T
    t3 = np.array(t3.transpose(3, 4, 5, 0, 1, 2))
    t4 = np.array(t4.transpose(4, 5, 6, 7, 0, 1, 2, 3))

    # mo_slices = [o.start, o.stop, v.start, v.stop]
    ############################

    # 4. set up T amplitudes in full MO space
    t1_mo = np.zeros((nocc, nvirt))
    t2_mo = (eri_phys_asymm[v, v, o, o] * e_abij).transpose(2, 3, 0, 1)

    # optionally use CAS T1/T2 as guess
    if guess_t1_t2_from_ci:
        t1_mo[occslice, virtslice] = t1.T
        t2_mo[occslice, occslice, virtslice, virtslice] = t2

    if t1_guess is not None:
        t1_mo = t1_guess.copy().T
    if t2_guess is not None:
        t2_mo = t2_guess.transpose(2, 3, 0, 1)

    r1_mo = np.zeros_like(t1_mo.T)
    r2_mo = np.zeros_like(t2_mo.transpose(2, 3, 0, 1))
    if not static_t3_t4:
        t3_mo = np.zeros((nvirt, nvirt, nvirt, nocc, nocc, nocc))
        t4_mo = np.zeros((nvirt, nvirt, nvirt, nvirt, nocc, nocc, nocc, nocc))
        t3_mo[virtslice, virtslice, virtslice, occslice, occslice, occslice] = t3
        t4_mo[
            virtslice, virtslice, virtslice, virtslice, occslice, occslice, occslice, occslice
        ] = t4
        if t1_mo.shape != t1.shape:
            print("Full space")
            occslice = slice(None)
            virtslice = slice(None)
        kwargs.update(dict(t3=t3_mo, t4=t4_mo, occslice=occslice, virtslice=virtslice))
    else:
        r1, r2 = static_t3_t4_contractions_subspace(
            t1, t3, t4, fock, eri_phys_asymm, occslice, virtslice, o, v
        )
        r1_mo[virtslice, occslice] = r1
        r2_mo[virtslice, virtslice, occslice, occslice] = r2

    # 5. solve ec-CC projection equations
    t1f, t2f, e_ecc, converged = solve_ec_cc(
        t1_mo.T,
        t2_mo.transpose(2, 3, 0, 1),
        r1_mo,
        r2_mo,
        fock,
        eri_phys_asymm,
        o,
        v,
        e_ai,
        e_abij,
        **kwargs,
    )
    e_ecc = float(e_ecc)
    t1f = np.array(t1f)
    t2f = np.array(t2f)
    e_corr = e_ecc - hf_energy
    ret = TCC(scfres, t1f, t2f, hf_energy + mol.energy_nuc(), 0.0, e_corr, converged)
    return ret


_tccsd_map: Dict[str, Callable] = {
    "oe": tccsd_opt_einsum,
    "pyscf": tccsd_pyscf,
}
