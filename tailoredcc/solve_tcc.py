# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import time

import numpy as np


def _solve_tccsd_oe(
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
    maxiter=100,
    conv_tol=1.0e-8,
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

    from .ccsd import oe as cc

    mo_slices = [o.start, o.stop, v.start, v.stop]
    old_energy = cc.ccsd_energy(t1, t2, fock, g, *mo_slices)
    print(f"\tInitial CCSD energy: {old_energy}")
    for idx in range(maxiter):
        start = time.time()
        singles_res = np.array(cc.singles_residual(t1, t2, fock, g, *mo_slices))
        doubles_res = np.array(cc.doubles_residual(t1, t2, fock, g, *mo_slices))

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

        current_energy = cc.ccsd_energy(new_singles, new_doubles, fock, g, *mo_slices)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < conv_tol:
            print(f"\tConverged in iteration {idx}.")
            return new_singles, new_doubles
        else:
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy
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


def solve_tccsd(
    mp,
    occslice=None,
    virtslice=None,
    tguess=None,
    max_iter=100,
    stopping_eps=1.0e-8,
    diis_size=7,
    diis_start_cycle=4,
    backend="libcc",
):
    freeze_amplitude_slices = False
    if occslice is not None and virtslice is not None:
        freeze_amplitude_slices = True

    import adcc
    from adcc.functions import direct_sum

    from .ccsd import DISPATCH
    from .ccsd.equations_adcc import CCSDIntermediates

    cc = DISPATCH[backend]
    print(f"Using '{backend}' residual equations.")

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

    if tguess is None:
        t = adcc.AmplitudeVector(ov=mp.mp2_diffdm.ov, oovv=mp.t2oo)
    else:
        t = tguess

    if diis_size is not None:
        from .diis import DIIS

        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        old_vec = t

    old_energy = cc.ccsd_energy(mp, t)
    print(f"\tInitial CCSD energy: {old_energy}")
    fmt = "{:>10d}{:>24.15f}{:>15.3e}{:>15.3e}{:>20.6f}"
    # print header for CCSD iterations
    print(
        "\t{:>10s}{:>24s}{:>15s}{:>15s}{:>20s}".format(
            "Iteration", "Energy [Eh]", "Delta E [Eh]", "|r|", "time/iteration (s)"
        )
    )
    for idx in range(max_iter):
        start = time.time()
        if backend == "libcc":
            im = CCSDIntermediates(mp, t)
            singles_res = cc.singles_residual(mp, t, im)
            doubles_res = cc.doubles_residual(mp, t, im)
        else:
            singles_res = cc.singles_residual(mp, t)
            doubles_res = cc.doubles_residual(mp, t)

        # set the CAS-only residual to zero
        if freeze_amplitude_slices:
            zero_slices(singles_res, doubles_res, occslice, virtslice)

        new_singles = t.ov + singles_res / e_ia
        new_doubles = t.oovv + doubles_res / e_ijab
        new_t = adcc.AmplitudeVector(ov=new_singles, oovv=new_doubles)
        # print(new_t.oovv.describe_symmetry())
        # rnorm = np.sqrt(singles_res.dot(singles_res) + doubles_res.dot(doubles_res))

        # diis update
        if diis_size is not None:
            vectorized_iterate = new_t
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(
                vectorized_iterate, error_vec
            ).evaluate()
            new_t = new_vectorized_iterate
            old_vec = new_vectorized_iterate

        diff = new_t - t
        rnorm = np.sqrt(diff.dot(diff))
        current_energy = cc.ccsd_energy(mp, new_t)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps:
            print(f"\tConverged in iteration {idx}.")
            return new_t
        else:
            t = new_t
            old_energy = current_energy
            print("\t" + fmt.format(idx, old_energy, delta_e, rnorm, time.time() - start))
    else:
        print("Did not converge.")
        return new_t
