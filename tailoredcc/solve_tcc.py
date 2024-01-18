# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import time

import numpy as np


def solve_tccsd_oe(
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

    from .ccsd import equations_oe as cc

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
