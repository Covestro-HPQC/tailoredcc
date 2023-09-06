# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import time

import numpy as np
from opt_einsum import contract


def einsum(*args, **kwargs):
    kwargs["optimize"] = True
    # from opt_einsum import contract_path
    # path_info = contract_path(*args)
    # print(path_info[1])
    # print()
    return contract(*args, **kwargs)


def solve_ec_cc(
    t1,
    t2,
    r1,
    r2,
    fock,
    g,
    o,
    v,
    e_ai,
    e_abij,
    maxiter=100,
    conv_tol=1.0e-8,
    diis_size=7,
    diis_start_cycle=4,
    verbose=4,
):
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
    if verbose > 3:
        print(f"\tInitial ec-CC energy: {old_energy}")
    for idx in range(maxiter):
        start = time.time()
        singles_res = cc.singles_residual(t1, t2, fock, g, *mo_slices)
        doubles_res = cc.doubles_residual(t1, t2, fock, g, *mo_slices)

        # add the 'frozen' contributions for T3/T4 contractions
        singles_res += r1
        doubles_res += r2

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
        delta_e = current_energy - old_energy
        d1 = t1 - new_singles
        d2 = t2 - new_doubles
        rnorm = np.linalg.norm(d1) + np.linalg.norm(d2)

        if np.abs(delta_e) < conv_tol:
            if verbose > 3:
                print(f"\tConverged in iteration {idx}.")
            return new_singles, new_doubles, current_energy
        else:
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy
            if verbose > 3:
                print(
                    "\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}\t{: 5.15f}\t{: 5.3f}s".format(
                        idx, old_energy, delta_e, rnorm, time.time() - start
                    )
                )
    else:
        print("Did not converge.")
        return new_singles, new_doubles, current_energy


def static_t3_t4_contractions(t1, t3, t4, f, g, occslice, virtslice, o, v):
    oovv = (occslice, occslice, virtslice, virtslice)
    r1 = 0.25 * einsum("kjbc,bcaikj->ai", g[o, o, v, v][oovv], t3)
    # 	  1.0000 f(k,c)*t3(c,a,b,i,j,k)
    r2 = 1.0 * einsum("kc,cabijk->abij", f[o, v][occslice, virtslice], t3)
    # 	  0.5000 P(i,j)<l,k||c,j>*t3(c,a,b,i,l,k)
    contracted_intermediate = 0.5 * einsum(
        "lkcj,cabilk->abij", g[o, o, v, o][occslice, occslice, virtslice, occslice], t3
    )
    r2 += 1.0 * contracted_intermediate + -1.0 * einsum("abij->abji", contracted_intermediate)
    # 	  0.5000 P(a,b)<k,a||c,d>*t3(c,d,b,i,j,k)
    contracted_intermediate = 0.5 * einsum(
        "kacd,cdbijk->abij", g[o, v, v, v][occslice, virtslice, virtslice, virtslice], t3
    )
    r2 += 1.0 * contracted_intermediate + -1.0 * einsum("abij->baij", contracted_intermediate)
    # 	  0.2500 <l,k||c,d>*t4(c,d,a,b,i,j,l,k)
    r2 += 0.25 * einsum("lkcd,cdabijlk->abij", g[o, o, v, v][oovv], t4)
    # 	 -1.0000 <l,k||c,d>*t1(c,k)*t3(d,a,b,i,j,l)
    r2 += -1.0 * einsum(
        "lkcd,ck,dabijl->abij",
        g[o, o, v, v][oovv],
        t1,
        t3,
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    # 	 -0.5000 P(i,j)<l,k||c,d>*t1(c,j)*t3(d,a,b,i,l,k)
    contracted_intermediate = -0.5 * einsum(
        "lkcd,cj,dabilk->abij",
        g[o, o, v, v][oovv],
        t1,
        t3,
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    r2 += 1.0 * contracted_intermediate + -1.0 * einsum("abij->abji", contracted_intermediate)
    # 	 -0.5000 P(a,b)<l,k||c,d>*t1(a,k)*t3(c,d,b,i,j,l)
    contracted_intermediate = -0.5 * einsum(
        "lkcd,ak,cdbijl->abij",
        g[o, o, v, v][oovv],
        t1,
        t3,
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    r2 += 1.0 * contracted_intermediate + -1.0 * einsum("abij->baij", contracted_intermediate)
    return r1, r2
