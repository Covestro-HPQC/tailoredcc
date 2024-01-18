import os
import time

import covvqetools as cov
import h5py
import numpy as np
import pandas as pd
import pennylane as qml
from covvqetools.measurement.classical_shadows import MatchGateShadowTermwise

# from covvqetools.instant_vqes import QNPVQE
from covvqetools.pyscf import extract_state_dict
from pyscf import ao2mo, lib, mcscf, scf

lib.num_threads(24)
np.random.seed(42)

state_sz = {
    0: 98,
    1: 96,
    2: 96,
    3: 96,
    4: 96,
    5: 96,
    6: 96,
    7: 96,
    8: 96,
    9: 96,
    10: 96,
    11: 96,
    12: 96,
    13: 96,
    14: 96,
    15: 72,
    16: 56,
    17: 92,
    18: 56,
    19: 132,
    20: 136,
}


if __name__ == "__main__":
    params = {}
    data_vqe = []
    bla = {}
    # shadow_size = 100_000
    shots_n2 = np.loadtxt("shots_n2.txt")
    nsamples = 5
    for ii, d in enumerate(np.arange(0.8, 2.9, 0.1)):
        if d != 1.1:
            continue
        print(ii, d)
        mol, scf_dict = scf.chkfile.load_scf(f"mos/scf_{ii}.chk")
        mf = scf.RHF(mol)
        mf.__dict__.update(scf_dict)
        dx = np.load(f"mos/mo_{ii}.npz")
        mos = dx["mos"]
        mo_occ = dx["mo_occ"]
        assert d == dx["d"]

        np.testing.assert_allclose(mf.e_tot, dx["e_tot"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(mf.mo_coeff, mos, atol=1e-14, rtol=0)
        ncas = 6
        nalpha, nbeta = 3, 3

        mc = mcscf.CASCI(mf, nelecas=(nalpha, nbeta), ncas=ncas)
        mc.fcisolver.conv_tol = 1e-14
        h1, ecore = mc.get_h1eff()
        h2 = ao2mo.restore(1, mc.get_h2eff(), ncas)
        two_body_integrals = h2.transpose(0, 2, 3, 1)

        mc.kernel()

        state_dict = extract_state_dict(
            mc.fcisolver, mc.ci, mc.ncas, mc.nelecas[0], mc.nelecas[1], amplitude_cutoff=1e-5
        )

        qop = cov.operators.hamiltonian(
            core_energy=ecore,
            one_body_integrals=h1,
            two_body_integrals=two_body_integrals,
            convention="of",
            jw_ordering="seq",
            # jw_ordering="int",
            # qubit_ordering="seq",  # unused kwarg
            # qubit_ordering="int",
        )
        dev = qml.device("cov.qubit", wires=2 * mc.ncas)
        ms = cov.Hermitian(qop, dev=dev)

        def ansatz(*_, wires):
            cov.Superposition(state_dict, wires=wires)

        e = ms.expval(None, ansatz=ansatz)
        diff = np.abs(e - mc.e_tot)
        assert diff < 1e-7

        np.savez(
            f"states/state_{ii}.npz",
            state_dict=state_dict,
            core_energy=ecore,
            h1=h1,
            h2=h2,
            energy=e,
            energy_ci=mc.e_tot,
        )
        shadow_size = int(shots_n2[ii][1])
        assert shots_n2[ii][0] == d
        print("Recording shadow", ii, d, shadow_size)

        for k in range(nsamples):
            print(f"Sample {k}")
            shadow_method = MatchGateShadowTermwise(
                None,
                dev=dev,
                shot_distributor=("fixed_shots_per_shadow", {"shots_per_shadow": shadow_size}),
                # num_processes=32,
                # num_processes=16,
                num_processes=8,
                # num_processes=1,
                # num_processes=None,
                jw_ordering="seq",
            )
            start = time.time()
            states = [6 * "1" + 6 * "0"]

            # shadow_method.measure_shadow = None
            # shadow_method.load_shadow_cache(f"shadows/shadow_{ii}_{shadow_size}_{k}.npz")

            shadow_overlaps = np.array(
                shadow_method.overlap(
                    None, ansatz=ansatz, states=states, shots_per_distinct_circuit=1
                )
            )
            print(shadow_overlaps)

            stop = time.time()
            print(f"took {stop - start} seconds")
            shadow_method.save_shadow_cache(f"shadows/shadow_{ii}_{shadow_size}_{k}.npz")
