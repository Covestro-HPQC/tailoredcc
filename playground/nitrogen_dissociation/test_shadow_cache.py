import os
import time

import covvqetools as cov
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from covvqetools.instant_vqes import QNPVQE
from covvqetools.measurement.classical_shadows import MatchGateShadowTermwise
from covvqetools.pyscf import extract_state_dict
from pyscf import ao2mo, gto, lib, mcscf, scf

# lib.num_threads(24)
np.random.seed(42)


def test_shadow_cache():
    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.atom = """H 0 0 0
    H 0 0 1.23
    H 0 1.23 0
    H 0 1.23 1.23"""
    mol.verbose = 4
    mol.spin = 0
    mol.build()
    mol.symmetry = False
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-10
    # mf.chkfile = "scf.chk"
    mf.kernel()
    mo1 = mf.stability(verbose=0)[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability(verbose=1)

    mc = mcscf.CASCI(mf, ncas=4, nelecas=4)
    h1, ecore = mc.get_h1eff()
    h2 = mc.get_h2eff()
    h2 = ao2mo.restore(1, h2, mc.ncas).transpose(0, 2, 3, 1)
    mc.kernel()

    state_dict = extract_state_dict(
        mc.fcisolver, mc.ci, mc.ncas, mc.nelecas[0], mc.nelecas[1], amplitude_cutoff=0
    )
    print(len(state_dict))

    qop = cov.operators.hamiltonian(
        core_energy=ecore,
        one_body_integrals=h1,
        two_body_integrals=h2,
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
    print(e - mc.e_tot)

    states = list(state_dict.keys())

    shadow_size = 100000

    shadow_method = MatchGateShadowTermwise(
        None,
        dev=dev,
        shot_distributor=("fixed_shots_per_shadow", {"shots_per_shadow": shadow_size}),
        # num_processes=32,
        num_processes=8,
        jw_ordering="seq",
    )
    start = time.time()
    states = ["11110000"]
    shadow_overlaps = np.array(
        shadow_method.overlap(None, ansatz=ansatz, states=states, shots_per_distinct_circuit=1)
    )
    print(shadow_overlaps)

    stop = time.time()
    print(f"took {stop - start} seconds")

    shadow_method.save_shadow_cache("test.npz")

    shadow_method = MatchGateShadowTermwise(
        None,
        dev=dev,
        shot_distributor=("fixed_shots_per_shadow", {"shots_per_shadow": shadow_size}),
        # num_processes=32,
        num_processes=8,
        jw_ordering="seq",
    )
    shadow_method.measure_shadow = None
    shadow_method.load_shadow_cache("test.npz")

    shadow_overlaps2 = np.array(
        shadow_method.overlap(
            None,
            ansatz=ansatz,
            states=states,
            shots_per_distinct_circuit=1,
            use_past_measurements=True,
        )
    )
    print(shadow_overlaps2)

    shadow_overlaps3 = np.array(
        shadow_method.overlap(
            None,
            ansatz=ansatz,
            states=["11001100"],
            shots_per_distinct_circuit=1,
            use_past_measurements=True,
        )
    )
    print(shadow_overlaps3, state_dict["11001100"])
