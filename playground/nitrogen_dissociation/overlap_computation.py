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


if __name__ == "__main__":
    params = {}
    data_vqe = []
    bla = {}
    # shadow_size = 200_000
    # shadow_size = 1_000
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

        state_dict_current = extract_state_dict(
            mc.fcisolver, mc.ci, mc.ncas, mc.nelecas[0], mc.nelecas[1], amplitude_cutoff=1e-5
        )

        dep = np.load(f"states/state_{ii}.npz", allow_pickle=True)
        # state_dict=state_dict, core_energy=ecore, h1=h1, h2=h2,
        # energy=e, energy_ci=mc.e_tot,
        # )
        state_dict = dep["state_dict"].item()
        for k, v in state_dict.items():
            assert k in state_dict_current
            np.testing.assert_allclose(v, state_dict_current[k], atol=1e-10)

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

        shadow_size = int(shots_n2[ii][1])
        assert shots_n2[ii][0] == d
        for k in range(nsamples):
            retfile = f"overlaps/overlaps_{ii}_{shadow_size}_{k}.npz"
            if os.path.isfile(retfile):
                print(f"File {retfile} exists, continuing...")
                continue
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
            shadow_method.measure_shadow = None
            shadow_method.load_shadow_cache(f"shadows/shadow_{ii}_{shadow_size}_{k}.npz")

            nocca = nalpha
            nvirta = ncas - nocca
            from pyscf.ci.cisd import tn_addrs_signs

            t1addrs, t1signs = tn_addrs_signs(ncas, nocca, 1)
            t2addrs, t2signs = tn_addrs_signs(ncas, nocca, 2)

            from tailoredcc.amplitudes import (
                detstrings_doubles,
                detstrings_singles,
                interleave_strings,
            )

            hfdet = np.zeros(ncas, dtype=int)
            hfdet[:nocca] = 1
            _, detsa = detstrings_singles(nocca, nvirta)
            _, detsaa = detstrings_doubles(nocca, nvirta)

            cis_a = interleave_strings(detsa, hfdet)
            cis_b = interleave_strings(hfdet, detsa)
            # cis_a = cis_a.reshape(nocca, nvirta)
            # cis_b = cis_b.reshape(noccb, nvirtb)

            cid_ab = interleave_strings(detsa, detsa)
            # , vqe.params
            # ).reshape(nocca * nvirta, noccb * nvirtb)
            # cid_ab = np.einsum("ij,i,j->ij", cid_ab, t1signs, t1signs)
            # cid_ab = cid_ab.reshape(nocca, nvirta, noccb, nvirtb).transpose(0, 2, 1, 3)

            cid_aa = interleave_strings(detsaa, hfdet)
            cid_bb = interleave_strings(hfdet, detsaa)
            c0 = interleave_strings(hfdet, hfdet)

            states = {
                "0": c0,
                "a": cis_a,
                "b": cis_b,
                "ab": cid_ab,
                "aa": cid_aa,
                "bb": cid_bb,
            }

            ret = {}
            for block, state in states.items():
                start = time.time()
                print(block, len(state))
                ret[block] = np.array(
                    shadow_method.overlap(
                        None, ansatz=ansatz, states=state, shots_per_distinct_circuit=1
                    )
                )
                stop = time.time()
                print(f"took {stop - start} seconds")
            np.savez(retfile, **ret)
