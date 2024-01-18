import os
import time

import covvqetools as cov
import h5py
import numpy as np
import pandas as pd
import pennylane as qml

# from covvqetools.instant_vqes import QNPVQE
from covvqetools.pyscf import extract_state_dict
from pyscf import ao2mo, lib, mcscf, scf

from tailoredcc.amplitudes import amplitudes_to_spinorb, prepare_cas_slices
from tailoredcc.tailoredcc import tccsd_from_ci, tccsd_opt_einsum, tccsd_pyscf

lib.num_threads(24)


if __name__ == "__main__":
    params = {}
    data_vqe = []
    bla = {}
    # shadow_size = 200_000
    # shadow_size = 1_000
    shots_n2 = np.loadtxt("shots_n2.txt")
    nsamples = 5
    # nsamples = 1
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

        ncore = mc.ncore
        nvir = mf.mo_coeff.shape[1] - ncore - ncas

        mc.kernel()

        tccsd_exact = tccsd_from_ci(mc)

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
        tcc_ret = {}
        for k in range(nsamples):
            overlaps = np.load(f"overlaps/overlaps_{ii}_{shadow_size}_{k}.npz")
            nocca = nalpha
            noccb = nbeta
            nvirta = ncas - nocca
            nvirtb = ncas - noccb
            from pyscf.ci.cisd import tn_addrs_signs

            t1addrs, t1signs = tn_addrs_signs(ncas, nocca, 1)
            t2addrs, t2signs = tn_addrs_signs(ncas, nocca, 2)

            # deal with imaginary part
            c0 = overlaps["0"]
            print(c0)
            angle = np.angle(c0)
            overlaps_rot = {k: np.real(v / np.exp(1j * angle)) for k, v in overlaps.items()}
            print(overlaps_rot["0"])

            # overlaps_rot = {k: np.real(v) for k, v in overlaps.items()}
            # overlaps_rot = {k: np.abs(v) for k, v in overlaps.items()}
            # print(overlaps_rot['0'])

            del overlaps
            cis_a = overlaps_rot["a"] * t1signs
            cis_b = overlaps_rot["b"] * t1signs
            cis_a = cis_a.reshape(nocca, nvirta)
            cis_b = cis_b.reshape(noccb, nvirtb)

            cid_ab = overlaps_rot["ab"].reshape(nocca * nvirta, noccb * nvirtb)
            cid_ab = np.einsum("ij,i,j->ij", cid_ab, t1signs, t1signs)
            cid_ab = cid_ab.reshape(nocca, nvirta, noccb, nvirtb).transpose(0, 2, 1, 3)

            cid_aa = overlaps_rot["aa"] * t2signs
            cid_bb = overlaps_rot["bb"] * t2signs
            c0 = overlaps_rot["0"]
            ci_amps = {0: c0, "a": cis_a, "b": cis_b, "aa": cid_aa, "bb": cid_bb, "ab": cid_ab}

            c_ia, c_ijab = amplitudes_to_spinorb(ci_amps)
            # occslice, virtslice = prepare_cas_slices(nocca, noccb, nvirta, nvirtb, ncore, nvir, backend="oe")
            # tccsd_opt_einsum(mf, c_ia, c_ijab, occslice, virtslice)#, **kwargs)

            occslice, virtslice = prepare_cas_slices(
                nocca, noccb, nvirta, nvirtb, ncore, nvir, backend="pyscf"
            )
            ret = tccsd_pyscf(mf, c_ia, c_ijab, occslice, virtslice)  # , **kwargs)
            tcc_ret[k] = ret
            print(ret.e_cas, mc.e_tot - mf.e_tot)
            print("error CAS [mEh]", 1000 * (mf.e_tot + ret.e_cas - mc.e_tot))
            print("error [mEh] = ", 1000 * (ret.e_tot - tccsd_exact.e_tot))

        import pandas as pd

        energies = [[v.e_tot, v.e_cas + mf.e_tot] for k, v in tcc_ret.items()]
        df = pd.DataFrame(data=energies, columns=["energy_tcc", "energy_ci"])

        df["error_tcc"] = 1000 * np.abs(df.energy_tcc - tccsd_exact.e_tot)
        df["error_cas"] = 1000 * np.abs(df.energy_ci - mc.e_tot)

        print(df)
        print(df.describe())
