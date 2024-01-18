import time

import covvqetools as cov
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from covvqetools.instant_vqes import QNPVQE
from covvqetools.measurement.classical_shadows import MatchGateShadowTermwise
from covvqetools.pyscf import extract_state_dict
from pyscf import ao2mo, gto, mcscf, scf

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
mf.chkfile = "scf.chk"
ehf = mf.kernel()
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
sizes = [
    # 10,
    # 20, 50,
    # 100,
    # 200, 500,
    1000,
    2000,
    5000,
    1e4,
    # 2e4,
    # 5e4,
    # 1e5, 2e5, 5e5,
    # 1e6,
]
nsamp = 30

for shadow_size in sizes:
    shadow_size = int(shadow_size)
    for ii in range(nsamp):
        print(ii, shadow_size)
        shadow_method = MatchGateShadowTermwise(
            None,
            dev=dev,
            shot_distributor=("fixed_shots_per_shadow", {"shots_per_shadow": shadow_size}),
            # num_processes=32,
            num_processes=8,
            jw_ordering="seq",
        )
        start = time.time()
        shadow_overlaps = np.array(
            shadow_method.overlap(
                None, ansatz=ansatz, states=states, shots_per_distinct_circuit=1000
            )
        )

        overlap_shadow = shadow_method.get_shadow_from_cache(
            params=(None,),
            ansatz=ansatz,
            shots=shadow_size,
            shadow_type="overlaps",
            wires=dev.wires,
        )

        # skeys = list(shadow_method.shadow_cache.keys())
        # print(skeys)
        # assert len(skeys) == 1
        # skey = skeys[0]
        # overlap_shadow = shadow_method.shadow_cache[skey]['shadow']
        vars_overlap = np.array(
            shadow_method.estimate_slater_overlap_from_shadow(
                states,
                shadow=overlap_shadow,
                median_of_means_groups=1,
                calculate="var",
            )
        )
        # print(vars_overlap)
        stop = time.time()
        print(f"took {stop - start} seconds")
        np.savez(
            f"shadow_records/shadow_multi_h4_{shadow_size}_{ii}",
            shadow=shadow_overlaps,
            C=mf.mo_coeff,
            states=states,
            size=shadow_size,
            sample=ii,
            vars_overlap=vars_overlap,
        )
