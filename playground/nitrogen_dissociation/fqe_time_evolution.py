import fqe
import numpy as np
import pandas as pd
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from pyscf import ao2mo, lib, mcscf, scf

from tailoredcc import tccsd_from_fqe
from tailoredcc.utils import fqe_to_fake_ci

lib.num_threads(24)
np.random.seed(42)

params = {}
data_vqe = []
for ii, d in enumerate(np.arange(0.8, 2.9, 0.1)):
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

    cas = mcscf.CASCI(mf, nelecas=(nalpha, nbeta), ncas=ncas)
    h1, ecore = cas.get_h1eff()
    h2 = ao2mo.restore(1, cas.ao2mo(), ncas)
    two_body_integrals = h2.transpose(0, 2, 3, 1)
    fqe_ham = RestrictedHamiltonian((h1, np.einsum("ijlk", -0.5 * two_body_integrals)), e_0=ecore)

    cas.kernel()

    fqe_wf = fqe.Wavefunction([[nalpha + nbeta, 0, ncas]])
    fqe_wf.set_wfn(strategy="hartree-fock")
    fqe_wf.print_wfn()

    fqe_hf_energy = fqe_wf.expectationValue(fqe_ham).real
    np.testing.assert_allclose(fqe_hf_energy, mf.e_tot, atol=1e-12, rtol=0)

    fqe_energies = [fqe_hf_energy]
    nsteps = 200000
    threshold = 5e-4
    for ii in range(nsteps):
        fqe_wf = fqe_wf.time_evolve(-1j * 0.085, fqe_ham)
        fqe_wf.normalize()
        # fqe_wf.print_wfn()
        fqe_energy = fqe_wf.expectationValue(fqe_ham).real

        previous = fqe_energies[-1]
        fqe_energies.append(fqe_energy)
        dE = previous - fqe_energy
        # print(f"FQE energy = {fqe_energy}, dE = {dE}, CAS energy error {fqe_energy - cas.e_tot}")
        if np.abs(dE) < threshold:
            steps = ii
            break

    mc_fqe = fqe_to_fake_ci(fqe_wf, mf)
    civec = mc_fqe.ci
    civec /= np.sign(cas.ci[0, 0]) * np.sign(civec[0, 0])
    diff = civec - cas.ci
    var = np.var(diff)
    print(fqe_energy - cas.e_tot, var)
    tcc = tccsd_from_fqe(mf, fqe_wf)

    data_vqe.extend(
        [
            [d, fqe_energy, tcc.e_tot, threshold, var, steps],
            # [d, "FQE-TCCSD", tcc_vqe.e_tot, depth],
        ]
    )
    # for _ in range(ncycles):
    #     tcc_vqe_noise = tccsd_from_vqe(mf, vqe, gaussian_noise=noise, maxiter=200)
    #     data_vqe.append(
    #         [d, "VQE-TCCSD (noise)", tcc_vqe_noise.e_tot, depth]
    #     )

df = pd.DataFrame(data=data_vqe, columns=["d", "FQE", "FQE-TCCSD", "threshold", "var", "steps"])
print(df)
df.to_hdf("n2_dissociation_fqe.h5", key="df")
