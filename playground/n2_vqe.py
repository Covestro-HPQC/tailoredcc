import covvqetools as cov
import numpy as np
import pandas as pd
from covvqetools.instant_vqes import QNPVQE
from pyscf import ao2mo, gto, mcscf, scf

depths = [
    # 6,
    12,
    # 18,
    # 24,
]
params = None
vqe_data = []

for d in np.arange(0.8, 2.9, 0.1):
    mol = gto.Mole()
    mol.atom = "N 0 0 0; N 0 0 %f" % d
    mol.basis = "cc-pvdz"
    mol.verbose = 4
    mol.build()

    # Hartree Fock
    mf = scf.RHF(mol)
    mf.kernel()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    assert mf.converged

    # CASCI(6,6)
    ncas = 6
    nalpha, nbeta = 3, 3
    cas = mcscf.CASCI(mf, nelecas=(nalpha, nbeta), ncas=ncas)
    cas.kernel()
    # assert cas.converged

    h1, ecore = cas.get_h1eff()
    h2 = ao2mo.restore(1, cas.ao2mo(), ncas)
    two_body_integrals = h2.transpose(0, 2, 3, 1)

    maxiter_vqe = 1000
    # maxiter_vqe = 10
    for depth in depths:
        vqe = QNPVQE(
            depth=depth,
            nact=ncas,
            nalpha=nalpha,
            nbeta=nbeta,
            # NOTE: trick to get nocc correct even though we don't have nuclear charges
            nocc=cas.ncore,
            nchar=-sum(mol.atom_charges()),
            ###
            measurement_method=cov.CASBOX,
            core_energy=ecore,
            one_body_integrals=h1,
            two_body_integrals=two_body_integrals,
        )
        if params is None:
            np.random.seed(42)
            vqe.params += np.random.randn(*vqe.params.shape) * 1e-2
        else:
            vqe.params = params
        opt = cov.LBFGSB(atol=None, gtol=1e-8, ftol=None)
        vqe.vqe_optimize(opt=opt, maxiter=maxiter_vqe)
        energy_vqe = vqe.vqe_energy(vqe.params)
        # params = np.array(vqe.params.copy())

        vqe_data.append([d, depth, maxiter_vqe, np.array(vqe.params), float(energy_vqe)])


vqe_df = pd.DataFrame(data=vqe_data, columns=["d", "depth", "maxiter", "params", "energy"])
vqe_df.to_hdf("n2_vqe.h5", key="df")
