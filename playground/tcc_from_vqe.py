import covvqetools as cov
import numpy as np
from covvqetools.instant_vqes import QNPVQE
from pyscf import ao2mo, gto, mcscf, scf
from tailoredcc import tccsd_from_ci, tccsd_from_vqe

conv_tol_e = 1e-12
conv_tol_g = 1e-8

# xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/n2_real.xyz"
xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/p-benzyne.xyz"
# xyzfile = "benzene_blind.xyz"
# nact = 4
# nalpha, nbeta = (2, 2)
# depth = 4

nact = 6
nalpha, nbeta = (3, 3)
# depth = 32
# nact = 8
# nalpha, nbeta = (4, 4)
depth = 12

maxiter_vqe = 50
# maxiter_vqe = 1000


ncas = nact
nvirta = ncas - nalpha
nvirtb = ncas - nbeta
nocca = nalpha
noccb = nbeta

assert nocca == noccb
# basis = "cc-pvdz"
# basis = "sto-3g"
basis = "6-31g"

# measurement_method = cov.DoubleFactorized.with_kwargs(
#     num_leafs="max",
#     default_device_type="cov.quicksilver.casbox",
#     layout="square_non_autodiff",
# )
measurement_method = cov.CASBOX

mol = gto.M(atom=str(xyzfile), basis=basis, verbose=4)

# mol = gto.M(
#     atom="""
# N 0 0 0
# N 0 0 3.5
# """,
#     unit="bohr",
#     basis=basis,
#     verbose=4,
# )
scfres = scf.RHF(mol)
scfres.conv_tol = conv_tol_e
scfres.conv_tol_grad = conv_tol_g
scfres.kernel()

# run pyscf CASCI and generate integrals to start with
mci = mcscf.CASCI(scfres, nact, (nalpha, nbeta))
mci.kernel()
h1, ecore = mci.get_h1eff()
h2 = ao2mo.restore(1, mci.ao2mo(), nact)
two_body_integrals = h2.transpose(0, 2, 3, 1)

# run QNPVQE with fixed integrals
vqe = QNPVQE(
    depth=depth,
    nact=nact,
    nalpha=nalpha,
    nbeta=nbeta,
    # NOTE: trick to get nocc correct even though we don't have nuclear charges
    nocc=mci.ncore,
    nchar=-sum(mol.atom_charges()),
    ###
    measurement_method=measurement_method,
    core_energy=ecore,
    one_body_integrals=h1,
    two_body_integrals=two_body_integrals,
)
np.random.seed(42)
vqe.params += np.random.randn(*vqe.params.shape) * 1e-2
opt = cov.LBFGSB(atol=1e-10, gtol=1e-10, ftol=None)
vqe.vqe_optimize(opt=opt, maxiter=maxiter_vqe)
energy_vqe = vqe.vqe_energy(vqe.params)
print("CI energy error", energy_vqe - mci.e_tot)


backend = "pyscf"
# backend = "libcc"
# backend = "adcc"
# backend = "oe"
ret_ci = tccsd_from_ci(mci, backend=backend)

ret_vqe = tccsd_from_vqe(scfres, vqe, backend=backend)

print("CAS energy diff", ret_ci.e_cas - ret_vqe.e_cas)

print("TCC(CI)/TCC(VQE) diff = ", ret_vqe.e_tot - ret_ci.e_tot)

print(energy_vqe - (ret_vqe.e_cas + scfres.e_tot))
print(mci.e_tot - (ret_vqe.e_cas + scfres.e_tot))
print(mci.e_tot - (ret_ci.e_cas + scfres.e_tot))
# print(energy_vqe - ret_vqe.e_tot)
