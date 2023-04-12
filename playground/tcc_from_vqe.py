from covvqetools.instant_vqes import QNPVQE
import covvqetools as cov
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.ci.cisd import tn_addrs_signs
from tailoredcc.amplitudes import detstrings_singles, detstrings_doubles, amplitudes_to_spinorb
from tailoredcc.tailoredcc import tccsd, tccsd_from_ci, tccsd_from_vqe

conv_tol_e = 1e-12
conv_tol_g = 1e-8

xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/n2_real.xyz"
# xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/p-benzyne.xyz"
nact = 4
nalpha, nbeta = (2, 2)
depth = 16

nact = 6
nalpha, nbeta = (3, 3)
# depth = 32
# nact = 8
# nalpha, nbeta = (4, 4)
depth = 4

# maxiter_vqe = 10
maxiter_vqe = 2000


ncas = nact
nvirta = ncas - nalpha
nvirtb = ncas - nbeta
nocca = nalpha
noccb = nbeta

assert nocca == noccb
# basis = "cc-pvdz"
# basis = "sto-3g"
basis = "6-31g"

# measurement_method = cov.CASBOX
measurement_method = cov.DoubleFactorized.with_kwargs(
    num_leafs="max",
    default_device_type="cov.quicksilver.casbox",
    layout="square_non_autodiff",
)

# mol = gto.M(atom=str(xyzfile), basis=basis, verbose=4)
mol = gto.M(atom="""
N 0 0 0
N 0 0 3.5
""", unit="bohr", basis=basis, verbose=4)
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
opt = cov.LBFGSB(atol=1e-10, gtol=1e-10, ftol=None)

iter = 0
def callback(*args):
    print(iter)
    iter += 1

vqe.vqe_optimize(opt=opt, maxiter=maxiter_vqe)
energy_vqe = vqe.vqe_energy(vqe.params)

print(energy_vqe)
print("CI energy error", energy_vqe - mci.e_tot)

t1addrs, t1signs = tn_addrs_signs(ncas, nocca, 1)
t2addrs, t2signs = tn_addrs_signs(ncas, nocca, 2)

hfdet = np.zeros(ncas, dtype=int)
hfdet[:nocca] = 1
_, detsa = detstrings_singles(nocca, nvirta)
_, detsaa = detstrings_doubles(nocca, nvirta)

dets_ab = []
for det in detsa:
    cdet = np.zeros(2 * ncas, dtype=int)
    cdet[::2] = det
    cdet[1::2] = hfdet
    dets_ab.append("".join([str(w) for w in cdet]))
# det_arr = np.array([int(str(slater_det_int), 2) for slater_det_int in dets_ab])
cis_a_overlap = vqe.compute_vqe_basis_state_overlaps(dets_ab, vqe.params) * t1signs

dets_ab = []
for det in detsa:
    cdet = np.zeros(2 * ncas, dtype=int)
    cdet[1::2] = det
    cdet[::2] = hfdet
    dets_ab.append("".join([str(w) for w in cdet]))
# det_arr = np.array([int(str(slater_det_int), 2) for slater_det_int in dets_ab])
cis_b_overlap = vqe.compute_vqe_basis_state_overlaps(dets_ab, vqe.params) * t1signs

np.testing.assert_allclose(cis_a_overlap, cis_b_overlap, atol=1e-12, rtol=0)
cis_a_overlap = cis_a_overlap.reshape(nocca, nvirta)
cis_b_overlap = cis_b_overlap.reshape(noccb, nvirtb)

dets_ab = []
for det in detsa:
    for det2 in detsa:
        cdet = np.zeros(2 * ncas, dtype=int)
        cdet[::2] = det
        cdet[1::2] = det2
        dets_ab.append("".join([str(w) for w in cdet]))
cid_ab_overlap = vqe.compute_vqe_basis_state_overlaps(dets_ab, vqe.params).reshape(nocca*nvirta, noccb*nvirtb)
cid_ab_overlap = np.einsum("ij,i,j->ij", cid_ab_overlap, t1signs, t1signs)
cid_ab_overlap = cid_ab_overlap.reshape(nocca, nvirta, noccb, nvirtb).transpose(0, 2, 1, 3)

dets_ab = []
for det in detsaa:
    cdet = np.zeros(2 * ncas, dtype=int)
    cdet[::2] = det
    cdet[1::2] = hfdet
    dets_ab.append("".join([str(w) for w in cdet]))
cid_aa_overlap = vqe.compute_vqe_basis_state_overlaps(dets_ab, vqe.params) * t2signs

dets_ab = []
for det in detsaa:
    cdet = np.zeros(2 * ncas, dtype=int)
    cdet[1::2] = det
    cdet[::2] = hfdet
    dets_ab.append("".join([str(w) for w in cdet]))
cid_bb_overlap = vqe.compute_vqe_basis_state_overlaps(dets_ab, vqe.params) * t2signs

np.testing.assert_allclose(cid_aa_overlap, cid_bb_overlap, atol=1e-12, rtol=0)

hf = np.zeros(2 * ncas, dtype=int)
hf[::2] = hfdet
hf[1::2] = hfdet
hf = "".join([str(w) for w in hf])
c0 = vqe.compute_vqe_basis_state_overlaps([hf], vqe.params)[0]
print("c0", c0)
c_ia, c_ijab = amplitudes_to_spinorb(c0, cis_a_overlap, cis_b_overlap, cid_aa_overlap, cid_ab_overlap, cid_bb_overlap)

occslice = slice(2 * mci.ncore, 2 * mci.ncore + nocca + noccb)
virtslice = slice(0, nvirta + nvirtb)
print("Starting TCCSD")
ret_vqe = tccsd(mci._scf, c_ia, c_ijab, occslice, virtslice)

ret_ci = tccsd_from_ci(mci)

print("TCC(CI)/TCC(VQE) diff = ", ret_vqe.e_tot - ret_ci.e_tot)

print(energy_vqe - (ret_vqe.e_cas + scfres.e_tot))
print(energy_vqe - ret_vqe.e_tot)


ret_vqe_int = tccsd_from_vqe(scfres, vqe)