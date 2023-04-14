import covvqetools as cov
import numpy as np
import pennylane as qml
from pyscf import gto, mcscf, scf

from tailoredcc.amplitudes import detstrings_doubles, detstrings_singles


def assert_allclose_signfix(actual, desired, atol=0, **kwargs):
    """
    Call assert_allclose, but beforehand normalize the sign
    of the involved arrays (i.e. the two arrays may differ
    up to a sign factor of -1)
    """
    actual, desired = normalize_sign(actual, desired, atol=atol)
    np.testing.assert_allclose(actual, desired, atol=atol, **kwargs)


def normalize_sign(*items, atol=0):
    """
    Normalise the sign of a list of numpy arrays
    """

    def sign(item):
        flat = np.ravel(item)
        flat = flat[np.abs(flat) > atol]
        if flat.size == 0:
            return 1
        else:
            return np.sign(flat[0])

    desired_sign = sign(items[0])
    return tuple(desired_sign / sign(item) * item for item in items)


mol = gto.Mole()
mol.build(
    verbose=4,
    atom="N, 0., 0., 0. ; N,  0., 0., 1.4",
    # atom="H, 0., 0., 0. ; H,  0., 0., 1.0",
    # atom="He, 0., 0., 0. ; He,  0., 0., 1.0",
    # atom="Li, 0., 0., 0. ; Li,  0., 0., 1.0",
    # basis="minao",
    # basis="sto-3g",
    basis="3-21g",
    # basis="6-31g",
    # basis="cc-pvdz",
    # symmetry = True,
)
m = scf.RHF(mol)
m.kernel()

ncas = mol.nao_nr()
nelec = sum(mol.nelec)

# ncas = 2
# nelec = 2

# ncas = 4
# nelec = 2

# ncas = 8
# nelec = 6

# ncas = 10
# nelec = 6

# ncas = 6
# nelec = 6

ncas = 8
nelec = 6

nocca = nelec // 2
noccb = nocca
nvirta = ncas - nocca
nvirtb = ncas - noccb

print(f"CAS({nelec}, {ncas})")
mc = mcscf.CASCI(m, ncas, nelec)
mc.kernel()


state = cov.pyscf.extract_state_dict(mc.fcisolver, mc.ci, ncas, nocca, noccb, amplitude_cutoff=1e-8)
print("extraction done")

from pyscf.ci.cisd import tn_addrs_signs

t1addrs, t1signs = tn_addrs_signs(ncas, nocca, 1)
t2addrs, t2signs = tn_addrs_signs(ncas, nocca, 2)

from pathlib import Path

cifile = f"fcivec_{nelec}_{ncas}.npy"
if Path(cifile).is_file():
    print("Loading CI vector from disk")
    fcivec = np.load(cifile)
else:
    fcivec = mc.ci
    np.save(cifile, fcivec)

# CIS includes two types of amplitudes: alpha -> alpha and beta -> beta
cis_a = t1signs * fcivec[t1addrs, 0]
cis_b = t1signs * fcivec[0, t1addrs]
cis_a = cis_a.reshape(nocca, nvirta)
cis_b = cis_b.reshape(noccb, nvirtb)

cid_aa = fcivec[t2addrs, 0] * t2signs
cid_bb = fcivec[0, t2addrs] * t2signs

# alpha/beta -> alpha/beta
cid_ab = np.zeros((nocca, noccb, nvirta, nvirtb))
if len(t1addrs):
    cid_ab = np.einsum("ij,i,j->ij", fcivec[t1addrs[:, None], t1addrs], t1signs, t1signs)
    cid_ab = cid_ab.reshape(nocca, nvirta, noccb, nvirtb)
    # order is now occa, occb, virta, virtb
    cid_ab = cid_ab.transpose(0, 2, 1, 3)


dev = qml.device("lightning.qubit", wires=2 * ncas)


@qml.qnode(dev)
def ansatz(wires):
    cov.Superposition(state, wires=wires)
    return qml.state()


fname = f"statevec_{nelec}_{ncas}.npy"
if Path(fname).is_file():
    print("loading state vector from disk")
    statevec = np.load(fname)
else:
    wires = range(2 * ncas)
    statevec = ansatz(wires)
    np.save(fname, statevec)


def compute_overlap_with_dets(state, dets):
    return np.array([state[det] for det in dets]).real


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
det_arr = np.array([int(str(slater_det_int), 2) for slater_det_int in dets_ab])
cis_a_overlap = compute_overlap_with_dets(statevec, det_arr) * t1signs
cis_a_overlap = cis_a_overlap.reshape(nocca, nvirta)

cis_b_overlap = cis_a_overlap


dets_ab = []
for det in detsa:
    for det2 in detsa:
        cdet = np.zeros(2 * ncas, dtype=int)
        cdet[::2] = det
        cdet[1::2] = det2
        dets_ab.append("".join([str(w) for w in cdet]))
det_arr = np.array([int(str(slater_det_int), 2) for slater_det_int in dets_ab])
cid_ab_overlap = compute_overlap_with_dets(statevec, det_arr).reshape(
    nocca * nvirta, noccb * nvirtb
)
cid_ab_overlap = np.einsum("ij,i,j->ij", cid_ab_overlap, t1signs, t1signs)
cid_ab_overlap = cid_ab_overlap.reshape(nocca, nvirta, noccb, nvirtb).transpose(0, 2, 1, 3)

dets_ab = []
for det in detsaa:
    cdet = np.zeros(2 * ncas, dtype=int)
    cdet[::2] = det
    cdet[1::2] = hfdet
    dets_ab.append("".join([str(w) for w in cdet]))
det_arr = np.array([int(str(slater_det_int), 2) for slater_det_int in dets_ab])
cid_aa_overlap = compute_overlap_with_dets(statevec, det_arr) * t2signs

assert_allclose_signfix(cid_aa, cid_aa_overlap, atol=1e-5)
assert_allclose_signfix(cis_a, cis_a_overlap, atol=1e-5)
assert_allclose_signfix(cis_b, cis_b_overlap, atol=1e-5)
assert_allclose_signfix(cid_ab_overlap, cid_ab, atol=1e-4)
