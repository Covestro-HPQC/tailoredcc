# %%
from math import factorial

import numpy as np
from pyscf import fci, gto, scf
from pyscf.ci.cisd import tn_addrs_signs

from tailoredcc.amplitudes import (
    remove_index_restriction_doubles,
    remove_index_restriction_quadruples,
    remove_index_restriction_triples,
)

# %%
mol = gto.Mole()
mol.atom = [
    ["O", (0.0, 0.0, 0.0)],
    ["H", (0.0, -0.857, 0.587)],
    ["H", (0.0, 0.757, 0.687)],
]
mol.basis = "321g"
mol.build()
mol.verbose = 4
mf = scf.RHF(mol).run()
e, fcivec = fci.FCI(mf).kernel(verbose=5)

nmo = mol.nao
nocc = mol.nelectron // 2

t1addrs, t1signs = tn_addrs_signs(nmo, nocc, 1)
t2addrs, t2signs = tn_addrs_signs(nmo, nocc, 2)
t3addrs, t3signs = tn_addrs_signs(nmo, nocc, 3)
t4addrs, t4signs = tn_addrs_signs(nmo, nocc, 4)

cit_aaa = fcivec[t3addrs, 0] * t3signs
cit_bbb = fcivec[0, t3addrs] * t3signs
cit_aab = np.einsum("ij,i,j->ij", fcivec[t2addrs[:, None], t1addrs], t2signs, t1signs)
cit_abb = np.einsum("ij,i,j->ij", fcivec[t1addrs[:, None], t2addrs], t1signs, t2signs)

ciq_aaaa = fcivec[t4addrs, 0] * t4signs
ciq_bbbb = fcivec[0, t4addrs] * t4signs
ciq_aaab = np.einsum("ij,i,j->ij", fcivec[t3addrs[:, None], t1addrs], t3signs, t1signs)
ciq_aabb = np.einsum("ij,i,j->ij", fcivec[t2addrs[:, None], t2addrs], t2signs, t2signs)
ciq_abbb = np.einsum("ij,i,j->ij", fcivec[t1addrs[:, None], t3addrs], t1signs, t3signs)


# %%
def compute_nonredundant(nmo, nocc, exci):
    nvirt = nmo - nocc
    ret = 1
    for e in range(exci):
        ret *= (nocc - e) * (nvirt - e)
    ret //= (factorial(exci)) ** 2
    return ret


# nocc = 10
# nmo = 22
nvirt = nmo - nocc
assert compute_nonredundant(nmo, nocc, 1) == nocc * nvirt
assert compute_nonredundant(nmo, nocc, 2) == nocc * (nocc - 1) // 2 * nvirt * (nvirt - 1) // 2
assert (
    compute_nonredundant(nmo, nocc, 3)
    == nocc * (nocc - 1) * (nocc - 2) // 6 * nvirt * (nvirt - 1) * (nvirt - 2) // 6
)
assert compute_nonredundant(nmo, nocc, 4) == len(t4addrs)
# %%

print(t3addrs.size)
print(compute_nonredundant(nmo, nocc, 3))

print(cit_aaa.size)

print(cit_aab.size)
print(t2addrs.size * t1addrs.size)
print(compute_nonredundant(nmo, nocc, 2))

nvirt = nmo - nocc
nocca = noccb = nocc
nvirta = nvirtb = nvirt

cit_aab_full = np.zeros((nocca, nocca, nvirta, nvirta, noccb * nvirtb))
cit_abb_full = np.zeros((noccb, noccb, nvirtb, nvirtb, nocca * nvirta))
# print(cit_aab.shape)
for row in range(cit_aab.shape[1]):
    cit_aab_full[:, :, :, :, row] = remove_index_restriction_doubles(cit_aab[:, row], nocca, nvirta)

for row in range(cit_abb.shape[0]):
    cit_abb_full[:, :, :, :, row] = remove_index_restriction_doubles(cit_abb[row, :], noccb, nvirtb)

# cit_aab_full = cit_aab_full.reshape(nocca, nocca, nvirta, nvirta, noccb, nvirtb).transpose(0, 1, 4, 2, 3, 5)
# cit_abb_full = cit_abb_full.reshape(noccb, noccb, nvirtb, nvirtb, nocca, nvirta).transpose(4, 0, 1, 5, 2, 3)


print(ciq_aaab.shape)
assert ciq_aaab.shape == (compute_nonredundant(nmo, nocc, 3), compute_nonredundant(nmo, nocc, 1))

ciq_aaab_full = np.zeros((nocca, nocca, nocca, nvirta, nvirta, nvirta, noccb * nvirtb))
for row in range(ciq_aaab.shape[1]):
    ciq_aaab_full[..., row] = remove_index_restriction_triples(ciq_aaab[:, row], nocc, nvirt)

print(ciq_abbb.shape)
ciq_abbb_full = np.zeros((noccb, noccb, noccb, nvirtb, nvirtb, nvirtb, nocca * nvirta))
for row in range(ciq_abbb.shape[0]):
    ciq_abbb_full[..., row] = remove_index_restriction_triples(ciq_abbb[row, :], nocc, nvirt)

# %%

cid = np.arange(compute_nonredundant(nmo, nocc, 2))
print(cid.shape)

c_ijab_ref = remove_index_restriction_doubles(cid, nocc, nvirt)

import time

start = time.time()

nmo = 10
nocc = 5
nvirt = nmo - nocc

array = np.arange(compute_nonredundant(nmo, nocc, 3))
remove_index_restriction_triples(array, nocc, nvirt)

array = np.arange(compute_nonredundant(nmo, nocc, 4))
remove_index_restriction_quadruples(array, nocc, nvirt)

# ref = remove_index_restriction_doubles(array, nocc, nvirt)
# np.testing.assert_allclose(ref, ret, atol=1e-14, rtol=0)
# assert idx == array.size
# stop = time.time()
# print(nmo, nocc, stop - start)
# %%
