from pyscf import gto, scf, mcscf
import numpy as np
from pyscf.fci import cistring
from numpy import einsum

from tailoredcc.ccsd_equations import singles_residual, doubles_residual, ccsd_energy


def solve_tccsd(
    t1,
    t2,
    fock,
    g,
    o,
    v,
    e_ai,
    e_abij,
    occslice,
    virtslice,
    max_iter=100,
    stopping_eps=1.0e-8,
    diis_size=None,
    diis_start_cycle=4,
):
    t1slice = (virtslice, occslice)
    t2slice = (virtslice, virtslice, occslice, occslice)
    t1cas = t1[t1slice].copy()
    t2cas = t2[t2slice].copy()

    t1[t1slice] = 0.0
    t2[t2slice] = 0.0
    # initialize diis if diis_size is not None
    # else normal iterate
    if diis_size is not None:
        from tailoredcc.diis import DIIS

        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_dim = t1.size
        old_vec = np.hstack((t1.flatten(), t2.flatten()))

    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    old_energy = ccsd_energy(t1, t2, fock, g, o, v)
    print(f"\tInitial CCSD energy: {old_energy}")
    for idx in range(max_iter):
        t1[t1slice] = 0.0
        t2[t2slice] = 0.0

        singles_res = singles_residual(t1, t2, fock, g, o, v) + fock_e_ai * t1
        doubles_res = doubles_residual(t1, t2, fock, g, o, v) + fock_e_abij * t2

        singles_res[t1slice] = 0.0
        doubles_res[t2slice] = 0.0

        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij

        assert np.all(new_singles[t1slice] == 0.0)
        assert np.all(new_doubles[t2slice] == 0.0)

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack((new_singles.flatten(), new_doubles.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate, error_vec)
            new_singles = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
            new_doubles = new_vectorized_iterate[t1_dim:].reshape(t2.shape)
            old_vec = new_vectorized_iterate

        new_singles[t1slice] = t1cas
        new_doubles[t2slice] = t2cas

        current_energy = ccsd_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps:
            print(f"\tConverged in iteration {idx}")
            return new_singles, new_doubles
        else:
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy
            print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx, old_energy, delta_e))
    else:
        print("Did not converge")
        return new_singles, new_doubles


def ccsd_energy_correlation(t1, t2, f, g, o, v):
    """
    < 0 | e(-T) H e(T) | 0>:
    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    energy = 0.0
    # 	  1.0000 f(i,i)
    # energy = 1.0 * einsum('ii', f[o, o])

    # 	  1.0000 f(i,a)*t1(a,i)
    energy += 1.0 * einsum('ia,ai', f[o, v], t1)

    # #	 -0.5000 <j,i||j,i>
    # energy += -0.5 * einsum('jiji', g[o, o, o, o])

    # 	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy += 0.25 * einsum("jiab,abji", g[o, o, v, v], t2)

    # 	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += -0.5 * einsum(
        "jiab,ai,bj", g[o, o, v, v], t1, t1, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    return energy


mol = gto.Mole()
mol.build(
    verbose=4,
    # atom="N, 0., 0., 0. ; N,  0., 0., 1.4",
    # atom="H, 0., 0., 0. ; H,  0., 0., 1.0",
    # atom="He, 0., 0., 0. ; He,  0., 0., 1.0",
    atom="Li, 0., 0., 0. ; Li,  0., 0., 1.0",
    # basis="minao",
    # basis="sto-3g",
    # basis="3-21g",
    # basis="6-31g",
    # basis="cc-pvdz",
    # symmetry = True,
)

# k = 2
# alpha = 4
# # alpha = 1
# # alpha = 1.5
# mol = gto.M(
#     atom=f"""
#     H 0 0 0
#     H 0 {k} 0
#     H {alpha} 0 0
#     H {alpha} {k} 0
#     """, unit="Bohr", basis="DZP", verbose=4, spin=0, symmetry=True
# )

m = scf.RHF(mol)
m.kernel()

ncas = mol.nao_nr()
nelec = mol.nelec

# ncas = 2
# nelec = 2

ncas = 4
nelec = 4

print(f"CAS({nelec}, {ncas})")
# mc = mcscf.CASCI(m, ncas, nelec)
mc = mcscf.CASSCF(m, ncas, nelec)
mc.conv_tol = 1e-10
mc.conv_tol_grad = 1e-7
mc.kernel()

# from clusterdec import dump_clusterdec
# dump_clusterdec(mc)

ncas = mc.ncas
nocca, noccb = mc.nelecas
assert nocca == noccb
nvirta = mc.ncas - nocca
nvirtb = mc.ncas - noccb
assert nvirta == nvirtb
size_doubles = nocca**2 * nvirta**2

from pyscf.ci.cisd import tn_addrs_signs

t1addrs, t1signs = tn_addrs_signs(ncas, nocca, 1)
t2addrs, t2signs = tn_addrs_signs(ncas, nocca, 2)


fcivec = mc.ci
# CIS includes two types of amplitudes: alpha -> alpha and beta -> beta
cis_a = fcivec[t1addrs, 0] * t1signs
cis_b = fcivec[0, t1addrs] * t1signs


# CID has:
#    alpha,alpha -> alpha,alpha
#    alpha,beta  -> alpha,beta
#    beta ,beta  -> beta ,beta
# For alpha,alpha -> alpha,alpha excitations, the redundant coefficients are
# excluded. The number of coefficients is nocc*(nocc-1)//2 * nvir*(nvir-1)//2,
# which corresponds to C2_{ijab}, i > j and a > b.
cid_aa = fcivec[t2addrs, 0] * t2signs
cid_bb = fcivec[0, t2addrs] * t2signs

cid_aa_full = np.zeros((nocca, nocca, nvirta, nvirta))
cid_bb_full = np.zeros((noccb, noccb, nvirtb, nvirtb))

if len(t2addrs):
    # TODO: cleanup
    detstrings = [bin(ds) for ds in cistring.addrs2str(ncas, nocca, t2addrs.ravel())]
    detstrings_ref = []
    detidx = 0
    for j in range(nocca):
        for i in range(j):
            for b in range(nvirta):
                for a in range(b):
                    coeff = cid_aa[detidx]
                    cid_aa_full[i, j, a, b] = coeff
                    cid_aa_full[j, i, a, b] = -1.0 * coeff
                    cid_aa_full[i, j, b, a] = -1.0 * coeff
                    cid_aa_full[j, i, b, a] = coeff
                    coeff = cid_bb[detidx]
                    cid_bb_full[i, j, a, b] = coeff
                    cid_bb_full[j, i, a, b] = -1.0 * coeff
                    cid_bb_full[i, j, b, a] = -1.0 * coeff
                    cid_bb_full[j, i, b, a] = coeff
                    # NOTE: following part is just for debugging purposes
                    string = np.zeros(ncas, dtype=int)
                    string[:nocca] = 1
                    string[j] = 0
                    string[i] = 0
                    string[nocca + b] = 1
                    string[nocca + a] = 1
                    without_zeros = np.trim_zeros(string[::-1], "f")
                    cstr = "".join([str(x) for x in without_zeros])
                    cstr = "0b" + cstr
                    detstrings_ref.append(cstr)
                    detidx += 1
    assert len(detstrings_ref) == len(detstrings)
    for ret, ref in zip(detstrings_ref, detstrings):
        assert ret == ref

if len(t1addrs):
    cid_ab = np.einsum("ij,i,j->ij", fcivec[t1addrs[:, None], t1addrs], t1signs, t1signs)
    assert cid_ab.size == size_doubles
    cid_ab = cid_ab.reshape(nocca, nvirta, noccb, nvirtb)
    # order is now occa, occb, virta, virtb
    cid_ab = cid_ab.transpose(0, 2, 1, 3)
else:
    cid_ab = np.zeros((nocca, noccb, nvirta, nvirtb))


from pyscf.cc.addons import spatial2spin

c0 = fcivec[0, 0]
print(f"c0 = {c0:.8f}")
print(f"|c0|^2 = {c0**2:.8f}")
cis_a = cis_a.reshape(nocca, nvirta)
cis_b = cis_b.reshape(noccb, nvirtb)

c_ia = spatial2spin((cis_a, cis_b)) / c0
c_ijab = spatial2spin((cid_aa_full, cid_ab, cid_bb_full)) / c0
np.testing.assert_allclose(c_ijab, -1.0 * c_ijab.transpose(0, 1, 3, 2))
np.testing.assert_allclose(c_ijab, -1.0 * c_ijab.transpose(1, 0, 2, 3))
np.testing.assert_allclose(c_ijab, c_ijab.transpose(1, 0, 3, 2))

c1_norm = np.vdot(c_ia, c_ia)
# 1/4 for comparing with ClusterDec
c2_norm = 0.25 * np.vdot(c_ijab, c_ijab)
print(c0, c1_norm, c2_norm)

# # NOTE: of uses alpha_1, beta_1, alpha_2, beta_2, ... MO ordering (like interleaved qubit ordering),
# see https://github.com/quantumlib/OpenFermion/blob/master/src/openfermion/chem/molecular_data.py#L222

t1 = c_ia.copy()
t2 = c_ijab - np.einsum("ia,jb->ijab", t1, t1) + np.einsum("ib,ja->ijab", t1, t1)
np.testing.assert_allclose(t2, -1.0 * t2.transpose(0, 1, 3, 2))
np.testing.assert_allclose(t2, t2.transpose(1, 0, 3, 2))

assert isinstance(mc.ncore, int)
occslice = slice(2 * mc.ncore, 2 * mc.ncore + nocca + noccb)
virtslice = slice(0, nvirta + nvirtb)
t1slice = (virtslice, occslice)
t2slice = (virtslice, virtslice, occslice, occslice)


from openfermionpyscf._run_pyscf import compute_integrals
from openfermion.chem.molecular_data import spinorb_from_spatial

print('maxdiff', np.max(np.abs(m.mo_coeff- mc.mo_coeff)))
print('energies', m.mo_energy, mc.mo_energy)
m.mo_coeff = mc.mo_coeff
m.mo_energy = mc.mo_energy

oei, eri_of_spatial = compute_integrals(mol, m)
soei, eri_of = spinorb_from_spatial(oei, eri_of_spatial)
eri_of_asymm = eri_of - eri_of.transpose(0, 1, 3, 2)
nocc = sum(mol.nelec)
nvirt = 2 * mol.nao_nr() - nocc
# to Physicists' notation <12|1'2'> (OpenFermion stores <12|2'1'>)
eri_phys_asymm = eri_of_asymm.transpose(0, 1, 3, 2)

n = np.newaxis
o = slice(None, nocc)
v = slice(nocc, None)

# (canonical) Fock operator
fock = soei + np.einsum("piqi->pq", eri_phys_asymm[:, o, :, o])
hf_energy = 0.5 * np.einsum("ii", (fock + soei)[o, o])
# np.testing.assert_allclose(hf_energy + mol.energy_nuc(), m.e_tot, atol=1e-12, rtol=0)

# orbital energy differences
eps = np.kron(m.mo_energy, np.ones(2))
e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
e_ai = 1 / (-eps[v, n] + eps[n, o])

# full MO space T amplitudes
t1_mo = np.zeros((nocc, nvirt))
t2_mo = np.zeros((nocc, nocc, nvirt, nvirt))
t1_mo[occslice, virtslice] = t1
t2_mo[occslice, occslice, virtslice, virtslice] = t2

np.testing.assert_allclose(t2_mo, -1.0 * t2_mo.transpose(0, 1, 3, 2))
np.testing.assert_allclose(t2_mo, t2_mo.transpose(1, 0, 3, 2))

# t_ai and t_{abij} in pdaggerq
e_corr = ccsd_energy(t1_mo.T, t2_mo.transpose(2, 3, 0, 1), fock, eri_phys_asymm, o, v)
# e_corr = ccsd_energy(t1_mo.T, t2_mo.transpose(2, 3, 0, 1), fock, eri_phys_asymm, o, v)
e_cas = mc.e_tot - mol.energy_nuc()
print(f"CCSD electronic energy from CI amplitudes {e_corr:>12}")
print(f"CI energy (excluding nuclear repulsion)   {e_cas:>12}")
np.testing.assert_allclose(e_cas, e_corr, atol=1e-10)


# run SR-CCSD calculation
from pyscf.cc import CCSD

cc = CCSD(m)
cc.conv_tol_normt = 1e-8
cc.conv_tol = 1e-10
cc.kernel()
t1_ref = spatial2spin(cc.t1)
t2_ref = spatial2spin(cc.t2)

if np.allclose(cc.e_tot, mc.e_tot, atol=1e-10):
    np.testing.assert_allclose(t1_ref, t1, atol=1e-10)
    np.testing.assert_allclose(t2_ref, t2, atol=1e-10)
    np.testing.assert_allclose(mc.e_tot, m.e_tot + e_corr, atol=1e-10)
else:
    print("CCSD != CI in the present case...")


# run TCCSD
t1f, t2f = solve_tccsd(
    t1_mo.T,
    t2_mo.transpose(2, 3, 0, 1),
    fock,
    eri_phys_asymm,
    o,
    v,
    e_ai,
    e_abij,
    occslice,
    virtslice,
    diis_size=8,
)
# test that the T_CAS amplitudes are still intact
np.testing.assert_allclose(t1.T, t1f[t1slice], atol=1e-14)
np.testing.assert_allclose(t2.transpose(2, 3, 0, 1), t2f[t2slice], atol=1e-14)

t1ext = t1f.copy()
t2ext = t2f.copy()
t1ext[t1slice] = 0.0
t2ext[t2slice] = 0.0
e_ext = ccsd_energy_correlation(t1ext, t2ext, fock, eri_phys_asymm, o, v)
e_tcc = ccsd_energy_correlation(t1f, t2f, fock, eri_phys_asymm, o, v)
np.testing.assert_allclose(e_tcc, e_ext + e_corr - hf_energy, atol=1e-12, rtol=0)
print("Ecas", e_corr)
print("Eext", e_ext)
print("Ecas + Eext", e_ext + e_corr)
print("E(TCCSD)", e_ext + e_corr + mol.energy_nuc())
