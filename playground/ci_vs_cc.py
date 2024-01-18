import numpy as np
from pyscf import ao2mo, cc, ci, fci, gto, mcscf, scf
from pyscf.cc.addons import spatial2spin, spin2spatial
from pyscf.ci.cisd import tn_addrs_signs


def remove_index_restriction_doubles(cid_aa, nocc: int, nvirt: int):
    assert cid_aa.ndim == 1
    assert cid_aa.size == nocc * (nocc - 1) // 2 * nvirt * (nvirt - 1) // 2
    cid_aa_full = np.zeros((nocc, nocc, nvirt, nvirt), dtype=cid_aa.dtype)
    idx = 0
    for j in range(nocc):
        for i in range(j):
            for b in range(nvirt):
                for a in range(b):
                    coeff = cid_aa[idx]
                    cid_aa_full[i, j, a, b] = coeff
                    cid_aa_full[j, i, a, b] = -1.0 * coeff
                    cid_aa_full[i, j, b, a] = -1.0 * coeff
                    cid_aa_full[j, i, b, a] = coeff
                    idx += 1
    return cid_aa_full


def ci_to_cc(civec, norb, nelec):
    nocca, noccb = nelec
    nvirta = norb - nocca
    nvirtb = norb - noccb

    t1addrs, t1signs = tn_addrs_signs(norb, nelec[0], 1)
    t2addrs, t2signs = tn_addrs_signs(norb, nelec[0], 2)

    c0 = civec[0, 0]

    c_a = civec[t1addrs, 0] * t1signs
    c_b = civec[0, t1addrs] * t1signs

    c_a = c_a.reshape(nocca, nvirta)
    c_b = c_b.reshape(noccb, nvirtb)

    c_aa = civec[t2addrs, 0] * t2signs
    c_bb = civec[0, t2addrs] * t2signs

    c_aa = remove_index_restriction_doubles(c_aa, nocca, nvirta)
    c_bb = remove_index_restriction_doubles(c_bb, noccb, nvirtb)

    c_ab = np.einsum("ij,i,j->ij", civec[t1addrs[:, None], t1addrs], t1signs, t1signs)
    c_ab = c_ab.reshape(nocca, nvirta, noccb, nvirtb)
    # order is now occa, occb, virta, virtb
    c_ab = c_ab.transpose(0, 2, 1, 3)

    c1 = spatial2spin((c_a, c_b)) / c0
    c2 = spatial2spin((c_aa, c_ab, c_bb)) / c0

    t1 = c1.copy()
    t2 = c2 - np.einsum("ia,jb->ijab", t1, t1) + np.einsum("ib,ja->ijab", t1, t1)
    return t1, t2, c1, c2


mol = gto.M(
    # atom="""
    # O 0 0 0
    # H 1.2 0 0
    # H 0 2.1 0
    # """,
    atom="""
    N 0 0 0
    N 0 0 2.0
    """,
    # atom="""
    # Li 0 0 0
    # H 1 0 0
    # """,
    # basis="6-31g",
    # basis="3-21g",
    basis="sto-3g",
    verbose=4,
)

scfres = scf.RHF(mol)
scfres.conv_tol = 1e-14
scfres.kernel()

mccsd = cc.CCSD(scfres).run()

norb = mol.nao_nr()
nelec = mol.nelec
nocca, noccb = nelec

orbspin = np.zeros(norb * 2, dtype=int)
orbspin[1::2] = 1

# fcires = fci.FCI(scfres, singlet=True)
fcires = mcscf.CASCI(scfres, ncas=norb, nelecas=nelec)
fcires.verbose = 4
fcires.fcisolver.verbose = 5
# fcires.max_cycle = 1
# fcires.conv_tol = 1e-12
# fcires.fcisolver.max_cycle = 30
fcires.fcisolver.conv_tol = 1e-12
fcires.kernel()
civec = fcires.ci

t1, t2, c1, c2 = ci_to_cc(civec, norb, nelec)
t1x = spin2spatial(t1, orbspin)
t2x = spin2spatial(t2, orbspin)

ccsd = cc.UCCSD(scfres)
e_cc = ccsd.energy(t1x, t2x) + scfres.e_tot
print(e_cc, fcires.e_tot, e_cc - fcires.e_tot, fcires.converged)


from openfermion.chem.molecular_data import spinorb_from_spatial

mo = scfres.mo_coeff
h1e = mo.conj().T @ scfres.get_hcore() @ mo
eri = ao2mo.kernel(scfres._eri, mo)

soei, stei = spinorb_from_spatial(h1e, ao2mo.restore(1, eri, norb).transpose(0, 2, 3, 1))
astei = np.einsum("ijkl", stei) - np.einsum("ijlk", stei)
gtei = astei.transpose(0, 1, 3, 2)

# eps = np.kron(molecule.orbital_energies, np.ones(2))
n = np.newaxis
nocc = nocca
o = slice(None, 2 * nocc)
v = slice(2 * nocc, None)

e_corr = 0.5 * np.einsum("ijab,ia,jb->", gtei[o, o, v, v], t1, t1) + 0.25 * np.einsum(
    "ijab,ijab->", gtei[o, o, v, v], t2
)
print("ALARM", e_corr + scfres.e_tot - fcires.e_tot)

e_corr = 0.25 * np.einsum("ijab,ijab->", gtei[o, o, v, v], c2)
print("ALARM", e_corr + scfres.e_tot - fcires.e_tot)

# exit(0)

cisd = ci.CISD(scfres)
# cisd.max_cycle = 5
cisd.verbose = 4
cisd.conv_tol = 1e-14
cisd.kernel()
civec_cisd = cisd.to_fcivec(cisd.ci)
print("Diff between CCSD and CISD energy", cisd.e_tot - mccsd.e_tot)

# del t1, t2, t1x, t2x
t1cisd, t2cisd, c1cisd, c2cisd = ci_to_cc(civec_cisd, norb, nelec)
t1x = spin2spatial(t1cisd, orbspin)
t2x = spin2spatial(t2cisd, orbspin)

ccsd = cc.UCCSD(scfres)
e_cc = ccsd.energy(t1x, t2x) + scfres.e_tot
print(e_cc, cisd.e_tot, e_cc - cisd.e_tot)

e_corr = 0.25 * np.einsum("ijab,ijab->", gtei[o, o, v, v], c2cisd)
e_corr_cc = 0.5 * np.einsum("ijab,ia,jb->", gtei[o, o, v, v], t1cisd, t1cisd) + 0.25 * np.einsum(
    "ijab,ijab->", gtei[o, o, v, v], t2cisd
)
print("ALARM", e_corr, e_corr_cc, e_corr - e_corr_cc)

exit(0)

ci_level = 2
alpha_occs = fci.cistring.gen_occslst(range(norb), nocca)
alpha_excitations = (alpha_occs >= nocca).sum(axis=1)
beta_occs = fci.cistring.gen_occslst(range(norb), noccb)
beta_excitations = (beta_occs >= noccb).sum(axis=1)

a_idx, b_idx = np.array(
    [
        [ia, ib]
        for ia, a in enumerate(alpha_excitations)
        for ib, b in enumerate(beta_excitations)
        if a + b > ci_level
    ]
).T

# a_idx, b_idx = np.array([[ia, ib] for ia, a in enumerate(alpha_excitations) for ib, b in enumerate(beta_excitations) if a+b == 2 ]).T
# a_idx = a_idx[2:]
# b_idx = b_idx[2:]

# a_idx, b_idx = np.array([[ia, ib] for ia, a in enumerate(alpha_excitations) for ib, b in enumerate(beta_excitations) if (a+b > 2 or a+b == 1) ]).T  # CID
# a_idx = a_idx[1:]
# b_idx = b_idx[1:]


fcires_tr = fci.FCI(scfres, singlet=True)
fcires_tr.verbose = 4
fcires_tr.max_cycle = 2
fcires_tr.conv_tol = 1e-14

contract = fcires_tr.contract_2e
make_hdiag = fcires_tr.make_hdiag
na = fci.cistring.num_strings(norb, nelec[0])
nb = na


def contract_truncated(erix, fcivecx, norb, nelec, link_index):
    hc = contract(erix, fcivecx, norb, nelec, link_index)
    hc[a_idx, b_idx] = 0.0
    return hc


def make_hdiag_truncated(h1e, eri, norb, nelec, compress=False):
    hdiag = make_hdiag(h1e, eri, norb, nelec, compress)
    hdiag = hdiag.reshape(na, nb)
    hdiag[a_idx, b_idx] = 0.0
    return hdiag.ravel()


fcires_tr.contract_2e = contract_truncated
fcires_tr.make_hdiag = make_hdiag_truncated
fcires_tr.kernel()

t1cisd, t2cisd, c1cisd, c2cisd = ci_to_cc(fcires_tr.ci, norb, nelec)
t1x = spin2spatial(t1cisd, orbspin)
t2x = spin2spatial(t2cisd, orbspin)

ccsd = cc.UCCSD(scfres)
e_cc = ccsd.energy(t1x, t2x) + scfres.e_tot
print(e_cc, fcires_tr.e_tot, e_cc - fcires_tr.e_tot, fcires_tr.converged)

e_corr = 0.25 * np.einsum("ijab,ijab->", gtei[o, o, v, v], c2cisd)
e_corr_cc = 0.5 * np.einsum("ijab,ia,jb->", gtei[o, o, v, v], t1cisd, t1cisd) + 0.25 * np.einsum(
    "ijab,ijab->", gtei[o, o, v, v], t2cisd
)
print("ASDF", e_corr, e_corr_cc, e_corr - e_corr_cc)


# exit(0)


import covvqetools as cov
from covvqetools.instant_vqes import QNPVQE

# depth = 50
depth = 10
# maxiter_vqe = 200
maxiter_vqe = 40
nocc = 0

vqe = QNPVQE(
    depth=depth,
    nact=norb,
    nalpha=nocca,
    nbeta=noccb,
    # NOTE: trick to get nocc correct even though we don't have nuclear charges
    nocc=nocc,
    nchar=-sum(mol.atom_charges()),
    ###
    measurement_method=cov.CASBOX,
    core_energy=mol.energy_nuc(),
    one_body_integrals=h1e,
    two_body_integrals=ao2mo.restore(1, eri, norb).transpose(0, 2, 3, 1),
)
vqe.params += np.random.randn(*vqe.params.shape) * 1e-2

opt = cov.LBFGSB(atol=None, gtol=1e-10, ftol=None)


def callback(epoch, _):
    print("vqe", epoch, vqe.vqe_energy(), np.max(np.abs(vqe.vqe_jacobian())))


vqe.vqe_optimize(opt=opt, maxiter=maxiter_vqe, callback=callback)
energy_vqe = float(vqe.vqe_energy(vqe.params))

from tailoredcc.amplitudes import (
    amplitudes_to_spinorb,
    extract_vqe_singles_doubles_amplitudes,
)
from tailoredcc.tailoredcc import tccsd_from_vqe

tcc = tccsd_from_vqe(scfres, vqe, backend="oe")
ci_amps = extract_vqe_singles_doubles_amplitudes(vqe)
c_ia, c_ijab = amplitudes_to_spinorb(ci_amps)
c_ia = c_ia.real
c_ijab = c_ijab.real

e_corr = 0.25 * np.einsum("ijab,ijab->", gtei[o, o, v, v], c_ijab)
print("vqe energy - cas t1/t2 energy", tcc.e_cas + scfres.e_tot - energy_vqe)
print("vqe energy -    cas c2 energy", e_corr + scfres.e_tot - energy_vqe)
print("ASDF", e_corr, tcc.e_cas, e_corr - tcc.e_cas)
