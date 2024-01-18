# import covvqetools as cov
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import openfermion as of
import seaborn as sns
from covvqetools.instant_vqes import QNPVQE
from covvqetools.pyscf import extract_state_dict
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.ci.cisd import tn_addrs_signs
from scipy.sparse.linalg import expm

from tailoredcc.amplitudes import (
    ci_to_cluster_amplitudes,
    extract_ci_singles_doubles_amplitudes,
    extract_vqe_singles_doubles_amplitudes,
    remove_index_restriction_doubles,
)


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


sns.set_theme(context="talk", style="ticks", font_scale=1.2)

mol = gto.M(
    # atom="/fs/home/cvsik/Projects/stack/covvqetools/examples/p-benzyne.xyz",
    # atom="/fs/home/cvsik/Projects/stack/covvqetools/examples/n2_real.xyz",
    atom="/fs/home/cvsik/Projects/stack/covvqetools/examples/h2o_real.xyz",
    # basis="sto-3g",
    basis="6-31g",
)
scfres = scf.RHF(mol)
scfres.kernel()

# ncas = 7
# nelecas = 10

ncas = 4
nelecas = 2

mc = mcscf.CASCI(scfres, ncas=ncas, nelecas=nelecas)
mc.fcisolver.conv_tol = 1e-14
mc.kernel()
print("HF energy", scfres.e_tot, mc.e_tot - scfres.e_tot)

h1, ecore = mc.get_h1eff()
h2 = ao2mo.restore(1, mc.ao2mo(), ncas)
two_body_integrals = h2.transpose(0, 2, 3, 1)

depth = 12
vqe = QNPVQE(
    nact=ncas,
    nalpha=mc.nelecas[0],
    nbeta=mc.nelecas[1],
    depth=depth,
    one_body_integrals=h1,
    two_body_integrals=two_body_integrals,
    core_energy=ecore,
    # measurement_method=cov.CASBOX,
)

sparse_ham = of.get_sparse_operator(vqe.hamiltonian)

hfvec = np.zeros(sparse_ham.shape[0], dtype=np.complex128)
all0 = np.zeros(sparse_ham.shape[0], dtype=np.complex128)
all0[...] = 1.0 + 0j

hfstr = nelecas * "1" + (2 * ncas - nelecas) * "0"
hfidx = int(hfstr, 2)
hfvec[hfidx] = 1.0 + 0j
e_hf = (hfvec.conj().T @ sparse_ham @ hfvec).real
print("hfdiff", scfres.e_tot - e_hf)


def cc_operator_hilbert(c_ia, c_ijab, nocca, noccb, nvirta, nvirtb):
    nocc = nocca + noccb
    nvirt = nvirta + nvirtb

    t1cas, t2cas = ci_to_cluster_amplitudes(c_ia, c_ijab)

    _, s = tn_addrs_signs(nocc + nvirt, nocc, 1)

    T1 = of.FermionOperator()
    T2 = of.FermionOperator()
    for i in range(nocc):
        for mo_a, a in enumerate(range(nvirt), nocc):
            sign_t1 = s[nocc * i + a]
            sign_t1 = 1
            T1 += of.FermionOperator(f"{mo_a}^ {i}", sign_t1 * t1cas[i, a])
            # print(i, a, sign_t1)
            for j in range(nocc):
                for mo_b, b in enumerate(range(nvirt), nocc):
                    T2 += 0.25 * of.FermionOperator(f"{mo_a}^ {mo_b}^ {j} {i}", t2cas[i, j, a, b])

    T = T1 + T2
    T_sparse = of.get_sparse_operator(T, n_qubits=(nocc + nvirt))
    T_array = T_sparse  # .toarray()
    # expT = expm_dense(T_array)
    # expTm = expm_dense(-T_array)
    expT = expm(T_array)
    # expTm = expm(-T_array)
    return expT


def ci_operator_hilbert(c_ia, c_ijab, nocca, noccb, nvirta, nvirtb):
    nocc = nocca + noccb
    nvirt = nvirta + nvirtb

    C1 = of.FermionOperator()
    C2 = of.FermionOperator()

    for i in range(nocc):
        for mo_a, a in enumerate(range(nvirt), nocc):
            sign_maxi = (-1) ** i * (-1) ** (nocc - 1)
            C1 += of.FermionOperator(f"{mo_a}^ {i}", -sign_maxi * c_ia[i, a])
            for j in range(nocc):
                for mo_b, b in enumerate(range(nvirt), nocc):
                    sign_maxi_doubles = 1
                    print(i, j, a, b, sign_maxi_doubles)
                    C2 += of.FermionOperator(
                        f"{mo_a}^ {mo_b}^ {j} {i}", 0.25 * c_ijab[i, j, a, b] * sign_maxi_doubles
                    )
    # C = 1 + C1 + C2
    # C = 1 + C2
    C = 1 + C1
    C_sparse = of.get_sparse_operator(C, n_qubits=(nocc + nvirt))
    # np.savetxt("C_sparse.txt", C_sparse.toarray().real)
    return C_sparse


def casci_operator_hilbert(state):
    C = of.FermionOperator()
    for det, coeff in state.items():
        occs = np.where(np.array([*det]) == "1")[0]
        # ops_a = "".join([f"{idx}^ " for idx in occs if idx % 2 == 1])
        # ops_b = "".join([f"{idx}^ " for idx in occs if idx % 2 == 0])
        # ops = ops_b + ops_a
        ops = "".join([f"{idx}^ " for idx in occs])
        C += of.FermionOperator(ops, coeff)

    C_sparse = of.get_sparse_operator(C)
    return C_sparse


nocca, noccb = mc.nelecas
assert nocca == noccb
nvirta = mc.ncas - nocca
nvirtb = mc.ncas - noccb
assert nvirta == nvirtb
assert isinstance(mc.ncore, int)
ncore = mc.ncore
ncas = mc.ncas

from tailoredcc import tccsd_from_ci

# tcc_ret = tccsd_from_ci(mc, backend="oe")
# print("CI energy error before TCC", tcc_ret.e_cas + scfres.e_tot - mc.e_tot)


c_ia, c_ijab = extract_ci_singles_doubles_amplitudes(mc)
expT = cc_operator_hilbert(c_ia, c_ijab, nocca, noccb, nvirta, nvirtb)
e_cc = (hfvec.conj().T @ sparse_ham @ expT @ hfvec).real
ortho = hfvec.conj().T @ expT @ hfvec
print("ortho", ortho)
print("e_cc - e_ci", e_cc - mc.e_tot)


C = ci_operator_hilbert(c_ia, c_ijab, nocca, noccb, nvirta, nvirtb)
ci_wfn = C @ hfvec
# ci_wfn = ci_wfn / (ci_wfn.dot(ci_wfn))
e_ci = (ci_wfn.conj().T @ sparse_ham @ ci_wfn).real / (ci_wfn.conj().T @ ci_wfn)
print("e_cc - e_ci", e_ci - mc.e_tot)

state_dict = extract_state_dict(
    mc.fcisolver, mc.ci, nact=ncas, nalpha=mc.nelecas[0], nbeta=mc.nelecas[1], amplitude_cutoff=0
)

C_true = casci_operator_hilbert(state_dict)

from collections import OrderedDict

sorted_state_dict = OrderedDict(
    sorted(state_dict.items(), key=lambda item: abs(item[1]), reverse=True)
)

coeffs = np.array(list(sorted_state_dict.values()))
x = np.arange(coeffs.size)
c0 = coeffs[0]
print("c0", c0)
coeffs = coeffs / c0

keys = list(sorted_state_dict.keys())

cc_wfn = expT @ hfvec
coeffs_cc = np.array([cc_wfn[int(key, 2)] for key in keys]).real
coeffs_ci = np.array([ci_wfn[int(key, 2)] for key in keys]).real

ci_wfn_true = C_true @ all0
e_ci_true = ci_wfn_true.conj().T @ sparse_ham @ ci_wfn_true / (ci_wfn_true.conj().T @ ci_wfn_true)
coeffs_ci_true = np.array([ci_wfn_true[int(key, 2)] for key in keys]).real
coeffs_ci_true /= coeffs_ci_true[0]
print(e_ci_true)
print(coeffs_ci_true, "blubber", e_ci_true - mc.e_tot)
# print


coeffs, coeffs_cc = normalize_sign(coeffs, coeffs_cc, atol=1e-12)
coeffs, coeffs_ci = normalize_sign(coeffs, coeffs_ci, atol=1e-12)
coeffs, coeffs_ci_true = normalize_sign(coeffs, coeffs_ci_true, atol=1e-12)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 6)
ax.plot(x[1:], coeffs[1:], "o-", label="CI")
ax.plot(x[1:], coeffs_cc[1:], label="CI->CCSD")
# ax.plot(x[1:], coeffs_ci[1:], label="CI->CI (Hilbert)")
ax.plot(x[1:], coeffs_ci_true[1:], "o--", label="CI->CI (true vac)")
plt.legend()
plt.savefig("blubber.png", dpi=300)

print("CI coeffs (correct)")
pprint(coeffs)
# print("CC coeffs")
# pprint(coeffs_cc)
print("CI coeffs")
pprint(coeffs_ci)

# pprint(keys)
absplit = [(k[::2], k[1::2]) for k in keys]
# pprint(absplit)

# TODO: remove!!!
coeffs_cc = coeffs_ci
mask = np.where(np.abs(coeffs - coeffs_cc) > 1e-4)[0]
for m in mask:
    if coeffs_cc[m] == 0.0:
        continue
    print(f"Different sign found for idx {m}")
    print(f"=> Det = {absplit[m]}, coeff = {coeffs[m]}, cc = {coeffs_cc[m]}")
    swapped = (absplit[m][1], absplit[m][0])
    swap_inter = np.zeros(2 * ncas, dtype=object)
    swap_inter[::2] = list(swapped[0])
    swap_inter[1::2] = list(swapped[1])
    swap_inter = "".join(swap_inter)
    ix = keys.index(swap_inter)
    print(f"Inverse a/b order (idx={ix})")
    print(f"=> Det = {swapped}, coeff = {coeffs[ix]}, cc = {coeffs_cc[ix]}")
    print("====================")

# pprint(sorted_state_dict)

exit(0)


maxiter_vqe = 1000
# opt = cov.LBFGSB(atol=1e-10, gtol=1e-10, ftol=None)
# vqe.vqe_optimize(opt=opt, maxiter=maxiter_vqe)
# energy_vqe = vqe.vqe_energy(vqe.params)
# print(energy_vqe - mc.e_tot)

# coeffs_vqe = np.array(vqe.compute_vqe_basis_state_overlaps(keys, vqe.params))
# coeffs_vqe /= coeffs_vqe[0]

# # get the CCSD wavefunction from VQE amplitudes
# del c_ia
# del c_ijab
# c_ia_vqe, c_ijab_vqe = extract_vqe_singles_doubles_amplitudes_spinorb(vqe)
# c_ia_vqe = np.array(c_ia_vqe)
# c_ijab_vqe = np.array(c_ijab_vqe)
# expT_vqe, expTm_vqe = cc_operator_hilbert(c_ia_vqe, c_ijab_vqe, nocca, noccb, nvirta, nvirtb)

# cc_vqe_wfn = expT_vqe @ hfvec
# coeffs_cc_vqe = np.array([cc_vqe_wfn[int(key, 2)] for key in keys])

# e_cc_vqe = (hfvec.conj().T @ expTm_vqe @ sparse_ham @ cc_vqe_wfn).real

energies = {
    "e_ci": mc.e_tot,
    # "e_vqe": energy_vqe,
    "e_cc_ci": e_cc,
    # "e_cc_vqe": e_cc_vqe,
}
for idx, (i, k) in enumerate(energies.items()):
    for idx2, (j, k2) in enumerate(energies.items()):
        if idx < idx2:
            print(f"{i + ' - ' + j:<20s} = {np.abs(k - k2):<20.10e}")


def plot_axis(ax):
    ax.plot(x, np.abs(coeffs), label="CI")
    ax.plot(x, np.abs(coeffs_cc), label="CI->CCSD")
    # ax.plot(x, np.abs(coeffs_vqe), label="VQE")
    # ax.plot(x, np.abs(coeffs_cc_vqe), label="VQE->CCSD")
    ax.set_yscale("log")


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(16, 9)

plot_axis(ax)

ax.set_ylim([1e-9, 1])
ax.set_xlim([0, coeffs.size])

xstep = coeffs.size // 10
if xstep == 0:
    xstep = 1
ax.set_xticks(np.arange(0, coeffs.size, xstep))
ax.set_ylabel("|coeff|")
ax.set_xlabel("comp basis state (CI)")

axin = ax.inset_axes([0.5, 0.32, 0.43, 0.43])
plot_axis(axin)

cutoff = 1e-7
populated_idx = np.argwhere(np.abs(coeffs) > cutoff)
axin.set_xlim(0, populated_idx[-1])
axin.set_ylim(cutoff, 1)
ax.indicate_inset_zoom(axin)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=4)
plt.tight_layout()
plt.title(f"CAS({nelecas},{ncas})")
plt.savefig("amplitude_compare.pdf", dpi=300)
plt.show()
