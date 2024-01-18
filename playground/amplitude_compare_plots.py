from collections import OrderedDict

import covvqetools as cov
import matplotlib.pyplot as plt
import numpy as np
import openfermion as of
import pandas as pd
import seaborn as sns
from covvqetools.instant_vqes import QNPVQE
from covvqetools.pyscf import extract_state_dict
from pyscf import ao2mo, gto, mcscf, scf
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply

from tailoredcc.amplitudes import (
    ci_to_cluster_amplitudes,
    extract_ci_singles_doubles_amplitudes,
    extract_vqe_singles_doubles_amplitudes,
)


def cc_operator_hilbert(c_ia, c_ijab, nocca, noccb, nvirta, nvirtb):
    # TODO: still some sign problem because of Fermi vs true vacuum I guess...
    # but the absolute values of comp. basis states are correct :)
    nocc = nocca + noccb
    nvirt = nvirta + nvirtb

    t1cas, t2cas = ci_to_cluster_amplitudes(c_ia, c_ijab)
    T1 = of.FermionOperator()
    T2 = of.FermionOperator()
    for i in range(nocc):
        for mo_a, a in enumerate(range(nvirt), nocc):
            T1 += of.FermionOperator(f"{mo_a}^ {i}", t1cas[i, a])
            for j in range(nocc):
                for mo_b, b in enumerate(range(nvirt), nocc):
                    T2 += 0.25 * of.FermionOperator(f"{mo_a}^ {mo_b}^ {j} {i}", t2cas[i, j, a, b])

    T = T1 + T2
    print("=> Computing sparse operator...")
    T_sparse = of.get_sparse_operator(T, n_qubits=(nocc + nvirt))
    return T_sparse


def ccsd_wavefunction_hilbert_from_vqe(vqe, hfvec):
    c_ia_vqe, c_ijab_vqe = extract_vqe_singles_doubles_amplitudes(vqe)
    c_ia_vqe = np.array(c_ia_vqe)
    c_ijab_vqe = np.array(c_ijab_vqe)
    T_vqe = cc_operator_hilbert(c_ia_vqe, c_ijab_vqe, nocca, noccb, nvirta, nvirtb)
    cc_vqe_wfn = expm_multiply(T_vqe, hfvec)
    return cc_vqe_wfn


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


sns.set_theme(context="talk", style="ticks", font_scale=1.3)

# mol = gto.M(
#     atom="/fs/home/cvsik/Projects/stack/covvqetools/examples/p-benzyne.xyz",
#     # atom="/fs/home/cvsik/Projects/stack/covvqetools/examples/n2_real.xyz",
#     # atom="/fs/home/cvsik/Projects/stack/covvqetools/examples/h2o_real.xyz",
#     # atom="benzene_blind.xyz",
#     # basis="sto-3g",
#     basis="6-31g",
#     # basis="6-31g*",
#     verbose=4,
# )


mol = gto.M(
    atom="""
   N 0 0 0
   N 0 0 2.8         
""",
    basis="cc-pvdz",
    verbose=4,
)

conv_tol_e = 1e-12
conv_tol_g = 1e-8
scfres = scf.RHF(mol)
scfres.conv_tol = conv_tol_e
scfres.conv_tol_grad = conv_tol_g
scfres.kernel()

# ncas = 7
# nelecas = 10

# ncas = 4
# nelecas = 4

ncas = 6
nelecas = 6

mc = mcscf.CASCI(scfres, ncas=ncas, nelecas=nelecas)
mc.fcisolver.conv_tol = 1e-14
mc.kernel()
print("HF energy", scfres.e_tot, mc.e_tot - scfres.e_tot)

h1, ecore = mc.get_h1eff()
h2 = ao2mo.restore(1, mc.ao2mo(), ncas)
two_body_integrals = h2.transpose(0, 2, 3, 1)

depth = 80
maxiter_vqe = 1000
vqe = QNPVQE(
    nact=ncas,
    nalpha=mc.nelecas[0],
    nbeta=mc.nelecas[1],
    depth=depth,
    one_body_integrals=h1,
    two_body_integrals=two_body_integrals,
    core_energy=ecore,
    measurement_method=cov.CASBOX,
    # NOTE: trick to get nocc correct even though we don't have nuclear charges
    nocc=mc.ncore,
    nchar=-sum(mol.atom_charges()),
)
vqe.data_recorder_enable()
np.random.seed(42)
vqe.params += np.random.randn(*vqe.params.shape) * 1e-2

nbits = 1 << (2 * ncas)
hfvec = csc_matrix((nbits, 1), dtype=np.complex128)
hfstr = nelecas * "1" + (2 * ncas - nelecas) * "0"
hfidx = int(hfstr, 2)
hfvec[hfidx] = 1.0 + 0j

# print("=> Constructing sparse Hamiltonian...")
# sparse_ham = of.get_sparse_operator(vqe.hamiltonian)
# e_hf = hfvec.conj().T @ sparse_ham @ hfvec
# e_hf = e_hf[0, 0].real
# print("hfdiff", scfres.e_tot - e_hf)

nocca, noccb = mc.nelecas
assert nocca == noccb
nvirta = mc.ncas - nocca
nvirtb = mc.ncas - noccb
assert nvirta == nvirtb
assert isinstance(mc.ncore, int)
ncore = mc.ncore
ncas = mc.ncas

backend = "pyscf"
from tailoredcc import tccsd_from_ci

tcc_ret_ci = tccsd_from_ci(mc, backend=backend, maxiter=0)
print("CI energy error before TCC", tcc_ret_ci.e_cas + scfres.e_tot - mc.e_tot)


c_ia, c_ijab = extract_ci_singles_doubles_amplitudes(mc)
T_sparse = cc_operator_hilbert(c_ia, c_ijab, nocca, noccb, nvirta, nvirtb)
print("=> Computing exponential...")
cc_wfn = expm_multiply(T_sparse, hfvec)

state_dict = extract_state_dict(
    mc.fcisolver, mc.ci, nact=ncas, nalpha=mc.nelecas[0], nbeta=mc.nelecas[1], amplitude_cutoff=0
)

sorted_state_dict = OrderedDict(
    sorted(state_dict.items(), key=lambda item: abs(item[1]), reverse=True)
)

coeffs = np.array(list(sorted_state_dict.values()))
x = np.arange(coeffs.size)
c0 = coeffs[0]
print("c0", c0)
coeffs = coeffs / c0

keys = list(sorted_state_dict.keys())
coeffs_cc = np.array([cc_wfn[int(key, 2)][0, 0] for key in keys]).real

plot_animation = False


if plot_animation:
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 9)
    from celluloid import Camera

    camera = Camera(fig)


def callback(epoch, params):
    if epoch % 10 == 0:
        print("epoch", epoch, params.shape)
        if plot_animation:
            coeffs_vqe = np.array(vqe.compute_vqe_basis_state_overlaps(keys, params))
            coeffs_vqe /= coeffs_vqe[0]

            # get the CCSD wavefunction from VQE amplitudes
            cc_vqe_wfn = ccsd_wavefunction_hilbert_from_vqe(vqe, hfvec)
            coeffs_cc_vqe = np.array([cc_vqe_wfn[int(key, 2)][0, 0] for key in keys])

            cutoff = 1e-9
            diff_vqe = np.abs(np.abs(coeffs) - np.abs(coeffs_vqe))
            diff_cc_vqe = np.abs(np.abs(coeffs) - np.abs(coeffs_cc_vqe))

            diff_vqe[diff_vqe < cutoff] = cutoff
            diff_cc_vqe[diff_cc_vqe < cutoff] = cutoff

            df = pd.DataFrame(columns=["vqe", "cc_vqe"])
            df.vqe = diff_vqe
            df.cc_vqe = diff_cc_vqe
            dfm = pd.melt(df, value_vars=["vqe", "cc_vqe"], var_name="mapping", value_name="diff")
            ax.set_xlim((cutoff, 1))
            # ax2 = ax.twinx()
            sns.kdeplot(
                dfm,
                x="diff",
                hue="mapping",
                common_norm=False,
                ax=ax,
                cumulative=False,
                bw_adjust=0.7,
                log_scale=True,
            )
            camera.snap()


opt = cov.LBFGSB(atol=1e-10, gtol=1e-10, ftol=None)
# opt = cov.NelderMead()
vqe.vqe_optimize(opt=opt, maxiter=maxiter_vqe, callback=callback)
energy_vqe = vqe.vqe_energy(vqe.params)
print("CI energy error", energy_vqe - mc.e_tot)

if plot_animation:
    animation = camera.animate()
    animation.save("animation.mp4")

from tailoredcc import tccsd_from_vqe

tcc_ret = tccsd_from_vqe(scfres, vqe, backend=backend, maxiter=0)
print("CAS energy diff", tcc_ret.e_cas - tcc_ret_ci.e_cas)

coeffs_vqe = np.array(vqe.compute_vqe_basis_state_overlaps(keys, vqe.params))
coeffs_vqe /= coeffs_vqe[0]

# get the CCSD wavefunction from VQE amplitudes
cc_vqe_wfn = ccsd_wavefunction_hilbert_from_vqe(vqe, hfvec)
coeffs_cc_vqe = np.array([cc_vqe_wfn[int(key, 2)][0, 0] for key in keys])

energies = {
    "e_ci": mc.e_tot,
    "e_vqe": energy_vqe,
}

for idx, (i, k) in enumerate(energies.items()):
    for idx2, (j, k2) in enumerate(energies.items()):
        if idx < idx2:
            print(f"{i + ' - ' + j:<20s} = {np.abs(k - k2):<20.10e}")


######################
# NOTE: det indices that have been 'measured' :)
from tailoredcc.amplitudes import (
    detstrings_doubles,
    detstrings_singles,
    interleave_strings,
)

hfdet = np.zeros(ncas, dtype=int)
hfdet[:nocca] = 1
_, detsa = detstrings_singles(nocca, nvirta)
_, detsaa = detstrings_doubles(nocca, nvirta)

singles = interleave_strings(detsa, hfdet)
singles += interleave_strings(hfdet, detsa)
singles_indices = [keys.index(s) for s in singles]
singles_data = coeffs_cc_vqe[singles_indices]

doubles = interleave_strings(detsaa, hfdet)
doubles += interleave_strings(detsa, detsa)
doubles += interleave_strings(hfdet, detsaa)
doubles_indices = [keys.index(d) for d in doubles]
doubles_data = coeffs_cc_vqe[doubles_indices]
#####################


def plot_axis(ax):
    ax.plot(x, np.abs(coeffs), "-", label="CI")
    ax.plot(x, np.abs(coeffs_cc), "-", label="CI->CCSD")
    ax.plot(x, np.abs(coeffs_vqe), "--", label="VQE")
    ax.plot(x, np.abs(coeffs_cc_vqe), "--", label="VQE->CCSD")

    ax.plot(singles_indices, np.abs(singles_data), "x", label="C1")
    ax.plot(doubles_indices, np.abs(doubles_data), "x", label="C2")

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
ax.set_ylabel(r"$|C_i|$")
ax.set_xlabel(r"Determinant Index $i$")

# axin = ax.inset_axes([0.55, 0.32, 0.43, 0.43])
# plot_axis(axin)
# cutoff = 1e-7
# populated_idx = np.argwhere(np.abs(coeffs) > cutoff)
# axin.set_xlim(0, populated_idx[-1])
# axin.set_ylim(cutoff, 1)
# from matplotlib.ticker import MaxNLocator
# axin.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.indicate_inset_zoom(axin)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=4)
plt.tight_layout()
plt.title(f"CAS({nelecas},{ncas})")
plt.savefig("amplitude_compare.pdf", dpi=300)
plt.savefig("amplitude_compare.png", dpi=300)


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(16, 9)

diff_ccsd = np.abs(np.abs(coeffs) - np.abs(coeffs_cc))
diff_vqe = np.abs(np.abs(coeffs) - np.abs(coeffs_vqe))
diff_cc_vqe = np.abs(np.abs(coeffs) - np.abs(coeffs_cc_vqe))

diff_cc_vqe_from_vqe = np.abs(np.abs(coeffs_vqe) - np.abs(coeffs_cc_vqe))

print(
    np.linalg.norm(diff_ccsd),
    np.linalg.norm(diff_vqe),
    np.linalg.norm(diff_cc_vqe),
    np.linalg.norm(diff_cc_vqe_from_vqe),
    np.max(diff_cc_vqe_from_vqe),
    np.max(diff_vqe),
)

cutoff = 1e-9
nbins = 50

diff_ccsd[diff_ccsd < cutoff] = cutoff
diff_vqe[diff_vqe < cutoff] = cutoff
diff_cc_vqe[diff_cc_vqe < cutoff] = cutoff

diff_cc_vqe_from_vqe[diff_cc_vqe_from_vqe < cutoff] = cutoff

# bins = np.logspace(np.log10(cutoff), np.log10(1e-1), nbins)
# ax.hist([diff_ccsd, diff_vqe, diff_cc_vqe], bins=bins, label=["CI->CCSD", "VQE", "VQE->CCSD"], alpha=0.7, histtype="bar")
# ax.set_yscale("log")
# ax.set_xscale("log")
# ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=4)
# plt.tight_layout()
# plt.title(f"CAS({nelecas},{ncas})")
# plt.savefig("amplitude_diff_compare.pdf", dpi=300)
# plt.savefig("amplitude_diff_compare.png", dpi=300)

df = pd.DataFrame(columns=["ccsd", "vqe", "cc_vqe"])
df.ccsd = diff_ccsd
df.vqe = diff_vqe
df.cc_vqe = diff_cc_vqe

dfm = pd.melt(df, value_vars=["vqe", "cc_vqe"], var_name="mapping", value_name="diff")
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(16, 9)
ax.set_xlim((cutoff, 1))
ax2 = ax.twinx()
ax.set_xlabel(r"$|\Delta \vec C_\mathrm{CI}|$")
labels = ["VQE", "CCSD from VQE"]
sns.histplot(dfm, x="diff", hue="mapping", ax=ax, log_scale=True, bins=50, multiple="layer")
sns.kdeplot(
    dfm,
    x="diff",
    hue="mapping",
    common_norm=False,
    ax=ax2,
    cumulative=False,
    bw_adjust=0.7,
    legend=False,
)
legend = ax.get_legend()
legend.set_title("Wavefunction")
for t, l in zip(legend.texts, labels):
    t.set_text(l)
# ax.set_yscale('log')
plt.savefig("blubbl.png")
