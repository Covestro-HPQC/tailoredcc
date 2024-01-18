from functools import partial

import covvqetools as cov
import fqe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import seaborn as sns
from covvqetools.instant_vqes import QNPVQE
from covvqetools.measurement.classical_shadows import MatchGateShadowTermwise
from covvqetools.pyscf import extract_state_dict
from pyscf import ao2mo, gto, mcscf, scf

from tailoredcc.tailoredcc import ec_cc_from_ci
from tailoredcc.utils import fqe_to_fake_ci


def state_dict_to_fqe_wfn(state_dict, nact, nalpha, nbeta):
    nel_total = nalpha + nbeta
    sz = nalpha - nbeta
    fqe_wf_ci = fqe.Wavefunction([[nel_total, sz, nact]])
    fqe_data_ci = fqe_wf_ci.sector((nel_total, sz))
    fqe_graph_ci = fqe_data_ci.get_fcigraph()
    fqe_orderd_coeff = np.zeros(
        (fqe_graph_ci.lena(), fqe_graph_ci.lenb()), dtype=list(state_dict.values())[0].dtype
    )
    for detstr, coeff in state_dict.items():
        astr = detstr[::2][::-1]
        bstr = detstr[1::2][::-1]
        assert len(detstr) == 2 * nact
        assert astr.count("1") == nalpha
        assert bstr.count("1") == nbeta
        abit = int(astr, 2)
        bbit = int(bstr, 2)
        fqe_orderd_coeff[fqe_graph_ci.index_alpha(abit), fqe_graph_ci.index_beta(bbit)] = coeff
    fqe_data_ci.coeff = fqe_orderd_coeff
    return fqe_wf_ci


mol = gto.Mole()
mol.basis = "sto3g"
mol.atom = """H 0 0 0
H 0 0 1.23
H 0 1.23 0
H 0 1.23 1.23"""
mol.verbose = 0
mol.spin = 0
mol.build()
mol.symmetry = False
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
mf.conv_tol_grad = 1e-10
mf.chkfile = "scf.chk"
ehf = mf.kernel()
mo1 = mf.stability(verbose=0)[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability(verbose=0)

from pyscf.fci.cistring import gen_occslst

strs = gen_occslst(range(4), 2)
excilevels = np.array([np.sum(x > 1) for x in strs])
excilevels = excilevels[:, None] + excilevels
excilevels = np.asarray(excilevels, dtype=int)
print(excilevels)

cisd_mask = excilevels <= 2

mc = mcscf.CASCI(mf, ncas=4, nelecas=4)
h1, ecore = mc.get_h1eff()
h2o = mc.get_h2eff()
h2 = ao2mo.restore(1, h2o, mc.ncas).transpose(0, 2, 3, 1)
mc.kernel()

e_ci = mc.fcisolver.energy(h1, h2o, mc.ci, 4, (2, 2))
print(mc.e_cas, e_ci)
# exit(0)

# state_dict = extract_state_dict(
#     mc.fcisolver, mc.ci, mc.ncas, mc.nelecas[0], mc.nelecas[1], amplitude_cutoff=1e-8
# )
# print(len(state_dict))

sizes = [
    # 10,
    # 20,
    # 50,
    # 100, 200, 500,
    # 1000, 2000,
    #
    # 5000,
    1e4,
    2e4,
    5e4,
    1e5,
    2e5,
    5e5,
    1e6,
]
nsamp = 30
# ec_rets = {}
data = []
for shadow_size in sizes:
    shadow_size = int(shadow_size)
    for ii in range(nsamp):
        try:
            # schatten = np.load(f"shadow_records/shadow_multi_h4_{shadow_size}_{ii}.npz")
            schatten = np.load(f"shadow_records/shadow_h4_{shadow_size}_{ii}.npz")
            # schatten = np.load(f"shadow_h4_{shadow_size}.npz")
        except:
            print(ii, shadow_size, "not found")
            continue
        shadow_overlaps = schatten["shadow"]

        max_idx = np.argmax(np.abs(shadow_overlaps))
        max_val = shadow_overlaps[max_idx]
        global_phase = np.exp(1j * np.angle(max_val))
        shadow_overlaps /= global_phase
        shadow_overlaps = shadow_overlaps.real

        vars_overlap = schatten["vars_overlap"]
        vars_overlap /= global_phase
        vars_real = np.abs(vars_overlap.real)

        shadow_state = {k: shadow_overlaps[i] for i, k in enumerate(schatten["states"])}
        # print(shadow_overlaps, '\n', np.sqrt(vars_real))
        mf.mo_coeff = schatten["C"]
        # mf.stability(verbose=0)
        np.testing.assert_allclose(mf.mo_coeff, schatten["C"])

        mc = mcscf.CASCI(mf, ncas=4, nelecas=4)
        # mc.verbose = 4
        mc.fcisolver.conv_tol = 1e-12
        h1, ecore = mc.get_h1eff()
        h2orig = mc.get_h2eff()
        h2 = ao2mo.restore(1, h2, mc.ncas).transpose(0, 2, 3, 1)
        mc.kernel(schatten["C"])

        # Davidson correction
        # compute_ci_energy = partial(mc.fcisolver.energy, h1e=h1, eri=h2orig, norb=mc.ncas, nelec=mc.nelecas)

        # def davidson_correction(fcivec):
        #     civec_cisd = fcivec.copy()
        #     civec_cisd[~cisd_mask] = 0.0
        #     # civec_cisd /= np.linalg.norm(civec_cisd)
        #     e_cisd = ecore + compute_ci_energy(fcivec=civec_cisd)
        #     E_DC = (1 - civec_cisd[0, 0]**2) * (e_cisd - mf.e_tot) / civec_cisd[0, 0]**2
        #     return E_DC
        # e_cas_exact = compute_ci_energy(fcivec=mc.ci)
        # np.testing.assert_allclose(e_cas_exact, mc.e_cas, atol=1e-14, rtol=0)

        fqe_wf = state_dict_to_fqe_wfn(shadow_state, mc.ncas, mc.nelecas[0], mc.nelecas[1])
        fake_mc = fqe_to_fake_ci(fqe_wf, mf, sz=0)
        del fqe_wf

        # civec = fake_mc.ci
        # signs = np.sign(civec)
        # civec = 0.5 * (np.abs(civec) + np.abs(civec).T) * signs
        # fake_mc.ci = civec

        # fake_mc.ci[np.abs(fake_mc.ci) < 1e-3] = 0.0
        # print(fake_mc.ci)
        # print(np.linalg.norm(fake_mc.ci), np.linalg.norm(shadow_overlaps))

        # try to compute the variance
        civec = fake_mc.ci
        civec /= np.sign(mc.ci[0, 0]) * np.sign(civec[0, 0])
        diff = civec - mc.ci
        var = np.var(diff)

        # dc_exact = davidson_correction(mc.ci)
        # dc_shadow = davidson_correction(civec)
        # print(dc_exact, dc_shadow)

        # NOTE:
        mask_std = np.abs(fake_mc.ci) <= 2.0 * np.sqrt(vars_real.reshape(fake_mc.ci.shape))
        # mask_std = np.abs(fake_mc.ci) <= 2.0 * np.mean(np.sqrt(vars_real))
        # mask_std = np.abs(fake_mc.ci) <= np.mean(np.sqrt(vars_real))
        # mask_std = np.abs(fake_mc.ci) <= (2.0 / np.sqrt(shadow_size))
        if np.sum(mask_std) >= fake_mc.ci.size:
            print("discarded all values")
            raise ValueError
        else:
            nmask = np.sum(mask_std)
            print(f"discarding {nmask} elements")
            fake_mc.ci[mask_std] = 0.0
        # fake_mc.ci[zero_mask] = 0.0

        print(shadow_size, ii)
        ec = None
        try:
            ec = ec_cc_from_ci(
                fake_mc,
                maxiter=2000,
                conv_tol=1e-8,
                verbose=0,
                guess_t1_t2_from_ci=True,
                zero_companion_threshold=1e-6,
            )
            e_tot = ec.e_tot
        except Exception as e:
            print(str(e), "continuing...")
            e_tot = np.nan
        if ec is None or not ec.converged:
            print("NOOOOOO")
            e_tot = np.nan

        # if e_tot is np.nan:
        from tailoredcc.plot_utils import civec_scatter, sign_and_zero_errors

        # sns.set_theme(context="talk", palette="colorblind", font_scale=1.2, style="ticks")
        # fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(18, 9)
        # ax = civec_scatter({'exact': mc.ci, 'shadow': civec}, ax)
        # ax.set_title(f"Converged = {e_tot is not np.nan}")
        # plt.savefig(f"failures/h4_civec_{shadow_size}_{ii}.png")
        # num_sign_errors, sign_errors, nonzero_mask, num_zero_errors, zero_errors
        errors_signs_zeros = sign_and_zero_errors(mc.ci, civec)

        data.append([shadow_size, ii, e_tot, var, *errors_signs_zeros])


df = pd.DataFrame(
    data=data,
    columns=[
        "shots",
        "sample",
        "energy",
        "var",
        "num_sign_errors",
        "sign_errors",
        "nonzero_mask",
        "num_zero_errors",
        "zero_errors",
    ],
)
# df.dropna(inplace=True)
# df.drop(df[np.abs(df.energy) > 1e10].index, inplace=True)
df["std_numerical"] = np.sqrt(df["var"])
df["std_bound"] = 2 / np.sqrt(df.shots)
# df['std_diff'] = df.std_numerical - df.std_bound
df["error_abs"] = np.abs(df.energy - mc.e_tot)


# df.to_hdf("ec_cc_from_shadows_multi.h5", key="df")
df.to_hdf("ec_cc_from_shadows.h5", key="df")
