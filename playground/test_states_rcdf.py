from dataclasses import dataclass

import covvqetools as cov
import numpy as np
import pennylane as qml
import pytest
from covvqetools.pyscf import extract_state_dict
from pyscf import ao2mo, gto, lib, mcscf, scf
from pyscf.mcscf import avas


@dataclass
class ActiveSpaceSelection:
    molecule: gto.Mole
    scfres: scf.HF
    ncas: int
    nelecas_singlet: tuple
    nelecas_triplet: tuple
    mo: np.ndarray
    name: str = None


# @pytest.fixture(
#     name="as_system",
#     params=[
#         # ("p-benzyne.xyz", "cc-pvdz", ["C 2pz"])
#         # ("h2o.xyz", "cc-pvdz", ["O 2p", "O 2s", "H 1s"]),
#         # ('ferrocene.xyz', 'def2-svp', ['Fe 3d']),
#         ('naphthalene.xyz', 'def2-svp', ['C 2pz']),
#         # ('ferrocene.xyz', 'def2-svp', ['Fe 3d', 'C 2pz']),
#         # ('ls_cn2_bipy2.xyz', 'def2-svp', ['Fe 3d']),
#     ],
#     scope="module",
#     ids=lambda arg: f"{arg[0][:-4]}_{arg[1]}",
# )
@pytest.fixture
def _scf_avas_ci(request):
    lib.num_threads(24)
    d = 1.1
    ii = 3
    mol, scf_dict = scf.chkfile.load_scf(f"mos/scf_{ii}.chk")
    mf = scf.RHF(mol)
    mf.__dict__.update(scf_dict)
    dx = np.load(f"mos/mo_{ii}.npz")
    mos = dx["mos"]
    mo_occ = dx["mo_occ"]
    assert d == dx["d"]

    scfres = scf.HF(mol)
    scfres.max_cycle = 500
    scfres.conv_tol = 1e-10
    scfres.conv_tol_grad = 1e-7
    scfres.kernel()
    ncas, nelecas, mo = avas.avas(scfres, aolabels)
    singlet = 2 * (nelecas // 2,)
    triplet = (singlet[0] + 1, singlet[1] - 1)

    mc_singlet = mcscf.CASCI(scfres, ncas, singlet).run(mo)

    state_singlet = extract_state_dict(
        mc_singlet.fcisolver, mc_singlet.ci, ncas, *singlet, amplitude_cutoff=1e-9
    )
    return (
        ActiveSpaceSelection(mol, scfres, ncas, singlet, triplet, mo, name=request.param[0]),
        mc_singlet,
        mc_triplet,
        state_singlet,
        state_triplet,
    )


def active_space_integrals(mc, mo, ncas):
    h1, ecore = mc.h1e_for_cas(mo, ncas)
    h2 = ao2mo.restore(1, mc.ao2mo(mo), ncas).transpose(0, 2, 3, 1)
    return ecore, h1, h2


def energy_from_method(state, method, compute_std=False):
    @qml.template
    def ansatz(state, wires):
        cov.Superposition(state, wires)

    energy = method.expval(ansatz=ansatz, state=state).numpy()

    if compute_std:
        std = np.sqrt(method.var(ansatz=ansatz, state=state, use_past_measurements=False).numpy())
        return energy, std
    else:
        return energy


@pytest.mark.parametrize(
    "measurement_method, measurement_method_kwargs",
    [
        (
            cov.DoubleFactorized,
            {
                "num_leafs": i,
                "regularization_factor": rf,
                "flavor": flavor,
                "layout": "square_non_autodiff",
                "maxiter": 10000,
                "cdf_tol": 1e-7,
                "debug": True,
                "disp": True,
                "interface": "jax",
            },
        )
        # for i in [2, 4, 6, 8] + list(range(10, 55 + 1, 5))  # naphthalene (10, 10)
        for i in [
            20,
        ]
        for (flavor, rf) in [
            # ("XDF", None),
            # ("CDF", None),
            ("RC-DF", 1.0e-3),
        ]
    ],
)
def test_singlet_triplet_gaps(
    measurement_method, measurement_method_kwargs, as_system, results_bag
):
    sp, mcs, mct, state_singlet, state_triplet = as_system
    tl = active_space_integrals(mcs, sp.mo, sp.ncas)
    sl = active_space_integrals(mct, sp.mo, sp.ncas)
    np.testing.assert_allclose(tl[0], sl[0], atol=1e-11, rtol=0.0)
    np.testing.assert_allclose(tl[1], sl[1], atol=1e-11, rtol=0.0)
    np.testing.assert_allclose(tl[2], sl[2], atol=1e-11, rtol=0.0)

    core_energy, h1, h2 = sl

    device_singlet = qml.device(
        "cov.quicksilver.casbox",
        nalpha=sp.nelecas_singlet[0],
        nbeta=sp.nelecas_singlet[1],
        wires=2 * sp.ncas,
    )
    device_triplet = qml.device(
        "cov.quicksilver.casbox",
        nalpha=sp.nelecas_triplet[0],
        nbeta=sp.nelecas_triplet[1],
        wires=2 * sp.ncas,
    )

    method_exact = cov.DoubleFactorized(
        core_energy=core_energy,
        one_body_integrals=h1,
        two_body_integrals=h2,
        num_leafs="max",
        dev=device_singlet,
        layout="square_non_autodiff",
    )

    e_s0 = energy_from_method(state_singlet, method_exact)
    np.testing.assert_allclose(e_s0, mcs.e_tot, atol=1e-10, rtol=0.0)

    method_exact.dev = device_triplet
    e_t1 = energy_from_method(state_triplet, method_exact)
    np.testing.assert_allclose(e_t1, mct.e_tot, atol=1e-10, rtol=0.0)

    shots_for_distributor = 1e6
    # shots_for_distributor = 5e6
    sd = "according_to_term_group_weights"

    print("\nflavor: ", measurement_method_kwargs["flavor"])
    method = measurement_method(
        core_energy=core_energy,
        one_body_integrals=h1,
        two_body_integrals=h2,
        dev=device_singlet,
        **measurement_method_kwargs,
    )
    method.shot_distributor = getattr(method, sd)(shots_for_distributor)
    e_s0_approx, std_s0 = energy_from_method(state_singlet, method, compute_std=True)

    method.dev = device_triplet
    e_t1_approx, std_t1 = energy_from_method(state_triplet, method, compute_std=True)

    s0_error = e_s0 - e_s0_approx
    t1_error = e_t1 - e_t1_approx

    gap_ref = e_t1 - e_s0
    gap_approx = e_t1_approx - e_s0_approx
    error_gap = np.abs(gap_ref - gap_approx)

    delta_gap = error_gap + std_s0 + std_t1

    mse_gap = error_gap**2 + std_s0**2 + std_t1**2
    sqrt_mse = np.sqrt(mse_gap)

    print(f"leaves: {measurement_method_kwargs['num_leafs']}, ", s0_error)
    print(f"leaves: {measurement_method_kwargs['num_leafs']}, ", t1_error)
    print(gap_ref, gap_approx, error_gap)
    print("std s0", std_s0)
    print("std t1", std_t1)
    print(measurement_method_kwargs["num_leafs"], sqrt_mse)
    results_bag.energy_singlet = mcs.e_tot
    results_bag.energy_triplet = mct.e_tot
    results_bag.energy_singlet_approx = e_s0_approx
    results_bag.energy_triplet_approx = e_t1_approx
    results_bag.error_singlet = np.abs(s0_error)
    results_bag.error_triplet = np.abs(t1_error)
    results_bag.error_gap = error_gap
    results_bag.delta = delta_gap
    results_bag.sqrt_mse = sqrt_mse
    results_bag.std_singlet = std_s0
    results_bag.std_triplet = std_t1
    results_bag.num_leafs = measurement_method_kwargs["num_leafs"]
    results_bag.flavor = measurement_method_kwargs["flavor"]


def test_synthesis(module_results_df):
    pass
    # module_results_df.to_hdf("data.h5", key="df")
    # _plot_df(module_results_df)


def test_plot_from_disk():
    import pandas as pd

    df = pd.read_hdf("data.h5", key="df")
    _plot_df(df)


def _plot_df(df):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plt.rcParams.update(
        {
            #   "text.usetex": True,
            # "font.family": "Helvetica"
        }
    )
    # sns.set_theme(palette='Set2', context='talk')
    sns.set_theme(
        # palette='husl',
        font_scale=1.2,
        context="talk",
        style="ticks",
        rc={"lines.linewidth": 3.0},
    )

    # 3 column plot
    ncols = 3
    fig, axes = plt.subplots(nrows=1, ncols=ncols)
    fig.set_size_inches(12 * ncols, 10)

    ax1 = axes[0]
    sns.lineplot(x="num_leafs", y="error_gap", hue="flavor", data=df, ax=ax1, marker="o")
    ax1.set_xlabel(r"$n_t$")
    ax1.set_ylabel(r"$\Delta\Delta E$ [a.u.]")
    ax1.set(yscale="log")

    ax2 = axes[1]
    # ax2 = axes
    sns.lineplot(
        x="num_leafs", y="sqrt_mse", hue="flavor", data=df, ax=ax2, marker="o", markersize=15
    )
    ax2.set_xlabel(r"$n_t$")
    ax2.set_ylabel(r"$\sqrt{\rm{MSE}}$ [a.u.]")
    ax2.set(yscale="log")

    ax3 = axes[2]
    # print(df.columns)
    dfp = pd.melt(
        df,
        id_vars=["pytest_obj", "num_leafs", "flavor"],
        value_vars=["std_singlet", "std_triplet"],
        var_name="mult",
        value_name="std",
    )

    sns.lineplot(x="num_leafs", y="std", hue="flavor", style="mult", data=dfp, ax=ax3, marker="o")
    ax3.set(yscale="log")
    handles, labels = ax2.get_legend_handles_labels()
    lgd = fig.legend(
        handles, labels, title="Flavor", ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.00)
    )
    ax1.legend().remove()
    ax2.legend().remove()
    # sns.move_legend(ax2, labels=['X-DF', 'C-DF', 'RC-DF'], loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, title='Scheme')

    # ax1.set_xticks(np.unique(df.num_leafs.values))
    ax2.set_xticks(np.unique(df.num_leafs.values))
    ax2.axhline(1e-3, color="gray", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        "test_total_error.png",
        # bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )

    ncols = 1
    fig, axes = plt.subplots(nrows=1, ncols=ncols)
    fig.set_size_inches(12 * ncols, 9)

    ax2 = axes
    # pal = sns.color_palette()
    # pal = [pal[4], pal[0], pal[5]]
    print(df.flavor)
    sns.lineplot(
        x="num_leafs",
        y="sqrt_mse",
        hue="flavor",
        data=df,
        ax=ax2,
        marker="o",
        markersize=15,  # palette="Set2"
        hue_order=["CDF", "XDF", "RC-DF"],
        # palette=sns.color_palette(),
    )
    ax2.set_xlabel(r"$n_t$")
    ax2.set_ylabel(r"$\sqrt{\rm{MSE}}$ [Eh]")
    ax2.set(yscale="log")

    sns.move_legend(
        ax2,
        labels=["C-DF", "X-DF", "RC-DF"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=3,
        title="Scheme",
    )
    ax2.set_xticks(np.unique(df.num_leafs.values))
    ax2.axhline(1e-3, color="gray", linestyle="--")

    plt.tight_layout()
    plt.savefig("naphthalene_10_10_mse.png", bbox_inches="tight")
    plt.savefig("naphthalene_10_10_mse.pdf", bbox_inches="tight")
