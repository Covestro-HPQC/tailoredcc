from pyscf import scf, mcscf, cc, gto
from tailoredcc import tccsd_from_ci
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ncas = 2
# nelecas = 2

ncas = 4
nelecas = 4

# ncas = 8
# nelecas = 8


def build_pes(r_hf, unit="Bohr", label=None):
    # mol = gto.M(
    #     atom=f"""
    #     H 0 0 0
    #     F 0 0 {r_hf}
    #     """,
    #     # symmetry=True,
    #     unit=unit,
    #     basis="cc-pVDZ"
    # )

    mol = gto.M(
        atom=f"""
        O
        H  1  {r_hf}
        H  1  {r_hf} 2 104.52
        """,
        unit=unit,
        basis="cc-pVDZ",
        # symmetry=True,
    )
    mol.tofile("h2o_bartlett.xyz", format="xyz")

    scfres = scf.HF(mol)
    scfres.max_cycle = 500
    scfres.conv_tol = 1e-10
    scfres.conv_tol_grad = 1e-6
    # scfres.DIIS = scf.EDIIS
    # scfres.diis_space = 14
    scfres.verbose = 4
    scfres.kernel()
    scfres.analyze()
   
    # swap = [3, 4]
    # scfres.mo_coeff[:, swap[::-1]] = scfres.mo_coeff[:, swap]
    # scfres.mo_energy[swap[::-1]] = scfres.mo_energy[swap]
    
    # CASCI
    mc = mcscf.CASCI(scfres, nelecas=nelecas, ncas=ncas)
    mc.kernel()

    # dump_cube = True
    # if label is None:
    #     label = str(r_hf)
    # if dump_cube:
    #     from pyscf.tools import cubegen
    #     for idx in range(0, mc.ncore + ncas):
    #         cubegen.orbital(mol, outfile=f"cubes/orb_{label}_{idx}.cube", coeff=scfres.mo_coeff[:, idx])
    
    # TCCSD
    tcc = tccsd_from_ci(mc)
    np.testing.assert_allclose(tcc.e_cas, mc.e_tot - scfres.e_tot, atol=1e-9, rtol=0)

    # ref_bartlett = -100.22884
    ref_bartlett = -76.24090
    diff = tcc.e_tot - ref_bartlett
    print("DIFF", diff)
    
    # CCSD
    ccsd = cc.CCSD(scfres)
    ccsd.kernel()
    
    # ref_ccsd = -100.22816
    ref_ccsd = -76.24008
    diff = ccsd.e_tot - ref_ccsd
    print("DIFF CCSD", diff)

    # CASSCF
    # mc_scf = mcscf.CASSCF(scfres, nelecas=nelecas, ncas=ncas)
    # mc_scf.kernel()
    
    # TCCSD/CAS orbitals
    # tcc_scf = tccsd_from_ci(mc_scf)

    return {
        "scf": scfres.e_tot,
        "casci": mc.e_tot,
        # "casscf": mc_scf.e_tot,
        "ccsd": ccsd.e_tot,
        "tccsd": tcc.e_tot,
        # "tccsd_cas": tcc_scf.e_tot,
        "r_hf": r_hf
    }

df = pd.DataFrame()


# NOTE: HF
# r0 = 1.733  # bohr
# r1 = 2.0 * r0
unit = "Bohr"
# r0 = 0.917
# r1 = 3.0
# for dist in np.linspace(0.7 * r0, 3.5 * r0, 20):
sz = 1

# NOTE: H2O
r0 = 1.809  # bohr
r1 = 4.0 * r0

for dist in np.linspace(r0, r1, sz):
    print(dist)
    ret = build_pes(dist, unit=unit, label=f"{dist / r0}")
    df = df.append(ret, ignore_index=True)

value_vars = [
    # "scf",
    "casci", "ccsd",
    "tccsd",
    # "casscf", "tccsd_cas"
]
# for val in value_vars:
#     df[val] -= np.min(df[val])

dfm = pd.melt(df, id_vars="r_hf", value_vars=value_vars, value_name="energy", var_name="method")
print(dfm)
dfm.r_hf /= r0

sns.lineplot(data=dfm, x="r_hf", y="energy", hue="method", markers=True, marker="o")
plt.tight_layout()
plt.savefig("dissociation_hf.png")

df.to_json("dissociation.json")
dfm.to_json("dissociation_melt.json")