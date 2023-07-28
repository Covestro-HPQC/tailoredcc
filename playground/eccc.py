# Proprietary and Confidential
# Covestro Deutschland AG, 2023

# %%
from pyscf import gto, scf, mcscf, cc
from tailoredcc.amplitudes import extract_ci_amplitudes
from tailoredcc.amplitudes import amplitudes_to_spinorb
from tailoredcc import ec_cc_from_ci

# mol = gto.M(atom="Be 0 0 0; Be 0 0 2.4", basis="cc-pvqz")
# mol = gto.M(atom="Be 0 0 0; Be 0 0 2.4", basis="sto-3g")
mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="3-21g")
# mol = gto.M(atom="F 0 0 0; F 0 0 1.1", basis="3-21g")
# mol = gto.M(atom="He 0 0 0; He 0 0 1.0", basis="cc-pvdz")
# mol = gto.M(atom="Li 0 0 0; Li 0 0 1.0", basis="sto-3g")
print(mol.nao_nr())
print(mol.nelec)
scfres = scf.RHF(mol)
scfres.kernel()

ccsd = cc.CCSD(scfres).run()


mc = mcscf.CASCI(scfres, nelecas=mol.nelec, ncas=mol.nao_nr())
# mc = mcscf.CASCI(scfres, nelecas=8, ncas=10)
mc.fcisolver.conv_tol = 1e-12
mc.kernel()

ci_amps = extract_ci_amplitudes(mc, exci=4)
# %%
c1, c2, c3, c4 = amplitudes_to_spinorb(ci_amps, exci=4)
# t1, t2, t3, t4 = ci_to_cc(c1, c2, c3, c4)
# %%

from tailoredcc.tailoredcc import ec_cc
from tailoredcc.amplitudes import prepare_cas_slices

nocca, noccb = mc.nelecas
nvirta = mc.ncas - nocca
nvirtb = mc.ncas - noccb
ncore = mc.ncore
nvir = mc.mo_coeff.shape[1] - ncore - mc.ncas
occslice, virtslice = prepare_cas_slices(nocca, noccb, nvirta, nvirtb, ncore, nvir, "oe")
ret = ec_cc(scfres, c1, c2, c3, c4, occslice, virtslice)

print(ret.e_tot, ret.e_tot - mc.e_tot)
print(ret.e_tot, ret.e_tot - ccsd.e_tot)

ec_cc_from_ci(mc, conv_tol=1e-14, guess_t1_t2_from_ci=True)
# %%
