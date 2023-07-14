
import numpy as np
from pyscf import ao2mo, gto, mcscf, scf, lib
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tailoredcc.tailoredcc import tccsd_from_ci, tccsd_pyscf
from tailoredcc.amplitudes import extract_ci_singles_doubles_amplitudes, add_gaussian_noise, amplitudes_to_spinorb, prepare_cas_slices


lib.num_threads(4)
conv_tol_e = 1e-12
conv_tol_g = 1e-8

xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/n2_real.xyz"
# xyzfile = "/fs/home/cvsik/Projects/stack/covvqetools/examples/p-benzyne.xyz"

# nact = 2
# nalpha, nbeta = (1, 1)

# nact = 4
# nalpha, nbeta = (2, 2)

nact = 6
nalpha, nbeta = (3, 3)

# nact = 8
# nalpha, nbeta = (4, 4)

# nact = 10
# nalpha, nbeta = (5, 5)

# nact = 12
# nalpha, nbeta = (6, 6)

ncas = nact
nvirta = ncas - nalpha
nvirtb = ncas - nbeta
nocca = nalpha
noccb = nbeta

assert nocca == noccb
basis = "cc-pvdz"
# basis = "sto-3g"
# basis = "6-31g"

mol = gto.M(atom=str(xyzfile), basis=basis, verbose=4)
# mol = gto.M(atom="""
# N 0 0 0
# N 0 0 3.5
# """, unit="bohr", basis=basis, verbose=4)
scfres = scf.RHF(mol)
scfres.conv_tol = conv_tol_e
scfres.conv_tol_grad = conv_tol_g
scfres.kernel()

mc = mcscf.CASCI(scfres, nact, (nalpha, nbeta))
mc.kernel()

# prerequisites
nocca, noccb = mc.nelecas
assert nocca == noccb
nvirta = mc.ncas - nocca
nvirtb = mc.ncas - noccb
assert nvirta == nvirtb
assert isinstance(mc.ncore, (int, np.int64))
ncore = mc.ncore
ncas = mc.ncas
nvir = mc.mo_coeff.shape[1] - ncore - ncas

ci_amps = extract_ci_singles_doubles_amplitudes(mc)
c_ia, c_ijab = amplitudes_to_spinorb(*ci_amps)
occslice, virtslice = prepare_cas_slices(nocca, noccb, nvirta, nvirtb, ncore, nvir, backend="pyscf")
ret_exact = tccsd_pyscf(mc._scf, c_ia, c_ijab, occslice, virtslice, verbose=4)

np.random.seed(42)
cycles = 50
npoints = 10
data = []
for std in np.logspace(-10, -1, num=npoints, endpoint=True):
    for cycle in tqdm(range(cycles)):
        ci_amps_noisy = add_gaussian_noise(*ci_amps, std=std)
        c_ia, c_ijab = amplitudes_to_spinorb(*ci_amps_noisy)
        ret = tccsd_pyscf(mc._scf, c_ia, c_ijab, occslice, virtslice, verbose=0)
        assert ret.converged
        data.append([std, ret.e_tot])


df = pd.DataFrame(data=data, columns=['std', 'e_tot'])
df['tcc_error'] = np.abs(df.e_tot - ret_exact.e_tot)

sns.set_theme(style="ticks")
ax = sns.lineplot(data=df, x="std", y="tcc_error", markers=True, err_style="bars", marker="o")
ax.set_yscale("log")
ax.set_xscale("log")
plt.savefig("nino.png")


from functools import partial
from scipy.optimize import curve_fit

def line(x, a, b):
    return a + x * b

def power_law(x, a, b):
    return a * np.power(x, b)

def power_law_fit(x, y):
    x_log = np.log10(x)
    y_log = np.log10(y)
    popt, _ = curve_fit(line, x_log, y_log)
    a=np.power(10, popt[0])
    b=popt[1]
    fitted_power_law = partial(power_law, a=a, b=b)
    return a, b, fitted_power_law


a, b, fun = power_law_fit(df['std'].values, df['tcc_error'].values)
print(a, b)