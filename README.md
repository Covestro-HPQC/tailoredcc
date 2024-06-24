TailoredCC
==============================
Tailored and Externally Corrected Coupled Cluster Code based on PySCF.

### Example

The following script first runs CASCI with PySCF, and then passes the `CASCI` object into
the top-level run function `tccsd_from_ci`:

```py
from pyscf import gto, scf, mcscf
from tailoredcc.workflow import tccsd_from_ci

mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="cc-pvdz", verbose=4)
scfres = scf.RHF(mol)
scfres.kernel()
mc = mcscf.CASCI(scfres, ncas=6, nelecas=6).run()
tcc = tccsd_from_ci(mc)
```

For other examples, please check the unit/functionality tests.


### Installation

Please run
```console
pip install -v .
```
to install `tailoredcc` together with its dependencies.


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
