# Proprietary and Confidential
# Covestro Deutschland AG, 2023

def _append0(detstring, norb):
    """Append zeros to a determinant string.
    The strings from pyscf are normal binary strings,
    so we fill up with trailing zeros up to norb

    Parameters
    ----------
    detstring : str
        determinant string (left to right)
    norb : int
        total number of orbitals

    Returns
    -------
    str
        determinant string with trailing zeros
    """
    diff = norb - len(detstring)
    if diff > 0:
        detstring += diff * "0"
    return detstring


def dump_clusterdec(mc, tol=1e-14, fname="wfs.dat"):
    """Dump the CI wavefunction from PySCF to ClusterDec format.
    More information can be found at:
    https://github.com/susilehtola/clusterdec

    Parameters
    ----------
    mc : pyscf.mcscf.CASCI
        CASCI/CASSCF object
    tol : float, optional
        amplitude cutoff tolerance, by default 1e-14
    fname : str, optional
        output file name, by default "wfs.dat"
    """
    determinants = []
    nelec = mc.nelecas
    for c, ia, ib in mc.fcisolver.large_ci(mc.ci, mc.ncas, nelec, tol=tol, return_strs=True):
        inva = _append0(ia[2:][::-1], mc.ncas)
        invb = _append0(ib[2:][::-1], mc.ncas)
        cstr = f"{c:.14f}"
        determinants.append([cstr, inva, invb])

    header = f"{len(determinants)} {mc.ncas} {nelec[0]} {nelec[1]}\n"
    dstring = header
    for d in determinants:
        dstring += " ".join(d)
        dstring += "\n"

    with open(fname, "w") as fo:
        fo.write(dstring)
