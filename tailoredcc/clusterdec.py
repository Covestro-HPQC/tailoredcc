# Copyright 2024 Covestro Deutschland AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess
from pathlib import Path
from shutil import which


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


def run_clusterdec(fname="wfs.dat"):
    cwd = os.getcwd()
    exepath = Path(which("clusterdec_bit.x")).parent
    os.chdir(exepath.resolve())
    out = subprocess.run(["clusterdec_bit.x", fname], capture_output=True)
    print(out.stderr.decode("ascii"))
    print(out.stdout.decode("ascii"))
    os.chdir(cwd)
    lines = [ll.strip() for ll in out.stdout.decode("ascii").split("\n")]
    for idx, ll in enumerate(lines):
        if "|C_n|        |T_n|  |T_n|/|C_n|" in ll:
            c1sq = float(lines[idx + 1].split(" ")[1])
            c2sq = float(lines[idx + 2].split(" ")[1])
            c3sq = float(lines[idx + 3].split(" ")[1])
            c4sq = float(lines[idx + 4].split(" ")[1])
            t1sq = float(lines[idx + 1].split(" ")[2])
            t2sq = float(lines[idx + 2].split(" ")[2])
            t3sq = float(lines[idx + 3].split(" ")[2])
            t4sq = float(lines[idx + 4].split(" ")[2])
            break
    return (c1sq, c2sq, c3sq, c4sq), (t1sq, t2sq, t3sq, t4sq)
