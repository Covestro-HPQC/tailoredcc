import covvqetools as cov
import h5py
import numpy as np
import pandas as pd
from covvqetools.instant_vqes import QNPVQE
from pyscf import ao2mo, lib, mcscf, scf

from tailoredcc import tccsd_from_vqe
from tailoredcc.amplitudes import determinant_strings, extract_from_dict


def tccsd_from_vqe(
    scfres: scf.hf.SCF,
    vqe: cov.vqe.ActiveSpaceChemistryVQE,
    backend="pyscf",
    **kwargs,
):
    from tailoredcc.amplitudes import amplitudes_to_spinorb, prepare_cas_slices
    from tailoredcc.tailoredcc import _tccsd_map

    # TODO: docs, type hints
    nocca, noccb = vqe.nalpha, vqe.nbeta
    assert nocca == noccb
    ncas = vqe.nact
    nvirta = ncas - nocca
    nvirtb = ncas - noccb
    assert nvirta == nvirtb
    ncore = vqe.nocc
    if ncore == 0 and (nocca + noccb) != sum(scfres.mol.nelec):
        raise ValueError("The active space needs to contain all electrons if ncore=0.")
    ncas = vqe.nact
    nvir = scfres.mo_coeff.shape[1] - ncore - ncas

    ci_amps = extract_vqe_singles_doubles_amplitudes(vqe)
    c_ia, c_ijab = amplitudes_to_spinorb(ci_amps)

    c_ia = c_ia.real
    c_ijab = c_ijab.real

    occslice, virtslice = prepare_cas_slices(nocca, noccb, nvirta, nvirtb, ncore, nvir, backend)
    return _tccsd_map[backend](scfres, c_ia, c_ijab, occslice, virtslice, **kwargs)


def extract_vqe_singles_doubles_amplitudes(vqe: cov.vqe.ActiveSpaceChemistryVQE):
    if not hasattr(vqe, "compute_vqe_basis_state_overlaps"):
        raise NotImplementedError(
            "This VQE cannot compute overlap with computational basis states."
        )
    ncas = vqe.nact
    nocca, noccb = vqe.nalpha, vqe.nbeta
    if nocca != noccb:
        raise NotImplementedError(
            "Amplitude conversion only implemented for closed-shell active space."
        )
    nvirta = ncas - nocca
    nvirtb = ncas - noccb
    assert nvirta == nvirtb

    dets_tccsd = determinant_strings(ncas, nocca, level=2)

    def interleave_bits(int1, int2):
        max_length = max(int1.bit_length(), int2.bit_length())
        result = 0
        for i in range(max_length):
            bit1 = (int1 >> i) & 1
            bit2 = (int2 >> i) & 1
            result |= (bit2 << (2 * i + 1)) | (bit1 << (2 * i))
        return result

    overlaps = {}
    for key, dets in dets_tccsd.items():
        ret = []
        for d in dets:
            id = interleave_bits(int(d[0], 2), int(d[1], 2))
            detstring = bin(id)[2:][::-1]
            diff = 2 * ncas - len(detstring)
            if diff > 0:
                detstring += diff * "0"
            overlap = vqe.compute_vqe_basis_state_overlaps([detstring], vqe.params)[0]
            ret.append(overlap)
        overlaps[key] = np.asarray(ret)
    return extract_from_dict(overlaps, ncas, nocca, noccb)


lib.num_threads(24)
np.random.seed(42)

params = {}
data_vqe = []
for ii, d in enumerate(np.arange(0.8, 2.9, 0.1)):
    print(ii, d)
    mol, scf_dict = scf.chkfile.load_scf(f"mos/scf_{ii}.chk")
    mf = scf.RHF(mol)
    mf.__dict__.update(scf_dict)
    dx = np.load(f"mos/mo_{ii}.npz")
    mos = dx["mos"]
    mo_occ = dx["mo_occ"]
    assert d == dx["d"]

    np.testing.assert_allclose(mf.e_tot, dx["e_tot"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(mf.mo_coeff, mos, atol=1e-14, rtol=0)
    ncas = 6
    nalpha, nbeta = 3, 3

    cas = mcscf.CASCI(mf, nelecas=(nalpha, nbeta), ncas=ncas)
    h1, ecore = cas.get_h1eff()
    h2 = ao2mo.restore(1, cas.ao2mo(), ncas)
    two_body_integrals = h2.transpose(0, 2, 3, 1)

    cas.kernel()

    data = h5py.File(f"mos/vqe_{ii}.h5", "w")

    depths = [
        # 4,
        # 6,
        10,
        # 14,
        # 18,
        # 12,
        # 18,
        # 24,
        # 30,
        # 36,
    ]
    maxiter_vqe = 3000
    for depth in depths:
        vqe = QNPVQE(
            depth=depth,
            nact=ncas,
            nalpha=nalpha,
            nbeta=nbeta,
            # NOTE: trick to get nocc correct even though we don't have nuclear charges
            nocc=cas.ncore,
            nchar=-sum(mol.atom_charges()),
            ###
            measurement_method=cov.CASBOX,
            core_energy=ecore,
            one_body_integrals=h1,
            two_body_integrals=two_body_integrals,
        )
        # print(vqe.params.size)
        # exit(0)
        # if os.path.isfile(f"mos/vqe_{ii}.h5"):
        #     data_read = h5py.File(f"mos/vqe_{ii}.h5", "r")
        #     if f'{depth}' in data_read.keys():
        #         print("yes")
        # exit(0)
        if params.get(depth, None) is None:
            print("Fresh start")
        else:
            print("Re-using params from previous distance.")
            vqe.params = params[depth]
        vqe.params += np.random.randn(*vqe.params.shape) * 1e-2

        # opt = cov.LBFGSB(atol=None, gtol=1e-8, ftol=None)
        # opt = cov.LBFGSB(atol=1e-8, gtol=None, ftol=None)
        # opt = cov.LBFGSB(atol=None, gtol=1e-6, ftol=None)
        opt = cov.LBFGSB(atol=None, gtol=1e-8, ftol=None)

        def callback(epoch, _):
            print("vqe", epoch, vqe.vqe_energy(), np.max(np.abs(vqe.vqe_jacobian())))

        vqe.vqe_optimize(opt=opt, maxiter=maxiter_vqe, callback=callback)
        energy_vqe = float(vqe.vqe_energy(vqe.params))
        params[depth] = np.array(vqe.params.copy())

        dsg = data.create_group(f"{depth}")
        dsg.create_dataset("params", data=vqe.params)
        dsg["energy"] = vqe.vqe_energy()

        tcc_vqe = tccsd_from_vqe(mf, vqe, maxiter=200)
        # data_vqe.extend([
        #     [d, "VQE", energy_vqe, depth],
        #     [d, "VQE-TCCSD", tcc_vqe.e_tot, depth],
        # ])
        data_vqe.append([d, depth, energy_vqe, tcc_vqe.e_tot, tcc_vqe.e_cas, tcc_vqe.e_corr])
    print(list(data.keys()))
    data.close()

# df = pd.DataFrame(data=data_vqe, columns=["d", "method", "energy", "depth"])
df = pd.DataFrame(
    data=data_vqe, columns=["d", "depth", "VQE", "VQE-TCCSD", "tcc_e_cas", "tcc_e_corr"]
)
df.to_hdf("n2_dissociation_vqe.h5", key="df")
