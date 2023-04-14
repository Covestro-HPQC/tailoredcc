# Proprietary and Confidential
# Covestro Deutschland AG, 2023

import libadcc
from adcc.functions import einsum


def ccsd_energy_correlation_adcc(mp, t):
    hf = mp.reference_state
    ret = hf.fov.dot(t.ov)
    ret += 0.25 * hf.oovv.dot(t.oovv + 2.0 * einsum("ia,jb->ijab", t.ov, t.ov))
    return ret


def ccsd_energy_adcc(mp, t):
    hf = mp.reference_state
    # e = einsum("ii", hf.foo) - 0.5 * einsum("jiji->", hf.oooo)
    e = libadcc.trace("ii", hf.foo) - 0.5 * libadcc.trace("jiji", hf.oooo)
    e += ccsd_energy_correlation_adcc(mp, t)
    return e


def singles_residual_adcc(mp, t):
    hf = mp.reference_state
    t1 = t.ov
    t2 = t.oovv

    t1new = 1.0 * einsum("em->me", hf.fvo)
    t1new += -1.0 * einsum("im,ie->me", hf.foo, t1)
    t1new += 1.0 * einsum("ea,ma->me", hf.fvv, t1)
    t1new += -1.0 * einsum("ia,ma,ie->me", hf.fov, t1, t1)
    t1new += 1.0 * einsum("iema,ia->me", -hf.ovov, t1)
    t1new += 1.0 * einsum("jima,ia,je->me", -hf.ooov, t1, t1)
    t1new += 1.0 * einsum("ieab,ia,mb->me", hf.ovvv, t1, t1)
    t1new += 1.0 * einsum("jiab,ia,mb,je->me", hf.oovv, t1, t1, t1)
    t1new += -1.0 * einsum("ia,miae->me", hf.fov, t2)
    t1new += -0.5 * einsum("jima,jiae->me", -hf.ooov, t2)
    t1new += -0.5 * einsum("ieab,miab->me", hf.ovvv, t2)
    t1new += -0.5 * einsum("jiab,je,miab->me", hf.oovv, t1, t2)
    t1new += -0.5 * einsum("jiab,mb,jiae->me", hf.oovv, t1, t2)
    t1new += 1.0 * einsum("jiab,jb,miae->me", hf.oovv, t1, t2)
    return t1new.evaluate()


def doubles_residual_adcc(mp, t):
    hf = mp.reference_state
    t1 = t.ov
    t2 = t.oovv

    t2new = hf.oovv
    t2new += 1.0 * einsum("mnie,if->mnef", hf.ooov, t1).antisymmetrise(2, 3) * 2
    t2new.evaluate()
    t2new += 1.0 * einsum("naef,ma->mnef", -hf.ovvv, t1).antisymmetrise(0, 1) * 2
    t2new += -1.0 * einsum("jimn,ie,jf->mnef", hf.oooo, t1, t1).antisymmetrise(2, 3)
    t2new += (
        1.0
        * einsum("iena,ma,if->mnef", -hf.ovov, t1, t1).antisymmetrise(0, 1).antisymmetrise(2, 3)
        * 4
    )
    t2new += -1.0 * einsum("efab,na,mb->mnef", hf.vvvv, t1, t1).antisymmetrise(0, 1)
    t2new += (
        -1.0
        * einsum("jina,ma,ie,jf->mnef", -hf.ooov, t1, t1, t1)
        .antisymmetrise(0, 1)
        .antisymmetrise(2, 3)
        * 2
    )
    t2new += (
        -1.0
        * einsum("ieab,na,mb,if->mnef", hf.ovvv, t1, t1, t1)
        .antisymmetrise(0, 1)
        .antisymmetrise(2, 3)
        * 2
    )
    t2new += 1.0 * einsum("jiab,na,mb,ie,jf->mnef", hf.oovv, t1, t1, t1, t1).antisymmetrise(
        0, 1
    ).antisymmetrise(2, 3)
    t2new += -1.0 * einsum("in,mief->mnef", hf.foo, t2).antisymmetrise(0, 1) * 2
    t2new += 1.0 * einsum("ea,mnaf->mnef", hf.fvv, t2).antisymmetrise(2, 3) * 2
    t2new += 1.0 * einsum("ia,if,mnae->mnef", hf.fov, t1, t2).antisymmetrise(2, 3) * 2
    t2new += 1.0 * einsum("ia,ma,nief->mnef", hf.fov, t1, t2).antisymmetrise(0, 1) * 2
    t2new += 0.5 * einsum("jimn,jief->mnef", hf.oooo, t2)
    t2new += (
        1.0 * einsum("iena,miaf->mnef", -hf.ovov, t2).antisymmetrise(0, 1).antisymmetrise(2, 3) * 4
    )
    t2new += 0.5 * einsum("efab,mnab->mnef", hf.vvvv, t2)
    t2new += (
        -1.0
        * einsum("jina,jf,miae->mnef", -hf.ooov, t1, t2).antisymmetrise(0, 1).antisymmetrise(2, 3)
        * 4
    )
    t2new += 0.5 * einsum("jina,ma,jief->mnef", -hf.ooov, t1, t2).antisymmetrise(0, 1) * 2
    t2new += -1.0 * einsum("jina,ja,mief->mnef", -hf.ooov, t1, t2).antisymmetrise(0, 1) * 2
    t2new += 0.5 * einsum("ieab,if,mnab->mnef", hf.ovvv, t1, t2).antisymmetrise(2, 3) * 2
    t2new += (
        -1.0
        * einsum("ieab,mb,niaf->mnef", hf.ovvv, t1, t2).antisymmetrise(0, 1).antisymmetrise(2, 3)
        * 4
    )
    t2new += -1.0 * einsum("ieab,ib,mnaf->mnef", hf.ovvv, t1, t2).antisymmetrise(2, 3) * 2
    t2new += -0.5 * einsum("jiab,ie,jf,mnab->mnef", hf.oovv, t1, t1, t2).antisymmetrise(2, 3)
    t2new += (
        1.0
        * einsum("jiab,mb,jf,niae->mnef", hf.oovv, t1, t1, t2)
        .antisymmetrise(0, 1)
        .antisymmetrise(2, 3)
        * 4
    )
    t2new += 1.0 * einsum("jiab,ib,jf,mnae->mnef", hf.oovv, t1, t1, t2).antisymmetrise(2, 3) * 2
    t2new += -0.5 * einsum("jiab,na,mb,jief->mnef", hf.oovv, t1, t1, t2).antisymmetrise(0, 1)
    t2new += 1.0 * einsum("jiab,ja,mb,nief->mnef", hf.oovv, t1, t1, t2).antisymmetrise(0, 1) * 2
    t2new += -0.5 * einsum("jiab,niab,mjef->mnef", hf.oovv, t2, t2).antisymmetrise(0, 1) * 2
    t2new += 0.25 * einsum("jiab,mnab,jief->mnef", hf.oovv, t2, t2)
    t2new += -0.5 * einsum("jiab,jiae,mnbf->mnef", hf.oovv, t2, t2).antisymmetrise(2, 3)
    t2new += (
        1.0
        * einsum("jiab,niae,mjbf->mnef", hf.oovv, t2, t2).antisymmetrise(0, 1).antisymmetrise(2, 3)
        * 2
    )
    t2new += -0.5 * einsum("jiab,mnae,jibf->mnef", hf.oovv, t2, t2).antisymmetrise(2, 3)
    return t2new.evaluate()
