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
    t2new += -0.5 * einsum("jiab,mnae,jibf->mnef", hf.oovv, t2, t2).antisymmetrise(2, 3)
    t2new += (
        1.0
        * einsum("jiab,niae,mjbf->mnef", hf.oovv, t2, t2).antisymmetrise(0, 1).antisymmetrise(2, 3)
        * 2
    )
    return t2new.evaluate()


def singles_residual_libcc(mp, t):
    # adapted from libcc, http://iopenshell.usc.edu/downloads/tensor/,
    # see also Table V of Levchenko and Krylov, J. Chem. Phys. 120, 175 (2004)
    hf = mp.reference_state
    t1 = t.ov
    t2 = t.oovv

    f2ov = ccsd_f2ov(mp, t)
    f3oo = ccsd_f3oo(mp, t, f2ov)
    f1vv = ccsd_f1vv(mp, t)

    t1new = (
        +1.0 * hf.fov
        + 1.0 * einsum("ad,id->ia", f1vv, t1)
        - 1.0 * einsum("li,la->ia", f3oo, t1)
        - 1.0 * einsum("icka,kc->ia", hf.ovov, t1)
        + 1.0 * einsum("ikac,kc->ia", t2, f2ov)
        + 0.5 * einsum("kacd,kicd->ia", hf.ovvv, t2)
        - 0.5 * einsum("klic,klac->ia", hf.ooov, t2)
    )
    return t1new.evaluate()


def ccsd_f3oo(mp, t, f2ov):
    hf = mp.reference_state
    t1 = t.ov
    t2 = t.oovv

    f3oo = (
        +1.0 * hf.foo
        + 1.0 * einsum("kc,ic->ki", f2ov, t1)
        + 0.5 * einsum("kjab,ijab->ki", hf.oovv, t2)
        + 1.0 * einsum("klic,lc->ki", hf.ooov, t1)
    )
    return f3oo.evaluate()


def ccsd_f2ov(mp, t):
    hf = mp.reference_state
    t1 = t.ov

    f2ov = +1.0 * hf.fov + 1.0 * einsum("jb,ijab->ia", t1, hf.oovv)
    return f2ov.evaluate()


def ccsd_f1vv(mp, t):
    hf = mp.reference_state
    t1 = t.ov
    t2 = t.oovv

    f1vv = (
        +1.0 * hf.fvv
        - 0.5 * einsum("klcd,klbd->bc", hf.oovv, t2)
        + 1.0 * einsum("kbdc,kd->bc", hf.ovvv, t1)
    )
    return f1vv.evaluate()


def doubles_residual_libcc(mp, t):
    # adapted from libcc, http://iopenshell.usc.edu/downloads/tensor/,
    # see also Table V of Levchenko and Krylov, J. Chem. Phys. 120, 175 (2004)
    hf = mp.reference_state
    t1 = t.ov
    t2 = t.oovv

    f2oo = ccsd_f2oo(mp, t)

    f1vv = ccsd_f1vv(mp, t)
    f2vv = ccsd_f2vv(mp, t, f1vv)

    tt_oovv = ccsd_tt_oovv(t)
    i4_oooo = ccsd_i4_oooo(mp, t, tt_oovv)

    i2a_ooov = ccsd_i2a_ooov(mp, t, i4_oooo, tt_oovv)
    i1a_ovov = ccsd_i1a_ovov(mp, t)

    t2new = (
        +1.0 * hf.oovv
        + 1.0
        * 2.0
        * (
            +1.0 * einsum("ijac,bc->ijab", t2, f2vv)
            - 1.0 * einsum("ijkb,ka->ijab", i2a_ooov, t1)
            + 1.0 * 2.0 * einsum("kbic,jkac->ijab", i1a_ovov, t2).antisymmetrise(0, 1)
        ).antisymmetrise(2, 3)
        + 1.0
        * 2.0
        * (
            +1.0 * einsum("jcba,ic->ijab", hf.ovvv, t1) - 1.0 * einsum("ikab,jk->ijab", t2, f2oo)
        ).antisymmetrise(0, 1)
        + 0.5 * einsum("abcd,ijcd->ijab", hf.vvvv, tt_oovv)
        + 0.5 * einsum("klab,ijkl->ijab", t2, i4_oooo)
    )
    return t2new.evaluate()


def ccsd_f2oo(mp, t):
    hf = mp.reference_state
    t1 = t.ov
    t2 = t.oovv

    f2oo = (
        +1.0 * hf.foo
        + 1.0 * einsum("ja,ia->ij", hf.fov, t1)
        + 1.0 * einsum("jkia,ka->ij", hf.ooov, t1)
        + 1.0 * einsum("jkab,ia,kb->ij", hf.oovv, t1, t1)
        + 0.5 * einsum("jkab,ikab->ij", hf.oovv, t2)
    )
    return f2oo.evaluate()


def ccsd_f2vv(mp, t, f1vv):
    hf = mp.reference_state
    t1 = t.ov

    f2vv = (
        +1.0 * f1vv
        - 1.0 * einsum("kc,kb->bc", hf.fov, t1)
        - 1.0 * einsum("klcd,kb,ld->bc", hf.oovv, t1, t1)
    )
    return f2vv.evaluate()


def ccsd_i1a_ovov(mp, t):
    hf = mp.reference_state
    t1 = t.ov
    t2 = t.oovv

    i1a_ovov = (
        +1.0 * hf.ovov
        - 1.0 * einsum("iabc,jc->iajb", hf.ovvv, t1)
        - 1.0 * einsum("ikjb,ka->iajb", hf.ooov, t1)
        - 0.5 * einsum("jkca,ikcb->iajb", t2 + 2.0 * einsum("jc,ka->jkca", t1, t1), hf.oovv)
    )
    return i1a_ovov.evaluate()


def ccsd_i2a_ooov(mp, t, i4_oooo, tt_oovv):
    hf = mp.reference_state
    t1 = t.ov

    i2a_ooov = (
        +1.0 * hf.ooov
        - 0.5 * einsum("ijkl,lb->ijkb", i4_oooo, t1)
        + 0.5 * einsum("ijcd,kbcd->ijkb", tt_oovv, hf.ovvv)
        + 1.0 * 2.0 * einsum("kbic,jc->ijkb", hf.ovov, t1).antisymmetrise(0, 1)
    )
    return i2a_ooov.evaluate()


def ccsd_i4_oooo(mp, t, tt_oovv):
    hf = mp.reference_state
    t1 = t.ov

    i4_oooo = (
        +1.0 * hf.oooo
        + 0.5 * einsum("klab,ijab->ijkl", hf.oovv, tt_oovv)
        + 1.0 * 2.0 * einsum("klia,ja->ijkl", hf.ooov, t1).antisymmetrise(0, 1)
    )
    return i4_oooo.evaluate()


def ccsd_tt_oovv(t):
    t1 = t.ov
    t2 = t.oovv

    tt_oovv = t2 + 4.0 * einsum("ia,jb->ijab", 0.5 * t1, t1).antisymmetrise(0, 1).antisymmetrise(
        2, 3
    )
    return tt_oovv.evaluate()
