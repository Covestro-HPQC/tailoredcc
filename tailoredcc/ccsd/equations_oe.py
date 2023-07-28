# Proprietary and Confidential
# Covestro Deutschland AG, 2023

from functools import partial

from jax import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp


def einsum(*args, optimize=None, **kwargs):
    if optimize is None or optimize is False:
        result = jnp.einsum(*args, **kwargs)
    elif optimize is True:
        result = jnp.einsum(*args, optimize=True, **kwargs)
    elif isinstance(optimize, list):
        result = jnp.einsum(*args, **kwargs, optimize=optimize[1:])
    else:
        raise NotImplementedError(f"Handing of optimize={optimize} in jax mode not yet implemented")
    return result


def ccsd_energy_correlation_oe(t1, t2, f, g, o, v):
    """
    < 0 | e(-T) H e(T) | 0> :
    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    # 	  1.0000 f(i,i)
    # energy = 1.0 * einsum("ii", f[o, o])

    # 	  1.0000 f(i,a)*t1(a,i)
    energy = 0.0
    energy += 1.0 * einsum("ia,ai", f[o, v], t1)

    # 	 -0.5000 <j,i||j,i>
    # energy += -0.5 * einsum("jiji", g[o, o, o, o])

    # 	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy += 0.25 * einsum("jiab,abji", g[o, o, v, v], t2)

    # 	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += -0.5 * einsum(
        "jiab,ai,bj", g[o, o, v, v], t1, t1, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    return energy


@partial(jax.jit, static_argnames=["o1", "o2", "v1", "v2"])
def ccsd_energy_oe(t1, t2, f, g, o1, o2, v1, v2):
    """
    < 0 | e(-T) H e(T) | 0> :
    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    o = slice(o1, o2)
    v = slice(v1, v2)
    # 	  1.0000 f(i,i)
    energy = 1.0 * einsum("ii", f[o, o])

    # 	  1.0000 f(i,a)*t1(a,i)
    energy += 1.0 * einsum("ia,ai", f[o, v], t1)

    # 	 -0.5000 <j,i||j,i>
    energy += -0.5 * einsum("jiji", g[o, o, o, o])

    # 	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy += 0.25 * einsum("jiab,abji", g[o, o, v, v], t2)

    # 	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += -0.5 * einsum(
        "jiab,ai,bj", g[o, o, v, v], t1, t1, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    return energy


@partial(jax.jit, static_argnames=["o1", "o2", "v1", "v2"])
def singles_residual_oe(t1, t2, f, g, o1, o2, v1, v2):
    """
    < 0 | m* e e(-T) H e(T) | 0>
    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    o = slice(o1, o2)
    v = slice(v1, v2)
    # 	  1.0000 f(e,m)
    singles_res = 1.0 * einsum("em->em", f[v, o])

    # 	 -1.0000 f(i,m)*t1(e,i)
    singles_res += -1.0 * einsum("im,ei->em", f[o, o], t1)

    # 	  1.0000 f(e,a)*t1(a,m)
    singles_res += 1.0 * einsum("ea,am->em", f[v, v], t1)

    # 	 -1.0000 f(i,a)*t2(a,e,m,i)
    singles_res += -1.0 * einsum("ia,aemi->em", f[o, v], t2)

    # 	 -1.0000 f(i,a)*t1(a,m)*t1(e,i)
    singles_res += -1.0 * einsum(
        "ia,am,ei->em", f[o, v], t1, t1, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    # 	  1.0000 <i,e||a,m>*t1(a,i)
    singles_res += 1.0 * einsum("ieam,ai->em", g[o, v, v, o], t1)

    # 	 -0.5000 <j,i||a,m>*t2(a,e,j,i)
    singles_res += -0.5 * einsum("jiam,aeji->em", g[o, o, v, o], t2)

    # 	 -0.5000 <i,e||a,b>*t2(a,b,m,i)
    singles_res += -0.5 * einsum("ieab,abmi->em", g[o, v, v, v], t2)

    # 	  1.0000 <j,i||a,m>*t1(a,i)*t1(e,j)
    singles_res += 1.0 * einsum(
        "jiam,ai,ej->em", g[o, o, v, o], t1, t1, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    # 	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    singles_res += 1.0 * einsum(
        "ieab,ai,bm->em", g[o, v, v, v], t1, t1, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    # 	  1.0000 <j,i||a,b>*t1(a,i)*t2(b,e,m,j)
    singles_res += 1.0 * einsum(
        "jiab,ai,bemj->em", g[o, o, v, v], t1, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    # 	  0.5000 <j,i||a,b>*t1(a,m)*t2(b,e,j,i)
    singles_res += 0.5 * einsum(
        "jiab,am,beji->em", g[o, o, v, v], t1, t2, optimize=["einsum_path", (0, 2), (0, 1)]
    )

    # 	  0.5000 <j,i||a,b>*t1(e,i)*t2(a,b,m,j)
    singles_res += 0.5 * einsum(
        "jiab,ei,abmj->em", g[o, o, v, v], t1, t2, optimize=["einsum_path", (0, 2), (0, 1)]
    )

    # 	  1.0000 <j,i||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    singles_res += 1.0 * einsum(
        "jiab,ai,bm,ej->em",
        g[o, o, v, v],
        t1,
        t1,
        t1,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    return singles_res


@partial(jax.jit, static_argnames=["o1", "o2", "v1", "v2"])
def doubles_residual_oe(t1, t2, f, g, o1, o2, v1, v2):
    """
     < 0 | m* n* f e e(-T) H e(T) | 0>
    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    o = slice(o1, o2)
    v = slice(v1, v2)
    # 	 -1.0000 P(m,n)f(i,n)*t2(e,f,m,i)
    contracted_intermediate = -1.0 * einsum("in,efmi->efmn", f[o, o], t2)
    doubles_res = 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->efnm", contracted_intermediate
    )

    # 	  1.0000 P(e,f)f(e,a)*t2(a,f,m,n)
    contracted_intermediate = 1.0 * einsum("ea,afmn->efmn", f[v, v], t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->femn", contracted_intermediate
    )

    # 	 -1.0000 P(m,n)f(i,a)*t1(a,n)*t2(e,f,m,i)
    contracted_intermediate = -1.0 * einsum(
        "ia,an,efmi->efmn", f[o, v], t1, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->efnm", contracted_intermediate
    )

    # 	 -1.0000 P(e,f)f(i,a)*t1(e,i)*t2(a,f,m,n)
    contracted_intermediate = -1.0 * einsum(
        "ia,ei,afmn->efmn", f[o, v], t1, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->femn", contracted_intermediate
    )

    # 	  1.0000 <e,f||m,n>
    doubles_res += 1.0 * einsum("efmn->efmn", g[v, v, o, o])

    # 	  1.0000 P(e,f)<i,e||m,n>*t1(f,i)
    contracted_intermediate = 1.0 * einsum("iemn,fi->efmn", g[o, v, o, o], t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->femn", contracted_intermediate
    )

    # 	  1.0000 P(m,n)<e,f||a,n>*t1(a,m)
    contracted_intermediate = 1.0 * einsum("efan,am->efmn", g[v, v, v, o], t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->efnm", contracted_intermediate
    )

    # 	  0.5000 <j,i||m,n>*t2(e,f,j,i)
    doubles_res += 0.5 * einsum("jimn,efji->efmn", g[o, o, o, o], t2)

    # 	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t2(a,f,m,i)
    contracted_intermediate = 1.0 * einsum("iean,afmi->efmn", g[o, v, v, o], t2)
    doubles_res += (
        1.00000 * contracted_intermediate
        + -1.00000 * einsum("efmn->efnm", contracted_intermediate)
        + -1.00000 * einsum("efmn->femn", contracted_intermediate)
        + 1.00000 * einsum("efmn->fenm", contracted_intermediate)
    )

    # 	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    doubles_res += 0.5 * einsum("efab,abmn->efmn", g[v, v, v, v], t2)

    # 	 -1.0000 <j,i||m,n>*t1(e,i)*t1(f,j)
    doubles_res += -1.0 * einsum(
        "jimn,ei,fj->efmn", g[o, o, o, o], t1, t1, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    # 	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t1(a,m)*t1(f,i)
    contracted_intermediate = 1.0 * einsum(
        "iean,am,fi->efmn", g[o, v, v, o], t1, t1, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += (
        1.00000 * contracted_intermediate
        + -1.00000 * einsum("efmn->efnm", contracted_intermediate)
        + -1.00000 * einsum("efmn->femn", contracted_intermediate)
        + 1.00000 * einsum("efmn->fenm", contracted_intermediate)
    )

    # 	 -1.0000 <e,f||a,b>*t1(a,n)*t1(b,m)
    doubles_res += -1.0 * einsum(
        "efab,an,bm->efmn", g[v, v, v, v], t1, t1, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    # 	  1.0000 P(m,n)<j,i||a,n>*t1(a,i)*t2(e,f,m,j)
    contracted_intermediate = 1.0 * einsum(
        "jian,ai,efmj->efmn", g[o, o, v, o], t1, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->efnm", contracted_intermediate
    )

    # 	  0.5000 P(m,n)<j,i||a,n>*t1(a,m)*t2(e,f,j,i)
    contracted_intermediate = 0.5 * einsum(
        "jian,am,efji->efmn", g[o, o, v, o], t1, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->efnm", contracted_intermediate
    )

    # 	 -1.0000 P(m,n)*P(e,f)<j,i||a,n>*t1(e,i)*t2(a,f,m,j)
    contracted_intermediate = -1.0 * einsum(
        "jian,ei,afmj->efmn", g[o, o, v, o], t1, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += (
        1.00000 * contracted_intermediate
        + -1.00000 * einsum("efmn->efnm", contracted_intermediate)
        + -1.00000 * einsum("efmn->femn", contracted_intermediate)
        + 1.00000 * einsum("efmn->fenm", contracted_intermediate)
    )

    # 	  1.0000 P(e,f)<i,e||a,b>*t1(a,i)*t2(b,f,m,n)
    contracted_intermediate = 1.0 * einsum(
        "ieab,ai,bfmn->efmn", g[o, v, v, v], t1, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->femn", contracted_intermediate
    )

    # 	 -1.0000 P(m,n)*P(e,f)<i,e||a,b>*t1(a,n)*t2(b,f,m,i)
    contracted_intermediate = -1.0 * einsum(
        "ieab,an,bfmi->efmn", g[o, v, v, v], t1, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += (
        1.00000 * contracted_intermediate
        + -1.00000 * einsum("efmn->efnm", contracted_intermediate)
        + -1.00000 * einsum("efmn->femn", contracted_intermediate)
        + 1.00000 * einsum("efmn->fenm", contracted_intermediate)
    )

    # 	  0.5000 P(e,f)<i,e||a,b>*t1(f,i)*t2(a,b,m,n)
    contracted_intermediate = 0.5 * einsum(
        "ieab,fi,abmn->efmn", g[o, v, v, v], t1, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->femn", contracted_intermediate
    )

    # 	 -0.5000 P(m,n)<j,i||a,b>*t2(a,b,n,i)*t2(e,f,m,j)
    contracted_intermediate = -0.5 * einsum(
        "jiab,abni,efmj->efmn", g[o, o, v, v], t2, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->efnm", contracted_intermediate
    )

    # 	  0.2500 <j,i||a,b>*t2(a,b,m,n)*t2(e,f,j,i)
    doubles_res += 0.25 * einsum(
        "jiab,abmn,efji->efmn", g[o, o, v, v], t2, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    # 	 -0.5000 <j,i||a,b>*t2(a,e,j,i)*t2(b,f,m,n)
    doubles_res += -0.5 * einsum(
        "jiab,aeji,bfmn->efmn", g[o, o, v, v], t2, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )

    # 	  1.0000 P(m,n)<j,i||a,b>*t2(a,e,n,i)*t2(b,f,m,j)
    contracted_intermediate = 1.0 * einsum(
        "jiab,aeni,bfmj->efmn", g[o, o, v, v], t2, t2, optimize=["einsum_path", (0, 1), (0, 1)]
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->efnm", contracted_intermediate
    )

    # 	 -0.5000 <j,i||a,b>*t2(a,e,m,n)*t2(b,f,j,i)
    doubles_res += -0.5 * einsum(
        "jiab,aemn,bfji->efmn", g[o, o, v, v], t2, t2, optimize=["einsum_path", (0, 2), (0, 1)]
    )

    # 	 -1.0000 P(m,n)<j,i||a,n>*t1(a,m)*t1(e,i)*t1(f,j)
    contracted_intermediate = -1.0 * einsum(
        "jian,am,ei,fj->efmn",
        g[o, o, v, o],
        t1,
        t1,
        t1,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->efnm", contracted_intermediate
    )

    # 	 -1.0000 P(e,f)<i,e||a,b>*t1(a,n)*t1(b,m)*t1(f,i)
    contracted_intermediate = -1.0 * einsum(
        "ieab,an,bm,fi->efmn",
        g[o, v, v, v],
        t1,
        t1,
        t1,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->femn", contracted_intermediate
    )

    # 	  1.0000 P(m,n)<j,i||a,b>*t1(a,i)*t1(b,n)*t2(e,f,m,j)
    contracted_intermediate = 1.0 * einsum(
        "jiab,ai,bn,efmj->efmn",
        g[o, o, v, v],
        t1,
        t1,
        t2,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->efnm", contracted_intermediate
    )

    # 	  1.0000 P(e,f)<j,i||a,b>*t1(a,i)*t1(e,j)*t2(b,f,m,n)
    contracted_intermediate = 1.0 * einsum(
        "jiab,ai,ej,bfmn->efmn",
        g[o, o, v, v],
        t1,
        t1,
        t2,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        "efmn->femn", contracted_intermediate
    )

    # 	 -0.5000 <j,i||a,b>*t1(a,n)*t1(b,m)*t2(e,f,j,i)
    doubles_res += -0.5 * einsum(
        "jiab,an,bm,efji->efmn",
        g[o, o, v, v],
        t1,
        t1,
        t2,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )

    # 	  1.0000 P(m,n)*P(e,f)<j,i||a,b>*t1(a,n)*t1(e,i)*t2(b,f,m,j)
    contracted_intermediate = 1.0 * einsum(
        "jiab,an,ei,bfmj->efmn",
        g[o, o, v, v],
        t1,
        t1,
        t2,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    doubles_res += (
        1.00000 * contracted_intermediate
        + -1.00000 * einsum("efmn->efnm", contracted_intermediate)
        + -1.00000 * einsum("efmn->femn", contracted_intermediate)
        + 1.00000 * einsum("efmn->fenm", contracted_intermediate)
    )

    # 	 -0.5000 <j,i||a,b>*t1(e,i)*t1(f,j)*t2(a,b,m,n)
    doubles_res += -0.5 * einsum(
        "jiab,ei,fj,abmn->efmn",
        g[o, o, v, v],
        t1,
        t1,
        t2,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )

    # 	  1.0000 <j,i||a,b>*t1(a,n)*t1(b,m)*t1(e,i)*t1(f,j)
    doubles_res += 1.0 * einsum(
        "jiab,an,bm,ei,fj->efmn",
        g[o, o, v, v],
        t1,
        t1,
        t1,
        t1,
        optimize=["einsum_path", (0, 1), (0, 3), (0, 2), (0, 1)],
    )

    return doubles_res
