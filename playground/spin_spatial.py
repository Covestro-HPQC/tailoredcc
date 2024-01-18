# %%
import operator
from collections import defaultdict
from itertools import combinations_with_replacement, permutations, product

import numpy as np
from tqdm import tqdm


def get_canonical_block(block, canonical_blocks):
    canonical_block = "".join(sorted(block))
    assert tuple(canonical_block) in canonical_blocks
    perm = list(permutations(range(len(block))))
    # NOTE: need the permutation of the canonical block to the non-canonical one,
    # so that we know how to transpose the canonical block and write it to the output tensor
    block_tuple = tuple(block)
    for p in perm:
        if operator.itemgetter(*p)(canonical_block) == block_tuple:
            return canonical_block, list(p)
    raise ValueError(f"No canonical block transposition found for '{block}'.")


def compute_parity(perm):
    inversions = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def spinorb_to_spatial(tensor, exci_level, nocc_a, nocc_b, nvirt_a, nvirt_b):
    nocc = nocc_a + nocc_b
    nvirt = nvirt_a + nvirt_b
    shape = exci_level * (nocc,) + exci_level * (nvirt,)
    assert tensor.shape == shape

    spatial_strings = list(combinations_with_replacement("ab", exci_level))
    slices = {"a": slice(0, None, 2), "b": slice(1, None, 2)}
    ret = {}
    for canblock in spatial_strings:
        blockstr = "".join(canblock)
        both_blocks = canblock + canblock
        tmpslices = tuple(slices[b] for b in both_blocks)
        ret[blockstr] = tensor[tmpslices]
    return ret


def spatial_to_spinorb(tensors_dict, exci_level, nocc_a, nocc_b, nvirt_a, nvirt_b):
    assert nocc_a == nocc_b
    assert nvirt_a == nvirt_b
    nocc = nocc_a + nocc_b
    nvirt = nvirt_a + nvirt_b

    orbspin = np.zeros(nocc + nvirt, dtype=int)
    orbspin[1::2] = 1

    occa = np.where(orbspin[:nocc] == 0)[0]
    occb = np.where(orbspin[:nocc] == 1)[0]
    virta = np.where(orbspin[nocc:] == 0)[0]
    virtb = np.where(orbspin[nocc:] == 1)[0]

    block_to_indices = {
        "a": (occa, virta),
        "b": (occb, virtb),
    }

    # all possible alpha/beta combinations (also non-canonical ones)
    spinblocks = list(product("ab", repeat=exci_level))
    spatial_strings = list(combinations_with_replacement("ab", exci_level))
    slices_spinorb = defaultdict(dict)
    for block in spinblocks:
        blockstr = "".join(block)
        slices_spinorb["occ"][blockstr] = [block_to_indices[b][0] for b in block]
        slices_spinorb["virt"][blockstr] = [block_to_indices[b][1] for b in block]

    ooo = slices_spinorb["occ"]
    vvv = slices_spinorb["virt"]

    t_out = np.zeros(exci_level * (nocc,) + exci_level * (nvirt,), dtype=float)
    for comb in product(list(spinblocks), repeat=2):
        ospin, vspin = comb
        ostr = "".join(ospin)
        vstr = "".join(vspin)
        label = f"{ostr}|{vstr}"
        if ospin.count("a") != vspin.count("a") or ospin.count("b") != vspin.count("b"):
            # spin block not allowed
            continue

        ocan, operm = get_canonical_block(ostr, spatial_strings)
        vcan, vperm = get_canonical_block(vstr, spatial_strings)

        t_get = tensors_dict.get(ocan, tensors_dict.get(vcan, None))
        if t_get is None:
            raise ValueError(f"No tensor found: {comb}")

        total_perm = np.concatenate([np.array(operm), np.array(vperm) + len(ospin)])
        sign = compute_parity(operm) * compute_parity(vperm)
        # print(f"Current block: {label}")
        # if ocan != ostr or vcan != vstr:
        #     print(f"Canonical form:", ocan, vcan, operm, vperm)
        #     print("=> Permutation", total_perm, sign)
        t_out[np.ix_(*ooo[ostr], *vvv[vstr])] = sign * t_get.transpose(*total_perm)

    return t_out


# %%
# nocc_a = 4
# nocc_b = 4
# nvirt_a = 5
# nvirt_b = 5

# np.random.seed(42)
# t2ab = np.random.randn(nocc_a, nocc_b, nvirt_a, nvirt_b)

# t2aa = np.random.randn(nocc_a, nocc_a, nvirt_a, nvirt_a)
# t2bb = np.random.randn(nocc_b, nocc_b, nvirt_b, nvirt_b)

# t2aa = 0.5 * (t2aa - t2aa.transpose(1, 0, 2, 3))
# t2aa = 0.5 * (t2aa - t2aa.transpose(0, 1, 3, 2))

# t2bb = 0.5 * (t2bb - t2bb.transpose(1, 0, 2, 3))
# t2bb = 0.5 * (t2bb - t2bb.transpose(0, 1, 3, 2))

# t_ref = spatial2spin([t2aa, t2ab, t2bb])
# assert_spinorb_antisymmetric(t_ref)

# t_lookup = {
#     "aa": t2aa,
#     "ab": t2ab,
#     "bb": t2bb,
# }

# t_out = spatial_to_spinorb(t_lookup, 2, nocc_a, nocc_b, nvirt_a, nvirt_b)
# np.testing.assert_allclose(t_ref, t_out, atol=1e-14, rtol=0)
# assert_spinorb_antisymmetric(t_out)

# t_spatial = spinorb_to_spatial(t_out, 2, nocc_a, nocc_b, nvirt_a, nvirt_b)

# for bb in t_spatial:
#     assert bb in t_lookup
#     np.testing.assert_allclose(t_spatial[bb], t_lookup[bb], atol=1e-14, rtol=0)

nocc_a = 4
nocc_b = 4
nvirt_a = 5
nvirt_b = 5

nocc = nocc_a + nocc_b
nvirt = nvirt_a + nvirt_b
exci = 4
tt = np.random.randn(*(exci * (nocc,) + exci * (nvirt,)))
print(tt.shape, tt.size)

slices = {"a": slice(0, None, 2), "b": slice(1, None, 2)}

print("==> Setting up anti-symmetry <==")

perms = list(permutations(range(exci)))
parities = {p: compute_parity(p) for p in perms}
for _ in range(3):
    for p1 in tqdm(perms):
        sign1 = parities[p1]
        for p2 in perms:
            sign2 = parities[p2]
            perm_total = np.concatenate([np.array(p1), exci + np.array(p2)])
            tt = 0.5 * (tt + sign1 * sign2 * tt.transpose(perm_total))


for p1 in permutations(range(exci)):
    sign1 = compute_parity(p1)
    for p2 in permutations(range(exci)):
        sign2 = compute_parity(p2)
        perm_total = np.concatenate([np.array(p1), exci + np.array(p2)])
        # print(perm_total, sign1 * sign2)
        np.testing.assert_allclose(
            tt.ravel(),
            (sign1 * sign2 * tt.transpose(perm_total)).flatten(),
            err_msg=f"{perm_total}",
            atol=1e-8,
        )
        # np.testing.assert_allclose(t_out, sign1 * sign2 * t_out.transpose(perm_total), err_msg=f"{perm_total}", atol=1e-8)


spinblocks = list(product("ab", repeat=exci))
for comb in product(list(spinblocks), repeat=2):
    ospin, vspin = comb
    ostr = "".join(ospin)
    vstr = "".join(vspin)
    if ospin.count("a") != vspin.count("a") or ospin.count("b") != vspin.count("b"):
        tt[tuple(slices[ii] for ii in ostr + vstr)] = 0.0
t_spatial = spinorb_to_spatial(tt, exci, nocc_a, nocc_b, nvirt_a, nvirt_b)

t_out = spatial_to_spinorb(t_spatial, exci, nocc_a, nocc_b, nvirt_a, nvirt_b)

for comb in product(list(spinblocks), repeat=2):
    ospin, vspin = comb
    ostr = "".join(ospin)
    vstr = "".join(vspin)
    viewslice = tuple(slices[ii] for ii in ostr + vstr)
    view = t_out[viewslice]
    label = f"{ostr}|{vstr}"
    if ospin.count("a") != vspin.count("a") or ospin.count("b") != vspin.count("b"):
        np.testing.assert_allclose(view, np.zeros_like(view), atol=1e-14, rtol=0)
        np.testing.assert_allclose(tt[viewslice], np.zeros_like(view), atol=1e-14, rtol=0)
    else:
        try:
            np.testing.assert_allclose(view, tt[viewslice], atol=1e-10, rtol=0, err_msg=label)
        except AssertionError as e:
            print(label, "FAILED")
            # print(label, "FAILED", e)
            raise ValueError()
        else:
            print(label, "OK")


np.testing.assert_allclose(tt, t_out, atol=1e-10, rtol=0)

t_rnd = {}
for blk in t_spatial:
    t_rnd[blk] = np.random.randn(*t_spatial[blk].shape)

t_out = spatial_to_spinorb(t_rnd, exci, nocc_a, nocc_b, nvirt_a, nvirt_b)
t_spatial = spinorb_to_spatial(t_out, exci, nocc_a, nocc_b, nvirt_a, nvirt_b)

for bb in t_spatial:
    assert bb in t_rnd
    np.testing.assert_allclose(t_spatial[bb], t_rnd[bb])
