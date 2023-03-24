# Proprietary and Confidential
# Covestro Deutschland AG, 2023
import numpy as np


def spinorb_from_spatial(oei, tei):
    # TODO: docs
    nso = 2 * oei.shape[0]
    soei = np.zeros(2 * (nso,))
    soei[::2, ::2] = oei  # (a|a)
    soei[1::2, 1::2] = oei  # (b|b)

    eri_of = np.zeros(4 * (nso,))
    eri_of[::2, ::2, ::2, ::2] = tei  # <aa|aa>
    eri_of[1::2, 1::2, 1::2, 1::2] = tei  # <bb|bb>
    eri_of[::2, 1::2, 1::2, ::2] = tei  # <ab|ba>
    eri_of[1::2, ::2, ::2, 1::2] = tei  # <ba|ab>
    return soei, eri_of
