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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def civec_scatter(coeffs: dict, ax, s=150, excilevels=None):
    assert "exact" in coeffs
    ci_coeffs = coeffs["exact"].flatten()
    sort_idx = np.argsort(np.abs(ci_coeffs))[::-1]
    # sns.set_theme(context="talk", palette="colorblind", font_scale=1.2, style="ticks")
    # fig, ax = plt.subplots(1, 1)
    # fig.set_size_inches(18, 9)
    sorted_dict = {k: np.abs(coeff.flatten())[sort_idx] for k, coeff in coeffs.items()}
    if excilevels is not None:
        excilevels = excilevels.flatten()[sort_idx]
    else:
        excilevels = np.zeros_like(ci_coeffs)
    keys = list(sorted_dict.keys())
    vals = list(sorted_dict.values())
    df = pd.DataFrame(
        columns=["index", *keys, "excitation_level"],
        data=np.vstack([np.arange(ci_coeffs.size), *vals, excilevels]).T,
    )
    # df['coeffs_shadow'] = np.abs(civec.flatten()[sort_idx])
    dfm = pd.melt(
        df,
        id_vars=["index", "excitation_level"],
        value_vars=keys,
        value_name="coeff",
        var_name="method",
    )
    sns.scatterplot(
        x="index",
        y="coeff",
        hue="excitation_level",
        style="method",
        ax=ax,
        data=dfm,
        palette="Set2",
        s=s,
    )
    ax.set_yscale("log")
    # ax.set_title(f"{e_tot}")
    ax.set_ylim((1e-12, 1.1))
    plt.tight_layout()
    return ax


def sign_and_zero_errors(ref, civec, threshold=1e-6):
    nonzero_mask = np.abs(ref) > threshold
    zero_mask = ~nonzero_mask
    # if _nonzeros is not None:
    #     assert np.sum(nonzero_mask) == _nonzeros
    # else:
    #     _nonzeros = np.sum(nonzero_mask)
    signs_exact = np.sign(ref)
    signs_shadow = np.sign(civec)
    sign_errors = signs_exact != signs_shadow

    zeros_shadow = np.abs(civec) < threshold
    zero_errors = zeros_shadow != zero_mask
    num_zero_errors = np.sum(zero_errors)
    sign_errors_nonzero = sign_errors[nonzero_mask]
    num_sign_errors = np.sum(sign_errors_nonzero)
    return num_sign_errors, sign_errors, nonzero_mask, num_zero_errors, zero_errors
