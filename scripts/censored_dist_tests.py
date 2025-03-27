"""
Here, we implement some naive and rudimentary tests for testing if two sets of samples come from the same distribution, when the samples are both right-censored.

We follow the general structure of Chernobai et al. (2015), but adapt for a simple Type I right-censoring.

A. Chernobai, S. T. Rachev, F.J. Fabozzi (2015) "Composite Goodness-of-Fit Tests for Left-Truncated Loss Samples" in "Handbook of Financial Econometrics and Statistics" Springer Science+Business Media, New York.
"""

import numpy as np
from scipy import stats


def ks_2samp_rc(data1, data2, threshold1, threshold2, mode='auto'):
    """implements the two-sided kolmogorov-smirnov test for two right-censored distributions"""

    if mode not in ('atuo', 'exact', 'asymp'):
        raise ValueError(f"Invalid mode: {mode}")
    MAX_AUTO_N = 10_000

    data1 = np.sort(data1)
    data2 = np.sort(data2)

    n1 = data1.shape[0]
    n2 = data2.shape[0]

    if min(n1, n2) == 0:
        raise ValueError("Sample data must not be empty")

    v1 = np.isfinite(data1)
    v2 = np.isfinite(data2)
    vcat = np.concatenate([v1, v2])

    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    cdfdiffs = cdf1[vcat] - cdf2[vcat]
    