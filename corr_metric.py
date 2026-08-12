import numpy as np

def mean_cross_corr(x):
    """Mean pairwise correlation, skipping channels with no variance."""
    live = x.std(axis=0) > 0
    if live.sum() < 2:
        return float("nan")
    c = np.corrcoef(x[:, live].T)
    iu = np.triu_indices(live.sum(), 1)
    return float(np.nanmean(c[iu]))
