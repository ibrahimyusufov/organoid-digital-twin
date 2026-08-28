"""Real-data-only check: does higher stimulation amplitude produce more
separable per-channel response fingerprints across different patterns?

For each amplitude, take real stimulation patterns delivered at that
amplitude (>=100 trials), compute each pattern's mean per-channel response
in the first 0.5s after onset (same PSTH method as pattern_response.py,
reused directly), and report the mean pairwise cosine similarity across
those patterns' response vectors. No model is used anywhere -- this is
purely about the real organoid's recorded responses.

"Amplitude" is only well-defined for a pattern instance when every
electrode in that stim event fired at the same a1 -- about a third of real
stim events mix amplitudes across their electrodes (see
uniform_amplitude_patterns). Mixed-amplitude events are excluded rather
than assigned to any single amplitude bucket.

Uses the full ~5.7-day recording (not the 48h window pattern_response.py
happens to also restrict to, which exists there only because that script
was comparing against the 48h window used for model training -- irrelevant
here since there's no model).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_events, load_stimulations
from pattern_response import psth, LIVE

AMPLITUDES = [0.01, 1, 2, 3, 4, 5, 10]
MIN_TRIALS = 100
FIG = Path("figures/amplitude_separability.png")


def uniform_amplitude_patterns(st):
    """Stim events where every electrode fired at the same amplitude,
    grouped by (electrode pattern, amplitude)."""
    nunique = st.groupby("time_of_stim")["a1"].nunique()
    uniform_times = nunique[nunique == 1].index
    st_u = st[st["time_of_stim"].isin(uniform_times)]
    dropped = len(nunique) - len(uniform_times)
    print(f"{dropped}/{len(nunique)} stim events dropped: mixed amplitude across electrodes")
    return st_u.groupby("time_of_stim").agg(
        pattern=("electrode", lambda x: tuple(sorted(x))),
        amp=("a1", "first"),
    )


def pattern_response_vector(ev_ns, ev_ch, stim_times):
    """Mean per-live-channel spike count in the 0.5s after stim onset."""
    edges, counts = psth(ev_ns, ev_ch, stim_times, len(LIVE))
    post = (edges[:-1] >= 0) & (edges[:-1] < 0.5)
    return counts[:, post].sum(axis=1) / max(len(stim_times), 1)


def mean_pairwise_cosine(vecs):
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    cm = vn @ vn.T
    iu = np.triu_indices(len(vecs), 1)
    return float(cm[iu].mean())


def main():
    ev = load_events().sort_values("time_of_event")
    st = load_stimulations()
    ev_ns = ev["time_of_event"].values.astype("datetime64[ns]").astype("int64")
    ev_ch = ev["electrode"].values
    t0, t1 = ev["time_of_event"].min(), ev["time_of_event"].max()

    groups = uniform_amplitude_patterns(st)

    results = []
    for amp in AMPLITUDES:
        sub = groups[groups["amp"] == amp]
        counts = sub["pattern"].value_counts()
        qualifying = counts[counts >= MIN_TRIALS]
        print(f"\namp={amp:>5}: {len(sub)} uniform-amplitude events, "
              f"{len(counts)} distinct patterns, {len(qualifying)} with >={MIN_TRIALS} trials")

        if len(qualifying) < 2:
            print(f"  skipped: need >=2 qualifying patterns for a pairwise comparison, "
                  f"got {len(qualifying)}")
            results.append((amp, None, len(qualifying), "too few patterns"))
            continue

        vecs = []
        for pat, n in qualifying.items():
            times = sub.index[sub["pattern"] == pat].values
            vecs.append(pattern_response_vector(ev_ns, ev_ch, times))
            print(f"  {pat} (n={n})")
        vecs = np.array(vecs)

        n_zero = int((vecs.sum(axis=1) == 0).sum())
        if n_zero:
            # A pattern with literally zero spikes on every live channel across
            # 100+ trials isn't "distinguishable" -- it's absent signal. Cosine
            # similarity against an all-zero vector is a degenerate 0/eps -> 0,
            # which would misleadingly read as "highly separable" if plotted at
            # face value. Check whether this is a real in-recording silence or
            # just trials that predate/postdate the events-covered window
            # entirely (see blanking_check.py, which found exactly this for
            # amplitude 10: 100% of trials fell ~12-13.5h before recording
            # start, so there was never any data to have shown a response).
            all_times = pd.Series(sub.index[sub["pattern"].isin(qualifying.index)])
            outside = int(((all_times < t0) | (all_times > t1)).sum())
            if outside == len(all_times):
                cause = "100% of trials fall outside the events-covered window -- no data ever existed for these, not silence"
            elif outside:
                cause = f"{outside}/{len(all_times)} trials fall outside the events-covered window -- see blanking_check.py before trusting this as a response finding"
            else:
                cause = "all trials are within the events-covered window -- a genuine in-recording zero, worth a blanking_check.py-style raw-trace look"
            print(f"  WARNING: {n_zero}/{len(qualifying)} patterns have all-zero response "
                  f"vectors (no live-channel spikes in 0.5s post-onset, across 100+ trials "
                  f"each) -- {cause} -- excluding this amplitude from the plot")
            results.append((amp, None, len(qualifying), cause))
            continue

        mean_cos = mean_pairwise_cosine(vecs)
        print(f"  mean pairwise cosine similarity: {mean_cos:.3f}")
        results.append((amp, mean_cos, len(qualifying), None))

    plot_amps = [a for a, m, _, _ in results if m is not None]
    plot_means = [m for a, m, _, _ in results if m is not None]
    skipped = [(a, n, reason) for a, m, n, reason in results if m is None]
    if skipped:
        print(f"\nskipped from plot: {skipped}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_amps, plot_means, "o-", label="mean pairwise cosine similarity")
    ax.set_xlabel("stimulation amplitude (a1)")
    ax.set_ylabel("mean pairwise cosine similarity")
    ax.set_title("Real stimulation-response separability vs amplitude\n"
                  "(lower = more distinguishable patterns)")
    if skipped:
        note = "; ".join(f"amp={a}: {reason} ({n} patterns)" for a, n, reason in skipped)
        ax.text(0.5, -0.22, f"not plotted -- {note}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color="firebrick", wrap=True)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"\nsaved {FIG}")

    return results


if __name__ == "__main__":
    main()
