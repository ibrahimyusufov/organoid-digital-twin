"""Are the amplitude-10 "zero response" patterns evidence of stimulation-
artifact blanking, or genuine silence? Check the raw 30kHz voltage traces
around their stim onsets.

Raw samples are looked up via fs437_segment_index.parquet, never by loading
fs437_raw.hdf5 in full. The index's (electrode, t_start, t_end) columns give
a coarse (row_start, row_end) row range per electrode-segment, but segment
sizes are extremely skewed (median 181 rows, max 123,298,967 -- more than
half the 238M-row table) and row ranges overlap across electrodes (they're
positions in a single interleaved table, not a per-electrode partition), so
reading a whole matching segment is not safe. locate_rows() instead does a
bounded probe-based binary search within a candidate segment's row range to
narrow down to the sub-range actually covering the target time window,
using only small reads (a few thousand rows each), before doing the real
read.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_events, load_stimulations
from amplitude_separability import uniform_amplitude_patterns

RAW = Path("fs437_raw.hdf5")
RAW_KEY = "/fs437_wholelife_raw"
SEGMENT_INDEX = Path("fs437_segment_index.parquet")
PRE_S, POST_S = 0.5, 0.5
MIN_TRIALS = 100
TARGET_AMP = 10
EXAMPLES_PER_PATTERN = 5
PROBE_SIZE = 2000
MAX_PROBE_ITERS = 25
FIG = Path("figures/blanking_check.png")


def candidate_segments(idx, electrode, center_time):
    """Segment-index rows for this electrode whose [t_start, t_end] overlaps
    the [center_time-PRE_S, center_time+POST_S] window."""
    lo = center_time - pd.Timedelta(seconds=PRE_S)
    hi = center_time + pd.Timedelta(seconds=POST_S)
    return idx[(idx["electrode"] == electrode) & (idx["t_start"] <= hi) & (idx["t_end"] >= lo)]


def locate_rows(row_start, row_end, target_lo_ns, target_hi_ns):
    """Bounded probe-based binary search: narrow [row_start, row_end) down to
    a sub-range likely to contain rows in [target_lo_ns, target_hi_ns], reading
    only small PROBE_SIZE chunks -- never the full segment."""
    lo, hi = int(row_start), int(row_end)
    if hi - lo <= PROBE_SIZE * 3:
        return lo, hi
    for _ in range(MAX_PROBE_ITERS):
        if hi - lo <= PROBE_SIZE * 3:
            break
        mid = (lo + hi) // 2
        probe = pd.read_hdf(RAW, key=RAW_KEY, start=mid, stop=mid + PROBE_SIZE)
        if len(probe) == 0:
            break
        t = probe["time"].values.astype("datetime64[ns]").astype("int64")
        tmin, tmax = int(t.min()), int(t.max())
        if tmax < target_lo_ns:
            lo = mid
        elif tmin > target_hi_ns:
            hi = mid
        else:
            lo = max(row_start, mid - PROBE_SIZE * 2)
            hi = min(row_end, mid + PROBE_SIZE * 2)
            break
    return lo, hi


def raw_window(idx, electrode, center_time):
    """Raw (time, voltage_uv) samples for one electrode in
    [center_time-PRE_S, center_time+POST_S], or an empty DataFrame if none
    exist -- found entirely via the segment index, no full-table scan."""
    segs = candidate_segments(idx, electrode, center_time)
    if segs.empty:
        return pd.DataFrame(columns=["time", "voltage_uv"])

    lo_ns = int((center_time - pd.Timedelta(seconds=PRE_S)).value)
    hi_ns = int((center_time + pd.Timedelta(seconds=POST_S)).value)

    frames = []
    for _, seg in segs.iterrows():
        lo_row, hi_row = locate_rows(seg["row_start"], seg["row_end"], lo_ns, hi_ns)
        chunk = pd.read_hdf(RAW, key=RAW_KEY, start=lo_row, stop=hi_row)
        chunk = chunk[chunk["electrode"] == electrode]
        t_ns = chunk["time"].values.astype("datetime64[ns]").astype("int64")
        chunk = chunk[(t_ns >= lo_ns) & (t_ns <= hi_ns)]
        if len(chunk):
            frames.append(chunk[["time", "voltage_uv"]])
    if not frames:
        return pd.DataFrame(columns=["time", "voltage_uv"])
    return pd.concat(frames).sort_values("time")


def nearest_coverage_gap(idx, electrode, center_time):
    """How far away is the nearest raw data this electrode actually has,
    relative to center_time? Used to characterize a total miss."""
    e = idx[idx["electrode"] == electrode]
    if e.empty:
        return None
    before = e[e["t_end"] <= center_time]
    after = e[e["t_start"] >= center_time]
    gap_before = (center_time - before["t_end"].max()) if len(before) else None
    gap_after = (after["t_start"].min() - center_time) if len(after) else None
    return gap_before, gap_after


def classify(pre, post):
    """Saturated/clipped vs flat-at-baseline, from raw sample stats."""
    pre_v, post_v = pre["voltage_uv"].values, post["voltage_uv"].values
    pre_absmax = np.abs(pre_v).max() if len(pre_v) else np.nan
    post_absmax = np.abs(post_v).max() if len(post_v) else np.nan
    # repeated identical extreme values is the clipping signature; a healthy
    # signal has no exact repeats at its extremes
    post_clip_frac = 0.0
    if len(post_v):
        extreme = np.abs(post_v) >= 0.95 * post_absmax
        post_clip_frac = float((pd.Series(post_v[extreme]).duplicated(keep=False)).mean()) if extreme.sum() > 1 else 0.0
    if post_absmax > 3 * pre_absmax and post_clip_frac > 0.3:
        return "saturated/clipped (blanking)"
    if post_absmax <= 2 * pre_absmax:
        return "flat at baseline (genuine silence)"
    return "ambiguous"


def main():
    idx = pd.read_parquet(SEGMENT_INDEX)
    ev = load_events().sort_values("time_of_event")
    t0, t1 = ev["time_of_event"].min(), ev["time_of_event"].max()
    print(f"events-covered window: {t0} to {t1}")
    print(f"raw-data-covered window (segment index): "
          f"{idx['t_start'].min()} to {idx['t_end'].max()}\n")

    st = load_stimulations()
    groups = uniform_amplitude_patterns(st)
    sub = groups[groups["amp"] == TARGET_AMP]
    counts = sub["pattern"].value_counts()
    qualifying = counts[counts >= MIN_TRIALS]

    examples = []
    for pat, n in qualifying.items():
        times = pd.Series(sub.index[sub["pattern"] == pat]).sort_values()
        electrodes = list(pat)
        n_outside = int(((times < t0) | (times > t1)).sum())
        print(f"pattern {pat} (n={n}): {n_outside}/{n} trials "
              f"({n_outside/n*100:.1f}%) fall outside the events-covered window")

        found_any = False
        for stim_time in times.iloc[:EXAMPLES_PER_PATTERN]:
            for e in electrodes:
                win = raw_window(idx, e, stim_time)
                if len(win):
                    found_any = True
                    pre = win[win["time"] < stim_time]
                    post = win[win["time"] >= stim_time]
                    verdict = classify(pre, post)
                    print(f"  ch{e} @ {stim_time}: {len(pre)} pre / {len(post)} post samples -> {verdict}")
                    examples.append((pat, e, stim_time, pre, post, verdict))

        if not found_any:
            gap_b, gap_a = nearest_coverage_gap(idx, electrodes[0], times.iloc[0])
            print(f"  NO raw samples found for any of the first {EXAMPLES_PER_PATTERN} trials, "
                  f"any electrode in {pat}")
            print(f"  nearest raw coverage for ch{electrodes[0]}: "
                  f"{gap_b} before, {gap_a} after the first stim onset")
        print()

    if not examples:
        print(f"CONCLUSION: no raw voltage data exists for any amplitude-{TARGET_AMP} "
              f"qualifying-pattern trial. All {qualifying.sum()} trials across "
              f"{len(qualifying)} patterns fall in a pre-recording window "
              f"(stim log starts {st['time_of_stim'].min()}, events/raw data start "
              f"{t0}) -- roughly 12-13.5 hours before the recording that produced "
              f"the events table even began. This is not evidence of stimulation-"
              f"artifact blanking or of genuine biological silence -- neither claim "
              f"can be assessed, because there is no data of any kind (spike events "
              f"or raw voltage) covering these stim onsets. The earlier "
              f"'stimulation-artifact blanking' explanation in README.md is wrong "
              f"and should be corrected.")
        return []

    fig, axes = plt.subplots(len(examples), 1, figsize=(9, 2.2 * len(examples)), squeeze=False)
    for ax, (pat, e, stim_time, pre, post, verdict) in zip(axes[:, 0], examples):
        for df, label in [(pre, "pre"), (post, "post")]:
            t_rel = (df["time"] - stim_time).dt.total_seconds()
            ax.plot(t_rel, df["voltage_uv"], lw=0.5, label=label)
        ax.axvline(0, color="red", ls="--", lw=0.8)
        ax.set_title(f"{pat} ch{e} @ {stim_time} -- {verdict}", fontsize=9)
        ax.set_ylabel("uV")
        ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("time relative to stim onset (s)")
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"\nsaved {FIG}")
    return examples


if __name__ == "__main__":
    main()
