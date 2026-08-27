# Organoid Spiking Dynamics: A Generative Model

This repo fits an autoregressive generative model to multi-electrode spiking
activity recorded from a living cortical organoid (FinalSpark preparation
`fs437`, 32-electrode MEA). Given a short seed window of real spike counts,
the model rolls forward its own predicted activity, reseeding periodically
from real data to control drift.

**This is not a "digital twin."** A digital twin implies a real-time,
bidirectionally-coupled emulation that tracks a specific physical system as
it evolves. What's here is a statistical model of the *distribution* of
spiking patterns present in one 48-hour recording window from one organoid:
it reproduces population-level statistics (rate, burstiness, cross-channel
correlation, stimulation response) reasonably well in aggregate, but it does
not track the organoid's actual trial-by-trial trajectory, does not update
from a live feed, and — per the limitations below — misses substantial
structure (correlation, long-horizon drift) that a twin claim would require.
"Generative model of organoid dynamics" is the accurate description.

## Data

- Source: FinalSpark `fs437` organoid, 32-electrode MEA, delivered as HDF5
  (`fs437_package.hdf5` for events/stimulations/metadata, `fs437_raw.hdf5`
  for raw 30kHz samples — not used here).
- All modeling uses the first 48 hours of the recording, binned into 50ms
  windows (`preprocess.py`).
- Electrode-targeted stimulation was delivered during this window in
  several fixed 4-electrode patterns; three are analyzed here:
  `A = (18,19,21,23)`, `B = (10,11,13,15)`, `C = (1,9,12,14)`.

Raw and intermediate arrays, and FinalSpark's own dataset documentation and
example notebook, are confidential and are not in this repository (see
[Data confidentiality](#data-confidentiality)).

## Methodological finding: near-silent channels destabilize the model

Of the 32 electrodes, most carry almost no signal. `channel_mask.py` keeps
only channels carrying ≥1% of total spikes each:

| | |
|---|---|
| Channels kept | 13 of 32 |
| Spikes retained | 95.1% of all spikes |

An early model trained on all 32 channels showed severe rate drift under
autoregressive rollout — generated rate climbed to **36.2 Hz** against a real
target of **9.28 Hz**. The cause wasn't a modeling bug in the usual sense: the
19 near-silent channels contribute almost nothing to the *loss*, but their
sparse, low-confidence predictions still feed back into the next timestep's
input, and small errors on 19 channels compound across an autoregressive
rollout even when each one is individually negligible. Masking those
channels before training dropped generated rate to **9.1 Hz** — within 2% of
the real target — with no other change to the model. Any generative model of
this kind of sparse multi-channel spiking data needs this masking step, or
an equivalent, before autoregression is attempted; it is not something the
loss function will discover on its own.

## Model

Four LSTM variants were trained in sequence, each addressing a specific
shortfall found in the previous one's evaluation:

| Model | File | Adds |
|---|---|---|
| `SpikeCloneLSTM` | `model.py` | Baseline: independent per-channel log-rates |
| `BurstCloneLSTM` | `model_burst.py` | + deterministic shared network-wide drive term |
| `LatentBurstLSTM` | `model_latent.py` | + *stochastic* shared latent drive (sampled per bin at generation time), with a learned per-channel gain on it |
| `CondCloneLSTM` | `model_cond.py` | `LatentBurstLSTM` + stimulation input, for evoked-response modeling |

The shared drive terms exist because per-channel-independent generation
under-coordinates activity across channels relative to the real organoid
(see below) — real channels rise and fall together more than an
independent model can produce. Sampling one shared latent value per bin and
scaling it per-channel by a learned gain (`LatentBurstLSTM`) was the
version that meaningfully closed this gap.

## Matched-slice evaluation

Reported statistics (rate, silence fraction, ISI) can look deceptively good
if real and generated activity are compared in aggregate, since a 48-hour
recording's statistics shift over time as the organoid's health changes. The
evaluation here instead always compares generated rollouts against the real
activity from the *same* time window they were seeded from (`eval_matched.py`,
`eval_latent.py`), averaged over 8 independent draws spanning the recording.

Current best model (`LatentBurstLSTM`), one run of 8 draws:

| Metric | Real | Generated | Ratio |
|---|---|---|---|
| Mean rate | 11.64 | 11.27 | 0.97 |
| Silent-bin fraction | 0.604 | 0.608 | 1.01 |
| Median ISI (bins) | 9.75 | 8.88 | 0.91 |
| Fano factor | 51.89 | 33.41 | 0.64 |
| Mean cross-channel correlation | 0.027 | 0.010 | 0.37 |

Rate, silence, and ISI — first-order, per-channel statistics — are matched
well. Fano factor and cross-channel correlation — both second-order
statistics that depend on coordination *across* time and *across* channels —
are still below real, even after the shared latent drive substantially
closed the gap versus the pre-latent-drive baseline (project history:
correlation roughly 26%→37% of real, fano factor also improved). Both
ratios vary somewhat run to run, since they depend on which hours of the
48h recording the 8 draws happen to land on — burstiness and correlation
in the real data are not constant across the recording (see
[Known limitations](#known-limitations)). The model is a substantially
better fit to single-channel statistics than to population-level
coordination in every run observed so far.

## Stimulation discrimination

The real organoid produces distinguishable per-channel response
fingerprints depending on which electrodes are stimulated
(`pattern_response.py`, computed from real recorded responses):

| Pattern | n stims | Dominant response channel(s) | Post/pre spike ratio |
|---|---|---|---|
| A (18,19,21,23) | 5965 | diffuse, no clear peak | 1.02 |
| B (10,11,13,15) | 3252 | ch15 (3.02 spikes/bin) | 1.88 |
| C (1,9,12,14) | 2617 | ch8, ch9 (2.51, 2.33 spikes/bin) | 1.58 |

Cosine similarity between per-channel response vectors: **A vs B = 0.966**
(nearly identical — A barely evokes a response at all, so it resembles
baseline/B trivially), **A vs C = 0.569**, **B vs C = 0.458** (the two
patterns that do evoke strong, distinct responses are clearly separable).

`CondCloneLSTM`, conditioned on the same stimulation input, reproduces the
qualitative targeting correctly — B drives ch15, C drives ch8/9, matching
the real organoid — but the separation between the two patterns is weaker
than real: cosine similarity **0.341** for the model's B-vs-C response
vectors, against **0.458** for the real organoid (`test_cond_response.py`).
The model gets *where* each pattern's effect concentrates right, but
under-states *how distinct* the two evoked states are from each other.

## Plasticity: reward-gated readout learns B/C discrimination

A linear readout (`plasticity.py`) is trained on top of `CondCloneLSTM`'s
simulated evoked responses to classify which stimulation pattern (B or C)
was just delivered, using a reward-gated (not gradient-based) update rule.

| Condition | Accuracy |
|---|---|
| Real reward (graded: +1 correct, +0.3 incorrect) | **~83%** |
| Shuffled reward (reward sign independent of correctness) | **~53–55%**, flat across training |

The shuffled-reward condition is the control that makes the 83% figure
meaningful: if the readout could reach high accuracy even when reward carries
no information about correctness, that would indicate it was learning from
some other structure in the response vectors (e.g. raw signal magnitude)
rather than from the reward signal itself. It's the **gap** between real and
shuffled performance — not the 83% number in isolation — that supports the
claim that reward-gated learning is doing the work here.

This shuffled control was not flat when first implemented — it climbed from
52% to 67% over the course of training, which would have invalidated the
result. The cause: the eligibility trace accumulated evidence across
trials with a slow decay (`TAU=0.7`), so by the time a given trial's reward
was applied, the trace being scaled by that reward already mixed in
contributions from several earlier, unrelated trials. Because reward should
only ever credit the trial that earned it, this let currently-unrelated
reward draws reinforce old trace directions regardless of their own sign —
a real drift, not sampling noise, since it occurred consistently across
runs. Setting `TAU=0` (trace reflects only the current trial) removed the
persistence and the drift with it; see `NOTES.md` for the fix and
before/after numbers.

## Known limitations

- **Coordination is still the weak point.** Even after the shared latent
  drive, fano factor (54%) and cross-channel correlation (37%) remain well
  below real. The model captures much more single-channel structure than
  population-level coordination.
- **No time-conditioning.** The organoid's real activity declines over its
  lifespan (see `lifespan_trend.py` / `figures/lifespan_trend.png`); this
  model is trained on a single fixed 48-hour window and has no mechanism to
  represent that trend. It describes the dynamics of that window, not the
  organoid's trajectory over its life.
- **13 of 32 electrodes.** The 19 masked channels are not modeled at all —
  not because they were confirmed silent, but because they carried too
  little signal to train on without destabilizing rollout (see above). What
  those channels are actually doing, if anything, is outside this model's
  scope.
- **Reseeded, not free-running.** Generation reseeds from real data every
  200 steps (10s) to control drift. This is a short-horizon local dynamics
  model exercised repeatedly from real starting points, not a simulator
  capable of long independent trajectories.
- **Single organoid, no replicate.** A second organoid recording (`FS369`)
  was never downloaded or evaluated. Every result here — the channel mask,
  the model fits, the stimulation fingerprints, the plasticity result — is
  from one animal's one recording. None of it has been checked for
  generalization across organoids.
- **Stimulation-discrimination and plasticity results are model-on-model.**
  The plasticity experiment's "evoked responses" are `CondCloneLSTM`'s own
  simulated reaction to simulated stimulation, not independently held-out
  real trials. It demonstrates that a reward-gated readout *can* learn to
  discriminate the model's simulated B/C responses — it does not demonstrate
  that the real organoid exhibits reward-modulated plasticity, which was
  never tested experimentally.
- **Not independently reproducible from this repo alone.** The raw/processed
  data and FinalSpark's own documentation are confidential and excluded (see
  below); reproducing any of these results requires separately obtaining the
  licensed FS437 dataset from FinalSpark.

## Data confidentiality

FinalSpark datasets must not be published. `.gitignore` excludes the raw
arrays and model checkpoints under `clone_data/`, both HDF5 files, and
FinalSpark's own `README.md`, `demo.ipynb`, and `demo_utils.py` (their
dataset guide and example notebook, distinct from this file) — these were
also removed from this repository's entire git history, not just the
current tree.

## Pipeline

```
load_data.py          → read events/stimulations/metadata from HDF5
preprocess.py          → bin spikes to 50ms, first 48h, all 32 channels
channel_mask.py         → identify 13 "live" channels (>=1% of spikes each)
preprocess_masked.py    → apply the channel mask
preprocess_stim.py      → build the aligned stimulation matrix
train.py / train_burst.py / train_latent.py / train_cond.py
                        → fit each model variant
evaluate.py / eval_matched.py / eval_latent.py
                        → real-vs-generated comparison (matched-slice)
pattern_response.py     → real stimulation-response fingerprints
test_cond_response.py   → CondCloneLSTM's simulated stimulation response
plasticity.py           → reward-gated B/C discrimination readout
```

---

Author: Ibrahim Yusufov. Implementation was carried out using Claude Code;
direction, experimental design, and interpretation of results are the
author's.
