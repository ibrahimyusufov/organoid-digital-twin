# FS437 organoid clone — status

## Working
- 13 live channels of 32 (channel_mask.py, >=1% of spikes)
- Clone matches real rate/silence/ISI within ~10%
- Conditional model responds to stimulation: B(10,11,13,15)->ch15, C(1,9,12,14)->ch8/9
- Plasticity layer learns B/C at 83% vs 60% shuffled control

## Bug to fix first
plasticity.py: shuffled-reward control climbs 52%->67% across trials.
Should stay flat at 50%. Suspect eligibility trace persisting between
trials (TAU=0.7) lets weights drift toward loudest channels (ch9, ch15)
regardless of reward sign. Try TAU=0 and normalizing response vectors.
The real-vs-shuffled gap is the defensible number, not 83% alone.

## Known gaps
- fano 54% of real, cross-channel corr 37% of real
- no time-conditioning; can't reproduce the 48h activity decline
- FS369 (second organoid) never downloaded

## Data is confidential
FinalSpark datasets must not be published. .gitignore covers the arrays,
README.md, demo.ipynb, demo_utils.py.
