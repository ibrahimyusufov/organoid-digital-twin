# FS437 organoid clone — status

## Working
- 13 live channels of 32 (channel_mask.py, >=1% of spikes)
- Clone matches real rate/silence/ISI within ~10%
- Conditional model responds to stimulation: B(10,11,13,15)->ch15, C(1,9,12,14)->ch8/9
- Plasticity layer learns B/C at ~83% vs ~53-55% flat shuffled control

## Fixed
plasticity.py: shuffled-reward control was climbing 52%->67% across trials
instead of staying flat at 50%. Cause: eligibility trace persisting between
trials (TAU=0.7) let a trial's reward get applied to trace contributions
built up from earlier, unrelated trials, drifting weights regardless of
reward sign. Fix: TAU=0 at plasticity.py:12 (trace no longer persists
across trials). Verified: real reward ~83%, shuffled reward ~53-55% flat
across quartiles, no systematic climb. The real-vs-shuffled gap is now the
defensible number, not 83% alone.

## Known gaps
- fano 54% of real, cross-channel corr 37% of real
- no time-conditioning; can't reproduce the 48h activity decline
- FS369 (second organoid) never downloaded

## Data is confidential
FinalSpark datasets must not be published. .gitignore covers the arrays,
README.md, demo.ipynb, demo_utils.py.
