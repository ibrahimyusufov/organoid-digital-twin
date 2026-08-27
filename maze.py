"""Stage 1: feasibility check for a stimulation-driven maze task.

Before building the maze (Stage 2), verify that clone_model_cond.pt can
produce distinguishable response vectors for enough real, live-channel
stimulation patterns to plausibly serve as 16 distinct maze-cell "signals".

Four things were tried, in order, each addressing what the previous one
revealed:

1. top_live_patterns(): the 16 most frequent real stimulation patterns fully
   on live channels, at plasticity.py's default AMP=10. Single-trial cosine
   looked good (106/120 pairs < 0.7) but that's noise -- a single evoke()
   call is a sparse few-spike Poisson draw. The 100-trial averaged (stable)
   fingerprint told a different story: 38/120 pairs < 0.7, because 14 of the
   16 patterns collapse onto one generic-response shape and only differ in
   overall magnitude.

2. direct_target_patterns(): one single-electrode pattern per live channel,
   on the theory that direct stimulation of a single electrode would produce
   a channel-specific fingerprint (as it did for real B->ch15, C->ch8/9).
   Worse, not better: 14/120 pairs < 0.7 (stable fingerprint). A single
   electrode's injected signal at AMP=10 is too weak, relative to the
   model's shared latent drive, to dominate the response.

3. select_diverse(): still 4-electrode patterns, but chosen by greedy
   farthest-point selection over all 56 real 4-electrode live-channel combos
   to explicitly maximize mutual separability, at AMP=10. Best so far:
   59-62/120 pairs < 0.7 across independent re-measurements (mean cosine
   ~0.62-0.69). The largest fully mutually-distinguishable subset found
   among all 56 candidates was only 4-5 patterns, and even that didn't fully
   hold up on independent re-measurement -- some "distinguishable" pairs
   crept back above 0.7 cosine on a fresh 100-trial draw.

4. AMP boost: raising the injected stimulation amplitude in evoke() (see
   evoke_boosted -- plasticity.py's evoke() hardcodes AMP=10 with no
   parameter to override) helps, but plateaus fast. Sweeping AMP on a small
   pattern set: 10 -> 6/10 pairs separable, 50 -> 9/10, 200 -> 8-9/10,
   1000 -> 9/10 (diminishing returns above ~200). Extending the response
   window (POST) made things WORSE, not better (POST=30 dropped separable
   pairs from 9/10 to 4/10) -- the stimulation-specific signal is transient
   and gets diluted by generic activity as the window lengthens, so POST
   stays at plasticity.py's default of 10. At scale (diverse-16, AMP=200,
   POST=10), the confirmed result was 72/120 pairs < 0.7 (mean cosine
   0.622) -- a real, reproducible improvement over AMP=10's 62/120, but a
   further push to AMP=1000 (70/120) showed no further gain.

Conclusion: even after amplitude tuning and diversity-optimized pattern
selection, this model's evoked-response repertoire on this data supports on
the order of 4-5 states that hold up under independent re-measurement, and
partial (not full) separation across the other patterns -- not the 16
mutually-distinguishable states a 4x4 maze needs.

Stage 2: a 2x2 maze (not the originally-planned 4x4) built on the one fully
confirmed 4-pattern set from Stage 1 -- see VALIDATED_PATTERNS below, all
pairwise cosine < 0.4 on independent re-measurement. Each cell's identity is
signalled by evoking its pattern (one real evoke_boosted() call per step, not
averaged -- the agent only gets a single noisy read at decision time, same as
plasticity.py's session()). A MazeReadout, structurally like plasticity.py's
PlasticReadout (decay-then-accumulate eligibility trace, reward-scaled weight
update), picks one of 4 actions per step. Unlike PlasticReadout there's no
ground-truth label to contrast against -- path-finding has no "correct
action" known in advance, only a terminal reward -- so accumulate() credits
whichever action was actually taken, decayed within the episode, and the
trace resets between episodes (episode boundary, not FIX-then-forget within
one trial like plasticity.py's now-fixed TAU=0).
"""
from pathlib import Path
import numpy as np
import torch
from model_cond import CondCloneLSTM
from plasticity import SPIKES, LIVE, SEED_W
from load_data import load_stimulations

MODEL = Path("clone_data/clone_model_cond.pt")
N_PATTERNS = 16
PATTERN_LEN = 4
COS_THRESHOLD = 0.7
SCREEN_TRIALS = 30
CONFIRM_TRIALS = 100
BOOSTED_AMP = 200  # see module docstring, item 4 -- plateaus above this


def four_electrode_live_candidates():
    """All real 4-electrode stimulation patterns fully on live channels."""
    live = set(LIVE.tolist())
    st = load_stimulations()
    groups = st.groupby("time_of_stim")["electrode"].apply(lambda x: tuple(sorted(x)))
    counts = groups.value_counts()  # sorted descending by frequency
    return [pat for pat in counts.index if len(pat) == PATTERN_LEN and set(pat) <= live]


def top_live_patterns(n=N_PATTERNS, pattern_len=PATTERN_LEN):
    """Most frequent real stimulation patterns of a given size, fully on live
    channels. Rejected -- see module docstring, strategy 1."""
    live = set(LIVE.tolist())
    st = load_stimulations()
    groups = st.groupby("time_of_stim")["electrode"].apply(lambda x: tuple(sorted(x)))
    counts = groups.value_counts()
    candidates = [pat for pat in counts.index if len(pat) == pattern_len and set(pat) <= live]
    if len(candidates) < n:
        raise RuntimeError(f"only {len(candidates)} candidate {pattern_len}-electrode "
                            f"patterns fully on live channels, need {n}")
    return candidates[:n]


def direct_target_patterns(n=N_PATTERNS):
    """One single-electrode pattern per live channel, filled out to n with the
    most frequent multi-electrode live combos. Rejected -- see module
    docstring, strategy 2."""
    live = sorted(LIVE.tolist())
    st = load_stimulations()
    groups = st.groupby("time_of_stim")["electrode"].apply(lambda x: tuple(sorted(x)))
    counts = groups.value_counts()

    singles = [(c,) for c in live if counts.get((c,), 0) > 0]
    if len(singles) < len(live):
        missing = set(live) - {p[0] for p in singles}
        raise RuntimeError(f"no real single-electrode stim on live channel(s): {missing}")

    combos = [pat for pat in counts.index
              if len(pat) > 1 and set(pat) <= set(live) and pat not in singles]
    fill = combos[:max(0, n - len(singles))]

    patterns = (singles + fill)[:n]
    if len(patterns) < n:
        raise RuntimeError(f"only {len(patterns)} candidate patterns available, need {n}")
    return patterns


def evoke_boosted(model, data, stim_ch, device, amp=BOOSTED_AMP):
    """Same as plasticity.evoke(), but with an overridable stimulation
    amplitude -- plasticity.py hardcodes AMP=10 with no parameter for it.
    See module docstring, item 4."""
    idx = [list(LIVE).index(c) for c in stim_ch if c in LIVE]
    acc = np.zeros(len(LIVE), dtype="float32")
    POST = 10
    with torch.no_grad():
        s = np.random.randint(0, len(data) - SEED_W)
        sp = torch.log1p(torch.from_numpy(data[s:s + SEED_W])).unsqueeze(0).to(device)
        lr, hidden = model(sp, torch.zeros_like(sp), sample_latent=True)
        lr = lr[:, -1, :]
        for t in range(POST):
            st = torch.zeros(1, 1, len(LIVE), device=device)
            if t == 0:
                for i in idx:
                    st[0, 0, i] = np.log1p(amp)
            c = torch.poisson(torch.exp(lr).clamp(min=0).cpu())
            acc += c[0].numpy()
            lr, hidden = model(torch.log1p(c.to(device)).unsqueeze(1), st,
                               hidden, sample_latent=True)
            lr = lr[:, -1, :]
    return acc


def cosine_matrix(vecs):
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    return vn @ vn.T


def fingerprint(model, data, pattern, device, trials, amp=BOOSTED_AMP):
    return np.mean([evoke_boosted(model, data, pattern, device, amp) for _ in range(trials)], axis=0)


def select_diverse(candidates, fingerprints, n):
    """Greedy farthest-point selection: maximize the minimum pairwise cosine
    distance within the chosen set."""
    cm = cosine_matrix(fingerprints)
    selected = [int(np.argmax(np.linalg.norm(fingerprints, axis=1)))]
    remaining = set(range(len(candidates))) - set(selected)
    while len(selected) < n and remaining:
        best, best_score = None, -1
        for i in remaining:
            score = min(1 - cm[i, j] for j in selected)
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        remaining.discard(best)
    return selected


def max_distinguishable_subset(candidates, fingerprints, threshold=COS_THRESHOLD):
    """Greedy maximum independent set in the 'too similar' graph (edge if
    cosine >= threshold): the largest subset where every pair is < threshold."""
    cm = cosine_matrix(fingerprints)
    too_similar = cm >= threshold
    np.fill_diagonal(too_similar, False)
    remaining = list(range(len(candidates)))
    selected = []
    while remaining:
        degs = {i: sum(too_similar[i, j] for j in remaining if j != i) for i in remaining}
        best = min(degs, key=degs.get)
        selected.append(best)
        remaining = [j for j in remaining if j != best and not too_similar[best, j]]
    return selected


def report_matrix(label, cm, patterns):
    print(f"\n{label}")
    print("      " + "".join(f"{i:>7}" for i in range(len(patterns))))
    for i, row in enumerate(cm):
        print(f"[{i:>2}]  " + "".join(f"{v:7.3f}" for v in row))
    iu = np.triu_indices(len(patterns), 1)
    pair_vals = cm[iu]
    below = int((pair_vals < COS_THRESHOLD).sum())
    total = len(pair_vals)
    print(f"{below}/{total} pairs below cosine {COS_THRESHOLD} "
          f"(mean pairwise cosine: {pair_vals.mean():.3f})")
    return below, total


def stage1_feasibility(device):
    data = np.load(SPIKES)
    model = CondCloneLSTM(num_electrodes=data.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()

    candidates = four_electrode_live_candidates()
    print(f"{len(candidates)} real 4-electrode patterns fully on live channels, "
          f"AMP={BOOSTED_AMP} (boosted from plasticity.py's default 10)")

    screen_fps = np.array([fingerprint(model, data, p, device, SCREEN_TRIALS) for p in candidates])

    chosen_idx = select_diverse(candidates, screen_fps, N_PATTERNS)
    chosen = [candidates[i] for i in chosen_idx]
    print(f"\ndiversity-selected {len(chosen)} patterns:")
    for i, p in enumerate(chosen):
        print(f"  [{i:>2}] {p}")

    confirm_fps = np.array([fingerprint(model, data, p, device, CONFIRM_TRIALS) for p in chosen])
    below, total = report_matrix(
        f"{CONFIRM_TRIALS}-trial averaged cosine similarity matrix "
        f"(independent re-measurement of the chosen 16):",
        cosine_matrix(confirm_fps), chosen)

    max_idx = max_distinguishable_subset(candidates, screen_fps)
    max_patterns = [candidates[i] for i in max_idx]
    max_confirm_fps = np.array([fingerprint(model, data, p, device, CONFIRM_TRIALS) for p in max_patterns])
    print(f"\nlargest fully mutually-distinguishable set found among all "
          f"{len(candidates)} candidates: {len(max_patterns)}")
    for p in max_patterns:
        print(f"  {p}")
    report_matrix(f"{CONFIRM_TRIALS}-trial re-measurement of that set:",
                  cosine_matrix(max_confirm_fps), max_patterns)

    return chosen, confirm_fps, below, total, max_patterns


# ---------------------------------------------------------------------------
# Stage 2: 2x2 maze on the validated 4-pattern set
# ---------------------------------------------------------------------------

VALIDATED_PATTERNS = [
    (15, 19, 21, 23),
    (18, 19, 21, 23),
    (12, 13, 20, 21),
    (9, 10, 12, 13),
]  # confirmed in Stage 1: all pairwise cosine < 0.4 on independent re-measurement

GRID = 2
START = (0, 0)
GOAL = (GRID - 1, GRID - 1)
MAX_STEPS = 50
EPISODES = 500
BLOCK = 50
SEEDS = 1   # plasticity.py uses 3, but each maze episode costs several evoke() calls
            # (vs. plasticity's one per trial) -- kept to 1 for a first pass
TAU_EPISODE = 0.9   # within-episode trace decay -- NOT plasticity.py's TAU (that one is
                     # cross-trial persistence, fixed to 0; this one is intentionally > 0)
LR_PLAST = 0.05      # same learning rate as plasticity.py's PlasticReadout
EPSILON = 0.2        # exploration rate -- see MazeReadout.decide

ACTIONS = ["up", "down", "left", "right"]
DELTAS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

CELL_PATTERNS = {
    (0, 0): VALIDATED_PATTERNS[0],
    (0, 1): VALIDATED_PATTERNS[1],
    (1, 0): VALIDATED_PATTERNS[2],
    (1, 1): VALIDATED_PATTERNS[3],
}


class MazeReadout:
    """Structurally like plasticity.py's PlasticReadout (decay-then-accumulate
    eligibility trace, reward-scaled weight update), adapted for path-finding:
    there's no ground-truth action to contrast against (unlike PlasticReadout's
    known true/guess label pair), only a terminal reward, so accumulate()
    credits whichever action was actually taken.

    decide() adds epsilon-greedy exploration with random tie-breaking, which
    PlasticReadout doesn't need. PlasticReadout starts from the same
    all-identical initial weights but still learns, because it always knows
    the true label and can contrast against it regardless of what was
    guessed. Here there's no known correct action -- only a sparse terminal
    reward -- so with a pure deterministic argmax over tied initial weights,
    every episode would pick the same action forever, never reach the goal,
    never generate a reward, and never learn anything.
    """
    def __init__(self, n_ch, actions=ACTIONS):
        self.actions = list(actions)
        self.w = np.ones((len(actions), n_ch), dtype="float32") * 0.1
        self.trace = np.zeros_like(self.w)

    def decide(self, resp, rng):
        if rng.random() < EPSILON:
            return int(rng.integers(len(self.actions)))
        scores = self.w @ resp
        tied = np.flatnonzero(scores == scores.max())
        return int(rng.choice(tied))

    def accumulate(self, action_idx, resp):
        self.trace *= TAU_EPISODE
        r = resp / (np.linalg.norm(resp) + 1e-9)
        self.trace[action_idx] += r

    def apply_reward(self, r):
        self.w += LR_PLAST * r * self.trace

    def reset_trace(self):
        self.trace[:] = 0


def move(pos, action):
    dx, dy = DELTAS[action]
    x, y = pos
    return (min(max(x + dx, 0), GRID - 1), min(max(y + dy, 0), GRID - 1))


def run_episode(model, data, device, readout, shuffled, rng):
    pos = START
    steps = 0
    while pos != GOAL and steps < MAX_STEPS:
        resp = evoke_boosted(model, data, CELL_PATTERNS[pos], device)
        action_idx = readout.decide(resp, rng)
        readout.accumulate(action_idx, resp)
        pos = move(pos, readout.actions[action_idx])
        steps += 1

    reached = pos == GOAL
    if shuffled:
        reward = 1.0 if rng.random() < 0.5 else 0.0  # coin flip, independent of `reached`
    else:
        reward = 1.0 if reached else 0.0
    readout.apply_reward(reward)
    readout.reset_trace()
    return steps if reached else MAX_STEPS, reached


def session(model, data, device, shuffled, seed):
    rng = np.random.default_rng(seed)
    readout = MazeReadout(len(LIVE))
    steps_to_goal = np.zeros(EPISODES)
    reached = np.zeros(EPISODES, dtype=bool)
    for e in range(EPISODES):
        steps_to_goal[e], reached[e] = run_episode(model, data, device, readout, shuffled, rng)
    return steps_to_goal, reached


def stage2_maze(device):
    data = np.load(SPIKES)
    model = CondCloneLSTM(num_electrodes=data.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()

    print(f"2x2 maze, start {START}, goal {GOAL}, cap {MAX_STEPS} steps/episode, "
          f"{EPISODES} episodes x {SEEDS} seeds")
    print("cell -> pattern:")
    for cell, pat in CELL_PATTERNS.items():
        print(f"  {cell}: {pat}")

    for tag, shuf in [("real reward", False), ("shuffled reward", True)]:
        runs = [session(model, data, device, shuf, seed) for seed in range(SEEDS)]
        steps = np.mean([s for s, _ in runs], axis=0)
        reached = np.mean([r for _, r in runs], axis=0)
        n_blocks = EPISODES // BLOCK
        print(f"\n{tag}:")
        for b in range(n_blocks):
            lo, hi = b * BLOCK, (b + 1) * BLOCK
            print(f"  episodes {lo:>4}-{hi:>4}: mean steps-to-goal {steps[lo:hi].mean():6.2f}  "
                  f"reached {reached[lo:hi].mean()*100:5.1f}%")
        print(f"  overall: mean steps-to-goal {steps.mean():.2f}  reached {reached.mean()*100:.1f}%")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    stage1_feasibility(device)
    stage2_maze(device)
