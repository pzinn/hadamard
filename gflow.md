# The mathematical framework

A GFlowNet defines a distribution over a terminal state space via a DAG of partial construction steps. For Hadamard matrices:

- States = partial sequences (prefixes) x_{1:t} of the length-n ±1 array.
- Actions = append a ±1 token.
- Initial state s_0 = empty sequence.
- Terminal states x = x_{1:n} carry a reward R(x) > 0.
- Forward policy p_F(x_t | x_{<t}) = your transformer output at position t.
- Goal: arrange p_F so that the induced terminal distribution satisfies p(x) ∝ R(x).

With the reward

R(x) = exp(-score(x) / τ)

a GFlowNet trained to convergence samples Hadamard matrices (score = 0) at an exponentially higher rate than near-misses. τ controls the sharpness — small τ concentrates mass, large τ explores.

Training objective

The cleanest objective is trajectory balance (Malkin et al., 2022). For every trajectory τ = (s_0, s_1, ..., s_n = x):

Z · Π_t p_F(x_t | x_{<t})  =  R(x) · Π_t p_B(x_{<t} | x_{1:t})

The loss is a single squared-log-ratio per sample:

L(τ) = ( log Z + Σ_t log p_F(x_t | x_{<t}) − log R(x) − Σ_t log p_B )²

In your setting, the "tokenized" sequence has a unique parent for every state (you know which position was just written), so the backward policy is trivial: p_B = 1 at every step. The objective collapses to:

L(x) = ( log Z + log p_F(x) + score(x)/τ )²

where log p_F(x) = Σ log_softmax(logits_t)[x_t] is the standard autoregressive log-likelihood, and log Z is a single learned scalar parameter (initialized to something reasonable, optimized with a separate learning rate if needed).

That's the whole training loss. It's a single MSE per sample, no variance reduction, no baselines, no KL penalties.

What this buys you versus REINFORCE and versus your current approach

- REINFORCE optimizes E[R(x)], which pushes toward one mode (argmax). GFlowNets optimize for p(x) ∝ R(x), which gives diverse samples across all high-reward modes. For Hadamard matrix search, diversity matters — you want lots of different candidates to feed into the improvement step.
- Decision Transformer / score-conditioning (what you have now) trains p(x | s) and hopes for extrapolation at s = 0. GFlowNet directly trains p(x) ∝ exp(-score/τ) — the target distribution is in the loss, no extrapolation gap.
- Score-weighted MLE (my suggestion #1) is a crude approximation to the GFlowNet objective: it tilts the training distribution toward low-score samples but doesn't normalize correctly, so the model isn't sampling from p(x) ∝ R(x). GFlowNet fixes this via log Z.

Data sources — on-policy vs off-policy

The TB loss is valid for any trajectory, regardless of where it came from. This is a big deal because it means you can use:

1. On-policy samples from the current transformer — standard, provides exploration scaled by current model entropy.
2. Off-policy samples from your improvement step — these are high-reward trajectories, very valuable for training. Just plug them into the loss.
3. Historical samples from the PatternBoost population — equivalent to your current training data.

In practice, a mix of 50% on-policy and 50% high-reward off-policy works well (Madan et al., 2023). This fits your PatternBoost loop naturally: the improved matrices from each generation are exactly the high-reward off-policy data.

Reward design

Your score S = -log det(M/√n) is already in log space and bounded below by 0. A reasonable reward:

log R(x) = -score(x) / τ

Details to work out:
- τ schedule: start at ~1 (soft, explores), anneal to ~0.1 (concentrates on Hadamard candidates) across generations.
- Numerical stability: finite log rewards should not be clipped, because that changes the target distribution by assigning every sufficiently bad state the same reward. A non-finite score is treated as an error instead of being silently hidden.
- Optional: subtract a baseline log R_ref to keep log Z in a reasonable range.

Exploration

The forward policy can collapse if training is too aggressive. Standard tricks:

- Tempered sampling: sample with temperature > 1 during data collection. This is an off-policy behaviour distribution; samples from the trained GFlowNet itself must use temperature 1.
- ε-uniform mixing: with probability ε, replace the forward policy with uniform over actions.
- Prioritized replay: oversample high-reward trajectories from a replay buffer.
- Backward-KL regularization against a pre-trained supervised model, if you want to stay close to a "reasonable" starting policy.

Mapping to your codebase

The changes are surprisingly small:

1. Add log_Z parameter: one scalar nn.Parameter on the model, maybe with its own optimizer at ~10× the learning rate (Malkin et al. note log_Z needs faster updates since it's a single scalar).
2. Replace the loss function: instead of cross-entropy against training targets, compute log p_F(x) (sum of log-softmax over chosen tokens — same computation as cross-entropy, just a sign and a sum), then compute the TB loss.
3. Reward function: wrap score() to return log R = -score/τ.
4. Data loader: no architectural change; the existing batch of (sequences, scores) is exactly what you need.

For the modified-transformer variant (transformer_uses_score=True), GFlowNet arguably replaces it entirely — you no longer need score conditioning because the model is trained to target low scores.

Concrete caveats for your problem

- Tree vs DAG: the current tree formulation is reward-proportional over encoded sequences, not symmetry orbits. An orbit with more encodings therefore receives more total probability. On-the-fly symmetry randomisation improves data coverage but does not correct this multiplicity bias. Correcting it would require an explicit orbit-size reward correction, canonical terminal states, or a more complicated quotient-DAG formulation.
- Sparse reward: if exp(-score/τ) is near zero for most samples, gradients are weak. The τ schedule matters a lot. Consider initializing τ ≈ mean population score, annealing slowly.
- Interaction with improvement step: the improvement step produces high-reward off-policy samples. This is ideal for GFlowNet training and much more valuable here than in the supervised setting (where the improved samples already look like the data distribution).
- Block structure (GS type): if you keep the block-by-block generation from your score-conditioned variant, each block becomes a sub-trajectory in the GFlowNet. Reward can be assigned either per-block (using the partial score) or only at the end. Per-block rewards give denser training signal but require care in the flow-matching condition.

Reading

- Bengio et al. 2021, "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation" — original paper, motivated exactly by drug/molecule discovery (similar structure to yours).
- Malkin et al. 2022, "Trajectory Balance: Improved Credit Assignment in GFlowNets" — the objective I described above.
- Madan et al. 2023, "Learning GFlowNets from Partial Episodes" — useful for off-policy / hindsight-relabeled data.
- Deleu et al. 2022, "Bayesian Structure Learning with Generative Flow Networks" — good example of combinatorial search.


# Let me sketch the GFlowNet changes.

## Architecture of the change

**Tokens** = packed `stacking`-bit groups of ±1 signs; `vocab_size = 2^stacking`, `block_size = nm * ⌈nn/stacking⌉`. Each trajectory emits `block_size` tokens. `log p_F(x)` is a sum over these `block_size` positions — not over the `na` individual signs.

**What stays identical**: `Transformer` class, tokenization, symmetry randomization, `save/load`, sampling loop in `generate()`, `sample()`.

**What changes**: the loss in `forward()`, a new learned `log_Z` parameter, the training loop (needs score per sample + a data source that mixes on-policy with off-policy samples), and a `temperature` parameter on the *training* forward policy (separate from the inference temperature).

## Sketch

### 1. Add `log_Z` and a reward function

```python
# params.py (new tunables)
gflow_tau_init = 2.0            # reward temperature, annealed
gflow_tau_min  = 0.2
gflow_tau_decay = 0.9           # tau *= decay each generation
gflow_logZ_lr  = 1e-2           # log_Z needs a fast LR
gflow_onpolicy_frac = 0.5       # fraction of each batch sampled on-policy
```

```python
# transformer.py, inside Transformer.__init__
self.log_Z = torch.nn.Parameter(torch.zeros(()))
```

### 2. Reward from score

```python
# transformer.py — use params.score() on the decoded array
def log_reward(arrays, tau):
    # arrays: (B, na) int8 ±1
    s = params.score(arrays)   # (B,)
    if not torch.isfinite(s).all():
        raise RuntimeError("non-finite score in GFlowNet reward")
    return -s / tau
```

### 3. Rewrite `forward()` to compute `log p_F(x)` instead of CE loss

The existing `forward()` already computes `logits` of shape `(B, block_size, vocab_size)`. Replace the loss path:

```python
def forward(self, batch0, compute_logpf=False):
    # ... unchanged through logits computation ...
    logits = self.lm_head(x)
    log_pf = None
    if compute_logpf:
        # logits[:, t, :] predicts batch0[:, t]
        log_probs = F.log_softmax(logits, dim=-1)              # (B, T, V)
        log_pf = log_probs.gather(-1, batch0.unsqueeze(-1)).squeeze(-1)  # (B, T)
        log_pf = log_pf.sum(dim=-1)                            # (B,)
    return logits, log_pf
```

Note: when `self.training`, the existing code drops the last input token. The logit at position `t` predicts `batch0[:, t]`, which is exactly what we gather — no off-by-one issue since `batch0` length equals `block_size`.

### 4. TB loss

```python
def tb_loss(log_pf, log_R, log_Z):
    return ((log_Z + log_pf - log_R) ** 2).mean()
```

### 5. Training loop: mixed on-policy + off-policy data

```python
def train(data, **kwargs):
    # ... setup ...
    tau = max(config.gflow_tau_init * (config.gflow_tau_decay ** params.gen),
              config.gflow_tau_min)

    # two parameter groups so log_Z has its own LR
    params_main = [p for n, p in model.named_parameters() if n != "log_Z"]
    optimiser = torch.optim.AdamW([
        {"params": params_main, "lr": lr_sched(0), "weight_decay": config.weight_decay},
        {"params": [model.log_Z], "lr": config.gflow_logZ_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.99))

    k_on = int(batch_size * config.gflow_onpolicy_frac)
    k_off = batch_size - k_on

    while True:
        # off-policy: from replay / improved population (existing data)
        off_idx = torch.randint(data_len, (k_off,))
        off_arrays = randomise_symmetry(data[off_idx].to(device), params.symmetry_ctx)
        off_strings = array_to_string(off_arrays)

        # on-policy: sample from current model
        model.eval()
        with torch.no_grad():
            on_strings = torch.empty(k_on, config.block_size, dtype=torch.int, device=device)
            on_arrays  = torch.empty(k_on, na, dtype=torch.int8, device=device)
            generate(on_strings, on_arrays, temperature=config.gflow_train_temp)
        model.train()

        # combine
        string_batch = torch.cat([off_strings, on_strings], dim=0)
        arrays_batch = torch.cat([off_arrays,  on_arrays],  dim=0)

        # forward pass -> log p_F
        _, log_pf = model(string_batch, compute_logpf=True)
        log_R = log_reward(arrays_batch, tau)
        loss = tb_loss(log_pf, log_R, model.log_Z)
        # ... optimiser.step() etc
```

A couple of points:

- **On-policy sampling is expensive** (it's a full autoregressive generate over the whole batch). Amortize by sampling less frequently: generate a big on-policy batch every K steps, replay it for K steps. Or start purely off-policy (like vanilla supervised training) for the first few hundredx steps and introduce on-policy only once `log_Z` stabilizes.
- **`generate()` currently reads `config.temperature` via a module-level closure**. Needs a `temperature` argument so training can use a different one than inference (typically training temp is higher for exploration, e.g. 1.0–1.5; inference temp stays low).

### 6. What to remove / simplify

- The MLE cross-entropy loss on line 92 — replaced by TB loss. You can keep it as a pre-training phase (supervised init then GFlowNet fine-tune), controlled by a flag.
- `best_from()` in `hadamard.py` (the top-k selection for the training set): less critical now — TB loss handles all samples, good or bad. But keeping a quality filter for the off-policy data is still worthwhile (bad samples contribute gradient via squared loss anyway, but they're numerically dominated by `-score/τ` which can be huge).

## Order in which to implement

1. **Shim layer, keep MLE training working.** Add `log_Z`, `log_reward`, and the `compute_logpf` path to `forward()`. Don't touch the training loop yet. Sanity-check that `log_pf` computed by this path matches `-cross_entropy · block_size` for a MLE-trained model.
2. **Add TB loss, pure off-policy.** Change the training loop to use `(log_Z + log_pf - log_R)²` on the existing replay-buffer samples. Keep everything else the same. Tune τ by watching `log_Z` stabilize.
3. **Add on-policy mixing.** Plug in the on-policy sampling branch. Tune `gflow_onpolicy_frac`.
4. **(Optional) Hybrid loss.** Weighted sum of MLE and TB during fine-tune, to prevent the model from wandering too far from the "reasonable sequences" manifold early in training:
   ```
   L = λ · L_TB + (1 - λ) · L_MLE
   ```
   Anneal λ from 0 to 1.

## Two things to watch

- **`log_Z` magnitude**: if `log R` is very negative (score-τ ratio huge), `log_Z` will try to track `E[log p_F - log_R]`, which can be hundreds. This is expected, but makes the separate learning rate and logging of `log_Z`, `log p_F`, `log R`, and the TB residual important.
- **Gradient signal from `log p_F`**: the sum of `block_size` log-softmax values dominates the loss numerically for long sequences. Some GFlowNet implementations normalize `log_pf` by trajectory length; I'd leave it alone for first experiments.
