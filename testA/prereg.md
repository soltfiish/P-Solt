# Test A — Pre-registration (FROZEN BEFORE RESULTS)

This file is committed **before** `run_testA.py` is ever run and before any
results CSV exists. The git history is the timestamp. No edit to this file is
permitted after results are generated; the verdict is whatever the frozen rule
returns.

---

## Honest scope (written before running, per §1)

The substitution labels were originally chosen by chemists reasoning from
chemical similarity. κ and every baseline are also functions of atomic
properties. Therefore the strongest claim Test A can support is **"recovers
expert substitution judgments,"** not "predicts physical reality." This is a
real, falsifiable claim — named correctly. The baselines and negative controls
are what separate "recovers expert judgment better than a dumb rule" from
"recovers nothing a dumb rule doesn't."

### Reconstruction caveat (specific to this repository)

No original κ source code, no Paper-14/Paper-15 artifact, and no original
100-pair dataset exist in this repository. Everything here is built from
scratch:

* The five κ axes are a **transparent reconstruction** of the claimed κ-map
  from its documented inputs (see `build_data.py` header). Each axis is a
  fixed, documented function of base atomic properties.
* The default `xi_e = IE1 · α_fs` (fine-structure-constant form, "14–15
  text"). The polarizability form `xi_e = IE1 / polarizability` is computed
  **only** inside the §8 ablation.
* `mendeleev_number` is a reconstructed Pettifor-*style* chemical scale, not a
  verified copy of Pettifor's published sequence. B1 = "a Pettifor-style
  scale."
* Pair `source_citation` values name an authoritative source **category**
  (USGS MCS, DOE Critical Materials Strategy, EU CRM reports, named reviews)
  for well-established materials-science substitutions. They are **not**
  verified DOIs. Pairs that could not be defensibly sourced are marked
  `sourced=False` and excluded from the primary metric (reported separately).

Freezing *a* fixed definition is all the falsifiability Test A requires: the
verdict is honest with respect to whatever is frozen here. If a reader has the
authentic κ definition, they can drop it into the `rank_candidates` interface
and re-run unchanged.

---

## §12 Pre-registration block

```
Date frozen: 2026-06-25
Data checksum:
  element_properties.csv  sha256  758bbe276af350073ee1da65c990cd90bf6cb70f96f97e545def6a46e056824f
  substitution_pairs.csv  sha256  57c77d26545724b6e5378638ebaaeb902063ce3707fdcddfd1ba1f282c53fa96

Primary metric: out-of-sample top-1 exact accuracy
  (rank-1 prediction ∈ frozen accepted-substitute set)

CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=10), stratified by block,
    seed = 1729   → 50 test evaluations.

kappa_full hyperparameter search space (exhaustive — no other DOF permitted):
  - axis weights: each of {xi_e, xi_n, xi_mag, xi_bond, xi_shell} chosen from
      {0.0, 0.5, 1.0, 2.0}, searched greedily (coordinate ascent from all-1.0,
      one pass), to keep the space disciplined and train-fold-selected only.
  - d10 gate: {on, off}
  - mercury gate: {on, off}
  - early-d / refractory extension: {on, off}
  (Gates apply a fixed 0.5× distance discount when source and candidate both
   belong to the gated class. The discount magnitude is fixed, not tuned.)

Selection rule (train-fold ONLY):
  Greedy/forward: start from equal weights, all gates off. Evaluate train-fold
  top-1. For each gate, turn it on iff it strictly raises train-fold top-1
  (frozen "include-a-fix-iff-it-helps-train" rule). Then one coordinate-ascent
  pass over axis weights, accepting a weight change iff it strictly raises
  train-fold top-1. The selected (weights, gates) are FROZEN and applied to the
  test fold. No test-fold error is ever inspected to choose hyperparameters.
  A surgical fix written after seeing test-fold errors voids the run (§5.3).

Baselines have 0 hyperparameters (B5 has none beyond z-scoring). Asymmetry is
intentional and fair: baselines are supposed to be dumb.

Meaningful margin over B5_multifeature: ≥ 5 percentage-points top-1,
  with the paired bootstrap CI on the mean per-fold difference excluding 0.
Alpha: 0.05

Controls that MUST pass first (else no §5 result is interpretable):
  - label-permutation ≈ chance for every method (incl. kappa_full)
  - N1_random_coords ≈ chance
  - N2_random_pick   ≈ chance
  chance = 1 / mean(candidate-pool size)
```

---

## §9 Decision rule (frozen)

* **PASS (real):** `kappa_full` top-1, paired bootstrap CI excludes 0 over
  **both** `B3_IE` **and** `B5_multifeature`, **and** mean top-1 beats
  `B0_same_group` and `B1_mendeleev`.
  → "κ recovers expert substitution judgments beyond standard descriptors."

* **PARTIAL:** `kappa_full` ≈ `B5_multifeature` (CI does not clear +5pp / does
  not exclude 0) but > single-descriptor baselines.
  → "κ is a compact re-encoding of standard descriptors; no gain over a naive
  multi-feature NN." Drop "predicts."

* **FAIL:** `kappa_full` ≈ `B3_IE`, or ≤ `B0_same_group` / `B1_mendeleev`.
  → substitution claim retired; κ ≈ ionization energy / ≈ periodic position.

Whatever the verdict, it is reported. No re-tuning to escape a FAIL.

---

## Methods under test (§3) and harness (§4) — frozen interface

All methods implement `rank_candidates(source, pool, ...) -> ranked list`.
Identical candidate pool (same-block minus source; full-table fallback if
pool < 5), identical local (pool) normalization, plain weighted Euclidean,
ties broken by lower Z. No method gets a private gate except `kappa_full`,
whose gates are train-fold-selected hyperparameters. No CLOSE / no
model-metric grading anywhere; soft metrics are defined in neutral space.
