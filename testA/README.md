# Test A — Ground Test Protocol (κ-substitution)

Held-out, pre-registered, blind test of the claim that a κ-map predicts
industrial element substitutes **better than its own inputs**. Built from
scratch (no prior κ code or dataset existed in this repo). **v2** folds in the
five fixes from the fix-handoff; the verdict is decided on a **fresh held-out
set**, not the set the protocol was developed against.

## Verdict (fresh held-out set): **INCONCLUSIVE**

> On the fresh set, `kappa_full` (0.326) vs `B5_star` (0.295) — a z-scored NN
> over κ's **identical raw inputs** — gives Δ **+3.1pp**, Nadeau–Bengio CI
> **[-11.2, +17.3]pp (includes 0)**. The **Minimum Detectable Effect is ≈20pp**,
> far above the 5pp target, so the 44-pair fresh set is **underpowered**: this
> is a *null of detection*, not demonstrated equivalence. The honest next step
> is a larger fresh set, not a claim.

What *is* established, robustly and across both sets:

- **κ's geometry adds nothing over its own inputs.** `kappa_full ≈ B5_star`
  everywhere — fresh (Δ+3.1pp, CI incl 0), full-table pool (Δ+0.6pp, CI incl 0),
  and even in-sample dev (B5\* 0.530 ≥ κ_full 0.518). The only thing κ claims —
  that its coordinate *geometry* beats a naive NN on the same numbers — is not
  observed.
- **The gates are inert patches.** gated − un-gated κ = **0.0pp**. The
  d10/mercury/refractory gates, built from dev-set misses, do nothing
  out-of-sample — exactly as predicted.
- **κ ≈ the same-group floor on fresh.** κ_full 0.326 barely exceeds
  `B0_same_group` 0.319; the periodic table alone is nearly as good.
- κ does beat the radius-only baseline (Δ+25.7pp, CI excludes 0) and trends
  above B3_IE (Δ+9.9pp) — but neither rescues a claim over B5\*.

| method | fresh top-1 | fresh 95% CI | dev top-1 (non-decisive) |
|---|---|---|---|
| kappa_full | 0.326 | [0.198, 0.459] | 0.518 |
| kappa_nogate | 0.326 | [0.198, 0.459] | 0.557 |
| **B5_star** (κ's raw inputs) | **0.295** | [0.159, 0.432] | **0.530** |
| B5_multifeature | 0.295 | [0.159, 0.432] | 0.454 |
| B0_same_group | 0.319 | [0.182, 0.455] | 0.339 |
| B3_IE | 0.226 | [0.114, 0.364] | 0.189 |
| chance floor | 0.053 | — | — |

(Controls pass: label-permutation/N1/N2 ≈ chance on the fresh set → no leak.)

## Why v2 changed the verdict from v1's "PARTIAL"

v1's k-fold CV held out *weight tuning* but not the *axis/gate design*, which
was fixed against the development pairs — so its PARTIAL ("demonstrated
equivalence to B5") over-claimed. v2:

1. **Fresh held-out set** is the verdict; the original 53 pairs are demoted to
   a non-decisive development set.
2. **B5\*** uses κ's *identical raw inputs*, isolating geometry (v1's B5 gave
   κ extra features, confounding the comparison).
3. **Nadeau–Bengio** corrected t-test replaces pooled McNemar (which
   double-counts overlapping folds → anticonservative); **MDE** is computed, so
   an underpowered null is correctly labeled **INCONCLUSIVE**, not PARTIAL.
4. **Pool sensitivity** (full-table) confirms no edge in either pool.

The lesson: with 44 fresh pairs you cannot demonstrate equivalence *or* a
5pp edge. PARTIAL requires power v1 never had.

## Reproduce

```bash
pip install numpy pandas scipy scikit-learn
python3 build_data.py    # freezes element_properties + dev + fresh CSVs (checksummed)
python3 run_testA.py     # controls → dev → fresh verdict → NB stats → sensitivity → ablations
```

Seed 1729; `RepeatedStratifiedKFold(5,10)`; one `rank_candidates` interface for
every method; same-block pool + local normalization for all.

## Files

- `prereg.md`, `prereg_v2.md` — frozen pre-registration (committed before the
  respective results). v2 freezes the fresh-set §9 rule and the four-outcome
  decision (PASS / PARTIAL / INCONCLUSIVE / FAIL).
- `build_data.py` → `data/{element_properties, substitution_pairs,
  substitution_pairs_fresh}.csv` + `CHECKSUMS.txt`. Provenance & limitations in
  the file header.
- `run_testA.py` — the harness.
- `results/` — `results_summary.csv`, `paired_stats.csv`, `controls.csv`,
  `misses.csv`, `ablations.csv`, `verdict.md`.

## Honest scope

The test can only support "**recovers expert substitution judgments**," never
"predicts physical reality" — labels and methods are all functions of atomic
properties. The κ axes, Mendeleev scale, and pair citations are reconstructions
(source *categories*, not verified DOIs); the fresh-set assembler had seen
dev-set misses, so it is not perfectly κ-naive (ideal: a zero-κ-exposure
sourcer). All caveats are in `prereg_v2.md`. To push toward a decisive verdict,
the single highest-value step is a larger, DOI-verified, κ-naive fresh set
(target: MDE ≤ 5pp).
