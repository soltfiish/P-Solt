# Test A — Ground Test Protocol (κ-substitution)

Held-out, pre-registered, blind test of the claim that a κ-map predicts
industrial element substitutes better than its own inputs. Built from scratch
(no prior κ code or dataset existed in this repo); see `prereg.md` →
"Reconstruction caveat" for exactly what is reconstructed and why that does
not weaken the falsifiability.

## Verdict: **PARTIAL**

> κ_full clears every single-descriptor baseline (it beats `B3_IE` by +0.33
> top-1, paired CI excluding 0, McNemar p<0.001 — so **κ is more than
> ionization energy**), **but** it is statistically indistinguishable from
> `B5_multifeature`, a naive z-scored 4-feature nearest-neighbor (Δ+0.064,
> CI [-0.051, +0.185] **includes 0**, McNemar p=0.27). The +5pp margin over
> B5 required by the frozen §9 rule is not cleared with CI excluding 0.

**Supported claim:** "κ is a compact re-encoding of standard chemical
descriptors." **Not supported:** "κ *predicts* substitution beyond standard
descriptors." Drop "predicts."

| method | top-1 | 95% CI | top-3 |
|---|---|---|---|
| **kappa_full** | **0.518** | [0.391, 0.640] | 0.805 |
| kappa_5D (equal wt) | 0.511 | [0.377, 0.642] | 0.793 |
| kappa_2D | 0.303 | [0.189, 0.434] | 0.473 |
| B5_multifeature | 0.454 | [0.321, 0.585] | 0.850 |
| B0_same_group | 0.339 | [0.208, 0.472] | 0.661 |
| B1_mendeleev | 0.301 | [0.189, 0.434] | 0.660 |
| B3_IE | 0.189 | [0.094, 0.302] | 0.529 |
| chance floor | 0.052 | — | — |

(N1 random-coords 0.075, N2 random-pick 0.042, label-permutation ≈ 0.02–0.03
for every method → **controls pass, harness does not leak**.)

## What the ablations settled

- **Polarizability-vs-α (the flagged bug):** `xi_e = IE·α_fs` and `xi_e = IE`
  give **identical** top-1 (0.511) — the fine-structure constant is an inert
  rescale removed by local normalization. The polarizability form (0.510) adds
  nothing either. **Confirmed empirically, not by argument.**
- **Drop-one-axis:** dropping the magnetic axis (`xi_mag`) *raises* top-1
  (+0.038); radius (`xi_n`) and valence (`xi_shell`) are the load-bearing axes.
  κ_full's gates/weights add only +0.007 over equal-weight κ_5D.
- **Misses** cluster on `xi_mag` as the axis of max disagreement and are mostly
  adjacent same-block homologs separable by a few pm of radius (e.g. Pt→Ir
  instead of Pd, Co→Fe instead of Ni) — see `results/misses.csv`.

## Reproduce

```bash
pip install numpy pandas scipy scikit-learn
python3 build_data.py      # re-freeze the two input CSVs (checksummed)
python3 run_testA.py       # controls → CV → paired stats → ablations → verdict
```

Fixed seed 1729; `RepeatedStratifiedKFold(5, 10)`; one `rank_candidates`
interface for every method; same-block pool + local normalization for all.

## Files

- `prereg.md` — frozen §12 block + §9 decision rule (committed before results).
- `build_data.py` → `data/element_properties.csv`, `data/substitution_pairs.csv`
  (+ `CHECKSUMS.txt`). Provenance/limitations in the file header.
- `run_testA.py` — the harness.
- `results/` — `results_summary.csv`, `paired_stats.csv`, `controls.csv`,
  `misses.csv`, `ablations.csv`, `verdict.md`.

## Honest scope (from `prereg.md`)

The test can only support "**recovers expert substitution judgments**," never
"predicts physical reality," because labels and methods are all functions of
atomic properties. The primary metric covers **53 sourced, in-pool pairs**;
7 cross-block pairs are unscoreable under the mandated same-block harness and
5 unsourced pairs are excluded — all reported separately and applied
identically to every method.
