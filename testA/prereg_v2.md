# Test A — Pre-registration v2 (FROZEN BEFORE FRESH-SET RESULTS)

This amends `prereg.md` with the five fixes from the fix-handoff. It is
committed **before** any fresh-set result exists. The v1 axes, gate menu, and
default `xi_e` definition are **unchanged** from the original frozen commit
(`element_properties.csv` checksum is identical) — only the *evaluation
protocol* is strengthened. The verdict moves to a fresh held-out set so that
the strengthened protocol is what decides.

Honesty note carried forward: v1 dev-set results were already seen before this
amendment was written. That is exactly the contamination Fix 1 targets, and the
mitigation is that **the fresh set is the verdict** and the hypothesis class
(axes + gates + xi_e) is frozen and unchanged. The assembler of the fresh set
has seen the dev-set misses, so this is not a perfectly κ-naive extraction; the
ideal remains a zero-κ-exposure sourcer. Stated as a limitation, not hidden.

---

## The five fixes (folded in)

**Fix 1 — Fresh held-out set is the verdict.** The original 65/53 pairs are
demoted to a **development set** (non-decisive): k-fold CV holds out weight
tuning but not the axis/gate *design*, so the dev number is optimistically
biased. A **fresh set** (`substitution_pairs_fresh.csv`, 44 sourced pairs,
disjoint in (source, application), all four blocks) is the verdict. Un-gated κ
(`kappa_nogate`, all gates forced off) is run alongside gated κ; if they tie,
the gates were overfit patches built from old misses.

**Fix 2 — B5_star.** A z-scored Euclidean NN over the *identical raw
quantities* feeding κ's five axes: {IE1, atomic_radius, unpaired_electrons,
Pauling χ, valence_electrons}. κ and B5_star see the same information, so
κ−B5_star isolates **geometry** — the only thing κ actually claims. (κ uses no
polarizability or magnetic-moment input beyond `unpaired_electrons`, which
B5_star includes, so the information sets match exactly.) **PASS keys on
beating B5_star, not B5.**

**Fix 3 — Corrected stats + power.** The headline paired test is the
**Nadeau–Bengio variance-corrected resampled t-test** on the 50 per-fold
differences (correction factor `1/J + n_test/n_train`, here `1/50 + 1/4`).
Pooled McNemar is demoted to a descriptive sidebar (it double-counts pairs
across overlapping folds → anticonservative). The **Minimum Detectable Effect**
(MDE at power 0.80, α 0.05) is computed for κ_full vs B5_star. A null with
MDE > 5pp is **INCONCLUSIVE**, not PARTIAL.

**Fix 4 — Pool sensitivity.** Same-block pool is primary (pre-registered);
a full-table-pool run on the same fresh pairs is reported. If κ's edge over
B5_star exists only in the narrow pool, that is a fragility, reported plainly.

**Fix 5 — Decision rule.** Verdict on the fresh set; PASS keys on B5_star;
fourth outcome INCONCLUSIVE added; gated and un-gated both reported.

---

## §12b Pre-registration block (frozen)

```
Date frozen (v2): 2026-06-25
Data checksums:
  element_properties.csv        sha256 758bbe276af350073ee1da65c990cd90bf6cb70f96f97e545def6a46e056824f
  substitution_pairs.csv  (dev) sha256 57c77d26545724b6e5378638ebaaeb902063ce3707fdcddfd1ba1f282c53fa96
  substitution_pairs_fresh.csv  sha256 0b772a24b4c24738fef3f71a48a75254c8a4cf73a3c4c77a00b563da06229ae5

Primary metric : out-of-sample top-1 exact accuracy on the FRESH set.
Verdict set    : fresh (44 sourced pairs). Dev set: reported, NON-DECISIVE.
CV             : RepeatedStratifiedKFold(5, 10), stratified by block, seed 1729.
Headline test  : Nadeau–Bengio corrected resampled t-test on 50 per-fold diffs.
McNemar        : descriptive sidebar only.
Decisive contrast : kappa_full − B5_star (geometry). Floors: B0, B1, B3_IE.
Margin         : ≥ 5 pp top-1 over B5_star, NB CI excluding 0.
Power          : MDE at power 0.80, α 0.05; MDE > 5pp ⇒ INCONCLUSIVE.
kappa hyperparameters (train-fold only, unchanged from v1):
  axis weights ∈ {0,0.5,1,2} greedy coordinate-ascent from all-1;
  gates d10/mercury/refractory ∈ {on,off}, fixed 0.5× co-class discount;
  kappa_full selects gates+weights; kappa_nogate forces gates off, selects weights.
Pool           : same-block primary; full-table sensitivity reported.
Controls       : label-perm / N1 / N2 ≈ chance on the fresh set, else INVALID.
```

---

## §9 Decision rule v2 (frozen; evaluated on FRESH set)

Let Δ* = mean per-fold (kappa_full − B5_star), with NB CI and MDE.

* **FAIL** — κ_full ≤ B0/B1 floor, **or** not separated from B3_IE
  (NB CI includes 0 and Δ < 5pp). → κ ≈ IE / ≈ periodic position.
* **PASS** — NB CI(Δ*) excludes 0 **and** Δ* ≥ 5pp **and** κ_full beats the
  floor and B3_IE. → "κ's geometry recovers expert substitution beyond
  standard descriptors."
* **PARTIAL** — Δ* CI includes 0 **and** MDE ≤ 5pp (adequately powered, no
  edge found) **and** κ_full > single-descriptor baselines. → "demonstrated
  equivalence: κ is a compact re-encoding; no geometric gain over a z-scored
  NN on its own inputs." Drop "predicts."
* **INCONCLUSIVE** — Δ* CI includes 0 **and** MDE > 5pp. → underpowered; cannot
  separate equivalence from an undetected edge. Enlarge the fresh set.

Gated-vs-ungated reported regardless. No re-tuning to change the outcome.
