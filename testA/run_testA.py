#!/usr/bin/env python3
"""
run_testA.py -- Ground Test Protocol for the kappa-substitution claim.

One script, fixed seed, every method behind a common
    rank_candidates(source, pool, ...) -> ranked list[symbol]
interface (prereg §3, §4). Implements:

  * shared harness: same-block candidate pool, local (pool) normalization,
    plain weighted Euclidean, lower-Z tie-break  (§4)
  * methods: kappa_2D / kappa_5D / kappa_full + baselines B0..B5 + controls
    N1/N2  (§3)
  * RepeatedStratifiedKFold(5,10) stratified by block, train-fold-only
    hyperparameter selection for kappa_full  (§5.3)
  * negative controls run FIRST: label permutation, random coords, random
    pick, chance floor  (§7)
  * paired bootstrap CIs, McNemar  (§6)
  * ablations: polarizability-vs-alpha xi_e, drop-one-axis  (§8)
  * deliverables: results_summary.csv, paired_stats.csv, controls.csv,
    misses.csv, ablations.csv, verdict.md  (§11)

Reads ONLY the frozen CSVs in data/. No CLOSE / no model-metric grading
anywhere. The verdict follows the frozen §9 rule -- it is whatever it is.
"""

import csv
import math
import os
import random
from collections import defaultdict

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

# ---------------------------------------------------------------------------
# Frozen config (prereg §12)
# ---------------------------------------------------------------------------
SEED = 1729
N_SPLITS = 5
N_REPEATS = 10
ALPHA_FS = 7.2973525693e-3
GATE_DISCOUNT = 0.5          # fixed, not tuned
WEIGHT_GRID = [0.0, 0.5, 1.0, 2.0]
MARGIN_PP = 5.0             # meaningful margin over B5, percentage points
N_BOOT = 10000
KAPPA_AXES = ["kappa_xi_e", "kappa_xi_n", "kappa_xi_mag",
              "kappa_xi_bond", "kappa_xi_shell"]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

# d10-closed-shell species (ns2 (n-1)d10 at the common industrial state)
D10_SET = {"Zn", "Cd", "Hg", "Cu", "Ag", "Au", "Ga", "In", "Ge", "Sn"}
# refractory / early-d transition metals
REFRACTORY_SET = {"Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re"}


# ---------------------------------------------------------------------------
# Load frozen data
# ---------------------------------------------------------------------------
def load_elements():
    path = os.path.join(DATA, "element_properties.csv")
    el = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            sym = row["symbol"]
            def num(k):
                v = row[k]
                return None if v == "" else float(v)
            el[sym] = {
                "Z": int(row["Z"]),
                "symbol": sym,
                "block": row["block"],
                "group": int(row["group"]),
                "period": int(row["period"]),
                "mendeleev_number": float(row["mendeleev_number"]),
                "IE1_eV": num("IE1_eV"),
                "polarizability_A3": num("polarizability_A3"),
                "electronegativity_pauling": num("electronegativity_pauling"),
                "atomic_radius_pm": num("atomic_radius_pm"),
                "ionic_radius_common_pm": num("ionic_radius_common_pm"),
                "valence_electrons": num("valence_electrons"),
                "unpaired_electrons": num("unpaired_electrons"),
            }
    return el


def load_pairs():
    path = os.path.join(DATA, "substitution_pairs.csv")
    pairs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            pairs.append({
                "pair_id": row["pair_id"],
                "source": row["source_symbol"],
                "accepted": row["accepted_substitutes"].split("|"),
                "application": row["application"],
                "block": row["block"],
                "citation": row["source_citation"],
                "sourced": row["sourced"].strip().lower() == "true",
            })
    return pairs


EL = load_elements()
PAIRS_ALL = load_pairs()


def radius_of(sym):
    e = EL[sym]
    r = e["ionic_radius_common_pm"]
    return r if r is not None else e["atomic_radius_pm"]


# ---------------------------------------------------------------------------
# Shared harness (§4)
# ---------------------------------------------------------------------------
def candidate_pool(source):
    """Same-block minus source; full-table fallback if pool < 5. Identical
    rule for every method."""
    blk = EL[source]["block"]
    pool = [s for s in EL if EL[s]["block"] == blk and s != source]
    if len(pool) < 5:
        pool = [s for s in EL if s != source]
    return sorted(pool, key=lambda s: EL[s]["Z"])


def kappa_axis_value(sym, axis, xi_e_mode="alpha"):
    """Return the raw (pre-normalization) value of a kappa axis.
    xi_e_mode: 'alpha' -> IE1*alpha_fs (default, 14-15 text);
               'poly'  -> IE1/polarizability (12-13 form, ablation only);
               'rawIE' -> IE1 (to show alpha is an inert constant)."""
    e = EL[sym]
    if axis == "kappa_xi_e":
        if xi_e_mode == "poly":
            return e["IE1_eV"] / e["polarizability_A3"]
        if xi_e_mode == "rawIE":
            return e["IE1_eV"]
        return e["IE1_eV"] * ALPHA_FS
    if axis == "kappa_xi_n":
        return e["atomic_radius_pm"]
    if axis == "kappa_xi_mag":
        return e["unpaired_electrons"]
    if axis == "kappa_xi_bond":
        return e["electronegativity_pauling"]
    if axis == "kappa_xi_shell":
        return e["valence_electrons"]
    raise KeyError(axis)


def local_minmax(values):
    """Return a transform normalizing over the given values (min-max)."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return lambda v: 0.0
    return lambda v: (v - lo) / span


def kappa_rank(source, pool, axes, weights, xi_e_mode="alpha",
               gates=None, return_axisgap=False):
    """Weighted-Euclidean kappa ranking with optional gates.
    Local (pool ∪ source) min-max normalization per axis."""
    gates = gates or {}
    members = pool + [source]
    norms = {}
    for a in axes:
        vals = [kappa_axis_value(m, a, xi_e_mode) for m in members]
        norms[a] = local_minmax(vals)
    s_vec = {a: norms[a](kappa_axis_value(source, a, xi_e_mode)) for a in axes}

    scored = []
    axisgap = {}
    for c in pool:
        c_vec = {a: norms[a](kappa_axis_value(c, a, xi_e_mode)) for a in axes}
        d2 = 0.0
        gaps = {}
        for a in axes:
            diff = s_vec[a] - c_vec[a]
            gaps[a] = abs(diff)
            d2 += weights.get(a, 1.0) * diff * diff
        dist = math.sqrt(d2)
        # gates: multiplicative discount when source & candidate co-class
        if gates.get("d10") and source in D10_SET and c in D10_SET:
            dist *= GATE_DISCOUNT
        if gates.get("mercury") and source == "Hg" and c in D10_SET:
            dist *= GATE_DISCOUNT
        if gates.get("refractory") and source in REFRACTORY_SET and c in REFRACTORY_SET:
            dist *= GATE_DISCOUNT
        scored.append((dist, EL[c]["Z"], c))
        axisgap[c] = gaps
    scored.sort()
    ranked = [c for _, _, c in scored]
    if return_axisgap:
        return ranked, axisgap
    return ranked


def nn_rank_scalar(source, pool, valfn):
    """Nearest neighbour on a single scalar; lower-Z tie-break."""
    sv = valfn(source)
    scored = [(abs(valfn(c) - sv), EL[c]["Z"], c) for c in pool]
    scored.sort()
    return [c for _, _, c in scored]


# ---------------------------------------------------------------------------
# Method registry  (§3) -- every method returns a ranked list of symbols
# ---------------------------------------------------------------------------
def m_kappa_2D(source, pool, ctx):
    axes = ["kappa_xi_e", "kappa_xi_n"]
    return kappa_rank(source, pool, axes, {a: 1.0 for a in axes},
                      xi_e_mode=ctx.get("xi_e_mode", "alpha"))


def m_kappa_5D(source, pool, ctx):
    return kappa_rank(source, pool, KAPPA_AXES, {a: 1.0 for a in KAPPA_AXES},
                      xi_e_mode=ctx.get("xi_e_mode", "alpha"))


def m_kappa_full(source, pool, ctx):
    hp = ctx["hp"]
    axes = ctx.get("axes", KAPPA_AXES)
    return kappa_rank(source, pool, axes, hp["weights"],
                      xi_e_mode=ctx.get("xi_e_mode", "alpha"),
                      gates=hp["gates"])


def m_B0_same_group(source, pool, ctx):
    sg, sp = EL[source]["group"], EL[source]["period"]
    scored = []
    for c in pool:
        cg, cp = EL[c]["group"], EL[c]["period"]
        scored.append((0 if cg == sg else 1, abs(cp - sp),
                       abs(cg - sg), EL[c]["Z"], c))
    scored.sort()
    return [t[-1] for t in scored]


def m_B1_mendeleev(source, pool, ctx):
    return nn_rank_scalar(source, pool, lambda s: EL[s]["mendeleev_number"])


def m_B2_electronegativity(source, pool, ctx):
    return nn_rank_scalar(source, pool, lambda s: EL[s]["electronegativity_pauling"])


def m_B3_IE(source, pool, ctx):
    return nn_rank_scalar(source, pool, lambda s: EL[s]["IE1_eV"])


def m_B4_radius(source, pool, ctx):
    return nn_rank_scalar(source, pool, radius_of)


def m_B5_multifeature(source, pool, ctx):
    feats = [
        lambda s: EL[s]["electronegativity_pauling"],
        lambda s: EL[s]["IE1_eV"],
        radius_of,
        lambda s: EL[s]["valence_electrons"],
    ]
    members = pool + [source]
    transforms = []
    for fn in feats:
        vals = np.array([fn(m) for m in members], dtype=float)
        mu, sd = vals.mean(), vals.std()
        transforms.append((fn, mu, sd if sd > 0 else 1.0))
    s_vec = np.array([(fn(source) - mu) / sd for fn, mu, sd in transforms])
    scored = []
    for c in pool:
        c_vec = np.array([(fn(c) - mu) / sd for fn, mu, sd in transforms])
        scored.append((float(np.linalg.norm(s_vec - c_vec)), EL[c]["Z"], c))
    scored.sort()
    return [c for _, _, c in scored]


def m_N1_random_coords(source, pool, ctx):
    rng = ctx["rng"]
    coords = ctx.setdefault("_n1coords", {})
    dim = 5
    def vec(s):
        if s not in coords:
            coords[s] = rng.normal(size=dim)
        return coords[s]
    sv = vec(source)
    scored = [(float(np.linalg.norm(vec(c) - sv)), EL[c]["Z"], c) for c in pool]
    scored.sort()
    return [c for _, _, c in scored]


def m_N2_random_pick(source, pool, ctx):
    rng = ctx["rng"]
    return list(rng.permutation(pool))


METHODS = {
    "kappa_2D": m_kappa_2D,
    "kappa_5D": m_kappa_5D,
    "kappa_full": m_kappa_full,
    "B0_same_group": m_B0_same_group,
    "B1_mendeleev": m_B1_mendeleev,
    "B2_electronegativity": m_B2_electronegativity,
    "B3_IE": m_B3_IE,
    "B4_radius": m_B4_radius,
    "B5_multifeature": m_B5_multifeature,
    "N1_random_coords": m_N1_random_coords,
    "N2_random_pick": m_N2_random_pick,
}
BASELINES = ["B0_same_group", "B1_mendeleev", "B2_electronegativity",
             "B3_IE", "B4_radius", "B5_multifeature"]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def topk_hit(ranked, accepted_set, k):
    return int(any(p in accepted_set for p in ranked[:k]))


def near_hit(pred, accepted_set):
    """Neutral soft metric (NOT kappa-distance): rank-1 prediction and some
    accepted answer share a group OR |Δ ionic radius| < 15 pm."""
    pg = EL[pred]["group"]
    pr = radius_of(pred)
    for a in accepted_set:
        if EL[a]["group"] == pg:
            return 1
        if abs(radius_of(a) - pr) < 15:
            return 1
    return 0


# ---------------------------------------------------------------------------
# kappa_full hyperparameter selection -- TRAIN FOLD ONLY (§5.3)
# ---------------------------------------------------------------------------
def train_top1(train_pairs, weights, gates, xi_e_mode="alpha"):
    hits = 0
    for p in train_pairs:
        ranked = kappa_rank(p["source"], p["pool"], KAPPA_AXES, weights,
                            xi_e_mode=xi_e_mode, gates=gates)
        hits += topk_hit(ranked, set(p["accepted"]), 1)
    return hits / len(train_pairs)


def select_hyperparams(train_pairs, xi_e_mode="alpha"):
    """Frozen forward rule: gates included iff they strictly raise train-fold
    top-1; then one coordinate-ascent pass over axis weights. No test fold is
    ever consulted."""
    gates = {"d10": False, "mercury": False, "refractory": False}
    weights = {a: 1.0 for a in KAPPA_AXES}
    base = train_top1(train_pairs, weights, gates, xi_e_mode)
    # gates
    for g in ["d10", "mercury", "refractory"]:
        trial = dict(gates); trial[g] = True
        if train_top1(train_pairs, weights, trial, xi_e_mode) > base:
            gates = trial
            base = train_top1(train_pairs, weights, gates, xi_e_mode)
    # axis weights (one coordinate-ascent pass)
    for a in KAPPA_AXES:
        best_w, best_s = weights[a], base
        for w in WEIGHT_GRID:
            trial = dict(weights); trial[a] = w
            s = train_top1(train_pairs, trial, gates, xi_e_mode)
            if s > best_s:
                best_s, best_w = s, w
        weights[a] = best_w
        base = best_s
    return {"weights": weights, "gates": gates}


# ---------------------------------------------------------------------------
# Scoreability: a pair is in-pool iff ≥1 accepted answer is in the source pool
# (method-independent harness-scope filter, applied identically to all)
# ---------------------------------------------------------------------------
def annotate_pairs(pairs):
    out = []
    for p in pairs:
        pool = candidate_pool(p["source"])
        p = dict(p)
        p["pool"] = pool
        p["in_pool"] = any(a in pool for a in p["accepted"])
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Main cross-validated evaluation (§5)
# ---------------------------------------------------------------------------
def run_cv(eval_pairs, methods=None, xi_e_mode="alpha", axes=None, seed=SEED,
           collect_misses=False):
    """Returns:
      fold_acc[method] -> list of 50 per-fold top-1 accuracies
      fold_acc3, fold_near similarly
      pair_hits[method][pair_id] -> list of per-repeat top-1 hits (held-out)
      misses (optional)
    """
    methods = methods or list(METHODS)
    rng = np.random.default_rng(seed)
    ctx_persistent = {m: {} for m in methods}  # for N1 fixed coords

    blocks = [p["block"] for p in eval_pairs]
    idx = np.arange(len(eval_pairs))
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=seed)

    fold_acc = {m: [] for m in methods}
    fold_acc3 = {m: [] for m in methods}
    fold_near = {m: [] for m in methods}
    pair_hits = {m: defaultdict(list) for m in methods}
    misses = []

    for train_idx, test_idx in rskf.split(idx, blocks):
        train_pairs = [eval_pairs[i] for i in train_idx]
        test_pairs = [eval_pairs[i] for i in test_idx]
        hp = select_hyperparams(train_pairs, xi_e_mode) if "kappa_full" in methods else None

        for m in methods:
            ctx = {"rng": np.random.default_rng(rng.integers(1 << 30)),
                   "hp": hp, "xi_e_mode": xi_e_mode}
            if axes is not None:
                ctx["axes"] = axes
            ctx.update(ctx_persistent[m])  # share N1 coords across folds
            h1 = h3 = hn = 0
            for p in test_pairs:
                ranked = METHODS[m](p["source"], p["pool"], ctx)
                acc_set = set(p["accepted"])
                t1 = topk_hit(ranked, acc_set, 1)
                t3 = topk_hit(ranked, acc_set, 3)
                nh = near_hit(ranked[0], acc_set)
                h1 += t1; h3 += t3; hn += nh
                pair_hits[m][p["pair_id"]].append(t1)
                if collect_misses and m == "kappa_full" and not t1:
                    pred = ranked[0]
                    _, axisgap = kappa_rank(p["source"], p["pool"], KAPPA_AXES,
                                            hp["weights"], xi_e_mode=xi_e_mode,
                                            gates=hp["gates"], return_axisgap=True)
                    gaps = axisgap[pred]
                    misses.append({
                        "pair_id": p["pair_id"], "source": p["source"],
                        "predicted": pred, "accepted_set": "|".join(p["accepted"]),
                        "same_group": int(any(EL[a]["group"] == EL[pred]["group"]
                                              for a in p["accepted"])),
                        "delta_ionic_radius_pm": round(min(abs(radius_of(a) -
                                                    radius_of(pred)) for a in p["accepted"]), 1),
                        "axis_of_max_disagreement": max(gaps, key=gaps.get),
                        "candidate_pool_size": len(p["pool"]),
                    })
            n = len(test_pairs)
            fold_acc[m].append(h1 / n)
            fold_acc3[m].append(h3 / n)
            fold_near[m].append(hn / n)
            # persist N1 coords
            if m == "N1_random_coords" and "_n1coords" in ctx:
                ctx_persistent[m]["_n1coords"] = ctx["_n1coords"]

    return fold_acc, fold_acc3, fold_near, pair_hits, misses


# ---------------------------------------------------------------------------
# Statistics (§6)
# ---------------------------------------------------------------------------
def bootstrap_ci_mean(per_pair_rate, n=N_BOOT, seed=SEED):
    """Bootstrap CI on the mean per-pair top-1 rate (resample pairs)."""
    rng = np.random.default_rng(seed)
    arr = np.array(per_pair_rate, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_diff(rate_a, rate_b, n=N_BOOT, seed=SEED):
    """Paired bootstrap CI on mean(a-b) over pairs (same pairs both methods)."""
    rng = np.random.default_rng(seed + 7)
    a = np.array(rate_a, dtype=float); b = np.array(rate_b, dtype=float)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(bin_a, bin_b):
    """McNemar on paired binary correctness. Returns (b01, b10, p, odds_ratio)."""
    from scipy.stats import binomtest
    b01 = sum(1 for x, y in zip(bin_a, bin_b) if x == 0 and y == 1)  # b better
    b10 = sum(1 for x, y in zip(bin_a, bin_b) if x == 1 and y == 0)  # a better
    nd = b01 + b10
    if nd == 0:
        return b01, b10, 1.0, 1.0
    p = binomtest(min(b01, b10), nd, 0.5).pvalue
    odds = (b10 + 0.5) / (b01 + 0.5)   # a-over-b odds (Haldane correction)
    return b01, b10, float(p), float(odds)


def per_pair_rate(pair_hits_method, pair_ids):
    """Mean held-out top-1 hit per pair (fractional for kappa_full)."""
    return [float(np.mean(pair_hits_method[pid])) if pair_hits_method[pid] else 0.0
            for pid in pair_ids]


def per_pair_binary(pair_hits_method, pair_ids):
    """Majority held-out correctness per pair (for McNemar)."""
    return [int(np.mean(pair_hits_method[pid]) >= 0.5) if pair_hits_method[pid] else 0
            for pid in pair_ids]


# ---------------------------------------------------------------------------
# Controls (§7) -- run FIRST
# ---------------------------------------------------------------------------
def chance_floor(eval_pairs):
    sizes = [len(p["pool"]) for p in eval_pairs]
    return float(np.mean([1.0 / s for s in sizes]))


def run_controls(eval_pairs):
    rows = []
    chance = chance_floor(eval_pairs)
    rows.append(("chance_floor", chance, "1/mean(pool size)"))

    # N2 random pick (analytic expectation + MC), N1 random coords
    n2_exp = float(np.mean([sum(1 for a in p["accepted"] if a in p["pool"]) /
                            len(p["pool"]) for p in eval_pairs]))
    rows.append(("N2_random_pick_expected", n2_exp, "E[|accepted∩pool|/|pool|]"))

    fa, _, _, ph, _ = run_cv(eval_pairs, methods=["N1_random_coords", "N2_random_pick"])
    for m in ["N1_random_coords", "N2_random_pick"]:
        rows.append((m + "_top1_mean", float(np.mean(fa[m])), "CV mean top-1"))

    # Label permutation: shuffle accepted across pairs, rerun EVERY method.
    rng = np.random.default_rng(SEED + 99)
    perm_methods = ["kappa_full", "B3_IE", "B5_multifeature", "B0_same_group"]
    n_perms = 5
    perm_acc = {m: [] for m in perm_methods}
    for _ in range(n_perms):
        shuffled = [dict(p) for p in eval_pairs]
        accs = [p["accepted"] for p in eval_pairs]
        order = rng.permutation(len(accs))
        for p, j in zip(shuffled, order):
            p["accepted"] = accs[j]
            p["in_pool"] = any(a in p["pool"] for a in p["accepted"])
        fa2, _, _, _, _ = run_cv(shuffled, methods=perm_methods, seed=SEED + 1)
        for m in perm_methods:
            perm_acc[m].append(float(np.mean(fa2[m])))
    for m in perm_methods:
        rows.append(("label_perm_" + m + "_top1", float(np.mean(perm_acc[m])),
                     f"mean over {n_perms} permutations (expect ≈ chance)"))
    return rows, chance


# ---------------------------------------------------------------------------
# Ablations (§8)
# ---------------------------------------------------------------------------
def run_ablations(eval_pairs):
    rows = []
    # Polarizability vs alpha vs rawIE for xi_e (kappa_5D, equal weight, isolates axis def)
    for mode, label in [("alpha", "xi_e=IE*alpha_fs (14-15 default)"),
                        ("rawIE", "xi_e=IE (alpha removed by normalization)"),
                        ("poly",  "xi_e=IE/polarizability (12-13 form)")]:
        fa, _, _, _, _ = run_cv(eval_pairs, methods=["kappa_5D"], xi_e_mode=mode)
        rows.append(("xi_e_def:" + mode, label, float(np.mean(fa["kappa_5D"]))))

    # Drop-one-axis for kappa_full (equal-weight 5D as reference, drop each)
    fa_full, _, _, _, _ = run_cv(eval_pairs, methods=["kappa_5D"])
    ref = float(np.mean(fa_full["kappa_5D"]))
    rows.append(("dropaxis:none(5D ref)", "all 5 axes equal weight", ref))
    for drop in KAPPA_AXES:
        axes = [a for a in KAPPA_AXES if a != drop]
        def make(axes):
            return lambda source, pool, ctx: kappa_rank(
                source, pool, axes, {a: 1.0 for a in axes},
                xi_e_mode=ctx.get("xi_e_mode", "alpha"))
        METHODS["_tmp_drop"] = make(axes)
        fa, _, _, _, _ = run_cv(eval_pairs, methods=["_tmp_drop"])
        acc = float(np.mean(fa["_tmp_drop"]))
        rows.append(("dropaxis:" + drop, f"Δacc vs 5D = {acc-ref:+.3f}", acc))
        del METHODS["_tmp_drop"]
    return rows


# ---------------------------------------------------------------------------
# Orchestration + deliverables (§11)
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    annotated = annotate_pairs(PAIRS_ALL)

    # scoreability / sourcing breakdown
    sourced = [p for p in annotated if p["sourced"]]
    primary = [p for p in sourced if p["in_pool"]]          # PRIMARY metric set
    excl_unsourced = [p for p in annotated if not p["sourced"]]
    excl_outpool = [p for p in sourced if not p["in_pool"]]

    print("=" * 70)
    print(f"Total pairs           : {len(annotated)}")
    print(f"  sourced=True        : {len(sourced)}")
    print(f"  -> in-pool (PRIMARY): {len(primary)}")
    print(f"  -> out-of-pool (excl, same-block harness): {len(excl_outpool)} "
          f"[{', '.join(p['pair_id']+':'+p['source']+'->'+'|'.join(p['accepted']) for p in excl_outpool)}]")
    print(f"  sourced=False (excl): {len(excl_unsourced)} "
          f"[{', '.join(p['pair_id'] for p in excl_unsourced)}]")
    print("=" * 70)

    pair_ids = [p["pair_id"] for p in primary]

    # ---- CONTROLS FIRST (§7) ----
    print("\n[1/4] Negative controls (must pass before any §5 result counts)...")
    ctrl_rows, chance = run_controls(primary)
    for name, val, note in ctrl_rows:
        print(f"   {name:42s} {val:7.4f}   {note}")
    with open(os.path.join(RESULTS, "controls.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["control", "value", "note"])
        for r in ctrl_rows:
            w.writerow([r[0], f"{r[1]:.4f}", r[2]])

    # control gate: permutation & randoms must be within ~2x chance
    controls_ok = True
    for name, val, _ in ctrl_rows:
        if name.startswith("label_perm_") or name.startswith("N1_") or \
           name in ("N2_random_pick_top1_mean",):
            if val > 2.0 * chance + 0.02:
                controls_ok = False
    print(f"   --> controls {'PASS' if controls_ok else 'FAIL (harness leak!)'} "
          f"(chance floor = {chance:.4f})")

    # ---- MAIN CV (§5) ----
    print("\n[2/4] Main cross-validated evaluation (RepeatedStratifiedKFold 5x10)...")
    methods = list(METHODS.keys())
    fa, fa3, fnear, ph, misses = run_cv(primary, methods=methods, collect_misses=True)

    rates = {m: per_pair_rate(ph[m], pair_ids) for m in methods}
    summary = {}
    for m in methods:
        mean, lo, hi = bootstrap_ci_mean(rates[m])
        summary[m] = {
            "top1_mean": float(np.mean(fa[m])), "top1_std": float(np.std(fa[m])),
            "top1_ci_lo": lo, "top1_ci_hi": hi,
            "top3_mean": float(np.mean(fa3[m])),
            "nearhit_mean": float(np.mean(fnear[m])),
        }
    with open(os.path.join(RESULTS, "results_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "top1_mean", "top1_std", "top1_ci_lo",
                    "top1_ci_hi", "top3_mean", "nearhit_mean"])
        for m in methods:
            s = summary[m]
            w.writerow([m, f"{s['top1_mean']:.4f}", f"{s['top1_std']:.4f}",
                        f"{s['top1_ci_lo']:.4f}", f"{s['top1_ci_hi']:.4f}",
                        f"{s['top3_mean']:.4f}", f"{s['nearhit_mean']:.4f}"])
    print(f"   {'method':22s} {'top1':>7s} {'95%CI':>16s} {'top3':>7s} {'near':>7s}")
    for m in methods:
        s = summary[m]
        print(f"   {m:22s} {s['top1_mean']:7.3f} "
              f"[{s['top1_ci_lo']:.3f},{s['top1_ci_hi']:.3f}] "
              f"{s['top3_mean']:7.3f} {s['nearhit_mean']:7.3f}")

    # ---- PAIRED STATS (§6): kappa_full vs every baseline ----
    print("\n[3/4] Paired statistics (kappa_full vs baselines)...")
    kf_rate = rates["kappa_full"]
    kf_bin = per_pair_binary(ph["kappa_full"], pair_ids)
    paired_rows = []
    for b in BASELINES:
        d, lo, hi = paired_bootstrap_diff(kf_rate, rates[b])
        b_bin = per_pair_binary(ph[b], pair_ids)
        b01, b10, p, odds = mcnemar(kf_bin, b_bin)
        win = "yes" if lo > 0 else "no"
        paired_rows.append([b, f"{d:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                            f"{p:.4f}", f"{odds:.3f}", b10, b01, win])
        print(f"   kappa_full - {b:22s} Δ={d:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
              f"McNemar p={p:.3f} OR={odds:.2f} CIexcl0={win}")
    with open(os.path.join(RESULTS, "paired_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "mean_diff(kappa_full-baseline)", "ci_lo", "ci_hi",
                    "mcnemar_p", "odds_ratio(kf/base)", "kf_only_correct",
                    "base_only_correct", "kf_wins_ci_excl_0"])
        w.writerows(paired_rows)

    # ---- ABLATIONS (§8) ----
    print("\n[4/4] Ablations...")
    abl_rows = run_ablations(primary)
    for name, note, val in abl_rows:
        print(f"   {name:30s} {val:7.3f}   {note}")
    with open(os.path.join(RESULTS, "ablations.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ablation", "top1_mean", "note"])
        for name, note, val in abl_rows:
            w.writerow([name, f"{val:.4f}", note])

    # ---- MISSES (§10) ----
    agg = {}
    for mrow in misses:
        agg.setdefault(mrow["pair_id"], mrow)  # one row per pair (stable)
    with open(os.path.join(RESULTS, "misses.csv"), "w", newline="") as f:
        cols = ["pair_id", "source", "predicted", "accepted_set", "same_group",
                "delta_ionic_radius_pm", "axis_of_max_disagreement",
                "candidate_pool_size"]
        w = csv.writer(f); w.writerow(cols)
        for pid in sorted(agg):
            w.writerow([agg[pid][c] for c in cols])

    # ---- VERDICT (§9) ----
    verdict, reason = decide_verdict(summary, paired_rows, controls_ok)
    write_verdict(summary, paired_rows, abl_rows, ctrl_rows, chance,
                  controls_ok, verdict, reason, primary, excl_outpool,
                  excl_unsourced)
    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(reason)
    print("=" * 70)


def decide_verdict(summary, paired_rows, controls_ok):
    if not controls_ok:
        return "INVALID", ("Negative controls did not collapse to chance -- the "
                           "harness leaks. No §5 result is interpretable (§7).")
    pr = {row[0]: row for row in paired_rows}
    def ci_excludes_0(b):
        return float(pr[b][2]) > 0  # ci_lo > 0
    kf = summary["kappa_full"]["top1_mean"]
    beats_b0 = kf > summary["B0_same_group"]["top1_mean"]
    beats_b1 = kf > summary["B1_mendeleev"]["top1_mean"]
    beats_b3_ci = ci_excludes_0("B3_IE")
    beats_b5_ci = ci_excludes_0("B5_multifeature")
    b5_margin = (kf - summary["B5_multifeature"]["top1_mean"]) * 100

    if beats_b3_ci and beats_b5_ci and beats_b0 and beats_b1 and b5_margin >= MARGIN_PP:
        return "PASS", ("kappa_full beats B3_IE and B5_multifeature with paired CI "
                        "excluding 0, clears the +5pp margin over B5, and beats "
                        "B0/B1. κ recovers expert substitution judgments beyond "
                        "standard descriptors.")
    if kf > summary["B3_IE"]["top1_mean"] and kf > summary["B0_same_group"]["top1_mean"] \
       and not (beats_b5_ci and b5_margin >= MARGIN_PP):
        return "PARTIAL", ("kappa_full ≈ B5_multifeature (margin %.1fpp, CI %s 0) but "
                           "exceeds single-descriptor / same-group floors. κ is a "
                           "compact re-encoding of standard descriptors; no gain over "
                           "a naive multi-feature NN. Drop \"predicts\"." %
                           (b5_margin, "excludes" if beats_b5_ci else "includes"))
    return "FAIL", ("kappa_full does not separate from B3_IE, or fails the "
                    "B0/B1 floor. Substitution claim retired: κ ≈ ionization "
                    "energy / ≈ periodic position.")


def write_verdict(summary, paired_rows, abl_rows, ctrl_rows, chance,
                  controls_ok, verdict, reason, primary, excl_outpool, excl_unsourced):
    lines = []
    lines.append("# Test A — Result & Verdict\n")
    lines.append("_Generated by `run_testA.py` against the frozen data. The "
                 "decision rule is the frozen §9 rule in `prereg.md`; no "
                 "re-tuning was done to change this outcome._\n")
    lines.append(f"\n**VERDICT: {verdict}**\n\n{reason}\n")
    lines.append("\n## Scope actually evaluated\n")
    lines.append(f"- Primary metric set (sourced & in-pool): **{len(primary)} pairs**\n")
    lines.append(f"- Excluded, out-of-pool under the same-block harness "
                 f"({len(excl_outpool)}): " +
                 ", ".join(f"{p['pair_id']} {p['source']}→{'|'.join(p['accepted'])}"
                           for p in excl_outpool) + "\n")
    lines.append(f"- Excluded, sourced=False ({len(excl_unsourced)}): " +
                 ", ".join(p["pair_id"] for p in excl_unsourced) + "\n")
    lines.append(f"- Chance floor = 1/mean(pool size) = **{chance:.3f}**\n")
    lines.append(f"- Negative controls: **{'PASS' if controls_ok else 'FAIL'}**\n")

    lines.append("\n## Top-1 accuracy (mean over 50 held-out evals, 95% bootstrap CI)\n")
    lines.append("| method | top-1 | 95% CI | top-3 | near-hit |\n|---|---|---|---|---|\n")
    order = ["kappa_full", "kappa_5D", "kappa_2D"] + BASELINES + \
            ["N1_random_coords", "N2_random_pick"]
    for m in order:
        s = summary[m]
        lines.append(f"| {m} | {s['top1_mean']:.3f} | "
                     f"[{s['top1_ci_lo']:.3f}, {s['top1_ci_hi']:.3f}] | "
                     f"{s['top3_mean']:.3f} | {s['nearhit_mean']:.3f} |\n")

    lines.append("\n## kappa_full vs baselines (paired)\n")
    lines.append("| baseline | Δ top-1 | paired 95% CI | McNemar p | OR (kf/base) | CI excl 0 |\n"
                 "|---|---|---|---|---|---|\n")
    for r in paired_rows:
        lines.append(f"| {r[0]} | {r[1]} | [{r[2]}, {r[3]}] | {r[4]} | {r[5]} | {r[8]} |\n")

    lines.append("\n## Ablations\n")
    lines.append("| ablation | top-1 | note |\n|---|---|---|\n")
    for name, note, val in abl_rows:
        lines.append(f"| {name} | {val:.3f} | {note} |\n")

    lines.append("\n## Controls\n")
    lines.append("| control | value | note |\n|---|---|---|\n")
    for name, val, note in ctrl_rows:
        lines.append(f"| {name} | {val:.4f} | {note} |\n")

    with open(os.path.join(RESULTS, "verdict.md"), "w") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
