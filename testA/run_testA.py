#!/usr/bin/env python3
"""
run_testA.py -- Ground Test Protocol for the kappa-substitution claim (v2).

v2 folds in the five fixes from the fix-handoff:
  FIX 1  fresh held-out set is the VERDICT; the original set is demoted to a
         non-decisive DEVELOPMENT set (k-fold holds out weight tuning, not the
         axis/gate *design*). Un-gated kappa is run alongside gated.
  FIX 2  B5_star: z-scored NN over the IDENTICAL raw inputs feeding kappa's
         axes -> isolates geometry (the only thing kappa claims). PASS keys on
         beating B5_star, not B5.
  FIX 3  Nadeau-Bengio corrected resampled t-test (variance-corrected for the
         overlapping CV folds) is the headline paired test; pooled McNemar is
         demoted to a descriptive sidebar. Minimum Detectable Effect (MDE) is
         computed; an underpowered null is INCONCLUSIVE, not PARTIAL.
  FIX 4  candidate pool is pre-registered (same-block) but a full-table
         sensitivity check is run and reported.
  FIX 5  decision rule (S9) updated: verdict on fresh set, PASS keys on
         B5_star, fourth outcome INCONCLUSIVE, gated-vs-ungated both reported.

Reads ONLY the frozen CSVs in data/. No CLOSE / no model-metric grading.
"""

import csv
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import t as tdist, binomtest
from sklearn.model_selection import RepeatedStratifiedKFold

# ---------------------------------------------------------------------------
# Frozen config (prereg §12 / prereg_v2 §12b)
# ---------------------------------------------------------------------------
SEED = 1729
N_SPLITS = 5
N_REPEATS = 10
ALPHA = 0.05
POWER = 0.80
ALPHA_FS = 7.2973525693e-3
GATE_DISCOUNT = 0.5
WEIGHT_GRID = [0.0, 0.5, 1.0, 2.0]
MARGIN_PP = 5.0
N_BOOT = 10000
KAPPA_AXES = ["kappa_xi_e", "kappa_xi_n", "kappa_xi_mag",
              "kappa_xi_bond", "kappa_xi_shell"]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

D10_SET = {"Zn", "Cd", "Hg", "Cu", "Ag", "Au", "Ga", "In", "Ge", "Sn"}
REFRACTORY_SET = {"Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re"}


# ---------------------------------------------------------------------------
# Load frozen data
# ---------------------------------------------------------------------------
def load_elements():
    el = {}
    with open(os.path.join(DATA, "element_properties.csv")) as f:
        for row in csv.DictReader(f):
            sym = row["symbol"]
            def num(k):
                v = row[k]
                return None if v == "" else float(v)
            el[sym] = {
                "Z": int(row["Z"]), "symbol": sym, "block": row["block"],
                "group": int(row["group"]), "period": int(row["period"]),
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


def load_pairs(fname):
    pairs = []
    with open(os.path.join(DATA, fname)) as f:
        for row in csv.DictReader(f):
            pairs.append({
                "pair_id": row["pair_id"], "source": row["source_symbol"],
                "accepted": row["accepted_substitutes"].split("|"),
                "application": row["application"], "block": row["block"],
                "citation": row["source_citation"],
                "sourced": row["sourced"].strip().lower() == "true",
            })
    return pairs


EL = load_elements()


def radius_of(sym):
    e = EL[sym]
    return e["ionic_radius_common_pm"] if e["ionic_radius_common_pm"] is not None \
        else e["atomic_radius_pm"]


# ---------------------------------------------------------------------------
# Shared harness (§4) -- pool mode is a pre-registered knob (Fix 4)
# ---------------------------------------------------------------------------
def candidate_pool(source, mode="block"):
    if mode == "full":
        pool = [s for s in EL if s != source]
    else:
        blk = EL[source]["block"]
        pool = [s for s in EL if EL[s]["block"] == blk and s != source]
        if len(pool) < 5:
            pool = [s for s in EL if s != source]
    return sorted(pool, key=lambda s: EL[s]["Z"])


def kappa_axis_value(sym, axis, xi_e_mode="alpha"):
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
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return lambda v: 0.0
    return lambda v: (v - lo) / span


def kappa_rank(source, pool, axes, weights, xi_e_mode="alpha",
               gates=None, return_axisgap=False):
    gates = gates or {}
    members = pool + [source]
    norms = {a: local_minmax([kappa_axis_value(m, a, xi_e_mode) for m in members])
             for a in axes}
    s_vec = {a: norms[a](kappa_axis_value(source, a, xi_e_mode)) for a in axes}
    scored, axisgap = [], {}
    for c in pool:
        c_vec = {a: norms[a](kappa_axis_value(c, a, xi_e_mode)) for a in axes}
        d2, gaps = 0.0, {}
        for a in axes:
            diff = s_vec[a] - c_vec[a]
            gaps[a] = abs(diff)
            d2 += weights.get(a, 1.0) * diff * diff
        dist = math.sqrt(d2)
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
    return (ranked, axisgap) if return_axisgap else ranked


def nn_rank_scalar(source, pool, valfn):
    sv = valfn(source)
    scored = sorted((abs(valfn(c) - sv), EL[c]["Z"], c) for c in pool)
    return [c for _, _, c in scored]


def zscore_nn(source, pool, featfns):
    members = pool + [source]
    transforms = []
    for fn in featfns:
        vals = np.array([fn(m) for m in members], dtype=float)
        mu, sd = vals.mean(), vals.std()
        transforms.append((fn, mu, sd if sd > 0 else 1.0))
    s_vec = np.array([(fn(source) - mu) / sd for fn, mu, sd in transforms])
    scored = sorted((float(np.linalg.norm(
        np.array([(fn(c) - mu) / sd for fn, mu, sd in transforms]) - s_vec)),
        EL[c]["Z"], c) for c in pool)
    return [c for _, _, c in scored]


# ---------------------------------------------------------------------------
# Methods (§3) -- common rank_candidates interface
# ---------------------------------------------------------------------------
def m_kappa_2D(s, p, ctx):
    ax = ["kappa_xi_e", "kappa_xi_n"]
    return kappa_rank(s, p, ax, {a: 1.0 for a in ax}, ctx.get("xi_e_mode", "alpha"))

def m_kappa_5D(s, p, ctx):
    return kappa_rank(s, p, KAPPA_AXES, {a: 1.0 for a in KAPPA_AXES},
                      ctx.get("xi_e_mode", "alpha"))

def m_kappa_full(s, p, ctx):       # gated, weights+gates train-selected
    hp = ctx["hp_gated"]
    return kappa_rank(s, p, ctx.get("axes", KAPPA_AXES), hp["weights"],
                      ctx.get("xi_e_mode", "alpha"), gates=hp["gates"])

def m_kappa_nogate(s, p, ctx):     # gates forced off, weights train-selected
    hp = ctx["hp_nogate"]
    return kappa_rank(s, p, ctx.get("axes", KAPPA_AXES), hp["weights"],
                      ctx.get("xi_e_mode", "alpha"), gates={})

def m_B0_same_group(s, p, ctx):
    sg, sp_ = EL[s]["group"], EL[s]["period"]
    return [t[-1] for t in sorted(
        (0 if EL[c]["group"] == sg else 1, abs(EL[c]["period"] - sp_),
         abs(EL[c]["group"] - sg), EL[c]["Z"], c) for c in p)]

def m_B1_mendeleev(s, p, ctx):
    return nn_rank_scalar(s, p, lambda x: EL[x]["mendeleev_number"])

def m_B2_electronegativity(s, p, ctx):
    return nn_rank_scalar(s, p, lambda x: EL[x]["electronegativity_pauling"])

def m_B3_IE(s, p, ctx):
    return nn_rank_scalar(s, p, lambda x: EL[x]["IE1_eV"])

def m_B4_radius(s, p, ctx):
    return nn_rank_scalar(s, p, radius_of)

def m_B5_multifeature(s, p, ctx):
    return zscore_nn(s, p, [
        lambda x: EL[x]["electronegativity_pauling"],
        lambda x: EL[x]["IE1_eV"], radius_of,
        lambda x: EL[x]["valence_electrons"]])

def m_B5_star(s, p, ctx):
    # IDENTICAL raw inputs to kappa's 5 axes (Fix 2): isolates geometry.
    return zscore_nn(s, p, [
        lambda x: EL[x]["IE1_eV"],                  # -> xi_e (alpha is inert)
        lambda x: EL[x]["atomic_radius_pm"],        # -> xi_n
        lambda x: EL[x]["unpaired_electrons"],      # -> xi_mag
        lambda x: EL[x]["electronegativity_pauling"],  # -> xi_bond
        lambda x: EL[x]["valence_electrons"]])      # -> xi_shell

def m_N1_random_coords(s, p, ctx):
    rng, coords = ctx["rng"], ctx.setdefault("_n1coords", {})
    def vec(x):
        if x not in coords:
            coords[x] = rng.normal(size=5)
        return coords[x]
    sv = vec(s)
    return [c for _, _, c in sorted(
        (float(np.linalg.norm(vec(c) - sv)), EL[c]["Z"], c) for c in p)]

def m_N2_random_pick(s, p, ctx):
    return list(ctx["rng"].permutation(p))


METHODS = {
    "kappa_full": m_kappa_full, "kappa_nogate": m_kappa_nogate,
    "kappa_5D": m_kappa_5D, "kappa_2D": m_kappa_2D,
    "B0_same_group": m_B0_same_group, "B1_mendeleev": m_B1_mendeleev,
    "B2_electronegativity": m_B2_electronegativity, "B3_IE": m_B3_IE,
    "B4_radius": m_B4_radius, "B5_multifeature": m_B5_multifeature,
    "B5_star": m_B5_star,
    "N1_random_coords": m_N1_random_coords, "N2_random_pick": m_N2_random_pick,
}
BASELINES = ["B0_same_group", "B1_mendeleev", "B2_electronegativity",
             "B3_IE", "B4_radius", "B5_multifeature", "B5_star"]
SINGLE_DESCR = ["B0_same_group", "B1_mendeleev", "B2_electronegativity",
                "B3_IE", "B4_radius"]
KAPPA_METHODS = {"kappa_full", "kappa_nogate"}


def topk_hit(ranked, accepted_set, k):
    return int(any(p in accepted_set for p in ranked[:k]))

def near_hit(pred, accepted_set):
    pg, pr = EL[pred]["group"], radius_of(pred)
    return int(any(EL[a]["group"] == pg or abs(radius_of(a) - pr) < 15
                   for a in accepted_set))


# ---------------------------------------------------------------------------
# Hyperparameter selection -- TRAIN FOLD ONLY (§5.3)
# ---------------------------------------------------------------------------
def train_top1(train_pairs, weights, gates, xi_e_mode="alpha"):
    hits = sum(topk_hit(kappa_rank(p["source"], p["pool"], KAPPA_AXES, weights,
                                   xi_e_mode, gates=gates), set(p["accepted"]), 1)
               for p in train_pairs)
    return hits / len(train_pairs)

def select_hyperparams(train_pairs, xi_e_mode="alpha", allow_gates=True):
    gates = {"d10": False, "mercury": False, "refractory": False}
    weights = {a: 1.0 for a in KAPPA_AXES}
    base = train_top1(train_pairs, weights, gates, xi_e_mode)
    if allow_gates:
        for g in ["d10", "mercury", "refractory"]:
            trial = dict(gates); trial[g] = True
            if train_top1(train_pairs, weights, trial, xi_e_mode) > base:
                gates = trial
                base = train_top1(train_pairs, weights, gates, xi_e_mode)
    for a in KAPPA_AXES:
        best_w, best_s = weights[a], base
        for w in WEIGHT_GRID:
            trial = dict(weights); trial[a] = w
            s = train_top1(train_pairs, trial, gates, xi_e_mode)
            if s > best_s:
                best_s, best_w = s, w
        weights[a], base = best_w, best_s
    return {"weights": weights, "gates": gates}


# ---------------------------------------------------------------------------
# Pair annotation + scoreability
# ---------------------------------------------------------------------------
def annotate_pairs(pairs, mode="block"):
    out = []
    for p in pairs:
        q = dict(p); q["pool"] = candidate_pool(p["source"], mode)
        q["in_pool"] = any(a in q["pool"] for a in p["accepted"])
        out.append(q)
    return out

def primary_subset(annotated):
    return [p for p in annotated if p["sourced"] and p["in_pool"]]


# ---------------------------------------------------------------------------
# Cross-validated evaluation (§5)
# ---------------------------------------------------------------------------
def run_cv(eval_pairs, methods=None, xi_e_mode="alpha", axes=None, seed=SEED,
           collect_misses=False):
    methods = methods or list(METHODS)
    rng = np.random.default_rng(seed)
    persistent = {m: {} for m in methods}
    blocks = [p["block"] for p in eval_pairs]
    idx = np.arange(len(eval_pairs))
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=seed)
    fold_acc = {m: [] for m in methods}
    fold_acc3 = {m: [] for m in methods}
    fold_near = {m: [] for m in methods}
    pair_hits = {m: defaultdict(list) for m in methods}
    test_sizes, train_sizes, misses = [], [], []

    for tr, te in rskf.split(idx, blocks):
        train_pairs = [eval_pairs[i] for i in tr]
        test_pairs = [eval_pairs[i] for i in te]
        test_sizes.append(len(test_pairs)); train_sizes.append(len(train_pairs))
        hp_gated = hp_nogate = None
        if "kappa_full" in methods:
            hp_gated = select_hyperparams(train_pairs, xi_e_mode, allow_gates=True)
        if "kappa_nogate" in methods:
            hp_nogate = select_hyperparams(train_pairs, xi_e_mode, allow_gates=False)
        for m in methods:
            ctx = {"rng": np.random.default_rng(rng.integers(1 << 30)),
                   "hp_gated": hp_gated, "hp_nogate": hp_nogate,
                   "xi_e_mode": xi_e_mode}
            if axes is not None:
                ctx["axes"] = axes
            ctx.update(persistent[m])
            h1 = h3 = hn = 0
            for p in test_pairs:
                ranked = METHODS[m](p["source"], p["pool"], ctx)
                acc = set(p["accepted"])
                t1 = topk_hit(ranked, acc, 1)
                h1 += t1; h3 += topk_hit(ranked, acc, 3); hn += near_hit(ranked[0], acc)
                pair_hits[m][p["pair_id"]].append(t1)
                if collect_misses and m == "kappa_full" and not t1:
                    pred = ranked[0]
                    _, ag = kappa_rank(p["source"], p["pool"], KAPPA_AXES,
                                       hp_gated["weights"], xi_e_mode,
                                       gates=hp_gated["gates"], return_axisgap=True)
                    g = ag[pred]
                    misses.append({
                        "pair_id": p["pair_id"], "source": p["source"], "predicted": pred,
                        "accepted_set": "|".join(p["accepted"]),
                        "same_group": int(any(EL[a]["group"] == EL[pred]["group"]
                                              for a in p["accepted"])),
                        "delta_ionic_radius_pm": round(min(abs(radius_of(a) -
                                                radius_of(pred)) for a in p["accepted"]), 1),
                        "axis_of_max_disagreement": max(g, key=g.get),
                        "candidate_pool_size": len(p["pool"])})
            n = len(test_pairs)
            fold_acc[m].append(h1 / n); fold_acc3[m].append(h3 / n); fold_near[m].append(hn / n)
            if m == "N1_random_coords" and "_n1coords" in ctx:
                persistent[m]["_n1coords"] = ctx["_n1coords"]
    meta = {"n_test": float(np.mean(test_sizes)), "n_train": float(np.mean(train_sizes))}
    return fold_acc, fold_acc3, fold_near, pair_hits, misses, meta


# ---------------------------------------------------------------------------
# Statistics (§6) -- Nadeau-Bengio corrected resampled t-test (Fix 3)
# ---------------------------------------------------------------------------
def nb_corrected(diffs, n_train, n_test, alpha=ALPHA, power=POWER):
    """Nadeau-Bengio variance-corrected paired t on per-fold differences.
    Returns mean, (ci_lo, ci_hi), t, p, se, MDE."""
    d = np.asarray(diffs, float)
    J = len(d)
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    corr = (1.0 / J) + (n_test / n_train)        # NB correction factor
    se = math.sqrt(corr * var) if var > 0 else 0.0
    df = J - 1
    tcrit = tdist.ppf(1 - alpha / 2, df)
    ci = (mean - tcrit * se, mean + tcrit * se)
    if se > 0:
        tstat = mean / se
        p = float(2 * tdist.sf(abs(tstat), df))
    else:
        tstat, p = 0.0, 1.0
    tbeta = tdist.ppf(power, df)
    mde = (tcrit + tbeta) * se                    # min detectable effect @ power
    return mean, ci, float(tstat), p, se, float(mde)

def bootstrap_ci_mean(rates, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); a = np.asarray(rates, float)
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def mcnemar(bin_a, bin_b):
    b01 = sum(1 for x, y in zip(bin_a, bin_b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(bin_a, bin_b) if x == 1 and y == 0)
    nd = b01 + b10
    p = 1.0 if nd == 0 else float(binomtest(min(b01, b10), nd, 0.5).pvalue)
    odds = (b10 + 0.5) / (b01 + 0.5)
    return b01, b10, p, float(odds)

def per_pair_rate(ph_m, ids):
    return [float(np.mean(ph_m[i])) if ph_m[i] else 0.0 for i in ids]
def per_pair_binary(ph_m, ids):
    return [int(np.mean(ph_m[i]) >= 0.5) if ph_m[i] else 0 for i in ids]


# ---------------------------------------------------------------------------
# Controls (§7)
# ---------------------------------------------------------------------------
def chance_floor(pairs):
    return float(np.mean([1.0 / len(p["pool"]) for p in pairs]))

def run_controls(pairs, tag):
    rows, chance = [], chance_floor(pairs)
    rows.append((f"{tag}:chance_floor", chance, "1/mean(pool size)"))
    rows.append((f"{tag}:N2_expected",
                 float(np.mean([sum(1 for a in p["accepted"] if a in p["pool"]) /
                                len(p["pool"]) for p in pairs])), "E[|acc∩pool|/|pool|]"))
    fa, *_ = run_cv(pairs, methods=["N1_random_coords", "N2_random_pick"])
    for m in ["N1_random_coords", "N2_random_pick"]:
        rows.append((f"{tag}:{m}_top1", float(np.mean(fa[m])), "CV mean top-1"))
    rng = np.random.default_rng(SEED + 99)
    pm = ["kappa_full", "B3_IE", "B5_star", "B0_same_group"]
    acc = {m: [] for m in pm}
    for _ in range(5):
        sh = [dict(p) for p in pairs]
        accs = [p["accepted"] for p in pairs]
        for p, j in zip(sh, rng.permutation(len(accs))):
            p["accepted"] = accs[j]
        fa2, *_ = run_cv(sh, methods=pm, seed=SEED + 1)
        for m in pm:
            acc[m].append(float(np.mean(fa2[m])))
    for m in pm:
        rows.append((f"{tag}:label_perm_{m}", float(np.mean(acc[m])),
                     "mean/5 perms (expect ≈ chance)"))
    return rows, chance


# ---------------------------------------------------------------------------
# Ablations (§8)
# ---------------------------------------------------------------------------
def run_ablations(pairs):
    rows = []
    for mode, label in [("alpha", "xi_e=IE*alpha_fs (14-15 default)"),
                        ("rawIE", "xi_e=IE (alpha removed by normalization)"),
                        ("poly",  "xi_e=IE/polarizability (12-13 form)")]:
        fa, *_ = run_cv(pairs, methods=["kappa_5D"], xi_e_mode=mode)
        rows.append(("xi_e_def:" + mode, label, float(np.mean(fa["kappa_5D"]))))
    fa_ref, *_ = run_cv(pairs, methods=["kappa_5D"])
    ref = float(np.mean(fa_ref["kappa_5D"]))
    rows.append(("dropaxis:none(5D ref)", "all 5 axes equal weight", ref))
    for drop in KAPPA_AXES:
        axes = [a for a in KAPPA_AXES if a != drop]
        METHODS["_tmp"] = (lambda axes: lambda s, p, ctx: kappa_rank(
            s, p, axes, {a: 1.0 for a in axes}, ctx.get("xi_e_mode", "alpha")))(axes)
        fa, *_ = run_cv(pairs, methods=["_tmp"])
        acc = float(np.mean(fa["_tmp"]))
        rows.append(("dropaxis:" + drop, f"Δacc vs 5D = {acc-ref:+.3f}", acc))
        del METHODS["_tmp"]
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(pairs, methods, collect_misses=False, tag=""):
    ids = [p["pair_id"] for p in pairs]
    fa, fa3, fn, ph, misses, meta = run_cv(pairs, methods=methods,
                                           collect_misses=collect_misses)
    summ = {}
    for m in methods:
        mean, lo, hi = bootstrap_ci_mean(per_pair_rate(ph[m], ids))
        summ[m] = {"top1_mean": float(np.mean(fa[m])), "top1_std": float(np.std(fa[m])),
                   "boot_lo": lo, "boot_hi": hi, "top3_mean": float(np.mean(fa3[m])),
                   "nearhit_mean": float(np.mean(fn[m])), "fold_acc": fa[m]}
    return summ, ph, ids, misses, meta


def paired_table(summ, ph, ids, meta, ref="kappa_full"):
    rows = []
    ref_bin = per_pair_binary(ph[ref], ids)
    for b in BASELINES:
        diffs = [summ[ref]["fold_acc"][j] - summ[b]["fold_acc"][j]
                 for j in range(len(summ[ref]["fold_acc"]))]
        mean, ci, tstat, p, se, mde = nb_corrected(diffs, meta["n_train"], meta["n_test"])
        b01, b10, mp, odds = mcnemar(ref_bin, per_pair_binary(ph[b], ids))
        rows.append({"baseline": b, "mean_diff": mean, "ci_lo": ci[0], "ci_hi": ci[1],
                     "t": tstat, "nb_p": p, "mde": mde, "mcnemar_p": mp, "odds": odds,
                     "ci_excl0": ci[0] > 0})
    return rows


# ---------------------------------------------------------------------------
# Verdict v2 (§9 / Fix 5)
# ---------------------------------------------------------------------------
def decide(summ, paired, controls_ok):
    if not controls_ok:
        return "INVALID", "Controls did not collapse to chance — harness leaks (§7)."
    pr = {r["baseline"]: r for r in paired}
    kf = summ["kappa_full"]["top1_mean"]
    bstar = pr["B5_star"]
    b3 = pr["B3_IE"]
    beats_floor = (kf > summ["B0_same_group"]["top1_mean"]
                   and kf > summ["B1_mendeleev"]["top1_mean"])
    margin_pp = bstar["mean_diff"] * 100
    mde_pp = bstar["mde"] * 100

    # FAIL: indistinguishable from IE, or below periodic floor
    if (not beats_floor) or (not b3["ci_excl0"] and b3["mean_diff"] * 100 < MARGIN_PP):
        return "FAIL", ("kappa_full does not clear the periodic floor (B0/B1) or is "
                        "not separated from B3_IE (NB CI %s 0, Δ=%.1fpp). κ ≈ "
                        "ionization energy / ≈ periodic position." %
                        ("excludes" if b3["ci_excl0"] else "includes", b3["mean_diff"]*100))
    # PASS: beats B5_star (geometry) with CI excluding 0 and ≥ margin
    if bstar["ci_excl0"] and margin_pp >= MARGIN_PP:
        return "PASS", ("kappa_full beats B5_star (identical raw inputs) by %.1fpp, "
                        "NB CI excludes 0, clears the floor and B3_IE. κ's *geometry* "
                        "recovers expert substitution beyond standard descriptors." % margin_pp)
    # No demonstrated edge over B5_star -> PARTIAL only if adequately powered
    if mde_pp <= MARGIN_PP:
        return "PARTIAL", ("kappa_full ≈ B5_star (Δ=%.1fpp, NB CI includes 0) and the "
                           "test was POWERED to detect %.0fpp (MDE=%.1fpp). Demonstrated "
                           "equivalence: κ adds no geometry over a z-scored NN on its own "
                           "raw inputs, while still beating single-descriptor baselines. "
                           "Drop \"predicts\"." % (margin_pp, MARGIN_PP, mde_pp))
    return "INCONCLUSIVE", ("kappa_full vs B5_star Δ=%.1fpp, NB CI includes 0, but MDE="
                            "%.1fpp > %.0fpp target — UNDERPOWERED. Cannot distinguish "
                            "equivalence from an undetected edge. Need a larger fresh set "
                            "(this is a null of detection, not a demonstrated equivalence)."
                            % (margin_pp, mde_pp, MARGIN_PP))


# ---------------------------------------------------------------------------
# Orchestration + deliverables (§11)
# ---------------------------------------------------------------------------
def fmt_summary_print(summ, order):
    print(f"   {'method':22s} {'top1':>7s} {'boot95%CI':>16s} {'top3':>7s} {'near':>7s}")
    for m in order:
        s = summ[m]
        print(f"   {m:22s} {s['top1_mean']:7.3f} [{s['boot_lo']:.3f},{s['boot_hi']:.3f}] "
              f"{s['top3_mean']:7.3f} {s['nearhit_mean']:7.3f}")

def main():
    os.makedirs(RESULTS, exist_ok=True)
    dev = annotate_pairs(load_pairs("substitution_pairs.csv"))
    fresh = annotate_pairs(load_pairs("substitution_pairs_fresh.csv"))
    dev_p, fresh_p = primary_subset(dev), primary_subset(fresh)
    methods = list(METHODS.keys())
    order = ["kappa_full", "kappa_nogate", "kappa_5D", "kappa_2D"] + BASELINES + \
            ["N1_random_coords", "N2_random_pick"]

    print("=" * 74)
    print(f"DEV set   : {len(dev)} pairs -> primary (sourced & in-pool) = {len(dev_p)} "
          f"[NON-DECISIVE]")
    print(f"FRESH set : {len(fresh)} pairs -> primary = {len(fresh_p)} [VERDICT]")
    print("=" * 74)

    # --- CONTROLS FIRST on the verdict (fresh) set (§7) ---
    print("\n[1/6] Controls on FRESH set (must pass before any verdict)...")
    ctrl_rows, chance = run_controls(fresh_p, "fresh")
    for n, v, note in ctrl_rows:
        print(f"   {n:34s} {v:7.4f}  {note}")
    controls_ok = all(v <= 2.0 * chance + 0.02 for n, v, _ in ctrl_rows
                      if ("label_perm" in n or "N1_random" in n or "N2_random_pick_top1" in n))
    print(f"   --> controls {'PASS' if controls_ok else 'FAIL (leak)'} (chance={chance:.4f})")

    # --- DEV evaluation (reported, non-decisive) ---
    print("\n[2/6] DEV-set evaluation (development; NON-DECISIVE)...")
    dev_summ, *_ = summarize(dev_p, methods)
    fmt_summary_print(dev_summ, order)

    # --- FRESH evaluation (VERDICT) ---
    print("\n[3/6] FRESH-set evaluation (VERDICT, same-block pool)...")
    fresh_summ, fresh_ph, fresh_ids, misses, meta = summarize(
        fresh_p, methods, collect_misses=True)
    fmt_summary_print(fresh_summ, order)

    # --- PAIRED STATS (Nadeau-Bengio) on FRESH ---
    print("\n[4/6] Paired NB-corrected stats on FRESH (kappa_full vs baselines)...")
    paired = paired_table(fresh_summ, fresh_ph, fresh_ids, meta, ref="kappa_full")
    for r in paired:
        print(f"   vs {r['baseline']:20s} Δ={r['mean_diff']*100:+5.1f}pp "
              f"NBci[{r['ci_lo']*100:+5.1f},{r['ci_hi']*100:+5.1f}] p={r['nb_p']:.3f} "
              f"MDE={r['mde']*100:4.1f}pp excl0={r['ci_excl0']} (McN p={r['mcnemar_p']:.3f})")
    # gated vs ungated
    gd = fresh_summ["kappa_full"]["top1_mean"] - fresh_summ["kappa_nogate"]["top1_mean"]
    print(f"   gated−ungated κ Δ={gd*100:+.1f}pp  (≈0 ⇒ gates were inert/overfit patches)")

    # --- POOL SENSITIVITY (Fix 4): full-table pool on same fresh pairs ---
    print("\n[5/6] Pool sensitivity: full-table pool on FRESH primary pairs...")
    fresh_full = annotate_pairs(
        [p for p in load_pairs("substitution_pairs_fresh.csv")
         if (p["source"], p["application"]) in
         {(q["source"], q["application"]) for q in fresh_p}], mode="full")
    full_summ, full_ph, full_ids, _, full_meta = summarize(
        fresh_full, ["kappa_full", "B5_star", "B3_IE", "B0_same_group"])
    fp = paired_table(full_summ, full_ph, full_ids, full_meta, ref="kappa_full")
    fp_star = next(r for r in fp if r["baseline"] == "B5_star")
    print(f"   full-pool chance={chance_floor(fresh_full):.4f}  "
          f"kappa_full={full_summ['kappa_full']['top1_mean']:.3f}  "
          f"B5_star={full_summ['B5_star']['top1_mean']:.3f}  "
          f"Δ={fp_star['mean_diff']*100:+.1f}pp NBci"
          f"[{fp_star['ci_lo']*100:+.1f},{fp_star['ci_hi']*100:+.1f}] excl0={fp_star['ci_excl0']}")

    # --- ABLATIONS on FRESH (§8) ---
    print("\n[6/6] Ablations on FRESH...")
    abl = run_ablations(fresh_p)
    for n, note, v in abl:
        print(f"   {n:28s} {v:7.3f}  {note}")

    # --- VERDICT ---
    verdict, reason = decide(fresh_summ, paired, controls_ok)

    # --- write deliverables ---
    write_all(dev_summ, fresh_summ, paired, fp_star, full_summ, abl, ctrl_rows,
              chance, controls_ok, verdict, reason, misses, gd, meta,
              dev, fresh, dev_p, fresh_p, order)
    print("\n" + "=" * 74)
    print(f"VERDICT (FRESH held-out set): {verdict}")
    print(reason)
    print("=" * 74)


def write_all(dev_summ, fresh_summ, paired, fp_star, full_summ, abl, ctrl_rows,
              chance, controls_ok, verdict, reason, misses, gd, meta,
              dev, fresh, dev_p, fresh_p, order):
    # results_summary.csv (fresh = verdict; dev columns prefixed)
    with open(os.path.join(RESULTS, "results_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "fresh_top1_mean", "fresh_top1_std", "fresh_boot_lo",
                    "fresh_boot_hi", "fresh_top3_mean", "fresh_nearhit_mean",
                    "dev_top1_mean"])
        for m in order:
            s, d = fresh_summ[m], dev_summ[m]
            w.writerow([m, f"{s['top1_mean']:.4f}", f"{s['top1_std']:.4f}",
                        f"{s['boot_lo']:.4f}", f"{s['boot_hi']:.4f}",
                        f"{s['top3_mean']:.4f}", f"{s['nearhit_mean']:.4f}",
                        f"{d['top1_mean']:.4f}"])
    # paired_stats.csv
    with open(os.path.join(RESULTS, "paired_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "mean_diff", "nb_ci_lo", "nb_ci_hi", "nb_t",
                    "nb_p", "mde", "mcnemar_p_sidebar", "odds_ratio", "ci_excl_0"])
        for r in paired:
            w.writerow([r["baseline"], f"{r['mean_diff']:.4f}", f"{r['ci_lo']:.4f}",
                        f"{r['ci_hi']:.4f}", f"{r['t']:.3f}", f"{r['nb_p']:.4f}",
                        f"{r['mde']:.4f}", f"{r['mcnemar_p']:.4f}", f"{r['odds']:.3f}",
                        r["ci_excl0"]])
    # controls.csv
    with open(os.path.join(RESULTS, "controls.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["control", "value", "note"])
        for n, v, note in ctrl_rows:
            w.writerow([n, f"{v:.4f}", note])
    # ablations.csv
    with open(os.path.join(RESULTS, "ablations.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ablation", "top1_mean", "note"])
        for n, note, v in abl:
            w.writerow([n, f"{v:.4f}", note])
    # misses.csv (fresh)
    agg = {}
    for mr in misses:
        agg.setdefault(mr["pair_id"], mr)
    with open(os.path.join(RESULTS, "misses.csv"), "w", newline="") as f:
        cols = ["pair_id", "source", "predicted", "accepted_set", "same_group",
                "delta_ionic_radius_pm", "axis_of_max_disagreement", "candidate_pool_size"]
        w = csv.writer(f); w.writerow(cols)
        for pid in sorted(agg):
            w.writerow([agg[pid][c] for c in cols])
    # verdict.md
    L = ["# Test A — Result & Verdict (v2, fresh held-out set)\n",
         "_Verdict computed on the FRESH set by the frozen `prereg_v2.md` §9 rule. "
         "Dev-set numbers are reported but NON-DECISIVE. No re-tuning._\n",
         f"\n**VERDICT (fresh held-out): {verdict}**\n\n{reason}\n",
         "\n## Scope\n",
         f"- Fresh primary (sourced & in-pool): **{len(fresh_p)} pairs** — the verdict.\n",
         f"- Dev primary (development, non-decisive): {len(dev_p)} pairs.\n",
         f"- Chance floor (fresh, same-block) = **{chance:.3f}**; controls "
         f"**{'PASS' if controls_ok else 'FAIL'}**.\n",
         f"- Avg fold sizes: n_test≈{meta['n_test']:.1f}, n_train≈{meta['n_train']:.1f} "
         f"(used in Nadeau–Bengio correction & MDE).\n",
         "\n## Top-1 accuracy — FRESH (verdict) vs DEV (non-decisive)\n",
         "| method | fresh top-1 | fresh boot95%CI | fresh top-3 | dev top-1 |\n|---|---|---|---|---|\n"]
    for m in order:
        s, d = fresh_summ[m], dev_summ[m]
        L.append(f"| {m} | {s['top1_mean']:.3f} | [{s['boot_lo']:.3f}, {s['boot_hi']:.3f}] "
                 f"| {s['top3_mean']:.3f} | {d['top1_mean']:.3f} |\n")
    L.append("\n## kappa_full vs baselines — FRESH, Nadeau–Bengio corrected (headline)\n")
    L.append("| baseline | Δ top-1 | NB 95% CI | NB p | MDE | CI excl 0 | McNemar p (sidebar) |\n"
             "|---|---|---|---|---|---|---|\n")
    for r in paired:
        L.append(f"| {r['baseline']} | {r['mean_diff']*100:+.1f}pp | "
                 f"[{r['ci_lo']*100:+.1f}, {r['ci_hi']*100:+.1f}]pp | {r['nb_p']:.3f} | "
                 f"{r['mde']*100:.1f}pp | {r['ci_excl0']} | {r['mcnemar_p']:.3f} |\n")
    L.append(f"\n**Gated − un-gated κ = {gd*100:+.1f}pp** "
             f"(≈0 ⇒ gates are inert/overfit patches, as predicted).\n")
    L.append("\n## Pool sensitivity (Fix 4): full-table pool, same fresh pairs\n")
    L.append(f"- kappa_full={full_summ['kappa_full']['top1_mean']:.3f}, "
             f"B5_star={full_summ['B5_star']['top1_mean']:.3f}, "
             f"Δ={fp_star['mean_diff']*100:+.1f}pp, NB CI "
             f"[{fp_star['ci_lo']*100:+.1f}, {fp_star['ci_hi']*100:+.1f}]pp, "
             f"excl 0 = {fp_star['ci_excl0']}. "
             f"{'Edge survives pool change.' if fp_star['ci_excl0'] else 'No edge under either pool.'}\n")
    L.append("\n## Ablations (fresh)\n| ablation | top-1 | note |\n|---|---|---|\n")
    for n, note, v in abl:
        L.append(f"| {n} | {v:.3f} | {note} |\n")
    L.append("\n## Controls (fresh)\n| control | value | note |\n|---|---|---|\n")
    for n, v, note in ctrl_rows:
        L.append(f"| {n} | {v:.4f} | {note} |\n")
    with open(os.path.join(RESULTS, "verdict.md"), "w") as f:
        f.write("".join(L))


if __name__ == "__main__":
    main()
