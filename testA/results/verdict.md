# Test A — Result & Verdict (v2, fresh held-out set)
_Verdict computed on the FRESH set by the frozen `prereg_v2.md` §9 rule. Dev-set numbers are reported but NON-DECISIVE. No re-tuning._

**VERDICT (fresh held-out): INCONCLUSIVE**

kappa_full vs B5_star Δ=3.1pp, NB CI includes 0, but MDE=20.3pp > 5pp target — UNDERPOWERED. Cannot distinguish equivalence from an undetected edge. Need a larger fresh set (this is a null of detection, not a demonstrated equivalence).

## Scope
- Fresh primary (sourced & in-pool): **44 pairs** — the verdict.
- Dev primary (development, non-decisive): 53 pairs.
- Chance floor (fresh, same-block) = **0.053**; controls **PASS**.
- Avg fold sizes: n_test≈8.8, n_train≈35.2 (used in Nadeau–Bengio correction & MDE).

## Top-1 accuracy — FRESH (verdict) vs DEV (non-decisive)
| method | fresh top-1 | fresh boot95%CI | fresh top-3 | dev top-1 |
|---|---|---|---|---|
| kappa_full | 0.326 | [0.198, 0.459] | 0.602 | 0.518 |
| kappa_nogate | 0.326 | [0.198, 0.459] | 0.602 | 0.557 |
| kappa_5D | 0.317 | [0.182, 0.455] | 0.612 | 0.511 |
| kappa_2D | 0.068 | [0.000, 0.159] | 0.386 | 0.303 |
| B0_same_group | 0.319 | [0.182, 0.455] | 0.659 | 0.339 |
| B1_mendeleev | 0.296 | [0.159, 0.432] | 0.681 | 0.301 |
| B2_electronegativity | 0.182 | [0.068, 0.295] | 0.477 | 0.189 |
| B3_IE | 0.226 | [0.114, 0.364] | 0.432 | 0.189 |
| B4_radius | 0.068 | [0.000, 0.159] | 0.408 | 0.208 |
| B5_multifeature | 0.295 | [0.159, 0.432] | 0.659 | 0.454 |
| B5_star | 0.295 | [0.159, 0.432] | 0.635 | 0.530 |
| N1_random_coords | 0.023 | [0.000, 0.068] | 0.227 | 0.075 |
| N2_random_pick | 0.046 | [0.025, 0.068] | 0.164 | 0.060 |

## kappa_full vs baselines — FRESH, Nadeau–Bengio corrected (headline)
| baseline | Δ top-1 | NB 95% CI | NB p | MDE | CI excl 0 | McNemar p (sidebar) |
|---|---|---|---|---|---|---|
| B0_same_group | +0.6pp | [-23.7, +24.9]pp | 0.958 | 34.6pp | False | 1.000 |
| B1_mendeleev | +3.0pp | [-21.7, +27.6]pp | 0.810 | 35.1pp | False | 0.824 |
| B2_electronegativity | +14.3pp | [-1.9, +30.6]pp | 0.082 | 23.1pp | False | 0.092 |
| B3_IE | +9.9pp | [-9.2, +29.1]pp | 0.303 | 27.3pp | False | 0.302 |
| B4_radius | +25.7pp | [+9.3, +42.2]pp | 0.003 | 23.4pp | True | 0.004 |
| B5_multifeature | +3.1pp | [-12.6, +18.7]pp | 0.697 | 22.3pp | False | 0.754 |
| B5_star | +3.1pp | [-11.2, +17.3]pp | 0.665 | 20.3pp | False | 0.754 |

**Gated − un-gated κ = +0.0pp** (≈0 ⇒ gates are inert/overfit patches, as predicted).

## Pool sensitivity (Fix 4): full-table pool, same fresh pairs
- kappa_full=0.280, B5_star=0.273, Δ=+0.6pp, NB CI [-8.8, +10.1]pp, excl 0 = False. No edge under either pool.

## Ablations (fresh)
| ablation | top-1 | note |
|---|---|---|
| xi_e_def:alpha | 0.317 | xi_e=IE*alpha_fs (14-15 default) |
| xi_e_def:rawIE | 0.317 | xi_e=IE (alpha removed by normalization) |
| xi_e_def:poly | 0.386 | xi_e=IE/polarizability (12-13 form) |
| dropaxis:none(5D ref) | 0.317 | all 5 axes equal weight |
| dropaxis:kappa_xi_e | 0.409 | Δacc vs 5D = +0.092 |
| dropaxis:kappa_xi_n | 0.316 | Δacc vs 5D = -0.001 |
| dropaxis:kappa_xi_mag | 0.249 | Δacc vs 5D = -0.068 |
| dropaxis:kappa_xi_bond | 0.272 | Δacc vs 5D = -0.045 |
| dropaxis:kappa_xi_shell | 0.272 | Δacc vs 5D = -0.045 |

## Controls (fresh)
| control | value | note |
|---|---|---|
| fresh:chance_floor | 0.0531 | 1/mean(pool size) |
| fresh:N2_expected | 0.0531 | E[|acc∩pool|/|pool|] |
| fresh:N1_random_coords_top1 | 0.0450 | CV mean top-1 |
| fresh:N2_random_pick_top1 | 0.0669 | CV mean top-1 |
| fresh:label_perm_kappa_full | 0.0046 | mean/5 perms (expect ≈ chance) |
| fresh:label_perm_B3_IE | 0.0090 | mean/5 perms (expect ≈ chance) |
| fresh:label_perm_B5_star | 0.0046 | mean/5 perms (expect ≈ chance) |
| fresh:label_perm_B0_same_group | 0.0137 | mean/5 perms (expect ≈ chance) |
