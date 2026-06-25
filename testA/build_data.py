#!/usr/bin/env python3
"""
build_data.py  --  Freeze the two CSV inputs for Test A.

Emits:
  data/element_properties.csv
  data/substitution_pairs.csv
  data/CHECKSUMS.txt   (sha256 of both files)

Run ONCE, then never edit the CSVs by hand. run_testA.py reads only the
frozen CSVs.

------------------------------------------------------------------------
PROVENANCE / HONESTY NOTE  (see prereg.md S2.1 and the limitations block)
------------------------------------------------------------------------
* Atomic properties (IE1, Pauling chi, atomic & ionic radii, polarizability,
  valence/unpaired electron counts) are standard public-table values
  (NIST Atomic Spectra DB for IE1; Pauling electronegativity; Shannon
  ionic radii at typical coordination; CRC static dipole polarizabilities;
  ground-state configurations for electron counts). Values are encoded to
  the precision needed for a similarity ranking, not to spectroscopic
  precision. Small per-value errors do not change the methodology and are
  averaged over in the CV.

* mendeleev_number is a *reconstructed* Pettifor-style chemical-ordering
  scale (alkali-low -> noble-high, group-ordered). It is NOT a verified copy
  of Pettifor's published 1984 sequence. It exists only to give baseline
  B1 a fair, monotone chemical-similarity coordinate. This is flagged as a
  limitation; B1 should be read as "a Pettifor-style scale", not "Pettifor".

* The five kappa axes (xi_e, xi_n, xi_mag, xi_bond, xi_shell) are a
  TRANSPARENT RECONSTRUCTION of the claimed kappa-map from its documented
  inputs, because no original kappa source code or Paper-14/15 artifact
  exists in this repository. Each axis is a fixed, documented function of
  the base atomic properties (below). The default xi_e uses the
  fine-structure-constant form from the "14-15 text"; the polarizability
  form is computed only inside the ablation (S8). Freezing *a* definition
  is all the falsifiability of Test A requires: the verdict is honest with
  respect to whatever definition is frozen here.
"""

import csv
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Fine-structure constant (CODATA), used by the default xi_e definition.
ALPHA_FS = 7.2973525693e-3  # ~ 1/137.036

# ---------------------------------------------------------------------------
# Base atomic property table.
# Columns:
#   Z, symbol, block, group, period,
#   mendeleev_number (reconstructed Pettifor-style, see note above),
#   IE1_eV, polarizability_A3, electronegativity_pauling,
#   atomic_radius_pm, ionic_radius_common_pm,
#   valence_electrons, unpaired_electrons
# ionic_radius_common_pm: Shannon radius at the typical industrial oxidation
#   state / coordination (6-coord unless noted in comment).
# ---------------------------------------------------------------------------
# fmt: off
ELEMENTS = [
    # Z   sym  blk grp per  MN     IE1     pol     chi    r_at  r_ion  ve  unp
    (2,  "He", "s", 18, 1,  1,   24.587,  0.205,  0.00,  31,   None,  2,  0),
    (3,  "Li", "s", 1,  2,  10,   5.392, 24.30,  0.98, 152,   76,    1,  1),
    (4,  "Be", "s", 2,  2,  17,   9.323,  5.60,  1.57, 112,   45,    2,  0),
    (5,  "B",  "p", 13, 2,  72,   8.298,  3.03,  2.04,  85,   27,    3,  1),
    (6,  "C",  "p", 14, 2,  77,  11.260,  1.76,  2.55,  77,   16,    4,  2),
    (7,  "N",  "p", 15, 2,  82,  14.534,  1.10,  3.04,  75,   13,    5,  3),
    (8,  "O",  "p", 16, 2,  87,  13.618,  0.80,  3.44,  73,  140,    6,  2),
    (9,  "F",  "p", 17, 2,  92,  17.423,  0.56,  3.98,  71,  133,    7,  1),
    (10, "Ne", "p", 18, 2,  2,   21.565,  0.40,  0.00,  69,   None,  8,  0),
    (11, "Na", "s", 1,  3,  11,   5.139, 24.11,  0.93, 186,  102,    1,  1),
    (12, "Mg", "s", 2,  3,  18,   7.646, 10.60,  1.31, 160,   72,    2,  0),
    (13, "Al", "p", 13, 3,  73,   5.986,  6.80,  1.61, 143,   54,    3,  1),
    (14, "Si", "p", 14, 3,  78,   8.152,  5.38,  1.90, 117,   40,    4,  2),
    (15, "P",  "p", 15, 3,  83,  10.487,  3.63,  2.19, 110,   38,    5,  3),
    (16, "S",  "p", 16, 3,  88,  10.360,  2.90,  2.58, 104,  184,    6,  2),
    (17, "Cl", "p", 17, 3,  93,  12.968,  2.18,  3.16,  99,  181,    7,  1),
    (18, "Ar", "p", 18, 3,  3,   15.760,  1.64,  0.00,  97,   None,  8,  0),
    (19, "K",  "s", 1,  4,  12,   4.341, 43.40,  0.82, 227,  138,    1,  1),
    (20, "Ca", "s", 2,  4,  19,   6.113, 22.80,  1.00, 197,  100,    2,  0),
    (21, "Sc", "d", 3,  4,  48,   6.561, 17.80,  1.36, 162,   75,    3,  1),
    (22, "Ti", "d", 4,  4,  51,   6.828, 14.60,  1.54, 147,   61,    4,  2),
    (23, "V",  "d", 5,  4,  54,   6.746, 12.40,  1.63, 134,   64,    5,  3),
    (24, "Cr", "d", 6,  4,  57,   6.767, 11.60,  1.66, 128,   62,    6,  5),  # Cr3+ r_ion
    (25, "Mn", "d", 7,  4,  60,   7.434,  9.40,  1.55, 127,   67,    7,  5),  # Mn2+ HS
    (26, "Fe", "d", 8,  4,  61,   7.902,  8.40,  1.83, 126,   65,    8,  4),  # Fe2+ HS
    (27, "Co", "d", 9,  4,  64,   7.881,  7.50,  1.88, 125,   65,    9,  3),  # Co2+ HS
    (28, "Ni", "d", 10, 4,  67,   7.640,  6.80,  1.91, 124,   69,   10,  2),
    (29, "Cu", "d", 11, 4,  70,   7.726,  6.10,  1.90, 128,   73,   11,  1),  # Cu2+; d10 for Cu1+
    (30, "Zn", "d", 12, 4,  76,   9.394,  5.75,  1.65, 134,   74,   12,  0),
    (31, "Ga", "p", 13, 4,  74,   5.999,  8.12,  1.81, 122,   62,    3,  1),
    (32, "Ge", "p", 14, 4,  79,   7.900,  5.84,  2.01, 122,   53,    4,  2),
    (33, "As", "p", 15, 4,  84,   9.789,  4.31,  2.18, 121,   58,    5,  3),
    (34, "Se", "p", 16, 4,  89,   9.752,  3.77,  2.55, 117,  198,    6,  2),
    (35, "Br", "p", 17, 4,  94,  11.814,  3.05,  2.96, 114,  196,    7,  1),
    (36, "Kr", "p", 18, 4,  4,   14.000,  2.48,  0.00, 110,   None,  8,  0),
    (37, "Rb", "s", 1,  5,  13,   4.177, 47.30,  0.82, 248,  152,    1,  1),
    (38, "Sr", "s", 2,  5,  20,   5.695, 27.60,  0.95, 215,  118,    2,  0),
    (39, "Y",  "d", 3,  5,  49,   6.217, 22.70,  1.22, 180,   90,    3,  1),
    (40, "Zr", "d", 4,  5,  52,   6.634, 17.90,  1.33, 160,   72,    4,  0),  # Zr4+ d0
    (41, "Nb", "d", 5,  5,  55,   6.759, 15.70,  1.60, 146,   64,    5,  0),  # Nb5+ d0
    (42, "Mo", "d", 6,  5,  58,   7.092, 12.80,  2.16, 139,   59,    6,  0),  # Mo6+ d0
    (43, "Tc", "d", 7,  5,  62,   7.280, 11.40,  1.90, 136,   56,    7,  3),
    (44, "Ru", "d", 8,  5,  65,   7.361,  9.60,  2.20, 134,   62,    8,  4),
    (45, "Rh", "d", 9,  5,  68,   7.459,  8.60,  2.28, 134,   60,    9,  3),
    (46, "Pd", "d", 10, 5,  69,   8.337,  4.80,  2.20, 137,   86,   10,  0),  # Pd2+
    (47, "Ag", "d", 11, 5,  71,   7.576,  7.20,  1.93, 144,  115,   11,  0),  # Ag+ d10
    (48, "Cd", "d", 12, 5,  75,   8.994,  7.36,  1.69, 151,   95,   12,  0),
    (49, "In", "p", 13, 5,  75,   5.786, 10.20,  1.78, 167,   80,    3,  0),  # In3+
    (50, "Sn", "p", 14, 5,  80,   7.344,  7.70,  1.96, 140,   69,    4,  0),  # Sn4+
    (51, "Sb", "p", 15, 5,  85,   8.608,  6.60,  2.05, 141,   76,    5,  2),
    (52, "Te", "p", 16, 5,  90,   9.010,  5.50,  2.10, 137,  221,    6,  2),
    (53, "I",  "p", 17, 5,  95,  10.451,  5.35,  2.66, 133,  220,    7,  1),
    (54, "Xe", "p", 18, 5,  5,   12.130,  4.04,  0.00, 130,   None,  8,  0),
    (55, "Cs", "s", 1,  6,  14,   3.894, 59.60,  0.79, 265,  167,    1,  1),
    (56, "Ba", "s", 2,  6,  21,   5.212, 39.70,  0.89, 222,  135,    2,  0),
    (57, "La", "f", 3,  6,  50,   5.577, 31.10,  1.10, 187,  103,    3,  0),  # La3+ f0
    (58, "Ce", "f", 19, 6,  30,   5.539, 29.60,  1.12, 182,  101,    3,  1),  # Ce3+
    (59, "Pr", "f", 19, 6,  31,   5.464, 28.20,  1.13, 182,   99,    3,  2),
    (60, "Nd", "f", 19, 6,  32,   5.525, 31.40,  1.14, 181,   98,    3,  3),
    (61, "Pm", "f", 19, 6,  33,   5.582, 30.10,  1.13, 183,   97,    3,  4),
    (62, "Sm", "f", 19, 6,  34,   5.644, 28.80,  1.17, 180,   96,    3,  5),  # Sm3+
    (63, "Eu", "f", 19, 6,  35,   5.670, 27.70,  1.20, 199,   95,    3,  6),  # Eu3+ (7 for half-fill core)
    (64, "Gd", "f", 19, 6,  36,   6.150, 23.50,  1.20, 179,   94,    3,  7),
    (65, "Tb", "f", 19, 6,  37,   5.864, 25.50,  1.20, 176,   92,    3,  6),
    (66, "Dy", "f", 19, 6,  38,   5.939, 24.50,  1.22, 175,   91,    3,  5),
    (67, "Ho", "f", 19, 6,  39,   6.022, 23.60,  1.23, 174,   90,    3,  4),
    (68, "Er", "f", 19, 6,  40,   6.108, 22.70,  1.24, 173,   89,    3,  3),
    (69, "Tm", "f", 19, 6,  41,   6.184, 21.80,  1.25, 172,   88,    3,  2),
    (70, "Yb", "f", 19, 6,  42,   6.254, 20.90,  1.10, 194,   87,    3,  1),  # Yb3+
    (71, "Lu", "f", 3,  6,  43,   5.426, 21.90,  1.27, 172,   86,    3,  0),  # Lu3+ f14
    (72, "Hf", "d", 4,  6,  53,   6.825, 16.20,  1.30, 159,   71,    4,  0),  # Hf4+
    (73, "Ta", "d", 5,  6,  56,   7.550, 13.10,  1.50, 146,   64,    5,  0),  # Ta5+
    (74, "W",  "d", 6,  6,  59,   7.864, 11.10,  2.36, 139,   60,    6,  0),  # W6+
    (75, "Re", "d", 7,  6,  63,   7.834,  9.70,  1.90, 137,   53,    7,  3),
    (76, "Os", "d", 8,  6,  66,   8.438,  8.50,  2.20, 135,   63,    8,  4),
    (77, "Ir", "d", 9,  6,  101,  8.967,  7.60,  2.20, 136,   68,    9,  3),
    (78, "Pt", "d", 10, 6,  102,  8.959,  6.50,  2.28, 139,   80,   10,  0),  # Pt2+
    (79, "Au", "d", 11, 6,  103,  9.226,  5.80,  2.54, 144,   85,   11,  0),  # Au3+/Au+ d10
    (80, "Hg", "d", 12, 6,  74,  10.438,  5.00,  2.00, 151,  102,   12,  0),  # Hg2+
    (81, "Tl", "p", 13, 6,  76,   6.108,  7.60,  1.62, 170,   89,    3,  0),
    (82, "Pb", "p", 14, 6,  81,   7.417,  6.80,  2.33, 175,  119,    4,  0),  # Pb2+
    (83, "Bi", "p", 15, 6,  86,   7.286,  7.40,  2.02, 156,  103,    5,  2),
]
# fmt: on

BASE_COLS = [
    "Z", "symbol", "block", "group", "period", "mendeleev_number",
    "IE1_eV", "polarizability_A3", "electronegativity_pauling",
    "atomic_radius_pm", "ionic_radius_common_pm",
    "valence_electrons", "unpaired_electrons",
]


def base_rows():
    rows = []
    for rec in ELEMENTS:
        rows.append(dict(zip(BASE_COLS, rec)))
    return rows


# ---------------------------------------------------------------------------
# kappa axis definitions (transparent reconstruction; see header note).
# These materialised values are the DEFAULT definition. run_testA.py
# recomputes from base columns for ablations (polarizability vs alpha).
#
#   xi_e   (electronic)  : IE1_eV * ALPHA_FS         <- "14-15 text" default
#   xi_n   (nuclear/size): atomic_radius_pm
#   xi_mag (magnetic)    : unpaired_electrons
#   xi_bond(bonding)     : electronegativity_pauling
#   xi_shell(shell)      : valence_electrons
#
# Axes are stored in raw physical units here. The harness applies LOCAL
# (candidate-pool) normalisation identically to every method (prereg S4),
# so the absolute scale stored here is immaterial; only the definition is.
# ---------------------------------------------------------------------------
def kappa_axes(row):
    ie = row["IE1_eV"]
    return {
        "kappa_xi_e": round(ie * ALPHA_FS, 6),
        "kappa_xi_n": float(row["atomic_radius_pm"]),
        "kappa_xi_mag": float(row["unpaired_electrons"]),
        "kappa_xi_bond": float(row["electronegativity_pauling"]),
        "kappa_xi_shell": float(row["valence_electrons"]),
    }


KAPPA_COLS = ["kappa_xi_e", "kappa_xi_n", "kappa_xi_mag",
              "kappa_xi_bond", "kappa_xi_shell"]
ALL_COLS = BASE_COLS + KAPPA_COLS


def write_element_properties(path):
    rows = base_rows()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ALL_COLS)
        for r in rows:
            ax = kappa_axes(r)
            out = []
            for c in ALL_COLS:
                v = r.get(c, ax.get(c))
                out.append("" if v is None else v)
            w.writerow(out)
    return len(rows)


# ---------------------------------------------------------------------------
# Substitution pairs.
# Schema: pair_id, source_symbol, accepted_substitutes (pipe-sep),
#         application, block, source_citation, sourced(bool)
#
# accepted_substitutes is the FROZEN accepted-substitute set per
# (source, application) -- many-to-many is honoured by listing all accepted
# answers. A prediction is a hit iff it lands in this set (prereg S2.2).
# This set is frozen here, BEFORE any model is run. It is never expanded
# after seeing predictions.
#
# 'sourced' flags whether the substitution is documented in a named
# authoritative source CATEGORY (USGS Mineral Commodity Summaries; DOE
# Critical Materials Strategy 2010/2011; EU Critical Raw Materials reports
# 2017/2020/2023; named review articles). Per S2.1, citations are to the
# authoritative source category for well-established materials-science
# substitutions, NOT verified DOIs -- this is stated as a limitation.
# sourced=False pairs are reported SEPARATELY and excluded from the primary
# metric.
# ---------------------------------------------------------------------------
# fmt: off
PAIRS = [
    # pid, source, accepted (pipe), application, block, citation, sourced
    ("p01", "Pt", "Pd",            "autocatalyst",            "d", "USGS MCS Platinum-group metals", True),
    ("p02", "Pd", "Pt",            "autocatalyst",            "d", "USGS MCS Platinum-group metals", True),
    ("p03", "Rh", "Pd|Pt",         "autocatalyst (3-way)",    "d", "USGS MCS Platinum-group metals", True),
    ("p04", "Pt", "Pd|Ir",         "electrolyser catalyst",   "d", "DOE Critical Materials Strategy 2011", True),
    ("p05", "Ir", "Ru",            "PEM anode catalyst",      "d", "DOE Hydrogen Program review", True),
    ("p06", "Ru", "Ir",            "electrocatalyst",         "d", "USGS MCS Platinum-group metals", True),
    ("p07", "Nd", "Pr",            "NdFeB permanent magnet",  "f", "DOE Critical Materials Strategy 2011", True),
    ("p08", "Pr", "Nd",            "permanent magnet",        "f", "DOE Critical Materials Strategy 2011", True),
    ("p09", "Dy", "Tb",            "magnet HT coercivity",    "f", "DOE Critical Materials Strategy 2011", True),
    ("p10", "Tb", "Dy",            "magnet HT coercivity",    "f", "DOE Critical Materials Strategy 2011", True),
    ("p11", "Sm", "Nd",            "SmCo vs NdFeB magnet",    "f", "USGS MCS Rare Earths", True),
    ("p12", "Eu", "Tb",            "phosphor activator",      "f", "EU CRM 2020 report", True),
    ("p13", "Y",  "Gd",            "phosphor / garnet host",  "d", "EU CRM 2020 report", True),
    ("p14", "La", "Ce",            "FCC catalyst / mischmetal","f", "USGS MCS Rare Earths", True),
    ("p15", "Ce", "La",            "polishing / mischmetal",  "f", "USGS MCS Rare Earths", True),
    ("p16", "Co", "Ni",            "Li-ion cathode",          "d", "USGS MCS Cobalt", True),
    ("p17", "Co", "Ni|Mn",         "NMC cathode",             "d", "DOE Critical Materials 2010", True),
    ("p18", "Ni", "Co",            "battery cathode",         "d", "USGS MCS Nickel", True),
    ("p19", "Li", "Na",            "rechargeable battery",    "s", "USGS MCS Lithium", True),
    ("p20", "Ta", "Nb",            "electrolytic capacitor",  "d", "USGS MCS Tantalum", True),
    ("p21", "Nb", "Ta",            "capacitor / HSLA steel",  "d", "USGS MCS Niobium", True),
    ("p22", "Ta", "Nb|Hf",         "superalloy / carbide",    "d", "USGS MCS Tantalum", True),
    ("p23", "W",  "Mo",            "carbide / high-temp",     "d", "USGS MCS Tungsten", True),
    ("p24", "Mo", "W",            "alloy steel / catalyst",   "d", "USGS MCS Molybdenum", True),
    ("p25", "Re", "W|Mo",          "superalloy / refractory", "d", "USGS MCS Rhenium", True),
    ("p26", "In", "Ga",           "transparent conductor",    "p", "USGS MCS Indium", True),
    ("p27", "Ga", "In",           "compound semiconductor",   "p", "USGS MCS Gallium", True),
    ("p28", "Ge", "Si",           "infrared optics / semi",   "p", "USGS MCS Germanium", True),
    ("p29", "Si", "Ge",           "semiconductor",            "p", "USGS MCS Silicon", True),
    ("p30", "Ga", "Al",           "LED / nitride substrate",  "p", "EU CRM 2017 report", True),
    ("p31", "V",  "Nb",           "microalloying steel",      "d", "USGS MCS Vanadium", True),
    ("p32", "V",  "Cr",           "alloy steel",              "d", "USGS MCS Vanadium", True),
    ("p33", "Cr", "V",            "stainless / hardening",    "d", "USGS MCS Chromium", True),
    ("p34", "Hf", "Zr",           "nuclear / refractory",     "d", "USGS MCS Hafnium", True),
    ("p35", "Zr", "Hf|Ti",        "alloy / cladding",         "d", "USGS MCS Zirconium", True),
    ("p36", "Sr", "Ba",           "ferrite magnet / pyro",    "s", "USGS MCS Strontium", True),
    ("p37", "Ba", "Sr",           "drilling / ferrite",       "s", "USGS MCS Barium", True),
    ("p38", "Be", "Al",           "lightweight alloy",        "s", "USGS MCS Beryllium", True),
    ("p39", "Mg", "Al",           "lightweight structural",   "s", "USGS MCS Magnesium", True),
    ("p40", "Al", "Mg",           "lightweight structural",   "p", "USGS MCS Aluminum", True),
    ("p41", "Sb", "Bi",           "flame retardant / alloy",  "p", "USGS MCS Antimony", True),
    ("p42", "Bi", "Sb|Pb",        "low-melt / free-cut alloy","p", "USGS MCS Bismuth", True),
    ("p43", "Sn", "Bi",           "solder (Pb-free)",         "p", "EU CRM 2020 report", True),
    ("p44", "Te", "Se",           "thin-film PV / alloy",     "p", "USGS MCS Tellurium", True),
    ("p45", "Se", "Te",           "pigment / semiconductor",  "p", "USGS MCS Selenium", True),
    ("p46", "Ag", "Cu",           "electrical conductor",     "d", "USGS MCS Silver", True),
    ("p47", "Au", "Pd",           "wire bonding / contact",   "d", "USGS MCS Gold", True),
    ("p48", "Cu", "Al",           "electrical conductor",     "d", "USGS MCS Copper", True),
    ("p49", "Rb", "Cs",           "specialty / research",     "s", "USGS MCS Rubidium", True),
    ("p50", "Cs", "Rb|K",         "drilling fluid / catalyst","s", "USGS MCS Cesium", True),
    ("p51", "Gd", "Dy",           "MRI contrast / magnet",    "f", "EU CRM 2020 report", True),
    ("p52", "Er", "Yb",           "fiber laser dopant",       "f", "named review (RE photonics)", True),
    ("p53", "Yb", "Er",           "laser / dopant",           "f", "named review (RE photonics)", True),
    ("p54", "Ho", "Dy",           "magnetostrictive / dopant","f", "named review (RE)", True),
    ("p55", "Lu", "Y",            "scintillator host",        "f", "named review (scintillators)", True),
    ("p56", "Os", "Ir",           "hard alloy / electrode",   "d", "USGS MCS Platinum-group metals", True),
    ("p57", "Ti", "Al",           "lightweight structural",   "d", "USGS MCS Titanium", True),
    ("p58", "Mn", "Fe",           "steel / battery additive", "d", "USGS MCS Manganese", True),
    ("p59", "K",  "Na",           "fertilizer / chemical",    "s", "USGS MCS Potash", True),
    ("p60", "P",  "As",           "n-type dopant",            "p", "named review (semiconductors)", True),
    # ---- weaker / less-cleanly-sourced pairs: flagged sourced=False ----
    ("p61", "As", "P|Sb",         "III-V dopant / alloy",     "p", "general chemistry analogy", False),
    ("p62", "Tl", "In",           "specialty (toxic, limited)","p", "weak analogy", False),
    ("p63", "Cd", "Zn",           "coating / pigment",        "d", "USGS MCS Cadmium (declining)", False),
    ("p64", "Pm", "Sm",           "radio / niche",            "f", "no industrial substitution", False),
    ("p65", "Tc", "Re",           "no civilian supply",       "d", "analogy only", False),
]
# fmt: on

PAIR_COLS = ["pair_id", "source_symbol", "accepted_substitutes",
             "application", "block", "source_citation", "sourced"]


def write_pairs(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PAIR_COLS)
        for rec in PAIRS:
            pid, src, acc, app, blk, cite, sourced = rec
            w.writerow([pid, src, acc, app, blk, cite, str(bool(sourced))])
    return len(PAIRS)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(DATA, exist_ok=True)
    ep = os.path.join(DATA, "element_properties.csv")
    sp = os.path.join(DATA, "substitution_pairs.csv")
    n_el = write_element_properties(ep)
    n_pr = write_pairs(sp)

    # Validate every symbol referenced by pairs exists in properties.
    syms = {r["symbol"] for r in base_rows()}
    missing = set()
    for rec in PAIRS:
        missing |= {s for s in ([rec[1]] + rec[2].split("|")) if s not in syms}
    if missing:
        raise SystemExit(f"ERROR: pairs reference unknown symbols: {sorted(missing)}")

    with open(os.path.join(DATA, "CHECKSUMS.txt"), "w") as f:
        f.write(f"element_properties.csv  sha256  {sha256(ep)}\n")
        f.write(f"substitution_pairs.csv  sha256  {sha256(sp)}\n")

    print(f"wrote {n_el} elements -> {ep}")
    print(f"wrote {n_pr} pairs    -> {sp}")
    print(f"  sourced=True pairs : {sum(1 for r in PAIRS if r[6])}")
    print(f"  sourced=False pairs: {sum(1 for r in PAIRS if not r[6])}")
    with open(os.path.join(DATA, "CHECKSUMS.txt")) as f:
        print(f.read().strip())


if __name__ == "__main__":
    main()
