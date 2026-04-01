"""
apastats: Example Usage
========================

Demonstrates the three main analyses in a realistic OB research scenario:
  1. Descriptives + correlations table (JAP "Table 1")
  2. Moderation analysis (X x W -> Y) with interaction plot and Johnson Neyman
  3. Mediation analysis (X -> M -> Y) with path diagram

Scenario: A study of N = 400 employees examining how perceived
organisational support (POS) predicts job performance, with
psychological empowerment moderating that link, and job engagement
mediating it.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripting
import matplotlib.pyplot as plt

from apastats import descriptives_table, moderation_analysis, mediation_analysis


# ─── Simulate realistic survey data ──────────────────────────────────────

rng = np.random.default_rng(2024)
N = 400

# True latent scores (correlated)
pos = rng.normal(3.5, 0.85, N)           # Perceived Org Support (1-5)
empowerment = rng.normal(3.8, 0.90, N)   # Psychological Empowerment (1-5)
engagement = 0.45 * pos + rng.normal(0, 0.70, N) + 2.0  # Job Engagement
performance = (
    0.30 * engagement
    + 0.15 * pos
    + 0.25 * (pos - 3.5) * (empowerment - 3.8)  # interaction
    + rng.normal(0, 0.65, N)
    + 1.5
)
age = rng.normal(35, 8, N).clip(21, 65)
tenure = (age - 21) * rng.uniform(0.1, 0.6, N)

df = pd.DataFrame({
    "pos": pos,
    "empowerment": empowerment,
    "engagement": engagement,
    "performance": performance,
    "age": age,
    "tenure": tenure,
})

# Simulate items for Cronbach's alpha
for var, n_items in [("pos", 8), ("empowerment", 6), ("engagement", 5), ("performance", 4)]:
    for i in range(n_items):
        df[f"{var}_item{i+1}"] = df[var] + rng.normal(0, 0.25, N)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS & CORRELATIONS TABLE (Table 1)
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("1. DESCRIPTIVE STATISTICS & CORRELATIONS TABLE")
print("=" * 70)

desc = descriptives_table(
    data=df,
    variables=["pos", "empowerment", "engagement", "performance", "age", "tenure"],
    labels=[
        "Perceived org. support",
        "Psych. empowerment",
        "Job engagement",
        "Job performance",
        "Age",
        "Tenure (years)",
    ],
    alphas={
        "pos": [f"pos_item{i}" for i in range(1, 9)],
        "empowerment": [f"empowerment_item{i}" for i in range(1, 7)],
        "engagement": [f"engagement_item{i}" for i in range(1, 6)],
        "performance": [f"performance_item{i}" for i in range(1, 5)],
        # age and tenure have no alpha → em-dash on diagonal
    },
)

print(desc)
print()

# The raw statistics are also available:
print(f"Mean POS: {desc.means['pos']:.2f}")
print(f"Cronbach's alpha for POS: {desc.alphas['pos']:.2f}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. MODERATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("2. MODERATION ANALYSIS")
print("=" * 70)

mod = moderation_analysis(
    data=df,
    x="pos",
    w="empowerment",
    y="performance",
    controls=["age", "tenure"],
    jn=True,
)

print(mod)
print()

# Simple slopes summary
print("Simple Slopes:")
for ss in mod.simple_slopes:
    sig = "significant" if ss.p < .05 else "not significant"
    print(
        f"  At {ss.w_label} of empowerment: "
        f"b = {ss.b:.3f}, SE = {ss.se:.3f}, "
        f"t({ss.df:.0f}) = {ss.t:.2f}, p = {ss.p:.3f} ({sig})"
    )
print()

# Johnson Neyman boundaries
if mod.jn and mod.jn.boundaries:
    for bnd in mod.jn.boundaries:
        print(f"  J-N boundary (centred): {bnd:.3f}")
print()

# Save interaction plot
fig_mod = mod.plot(
    x_label="Perceived Organisational Support",
    y_label="Job Performance",
    w_label="Psychological Empowerment",
)
fig_mod.savefig("interaction_plot.png", dpi=300, bbox_inches="tight")
print("Saved: interaction_plot.png")

# Save J-N plot
fig_jn = mod.plot_jn(w_label="Psychological Empowerment (centred)")
fig_jn.savefig("jn_plot.png", dpi=300, bbox_inches="tight")
print("Saved: jn_plot.png")
plt.close("all")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 3. MEDIATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("3. MEDIATION ANALYSIS")
print("=" * 70)

med = mediation_analysis(
    data=df,
    x="pos",
    m="engagement",
    y="performance",
    covariates=["age", "tenure"],
    n_boot=10_000,
    seed=42,
)

print(med)
print()

# Interpretation
ie = med.indirect_effects[0]
print(f"Indirect effect (ab): {ie.ab:.3f}")
print(f"  Bootstrap SE: {ie.boot_se:.3f}")
print(f"  95% CI: [{ie.ci_lower:.3f}, {ie.ci_upper:.3f}]")
print(f"  Significant: {ie.significant}")
print()
print(f"Direct effect (c'): {med.direct_effect.b:.3f}, p = {med.direct_effect.p:.3f}")
print(f"Total effect (c):   {med.total_effect.b:.3f}, p = {med.total_effect.p:.3f}")
print()

# Save path diagram
fig_path = med.plot(title="Mediation: POS → Engagement → Performance")
fig_path.savefig("path_diagram.png", dpi=300, bbox_inches="tight")
print("Saved: path_diagram.png")
plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# BONUS: Parallel mediation with two mediators
# ═══════════════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("BONUS: PARALLEL MEDIATION (two mediators)")
print("=" * 70)

med2 = mediation_analysis(
    data=df,
    x="pos",
    m=["engagement", "empowerment"],
    y="performance",
    n_boot=10_000,
    seed=42,
)

print(med2)
print()
for ie in med2.indirect_effects:
    ci = f"[{ie.ci_lower:.3f}, {ie.ci_upper:.3f}]"
    print(f"  Indirect via {ie.mediator}: ab = {ie.ab:.3f}, 95% CI {ci}, sig = {ie.significant}")
print(f"  Total indirect: ab = {med2.total_indirect.ab:.3f}")
