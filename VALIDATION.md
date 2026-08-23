# Verification and Validation

Two different questions, answered by two different references, kept apart on purpose.

| | Question | Reference |
|---|---|---|
| **Verification** | Did I implement the maths correctly? | CR-2144 Table IX-4, dimensional derivatives |
| **Validation** | Does the model describe a real aircraft? | CR-2144 Table IX-5, transfer-function factors |

Having both separates two failures that otherwise look identical — and that is not
theoretical here. It found a missing physical term. See §3.

Reference throughout: Heffley, R.K. and Jewell, W.F., *Aircraft Handling Qualities
Data*, NASA CR-2144, December 1972, Section IX. Flight Condition 2: power approach, sea
level, M = 0.249, 165 KTAS, 564,032 lb.

---

## 1. Verification — the implementation

The nonlinear model is trimmed, then linearised numerically about that trim, and the
resulting state matrix compared **element by element** against CR-2144's separately
tabulated dimensional derivatives.

| Term | numerical | CR-2144 | diff | |
|---|---:|---:|---:|---|
| Xu | −0.010598 | −0.010800 | −1.9% | speed damping |
| Xw | 0.102943 | 0.106000 | −2.9% | |
| Zu | −0.150562 | −0.150000 | +0.4% | |
| Zw | −0.611309 | −0.613000 | −0.3% | heave damping |
| Mu | 0.000578 | 0.000594 | −2.7% | |
| Mw | −0.006325 | −0.006332 | −0.1% | pitch stiffness |
| Mq | −0.436376 | −0.437000 | −0.1% | pitch damping |

**Standard applied:** every term within 5%. All seven pass, six within 3%.

Two further structural checks:

- **`∂θ/∂q = 1.000000`** exactly. This is a kinematic identity, not a modelling result;
  anything else means the Euler-chart wrapper is wrong.
- **Longitudinal/lateral coupling ratio = 0.0** to machine precision. A symmetric
  aircraft in symmetric flight must decouple exactly, so a non-zero value would indicate
  an asymmetry that should not exist.

---

## 2. Validation — the physics

Eigenvalues of the linearised longitudinal subsystem against the denominator factors of
CR-2144's bare-airframe elevator transfer function.

| Mode | | computed | CR-2144 | diff |
|---|---|---:|---:|---:|
| Short period | ζ | 0.5906 | 0.6290 | −6.1% |
| | ωₙ | 0.8923 | 0.9100 | **−1.9%** |
| Phugoid | ζ | 0.0168 | 0.0228 | −26.5% |
| | ωₙ | 0.1265 | 0.1270 | **−0.4%** |

**Frequencies agree to within 2%.**

### Where it disagrees, and why

**Phugoid damping is the worst, and this was predicted before the model was built.**
Phugoid damping is approximately `CD/(√2·CL)` — set almost entirely by drag. Our `CD`
carries no Mach term because CR-2144 does not tabulate `CD_M` for this condition
(ASSUMPTIONS §1.3), and thrust is speed-independent (§2). Both omissions land squarely
on this quantity.

The absolute error is **0.006 on a quantity of 0.023.** For a mode that is barely damped
at all, that is a small absolute discrepancy expressed as a large relative one.

**Short-period damping (−6.1%)** is more sensitive to `Cm_alphadot` and `Cm_q` than the
frequency is; both are transcribed values, and `Cm_alphadot` in particular is quoted to
only two significant figures in Table IX-2.

### Lateral modes — character only

CR-2144's lateral transfer functions were **not transcribed**, so these are checked for
physical character rather than against published numbers. This is a genuine gap, stated
rather than glossed.

| Mode | Result | Expected character |
|---|---|---|
| Dutch roll | ζ = 0.129, ωₙ = 0.760, T = 8.3 s | oscillatory, lightly damped ✓ |
| Roll subsidence | τ = 0.82 s | fast real root, stable ✓ |
| Spiral | τ = 27.0 s | very slow real root ✓ |

---

## 3. The finding that justifies keeping them separate

The first verification run produced this:

```
Zw   0.1%    Mw   0.4%    Mq  -0.1%    Xw  -2.7%     ← w and q derivatives
Xu -20.4%    Zu  14.2%    Mu -59.6%                   ← every u derivative
```

**The error was confined to one column.** All three failing terms were `∂/∂u` — how
forces change with speed — which pointed at exactly one cause: the coefficients had been
built without the **Mach derivatives**, `CL_M` and `Cm_M`. Adding them:

| | before | after |
|---|---:|---:|
| Xu | −20.4% | −1.9% |
| Zu | +14.2% | +0.4% |
| Mu | −59.6% | −2.7% |
| phugoid ωₙ | −3.4% | −0.4% |

**Validation alone would have missed this.** The modes had already looked acceptable —
frequencies within a few percent — and nothing about the pole map suggested a missing
physical term. Only element-by-element comparison against a *different table* localised
it, and the fact that the error was confined to one column is what identified *which*
term was missing rather than merely that something was wrong.

That is the entire argument for treating verification and validation as separate
evidence:

- Numerical Jacobian disagrees with Table IX-4 → **the implementation is wrong**, and
  the offending matrix element points at the equation.
- Matrices agree but eigenvalues miss Table IX-5 → the implementation is fine and the
  **model** is the limitation. That is a result, not a mistake.

---

## 4. Supporting verification

### 4.1 Trim

| Check | Result |
|---|---|
| Residuals | 3.6 × 10⁻¹⁵ |
| α at trim | 5.520° vs CR-2144's published 5.70° (−3.2%) |
| Holds 60 s, controls fixed | speed drift 0.000, altitude change 0.000 m |
| **Negative control** | a state 10% fast *does* drift, so the hold test has power |

### 4.2 Rigid-body mechanics — invariants, not stored outputs

| Property | Standard | Reference |
|---|---|---|
| Kinetic energy conserved, torque-free | < 10⁻¹¹ relative | exact invariant |
| Angular momentum conserved in the inertial frame | < 10⁻⁹ relative | exact invariant |
| Body-frame **H** components *not* constant | negative control | Euler's equation |
| Spherical inertia → constant body rates | < 10⁻¹² | ω × Iω vanishes identically |
| Intermediate-axis instability | perturbation grows > 10⁴ × | Dzhanibekov effect |
| Major/minor axis spin stable | bounded < 10⁻² | " |
| RK4 order of accuracy | ratios 15.95, 15.97, 15.97 | fourth order |
| Free fall | w = gt, h = ½gt² to 10⁻⁹ | closed form |
| Quaternion norm drift, 20k steps | < 10⁻¹² | skew-symmetric Ω |

The intermediate-axis test is the sharpest of these: it can only pass with a correctly
signed `ω × Iω` and distinct principal moments, and no plot would reveal its absence.

### 4.3 Attitude — against an independent implementation

Rotations are implemented from scratch, then checked against `scipy.spatial.transform`,
which uses scalar-*last* ordering and returns the *active* rotation. The comparison
therefore requires an explicit reorder and transpose — stating that mismatch is the
point, since an unexamined convention mismatch is precisely the bug the test exists to
catch.

Also: DCM orthogonality to 10⁻¹³, determinant +1 (not −1, which would be a mirrored
aircraft), Shepperd extraction stable through 180° rotations, and hand-computable
geometry — 90° yaw puts North on the left wing, 90° pitch puts North along body +z.

### 4.4 Atmosphere

Checked against the published ISA table at eight altitudes: temperature to 0.01 K,
pressure and density to 2 × 10⁻⁵ relative. Hydrostatic balance `dp/dh = −ρg` verified by
finite difference at every altitude, which tests the *derivation* rather than just the
endpoints.

**A finding worth recording:** five tests failed initially, but only at 5 km and 8 km,
while 0, 11, 15 and 20 km matched to 0.00%. A wrong exponent would have degraded smoothly
with altitude. The cause was that ISA is defined in **geopotential** altitude and two
table rows had been copied from a *geometric* table — 8000 m geometric is 7989.9 m
geopotential, worth 52 Pa, which closed the discrepancy exactly.

### 4.5 Source data transcription

| Cross-check | Result |
|---|---:|
| Lift vs weight (ties CL, q, S, mass) | −0.20% |
| Mach from airspeed and speed of sound | +0.00% |
| Dynamic pressure vs ½ρV² | −0.38% |
| `Zw`: Table IX-2 implies vs Table IX-4 states | −1.52% |
| Aspect ratio | 6.96 (747 is ~7.0) |

Every sign forced by physics is asserted individually, because a lost minus is the most
likely transcription error.

---

## 5. What has **not** been validated

- **Lateral modes against published values.** Character only. CR-2144's lateral transfer
  functions were not transcribed.
- **Any flight condition other than F/C 2.** One point in the envelope.
- **Cruise or high-altitude behaviour.** The validated condition is sea-level approach.
- **Against flight test data.** CR-2144 itself derives from a Boeing simulator
  description (D6-30643), so this is a model-to-model comparison, not a comparison
  against measured aircraft response. That is a real limitation of the reference, not
  just of this work.
- **Large-amplitude or post-stall behaviour.** The linear aerodynamic model forbids it,
  and the code warns rather than extrapolating.
- **`Cn_r` and `Cn_dr` signs**, which remain to be confirmed by eye against page 221 of
  the scan.

---

## 6. Reproducing this

```bash
pip install -e ".[dev]"
pytest                                  # 222 tests, ~45 s
python scripts/make_readme_figures.py
python scripts/make_visualiser.py
```
