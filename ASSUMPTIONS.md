# Assumptions and Limitations

Every modelling choice that could reasonably have been made differently, why it was
made, and what it costs. Ordered roughly by how much the result depends on it.

Anything here is a fair target for challenge. Where an assumption is a known weakness,
it says so.

---

## 1. Aerodynamics

### 1.1 The aerodynamic model is linear

Coefficients are built as a first-order expansion about the published trim point:

```
C = C0 + C_alpha·alpha + C_q·qhat + C_delta·delta + C_M·dM
```

**Why:** CR-2144 publishes *derivatives*, not lookup tables. Nothing else is supported
by the data.

**Cost:** the model is only valid within a few degrees of the reference condition. It
cannot represent stall, and a linear lift slope extrapolated past ~15° predicts lift
that does not exist. `AeroModel` warns rather than silently extrapolating.

**The claim this rules out:** this is **not a nonlinear aerodynamic model.** What *is*
nonlinear is the rigid-body mechanics that consumes these forces — the `ω × v` and
`ω × Iω` transport terms, the quaternion kinematics, gravity rotating through attitude.
That is what trim solves against and what linearisation exercises. Describing the
aerodynamics as nonlinear would be overselling it.

### 1.2 Zero-incidence intercepts are backed out from trim values

`CL0 = CL_trim − CL_alpha·alpha_trim`, and similarly for CD and Cm.

**Why:** CR-2144 tabulates the coefficient *at trim* and its slope, not the intercept.

**Cost:** this assumes the lift slope holds linearly all the way back to zero incidence.
For a flapped power-approach configuration it demonstrably does not. Acceptable only
because everything of interest happens within a few degrees of trim — and it means the
model should never be trusted at low incidence.

### 1.3 No drag rise with Mach

`CD` carries `CD_alpha` but no `CD_M` term, because CR-2144 does not tabulate one for
this condition.

**Cost:** this is the single largest contributor to the remaining phugoid damping error.
Phugoid damping is approximately `CD/(√2·CL)`, so it is set almost entirely by drag, and
a missing speed-dependence in drag shows up there before anywhere else. Measured
disagreement with the published value: −26.5%, an absolute error of 0.006 on 0.023.

### 1.4 Lateral derivatives are used but the resulting modes are not validated

CR-2144's lateral transfer functions were not transcribed from the scan.

**Cost:** dutch roll, roll subsidence and spiral are checked for **physical character
only** — oscillatory versus real, stable, correct ordering of timescales. They are not
compared against published numbers, and the README and tests say so explicitly.

---

## 2. Propulsion

Thrust acts along a line inclined 2.5° to the fuselage reference line, at a 10 ft
moment arm, with magnitude proportional to throttle.

**Excluded:** engine dynamics, spool-up lag, thrust lapse with altitude and Mach, and
any speed dependence.

**Why:** trim needs only the steady value, and at the fidelity of a derivative-based
model, engine dynamics would be false precision.

**Cost:** real engine thrust falls with airspeed, which contributes to `Xu`. Our `Xu`
agrees with CR-2144 to 1.9%, so the omission is evidently small here — but it would not
be for a case with a larger speed excursion.

---

## 3. Dynamics

### 3.1 Flat, non-rotating Earth

**Why:** over the timescale of a mode analysis at aircraft speeds, Earth rotation and
curvature contribute far less than the aerodynamic uncertainty.

**Cost:** invalid for long-range navigation or anything where Coriolis matters. Not a
limitation for this project's purpose.

### 3.2 Rigid airframe

No structural flexibility, no aeroelastic effects.

**Cost:** CR-2144's own cruise-condition plots are labelled *"Flexible"*, so the source
data for those conditions already embeds aeroelastic corrections that a rigid model
cannot reproduce. For the sea-level approach conditions used here the effect is small,
but a 747 is a large flexible aircraft and its first structural modes are not far above
the short period.

### 3.3 Constant mass and inertia

No fuel burn, no configuration change.

**Cost:** negligible over the seconds-to-minutes of a mode analysis.

### 3.4 Density frozen at the trim altitude during linearisation

`trimmed_derivative` evaluates the atmosphere once.

**Why:** so the linearisation measures the aircraft, not the atmosphere model. A
height-dependent density would add a term to the phugoid that CR-2144's state matrix
does not contain, making the verification comparison invalid.

**Cost:** the phugoid genuinely does involve altitude excursions, so this is a real
simplification — and it is one CR-2144 shares, which is why the comparison is fair.

---

## 4. Numerics

### 4.1 Fixed-step RK4

**Why:** deterministic and reproducible, and adaptive stepping injects noise into a
numerical Jacobian at exactly the scale being measured.

**Cost:** no automatic error control. Mitigated by an explicit fourth-order convergence
test, which also documents where the floating-point noise floor lies.

### 4.2 Quaternion renormalised every step

**Why:** the kinematic equation conserves the norm analytically, since Ω is
skew-symmetric, so any drift is pure integration error.

**Cost:** none of consequence. The drift is exposed by `quaternion_drift` as a
diagnostic; measured below 10⁻¹² over 20,000 steps.

### 4.3 Linearisation performed in an Euler chart

**Why:** a quaternion carries four components under one norm constraint, so linearising
in it produces a redundant direction and a spurious zero eigenvalue. At a trim point far
from gimbal lock the Euler chart is well conditioned and gives exactly nine dynamic
states.

**Cost:** none here, but the approach would fail near vertical pitch. **The physics is
untouched** — this is a change of coordinates for taking a derivative, not a change of
model, and integration always uses the quaternion.

### 4.4 Perturbation size scaled per state

`h = ε^(1/3) × max(|state|, scale)`.

**Cost:** the scale table is a judgement, not a derivation. The sweep in
`perturbation_sweep` demonstrates the optimum lands where theory predicts for u, and
also shows that the V-curve does *not* appear for states in which the dynamics are
nearly linear.

---

## 5. Source data

### 5.1 Transcribed from a 1972 scan

CR-2144's text layer is unusable OCR, so values were read from page images.

**Cost:** a mistyped derivative produces a model that trims perfectly and has entirely
wrong modes — a silent failure. Mitigated by cross-checks against physics rather than
against the transcription: lift equals weight to 0.20%, Mach reconciles with airspeed
and altitude to 0.00%, dynamic pressure to 0.38%, and `Zw` implied by Table IX-2 agrees
with Table IX-4 to 1.52%.

**Outstanding:** `Cn_r` and `Cn_dr` had faint minus signs in the scan. Both are set
negative because yaw damping and rudder yawing moment must be, and Table IX-1 shows them
unambiguously negative — but this has **not yet been confirmed by eye against page 221**
and is flagged in the source.

### 5.2 One flight condition

Only F/C 2 (power approach, sea level, 165 KTAS) is fully modelled.

**Why:** it is one of only two conditions with tabulated non-dimensional derivatives;
the other eight are published as plots against Mach, which cannot be read precisely.

**Cost:** the model is validated at one point in the envelope. Nothing here demonstrates
it works at cruise or at altitude.

### 5.3 Axis systems mixed by the source

Inertias are body-axis, derivative tables are labelled per-table, and CR-2144 warns that
lateral rolling and yawing derivatives share symbols between body and stability axes
while differing numerically.

**How handled:** Tables IX-4 and IX-5 are explicitly headed "BODY AXIS SYSTEM" and are
used as-is against a body-axis model. No axis transformation is applied, and none is
needed for the comparisons made.

---

## 6. What is not modelled at all

- Control-system dynamics, actuators, or the stability augmentation CR-2144 describes.
  All results are **bare airframe**, matching the tables used.
- Wind, gusts or turbulence.
- Ground effect, landing gear, or any configuration change.
- Sensor models or noise.

---

## 7. Reproducibility

- No randomness anywhere; results are deterministic.
- 222 tests, ~45 s.
- Figures and the visualiser regenerate from `scripts/`.
- Results are hardware-independent to the limits of IEEE-754 double arithmetic.
