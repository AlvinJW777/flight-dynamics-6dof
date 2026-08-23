# Specification

> **Written retrospectively.** The module process calls for this before implementation;
> in practice it was written after Unit 6, from the decisions actually taken. That is a
> deviation from the intended workflow and is recorded here rather than disguised. The
> content is accurate to what was built; it simply was not agreed in advance.

---

## 1. Objective

Build a nonlinear six-degree-of-freedom rigid-body aircraft simulator, find its
equilibrium flight condition by nonlinear root-finding, linearise it numerically, and
demonstrate that the resulting natural modes match published NASA flight data.

The deliverable is **evidence of engineering method**, not a flight simulator. The
project succeeds if verification and validation are separated, evidenced independently,
and every disagreement is explained.

## 2. Scope

**In scope**

- Rigid-body equations of motion, six degrees of freedom, flat non-rotating Earth
- Quaternion attitude representation with Euler output
- ISA atmosphere to 20 km
- Linear-derivative aerodynamics and a steady thrust model
- Straight-and-level trim as a nonlinear root-find
- Numerical linearisation about trim
- Eigenvalue extraction and identification of the five classical modes
- Verification and validation against NASA CR-2144

**Out of scope**

- Control system design (Unit 7 of the module guide — explicitly a stretch goal)
- Structural flexibility, aeroelasticity
- Wind, gust or turbulence models
- Stability augmentation, actuators, sensor models
- Any flight condition other than the one specified below

## 3. Reference data

Heffley, R.K. and Jewell, W.F., *Aircraft Handling Qualities Data*, **NASA CR-2144**,
December 1972, Section IX (Boeing 747).

| Table | PDF page | Provides | Role |
|---|---|---|---|
| IX-2 | 221 | Non-dimensional derivatives, per radian | Model input |
| IX-3 | 233 | Mass, inertia, geometry, flight conditions | Model input |
| IX-4 | 234 | Dimensional derivatives, body axis | **Verification target** |
| IX-5 | 235 | Elevator transfer-function factors | **Validation target** |
| Appendix A | 321+ | Axis systems, symbols, derivative definitions | Conventions |

**Flight Condition 2** — power approach, sea level, M = 0.249, 165 KTAS, 564,032 lb.
Selected because it is one of only two conditions with *tabulated* non-dimensional
derivatives; the remaining eight are published only as plots against Mach, which cannot
be read precisely.

## 4. Governing equations

Translational, body axes:

```
v̇ = F/m − ω × v
```

Rotational (Euler):

```
ω̇ = I⁻¹(M − ω × Iω)
```

Attitude kinematics:

```
q̇ = ½ Ω(ω) q
```

Navigation:

```
ṗ_NED = C_bnᵀ v_body
```

Aerodynamic coefficients:

```
C = C0 + C_α·α + C_q·(qc̄/2V) + C_δ·δ + C_M·ΔM
```

## 5. Conventions

| | |
|---|---|
| Units | SI and radians internally; one tested conversion layer at the data boundary |
| Quaternion | scalar-first `[q0,q1,q2,q3]`, Hamilton product |
| DCM | `C_bn` maps NED → body |
| Euler | 3-2-1 (yaw, pitch, roll); output and linearisation only, never integration |
| Angular velocity | `[p,q,r]`, body rates relative to Earth, in body axes |
| Inertia | full tensor including `Ixz`; products of inertia enter negatively |
| Derivatives | per radian; longitudinal rates by `c̄/2V`, lateral by `b/2V` |

## 6. State vector

Thirteen states — four quaternion components under one norm constraint:

```
[u, v, w,  p, q, r,  q0, q1, q2, q3,  pn, pe, pd]
```

## 7. Acceptance criteria

| # | Criterion | Standard | Result |
|---|---|---|---|
| A1 | Rotations match an independent implementation | < 10⁻¹³ vs scipy | ✅ |
| A2 | Energy and angular momentum conserved, torque-free | < 10⁻⁹ relative | ✅ |
| A3 | Intermediate-axis instability reproduced | growth > 10⁴ × | ✅ |
| A4 | RK4 demonstrates fourth-order convergence | ratio 14–18 per halving | ✅ 15.95–15.97 |
| A5 | Atmosphere matches published ISA | < 2 × 10⁻⁵ relative | ✅ |
| A6 | Trim residuals converged | < 10⁻¹⁰ | ✅ 3.6 × 10⁻¹⁵ |
| A7 | Trimmed state holds 60 s, controls fixed | no measurable drift | ✅ |
| A8 | Linearised matrix reproduces Table IX-4 | every term within 5% | ✅ 7/7, six within 3% |
| A9 | Longitudinal/lateral decouple | coupling ratio < 10⁻⁶ | ✅ 0.0 |
| A10 | Mode frequencies match Table IX-5 | within 5% | ✅ −1.9%, −0.4% |
| A11 | Mode damping matches Table IX-5 | short period within 12% | ✅ −6.1% |
| A12 | All five classical modes identified | present, correct character | ✅ |
| A13 | Disagreements explained, not just reported | narrative in VALIDATION.md | ✅ |

**Not met, and stated as such:** lateral modes are validated for character only, because
CR-2144's lateral transfer functions were not transcribed.

## 8. Deliverables

| | Status |
|---|---|
| `src/flightdyn/` — 11 modules | ✅ |
| Test suite | ✅ 222 tests, ~45 s |
| README.md with results and figures | ✅ |
| ASSUMPTIONS.md | ✅ |
| VALIDATION.md | ✅ |
| DEFENCE_PACK.md — 21 questions | ✅ |
| Interactive visualiser | ✅ `visualiser.html` |
| Repository published | ✅ private; public is a deliberate later step |
| Technical report PDF | ⬜ **outstanding** |
| Unit 7 — control system | ⬜ stretch, not attempted |

## 9. Known deviations from the module guide

1. **This specification was written after implementation**, not before.
2. **Per-unit `docs/0N_*.md` records were not produced.** Their content lives in the
   module docstrings, ASSUMPTIONS.md and VALIDATION.md instead. Writing them separately
   would have duplicated rather than added.
3. **No technical report PDF.** README, ASSUMPTIONS and VALIDATION cover the content;
   a formatted report remains outstanding.
4. **Unit 7 (control) not attempted.** Explicitly a stretch goal, gated on Units 0–6
   being complete and documented.
