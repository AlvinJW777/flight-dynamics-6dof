# Defence Pack

> **How to use this.** The code is AI-assisted; the understanding cannot be. After
> each unit, write your answer here **in your own words, without looking at the
> source**. If you can't, that unit isn't finished — tell me and we slow down
> rather than pressing on.
>
> Each question lists what a complete answer covers. That's a checklist, not the
> answer. Writing it yourself is the entire point.

**Status:** Complete. All seven learning outcomes answered, 21 questions.

Answers are written as **the shortest version that is still correct** — a few lines
you can reproduce cold, not prose you'll skim. If you can say these from memory,
you own the unit. Depth sits in the source docstrings when you want it.

---

## Outcome 1 — Reference frames

*Covered by `src/flightdyn/frames.py`.*

### 1.1 Why does the simulator use four different frames rather than one?

> **Each frame exists because one thing is constant in it.**
> NED — gravity is `[0,0,mg]`, always. Body — the inertia tensor, because the
> aircraft doesn't deform. Wind — drag, which is *defined* as opposing the relative
> wind. Stability — where the derivatives are published.
>
> You transform because each quantity has a natural home.

<details><summary>A complete answer covers…</summary>

- Each frame exists because some quantity is *simple* in it
- Gravity is constant in NED; the inertia tensor is constant in body; drag is aligned in wind
- Stability derivatives are conventionally quoted in stability axes
- Why `C_bn` was chosen as NED→body rather than the reverse

</details>

### 1.2 Why is the inertia tensor constant in body axes but not in Earth axes?

> **I describes where the mass sits *relative to the axes*.**
> Body axes are glued to the aircraft — the mass never moves, so I is fixed.
> Earth axes aren't — as the aircraft turns, the same bolt lands at different
> coordinates, so I changes.
>
> **Punchline:** `ω × Iω` is just `dI/dt` moved somewhere cheaper. The rotation
> never goes away; you only choose where it appears. One cross product in body
> axes, or a 3×3 tensor derivative every step in Earth axes.

<details><summary>A complete answer covers…</summary>

- The tensor describes how mass is distributed *relative to the axes*
- The aircraft doesn't deform, so that distribution is fixed in body axes
- In Earth axes the aircraft rotates, so `I` becomes time-varying — and `dH/dt` then needs a `dI/dt` term
- Consequence: the moment equation is only tractable in body axes

</details>

### 1.3 What are α and β, and why does one use `atan2` and the other `asin`?

> α is the angle **in** the plane of symmetry; β is the angle **out of** it.
>
> `α = atan2(w, u)` — a **ratio of two components**, so atan2 keeps it right through
> all four quadrants; backwards flight gives 180°, not 0°.
> `β = asin(v/V)` — measured against **total speed**, not a ratio. Hence arcsine.
>
> **α is a ratio. β is a fraction of the whole.**

<details><summary>A complete answer covers…</summary>

- α is measured in the plane of symmetry, β out of it
- α = atan2(w, u) — a ratio of two components, so `atan2` keeps it correct through all four quadrants including backwards flight
- β = asin(v / V) — measured against *total* speed, not a projection, so it's an arcsine

</details>

---

## Outcome 2 — Attitude representation

### 2.1 What specifically fails if you integrate Euler angles instead of a quaternion?

> **You can't cover a globe with one flat map** — at the poles, longitude dies.
> Every meridian meets there, so the coordinate has no value.
>
> Rotations are the same, one dimension up. **Any three angles must have a pole.**
> For 3-2-1 it's θ = ±90°, where roll and yaw become the same physical axis. The
> equations contain `tanθ` and `1/cosθ`, so they blow up — and go stiff *before*
> they get there.
>
> Change the sequence and you move the pole, never remove it.
> **Quaternions use 4 numbers with one constraint, so there's no flat grid to
> fold.** Cost: one redundant number and a normalisation.

<details><summary>A complete answer covers…</summary>

- Write down the Euler kinematic equations and point at the `tanθ` and `1/cosθ`
- At θ = ±90° roll and yaw become the same physical axis
- It's a *coordinate* singularity — the aircraft is fine, the representation isn't
- The equations go stiff *before* the singularity, so integration fails early
- The topological point: **no** three-parameter representation can avoid this, so the real choice is "accept a singularity or carry a fourth number"

</details>

### 2.2 Why does the quaternion need renormalising if the kinematic equation conserves its norm?

> **The equation conserves the norm. The integrator does not.**
> Ω is skew-symmetric so `d/dt(qᵀq) = 0` exactly — in the maths. RK4 is an
> approximation, so truncation error nudges q off the unit sphere.
>
> Because the equation is exact, **any drift is pure integration error** — a free
> accuracy diagnostic rather than a nuisance.

<details><summary>A complete answer covers…</summary>

- Ω is skew-symmetric, so d/dt(qᵀq) = qᵀΩq = 0 — the norm is conserved *analytically*
- The integrator is not exact, so truncation error accumulates off the unit sphere
- Therefore the drift measures integration error alone, which makes it a useful diagnostic rather than a nuisance

</details>

---

## Outcome 3 — Equations of motion

### 3.1 Derive `u̇ = Fx/m + rv − qw` from first principles.

> 1. Newton only holds in a **non-rotating** frame: `F = m(dv/dt)_inertial`
> 2. We want body axes, but the body frame rotates.
> 3. **Bridge — transport theorem:** `(dA/dt)_inertial = (dA/dt)_body + ω × A`
>    *Why:* a vector painted on the wing is fixed to you, but its tip sweeps at
>    `ω × A` to someone on the ground.
> 4. Sub in and rearrange: `(dv/dt)_body = F/m − ω × v`
> 5. Expand `ω × v = [qw−rv, ru−pw, pv−qu]`, take the first component:
>
> `u̇ = Fx/m + rv − qw`  ∎
>
> **Those terms aren't corrections. They *are* "the body frame rotates."**

<details><summary>A complete answer covers…</summary>

- Newton's second law holds in an **inertial** frame
- The transport theorem: (dA/dt)_inertial = (dA/dt)_body + ω × A
- Apply to velocity, rearrange to (dv/dt)_body = F/m − ω × v
- Expand the cross product and read off the first component
- The punchline: those terms *are* the statement that the body frame rotates — they are not corrections

</details>

### 3.2 Why is the intermediate-axis spin unstable?

> Torque-free Euler equations, spin fast about one axis, perturb the other two.
> Each perturbation drives the other, and stability comes down to **the sign of
> the product of two inertia differences**.
>
> With I₁ < I₂ < I₃, spinning about the **intermediate** axis makes both
> coefficients negative → product positive → **exponential growth**.
> About the major or minor axis, one flips sign → product negative → oscillation.
>
> That's the whole Dzhanibekov effect: one sign.

<details><summary>A complete answer covers…</summary>

- Torque-free Euler equations about principal axes, three coupled products
- Linearise about a fast spin, keep the two small perturbations
- The stability is decided by the **sign of the product** of the two coefficients
- Intermediate axis → both coefficients negative → product positive → exponential growth
- Major or minor axis → one flips → product negative → oscillation

</details>

### 3.3 Why must angular momentum be checked in the inertial frame, not the body frame?

> With no external moment, **H** is constant **in the inertial frame**. Its
> *body-frame components* still change, because the body rotates underneath it —
> which is exactly what Euler's equation describes.
>
> **H stands still; the aircraft turns around it.**
> A test asserting constant body-frame H would *fail on correct code*, which is why
> there is a negative control for precisely that.

<details><summary>A complete answer covers…</summary>

- With no external moment, **H** is constant in the inertial frame
- Its *body-frame components* still change, because the body rotates underneath it
- That's exactly what Euler's equation describes
- A test asserting constant body-frame H would fail on correct code

</details>

### 3.4 Why does dropping `Ixz` corrupt specific modes?

> Symmetry about the xz plane makes `Ixy = Iyz = 0`. **`Ixz` survives** — and it is
> the *inertial* coupling between roll and yaw.
>
> Drop it and roll and yaw decouple artificially. **Dutch roll IS coupled
> yaw-and-roll**, so it is hit hardest; the spiral follows.
>
> The point is not that Ixz is non-zero — it is *what Ixz does*.

<details><summary>A complete answer covers…</summary>

- Aircraft are symmetric about xz, so Ixy = Iyz = 0 but Ixz ≠ 0
- Ixz couples roll and yaw through the inertia tensor
- Dropping it artificially decouples them
- Dutch roll is *defined* by coupled yaw-roll motion, so it's the mode most affected; the spiral follows

</details>

---

## Outcome 4 — Trim

### 4.1 What does trim mean physically, and how did you prove you reached it?

> **Trim = every acceleration is zero**, so the aircraft holds the state forever
> with the controls fixed.
>
> Three unknowns — α, elevator, throttle — driving three residuals — u̇, ẇ, q̇ —
> to zero. No closed form, because gravity rotates with attitude and the aero
> forces rotate from wind to body axes. So it's a **nonlinear root-find**.
>
> `γ = θ − α = 0` for level flight. That constraint is what makes three unknowns
> match three equations instead of four.
>
> **Proof, in three parts:** residual 1×10⁻¹⁶; propagated 60 s with controls fixed
> and *nothing moved*; and a negative control — a state 10% fast *does* drift, so
> the hold test actually has power.

### 4.2 What does it mean when trim fails to converge, and why is tightening the tolerance the wrong response?

> It means **the condition isn't achievable with the model as built** — not enough
> thrust, or the linear aero extrapolated past where it's valid.
>
> Tightening the tolerance or nudging the guess until something emerges hides a
> real finding. The solver was right to fail; the question is why.

---

## Outcome 5 — Linearisation

### 5.1 How did you choose the perturbation size, and what happens either side?

> Central differences carry **truncation error ~h²** and **round-off ~ε/h**.
> Their sum minimises near **h ≈ ε^(1/3) × (the state's own scale)**.
>
> For u, scale 84 m/s → predicted 5.1×10⁻⁴. Measured minimum: 1×10⁻³. Within 2×.
>
> Scale **per state** — a step right for m/s is meaningless for radians.
>
> **The caveat I found by measuring:** the V-shape only appears where the function
> is genuinely nonlinear in that variable. In q the dynamics are nearly linear, so
> there's no truncation branch — the curve falls and flattens. Expecting a V there
> and "fixing" the code would be chasing a phantom.

### 5.2 Why does a fixed-step integrator matter more here than anywhere else?

> An adaptive solver **chooses its step based on the state**. Perturb the state by
> 10⁻⁶ and it may take different steps — injecting noise at exactly the scale
> you're trying to measure.

### 5.3 Why linearise in Euler angles when the simulator uses quaternions?

> A quaternion is 4 components under 1 constraint, so linearising in it gives a
> redundant direction and a **spurious zero eigenvalue** you'd have to identify
> and discard.
>
> At trim we're far from gimbal lock, so the Euler chart is well conditioned and
> gives exactly 9 dynamic states. **The physics is untouched** — it's a change of
> coordinates for taking a derivative, not a change of model.

---

## Outcome 6 — Modes

### 6.1 Extract ωₙ and ζ from an eigenvalue, and say what each means.

> `λ = −ζωₙ ± jωₙ√(1−ζ²)`, so **ωₙ = |λ|** and **ζ = −Re(λ)/|λ|**.
>
> ωₙ is how *fast*, ζ is how *quickly it dies*. ζ < 0 → divergent.

### 6.2 Explain the phugoid without equations.

> **A slow trade of speed for height.** Nose drops → speeds up → lift rises →
> climbs → slows → nose drops again.
>
> Energy sloshing between kinetic and potential. Barely damped because only *drag*
> removes energy — hence ζ = 0.02 and a 50-second period here.
>
> That's also why it's the mode our model matches worst: it's set almost entirely
> by drag, so a few percent on CD moves it a lot.

### 6.3 Why is the spiral often unstable, and why is that acceptable?

> Bank slightly → sideslip → **dihedral (Cl_β) rolls you level** while
> **weathercock (Cn_β) yaws you into the turn, which rolls you further in.**
> Whichever wins decides stability.
>
> Acceptable because it's *slow* — τ = 27 s here. A pilot corrects it without
> noticing. Certification limits time-to-double, not the sign.

---

## Outcome 7 — Verification vs validation

### 7.1 What did you verify, what did you validate, and why are they different?

> **Verification** — numerical Jacobian against Table IX-4's dimensional
> derivatives. *"Did I code the maths right?"* All seven longitudinal terms within
> 3%.
>
> **Validation** — eigenvalues against Table IX-5's transfer-function factors.
> *"Does the model describe a real aircraft?"* Frequencies within 2%.
>
> **They caught different things, and that's the whole point.** Validation looked
> fine — the modes were already within a few percent — while Xu, Zu and Mu were
> 14–60% wrong. Only element-by-element verification found the missing Mach
> derivatives, because the error was confined to one column of the matrix.
>
> One reference alone could not have told me whether a discrepancy meant a coding
> error or a modelling limitation.

---

## Outcome 7 — Verification vs validation, further reading

### 7.1 What have you verified, and what have you validated? Give one example of each from this project and say why they are different kinds of evidence.

> **Verification** — did I solve the equations *right*? Numerical Jacobian against
> Table IX-4's dimensional derivatives: all seven terms within 3%.
>
> **Validation** — did I solve the *right* equations? Eigenvalues against Table
> IX-5's transfer-function factors: frequencies within 2%.
>
> **Why both are needed, from this project:** validation looked fine — the modes were
> already within a few percent — while Xu, Zu and Mu were 14–60% wrong. Only
> element-by-element verification against a *different table* found the missing Mach
> derivatives, and the error being confined to one column identified *which* term was
> missing.
>
> **Verification = maths. Validation = physics. One can pass while the other fails.**

<details><summary>A complete answer covers…</summary>

- Verification = solving the equations right. Examples so far: scipy cross-check on the rotations, RK4 fourth-order convergence, conservation laws, the intermediate-axis test
- Validation = solving the right equations. Nothing yet — that arrives when eigenvalues meet CR-2144
- Why one cannot substitute for the other
- The diagnostic value of having both: which one fails tells you *where* the problem is

</details>

---

## Methodology

### M.1 Which parts were AI-assisted, and which did you verify yourself?
### M.2 You did not write this code by hand. How do you know it is right?

<details><summary>A complete answer covers…</summary>

- Honest description of the split: implementation and tests AI-written, conventions and architecture decided by you, physics understanding owned by you
- The verification is *in the repository*, not asserted about it
- Concrete example: the RK4 order test initially failed at ratios of 4.3 and 0.8 — not because the integrator was wrong but because the test measured below the floating-point noise floor. Being able to tell those apart is the skill.
- The standard: not whether you did the arithmetic, but whether you would notice if it were wrong

</details>
