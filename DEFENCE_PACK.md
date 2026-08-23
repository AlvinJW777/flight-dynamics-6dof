# Defence Pack

> **How to use this.** The code is AI-assisted; the understanding cannot be. After
> each unit, write your answer here **in your own words, without looking at the
> source**. If you can't, that unit isn't finished — tell me and we slow down
> rather than pressing on.
>
> Each question lists what a complete answer covers. That's a checklist, not the
> answer. Writing it yourself is the entire point.

**Status:** Units 0–2 complete, outcomes 1–3 answered.

Answers are written as **the shortest version that is still correct** — a few lines
you can reproduce cold, not prose you'll skim. If you can say these from memory,
you own the unit. Depth sits in the source docstrings when you want it.

---

## Outcome 1 — Reference frames

*Covered by `src/flightdyn/frames.py`.*

### 1.1 Why does the simulator use four different frames rather than one?

**Your answer:**

> _(write here)_

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

**Your answer:**

> _(write here)_

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

**Your answer:**

> _(write here)_

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

**Your answer:**

> _(write here)_

<details><summary>A complete answer covers…</summary>

- With no external moment, **H** is constant in the inertial frame
- Its *body-frame components* still change, because the body rotates underneath it
- That's exactly what Euler's equation describes
- A test asserting constant body-frame H would fail on correct code

</details>

### 3.4 Why does dropping `Ixz` corrupt specific modes?

**Your answer:**

> _(write here)_

<details><summary>A complete answer covers…</summary>

- Aircraft are symmetric about xz, so Ixy = Iyz = 0 but Ixz ≠ 0
- Ixz couples roll and yaw through the inertia tensor
- Dropping it artificially decouples them
- Dutch roll is *defined* by coupled yaw-roll motion, so it's the mode most affected; the spiral follows

</details>

---

## Outcome 4 — Trim *(unit not started)*

### 4.1 What does trim mean physically, and how did you prove you reached it?
### 4.2 What does it mean when trim fails to converge, and why is tightening the tolerance the wrong response?

---

## Outcome 5 — Linearisation *(unit not started)*

### 5.1 How did you choose the perturbation size, and what happens either side of that choice?
### 5.2 Why does a fixed-step integrator matter more here than anywhere else in the project?

---

## Outcome 6 — Modes *(unit not started)*

### 6.1 Extract ωₙ and ζ from a complex eigenvalue, and explain what each means physically.
### 6.2 Explain the phugoid without equations.
### 6.3 Why is the spiral mode often unstable, and why is that acceptable in a certificated aircraft?

---

## Outcome 7 — Verification vs validation *(ongoing)*

### 7.1 What have you verified, and what have you validated? Give one example of each from this project and say why they are different kinds of evidence.

**Your answer:**

> _(write here)_

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
