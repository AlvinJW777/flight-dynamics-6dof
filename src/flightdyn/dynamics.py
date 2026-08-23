"""Rigid-body equations of motion in six degrees of freedom.

Flat, non-rotating Earth. At aircraft speeds and over the timescales of a mode
analysis, Earth rotation and curvature contribute far less than the aerodynamic
uncertainty, so including them would add complexity without adding fidelity. This
is an assumption, not an oversight, and it is recorded in ASSUMPTIONS.md.

STATE VECTOR (13)
-----------------
    index  0  1  2   3  4  5   6   7   8   9   10  11  12
           u  v  w   p  q  r   q0  q1  q2  q3  pn  pe  pd
           body velocity        attitude quaternion  NED position

Thirteen rather than twelve because the quaternion carries four components under
one norm constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import body_to_ned, ned_to_body, normalise, quat_kinematics

# --- state layout -------------------------------------------------------------
IDX_VEL = slice(0, 3)
IDX_RATE = slice(3, 6)
IDX_QUAT = slice(6, 10)
IDX_POS = slice(10, 13)
N_STATES = 13

STATE_NAMES = ("u", "v", "w", "p", "q", "r", "q0", "q1", "q2", "q3", "pn", "pe", "pd")

STANDARD_GRAVITY = 9.80665


@dataclass(frozen=True)
class RigidBody:
    """Mass properties. SI throughout: kg and kg·m².

    An aircraft is symmetric about its xz plane, so Ixy = Iyz = 0 — but **Ixz is
    not zero**, and for a large swept-wing aircraft it is substantial. Textbook
    treatments often drop it for tractability. Dropping it artificially decouples
    roll and yaw, which specifically corrupts the dutch roll and spiral modes, so
    it is carried here.

    **Sign convention.** The inertia tensor is defined with *negative* products of
    inertia::

        I = [[ Ixx,   0,  -Ixz],
             [   0, Iyy,     0],
             [-Ixz,   0,   Izz]]

    where ``Ixz = ∫ x z dm``. Data sources vary: some publish the positive product
    of inertia (insert with the minus, as here), others publish an already-signed
    ``Jxz``. Read the source's definition — a sign error here flips the roll-yaw
    coupling without breaking anything visibly.
    """

    mass: float
    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float = 0.0

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        for name in ("Ixx", "Iyy", "Izz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        # Triangle inequality on principal moments — a physically impossible
        # inertia tensor otherwise passes silently and produces nonsense rates.
        a, b, c = self.Ixx, self.Iyy, self.Izz
        if not (a + b >= c and b + c >= a and c + a >= b):
            raise ValueError(
                f"inertias violate the triangle inequality (Ixx={a}, Iyy={b}, Izz={c}) — "
                "no rigid body can have these"
            )

    @property
    def inertia(self) -> np.ndarray:
        return np.array([
            [self.Ixx, 0.0, -self.Ixz],
            [0.0, self.Iyy, 0.0],
            [-self.Ixz, 0.0, self.Izz],
        ])

    @property
    def inertia_inverse(self) -> np.ndarray:
        return np.linalg.inv(self.inertia)


def gravity_body(quat: np.ndarray, mass: float, g: float = STANDARD_GRAVITY) -> np.ndarray:
    """Weight vector in body axes.

    Gravity is constant and simple in NED — straight down — which is exactly why
    the DCM is defined NED→body: this is the transformation performed most often.
    """
    return ned_to_body(quat, np.array([0.0, 0.0, mass * g]))


def rigid_body_derivative(
    state: np.ndarray,
    force_body: np.ndarray,
    moment_body: np.ndarray,
    body: RigidBody,
) -> np.ndarray:
    """State derivative for a rigid body in 6-DOF.

    ``force_body`` is the **total** force in body axes including weight; keeping
    gravity out of this function means the conservation tests can run with it
    switched off, which is what makes them sharp.

    Translational, in body axes::

        u̇ = rv − qw + Fx/m
        v̇ = pw − ru + Fy/m
        ẇ = qu − pv + Fz/m

    The ``rv − qw`` terms are not corrections — they are the transport terms that
    arise because the body frame rotates, so d/dt in the body frame is not the
    inertial derivative. Written vectorially the whole block is simply
    ``v̇ = F/m − ω × v``. Omit them and the aircraft still flies, but not like an
    aircraft, and nothing errors. This is the most common silent bug in a first
    6-DOF implementation.

    Rotational (Euler's equation)::

        ω̇ = I⁻¹ ( M − ω × Iω )

    The ``ω × Iω`` term is what makes free rotation interesting — it is the entire
    source of the intermediate-axis instability, and it vanishes only for a
    spherical inertia tensor.
    """
    v = state[IDX_VEL]
    omega = state[IDX_RATE]
    quat = state[IDX_QUAT]

    dv = np.asarray(force_body, dtype=float) / body.mass - np.cross(omega, v)

    Iw = body.inertia @ omega
    domega = body.inertia_inverse @ (np.asarray(moment_body, dtype=float) - np.cross(omega, Iw))

    dquat = quat_kinematics(quat, omega)
    dpos = body_to_ned(quat, v)

    out = np.empty(N_STATES)
    out[IDX_VEL] = dv
    out[IDX_RATE] = domega
    out[IDX_QUAT] = dquat
    out[IDX_POS] = dpos
    return out


def rk4_step(derivative, state: np.ndarray, dt: float, renormalise: bool = True) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step.

    ``derivative`` is a callable taking the state and returning its derivative.

    The quaternion is renormalised after each step. The kinematic equation
    conserves the norm analytically — Ω is skew-symmetric — so any drift comes
    purely from truncation, and the size of the correction is therefore a clean
    measure of integration error. :func:`quaternion_drift` exposes it.
    """
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if renormalise:
        nxt[IDX_QUAT] = normalise(nxt[IDX_QUAT])
    return nxt


def quaternion_drift(state: np.ndarray) -> float:
    """How far the quaternion has wandered from unit norm. Pure integration error."""
    return float(abs(np.linalg.norm(state[IDX_QUAT]) - 1.0))


def propagate(derivative, state: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """Integrate forward and return the trajectory, shape ``(n_steps + 1, 13)``."""
    traj = np.empty((n_steps + 1, N_STATES))
    traj[0] = state
    x = state.copy()
    for k in range(n_steps):
        x = rk4_step(derivative, x, dt)
        traj[k + 1] = x
    return traj


# --- diagnostics used by the verification tests -------------------------------


def kinetic_energy(state: np.ndarray, body: RigidBody) -> float:
    """Translational plus rotational kinetic energy, in the body frame.

    Conserved under no external work, so it is a sharp check on the ``ω × Iω``
    term: a sign error there pumps or drains energy without breaking anything else.
    """
    v = state[IDX_VEL]
    omega = state[IDX_RATE]
    return float(0.5 * body.mass * v @ v + 0.5 * omega @ (body.inertia @ omega))


def angular_momentum_ned(state: np.ndarray, body: RigidBody) -> np.ndarray:
    """Angular momentum in the inertial (NED) frame.

    Conserved when no external moment acts. It must be checked in the *inertial*
    frame — the body-frame vector rotates even when angular momentum is perfectly
    conserved, which is precisely the point of Euler's equation.
    """
    return body_to_ned(state[IDX_QUAT], body.inertia @ state[IDX_RATE])


def make_state(
    velocity=(0.0, 0.0, 0.0),
    rates=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    position=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """Assemble a state vector, so call sites never index by magic number."""
    x = np.empty(N_STATES)
    x[IDX_VEL] = velocity
    x[IDX_RATE] = rates
    x[IDX_QUAT] = normalise(np.asarray(quat, dtype=float))
    x[IDX_POS] = position
    return x
