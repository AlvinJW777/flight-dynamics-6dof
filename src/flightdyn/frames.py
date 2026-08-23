"""Reference frames, attitude representation and rotational kinematics.

CONVENTIONS — fixed here, assumed everywhere else
-------------------------------------------------
* **Quaternion layout:** scalar-first, ``q = [q0, q1, q2, q3]``, Hamilton product.
  Matches Stevens & Lewis and Etkin & Reid. Note ``scipy`` uses scalar-*last*, so
  anything crossing that boundary must be reordered explicitly.
* **DCM sense:** ``C_bn`` maps **NED into body**. Read the subscript as
  "body from NED". Gravity lives in NED and is needed in body axes for the force
  equations, which is the direction used most often.
* **Euler sequence:** 3-2-1 (yaw ψ, then pitch θ, then roll φ).
* **Angular velocity:** ``ω = [p, q, r]``, body rates relative to the Earth frame,
  expressed in body axes — which is what a rate gyro measures.

Nothing here imports a rotation library. Implementing this is the point of the
exercise; ``scipy`` appears only in the tests, as an independent reference to
verify against.
"""

from __future__ import annotations

import numpy as np


def normalise(q: np.ndarray) -> np.ndarray:
    """Return the unit quaternion. Raises if the input is degenerate.

    Numerical integration slowly violates the unit-norm constraint, so this gets
    applied every step. How much drift it is correcting is itself diagnostic —
    large corrections mean the timestep is too coarse.
    """
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError(f"quaternion has near-zero norm ({n:.3e}) and cannot be normalised")
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b, scalar-first.

    Not commutative — ``a ⊗ b`` composes b's rotation followed by a's. Getting the
    order backwards produces a valid rotation that is simply the wrong one, which
    is why the tests check composition against a known sequence.
    """
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.array([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate — the inverse rotation, for a unit quaternion."""
    q0, q1, q2, q3 = q
    return np.array([q0, -q1, -q2, -q3])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Direction cosine matrix C_bn, mapping a vector from NED into body axes.

    Derived from the passive-rotation form

        C = (q0² − |qv|²) I + 2 qv qvᵀ − 2 q0 [qv]×

    where ``[qv]×`` is the skew-symmetric cross-product matrix of the vector part.
    The minus sign on the last term is what makes this the *passive* (frame)
    rotation rather than the *active* (vector) one; flipping it transposes every
    transformation in the simulator.
    """
    q0, q1, q2, q3 = normalise(q)
    return np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 + q0*q3),             2*(q1*q3 - q0*q2)],
        [2*(q1*q2 - q0*q3),             q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 + q0*q1)],
        [2*(q1*q3 + q0*q2),             2*(q2*q3 - q0*q1),             q0*q0 - q1*q1 - q2*q2 + q3*q3],
    ])


def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quat_to_dcm`, via Shepperd's method.

    The naive extraction divides by ``sqrt(1 + trace)``, which loses precision and
    then fails outright when the trace approaches −1 (a rotation near 180°).
    Shepperd's method picks whichever of four formulations has the largest
    divisor, so it stays well-conditioned for every rotation.
    """
    C = np.asarray(C, dtype=float)
    trace = C[0, 0] + C[1, 1] + C[2, 2]
    candidates = np.array([trace, C[0, 0], C[1, 1], C[2, 2]])
    i = int(np.argmax(candidates))

    if i == 0:
        s = np.sqrt(1.0 + trace) * 2.0
        q = np.array([0.25 * s, (C[1, 2] - C[2, 1]) / s, (C[2, 0] - C[0, 2]) / s, (C[0, 1] - C[1, 0]) / s])
    elif i == 1:
        s = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
        q = np.array([(C[1, 2] - C[2, 1]) / s, 0.25 * s, (C[0, 1] + C[1, 0]) / s, (C[2, 0] + C[0, 2]) / s])
    elif i == 2:
        s = np.sqrt(1.0 - C[0, 0] + C[1, 1] - C[2, 2]) * 2.0
        q = np.array([(C[2, 0] - C[0, 2]) / s, (C[0, 1] + C[1, 0]) / s, 0.25 * s, (C[1, 2] + C[2, 1]) / s])
    else:
        s = np.sqrt(1.0 - C[0, 0] - C[1, 1] + C[2, 2]) * 2.0
        q = np.array([(C[0, 1] - C[1, 0]) / s, (C[2, 0] + C[0, 2]) / s, (C[1, 2] + C[2, 1]) / s, 0.25 * s])

    # q and -q are the same rotation; fix the sign so round-trips are stable.
    if q[0] < 0:
        q = -q
    return normalise(q)


def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """3-2-1 Euler angles (roll, pitch, yaw, radians) to a scalar-first quaternion."""
    cphi, sphi = np.cos(phi / 2), np.sin(phi / 2)
    cth, sth = np.cos(theta / 2), np.sin(theta / 2)
    cpsi, spsi = np.cos(psi / 2), np.sin(psi / 2)
    return np.array([
        cphi * cth * cpsi + sphi * sth * spsi,
        sphi * cth * cpsi - cphi * sth * spsi,
        cphi * sth * cpsi + sphi * cth * spsi,
        cphi * cth * spsi - sphi * sth * cpsi,
    ])


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Scalar-first quaternion to 3-2-1 Euler angles (roll, pitch, yaw), radians.

    For output and human interpretation only — never integrate Euler angles. At
    θ = ±90° roll and yaw become indistinguishable and the kinematic equations
    divide by cos θ → 0. That is gimbal lock, and it is the whole reason the state
    carries a quaternion instead.

    ``asin`` is clipped so that accumulated round-off just past ±1 returns ±90°
    rather than NaN.
    """
    C = quat_to_dcm(q)
    theta = -np.arcsin(np.clip(C[0, 2], -1.0, 1.0))
    phi = np.arctan2(C[1, 2], C[2, 2])
    psi = np.arctan2(C[0, 1], C[0, 0])
    return float(phi), float(theta), float(psi)


def quat_kinematics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion rate: q̇ = ½ Ω(ω) q, with ω = [p, q, r] the body rates.

    Ω is skew-symmetric, which is not a detail — it is what guarantees the norm is
    conserved analytically, since d/dt(qᵀq) = 2qᵀq̇ = qᵀΩq = 0 for skew Ω. Any
    norm drift you observe is therefore purely from the integrator, never from the
    equation, which makes it a clean measure of integration error.
    """
    p, qq, r = omega
    Omega = np.array([
        [0.0, -p,  -qq, -r],
        [p,    0.0, r,  -qq],
        [qq,  -r,   0.0, p],
        [r,    qq, -p,   0.0],
    ])
    return 0.5 * Omega @ np.asarray(q, dtype=float)


def ned_to_body(q: np.ndarray, v_ned: np.ndarray) -> np.ndarray:
    """Rotate a vector from NED into body axes."""
    return quat_to_dcm(q) @ np.asarray(v_ned, dtype=float)


def body_to_ned(q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """Rotate a vector from body into NED axes — the transpose of C_bn."""
    return quat_to_dcm(q).T @ np.asarray(v_body, dtype=float)


def airdata(v_body: np.ndarray) -> tuple[float, float, float]:
    """Airspeed, angle of attack and sideslip from the body-axis velocity.

    With ``v_body = [u, v, w]`` relative to the air mass::

        V = |v|,   α = atan2(w, u),   β = asin(v / V)

    α is the angle between the velocity's projection in the aircraft's plane of
    symmetry and the body x-axis; β is measured out of that plane. Note α uses
    ``atan2`` so it stays correct in backwards flight, while β uses ``asin``
    because it is defined against total speed rather than a projection.
    """
    u, v, w = np.asarray(v_body, dtype=float)
    V = float(np.linalg.norm([u, v, w]))
    if V < 1e-9:
        return 0.0, 0.0, 0.0
    alpha = float(np.arctan2(w, u))
    beta = float(np.arcsin(np.clip(v / V, -1.0, 1.0)))
    return V, alpha, beta
