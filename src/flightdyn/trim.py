"""Trim: finding the equilibrium flight condition.

Trim is the state at which every acceleration is zero, so the aircraft holds it
indefinitely with the controls fixed. For straight and level flight, three
unknowns — angle of attack, elevator and throttle — must drive three residuals to
zero:

    u_dot = 0     forward acceleration
    w_dot = 0     vertical acceleration
    q_dot = 0     pitching acceleration

That is a nonlinear root-finding problem. It has no closed form because the
gravity components rotate with attitude, the aerodynamic forces rotate from wind
to body axes through alpha, and thrust acts along an inclined line.

THE CONSTRAINT THAT CLOSES IT
-----------------------------
Flight path angle is ``gamma = theta - alpha``. For level flight gamma = 0, so
``theta = alpha``: the aircraft points exactly as far above the horizon as the
air meets it from below. That is what makes three unknowns match three equations
rather than four.

ON CONVERGENCE FAILURE
----------------------
If the solver does not converge, **do not tighten the tolerance or nudge the
initial guess until something emerges**. Non-convergence almost always means the
requested condition is not achievable with the model as built - insufficient
thrust, or the linear aerodynamic model extrapolated past where it is valid.
:class:`TrimResult` therefore carries the residuals and the achieved angle of
attack so the failure can be diagnosed rather than papered over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from .aerodynamics import AeroModel, Controls
from .atmosphere import isa
from .dynamics import (
    IDX_RATE,
    IDX_VEL,
    RigidBody,
    gravity_body,
    make_state,
    rigid_body_derivative,
)
from .frames import euler_to_quat


@dataclass(frozen=True)
class TrimResult:
    """A trimmed flight condition, with the evidence that it really is one."""

    state: np.ndarray
    controls: Controls
    alpha_rad: float
    theta_rad: float
    residuals: np.ndarray
    converged: bool
    message: str

    @property
    def max_residual(self) -> float:
        return float(np.max(np.abs(self.residuals)))

    def summary(self) -> str:
        return (
            f"trim {'converged' if self.converged else 'FAILED'}: "
            f"alpha = {math.degrees(self.alpha_rad):.3f} deg, "
            f"elevator = {math.degrees(self.controls.elevator):.3f} deg, "
            f"throttle = {self.controls.throttle:.4f}, "
            f"max residual = {self.max_residual:.3e}"
        )


def _build_state(alpha: float, speed: float, altitude: float) -> np.ndarray:
    """State vector for straight and level flight at a given incidence.

    Level flight means gamma = 0, hence theta = alpha. Body velocity components
    follow from resolving the airspeed through the incidence.
    """
    return make_state(
        velocity=(speed * math.cos(alpha), 0.0, speed * math.sin(alpha)),
        rates=(0.0, 0.0, 0.0),
        quat=euler_to_quat(0.0, alpha, 0.0),
        position=(0.0, 0.0, -altitude),
    )


def trim_straight_level(
    model: AeroModel,
    body: RigidBody,
    speed_m_s: float,
    altitude_m: float,
    alpha_guess_rad: float = 0.05,
    elevator_guess_rad: float = 0.0,
    throttle_guess: float = 0.3,
) -> TrimResult:
    """Solve for straight and level trim at a given airspeed and altitude."""
    density = isa(altitude_m).density_kg_m3

    def residuals(unknowns: np.ndarray) -> np.ndarray:
        alpha, elevator, throttle = unknowns
        state = _build_state(alpha, speed_m_s, altitude_m)
        controls = Controls(elevator=elevator, throttle=throttle)

        force, moment = model.forces_moments(
            state[IDX_VEL], state[IDX_RATE], controls, density
        )
        force = force + gravity_body(state[6:10], body.mass)

        dx = rigid_body_derivative(state, force, moment, body)
        # u_dot, w_dot, q_dot — the three that must vanish for level flight
        return np.array([dx[0], dx[2], dx[4]])

    sol = root(
        residuals,
        np.array([alpha_guess_rad, elevator_guess_rad, throttle_guess]),
        method="hybr",
        tol=1e-14,
    )

    alpha, elevator, throttle = sol.x
    controls = Controls(elevator=elevator, throttle=throttle)
    state = _build_state(alpha, speed_m_s, altitude_m)
    res = residuals(sol.x)

    return TrimResult(
        state=state,
        controls=controls,
        alpha_rad=float(alpha),
        theta_rad=float(alpha),
        residuals=res,
        converged=bool(sol.success) and float(np.max(np.abs(res))) < 1e-8,
        message=str(sol.message),
    )


def trimmed_derivative(model: AeroModel, body: RigidBody, altitude_m: float, controls: Controls):
    """Closure giving the state derivative with controls held fixed.

    This is what gets propagated to prove a trim really holds, and what gets
    perturbed to linearise about it. Density is frozen at the trim altitude so
    the linearisation is not contaminated by the atmosphere model - a deliberate
    simplification, recorded in ASSUMPTIONS.md.
    """
    density = isa(altitude_m).density_kg_m3

    def derivative(state: np.ndarray) -> np.ndarray:
        force, moment = model.forces_moments(state[IDX_VEL], state[IDX_RATE], controls, density)
        force = force + gravity_body(state[6:10], body.mass)
        return rigid_body_derivative(state, force, moment, body)

    return derivative
