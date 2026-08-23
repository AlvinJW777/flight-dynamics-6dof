"""Aerodynamic and propulsive forces from stability derivatives.

WHAT IS AND IS NOT NONLINEAR
----------------------------
The aerodynamics here is **linear in the perturbation variables** — CR-2144
publishes derivatives, not lookup tables, so nothing else is supported by the
data. Coefficients are built as

    C = C0 + C_alpha*alpha + C_q*qhat + C_delta*delta

What *is* genuinely nonlinear is the rigid-body mechanics that consumes these
forces: the ``omega x v`` and ``omega x I omega`` transport terms, the quaternion
kinematics, and gravity rotating through attitude. That is what trim solves
against and what linearisation exercises.

Do not describe this as a nonlinear aerodynamic model. It is a linear aerodynamic
model inside a nonlinear rigid-body simulation, and the distinction is a fair
interview question.

VALIDITY
--------
Derivatives are a first-order expansion about the published trim point, so they
degrade as you move away from it. :func:`AeroModel.forces_moments` warns outside
a stated band rather than silently extrapolating into stall, where a linear
lift slope predicts lift that does not exist.

AXES
----
Lift and drag are defined in **wind axes** — drag opposes the relative wind by
definition. Moments and the equations of motion live in **body axes**. The
rotation between them is by alpha and beta, and forgetting it is a standard way
to produce a model that trims and then behaves oddly in a climb.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from .aircraft.b747 import Geometry, LateralDerivatives, LongitudinalDerivatives
from .frames import airdata

#: Beyond this the linear lift slope is no longer a fair representation.
ALPHA_VALID_RAD = math.radians(15.0)
BETA_VALID_RAD = math.radians(10.0)


@dataclass(frozen=True)
class Controls:
    """Control inputs. Deflections in radians, throttle as a fraction of maximum."""

    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0


@dataclass(frozen=True)
class Propulsion:
    """A deliberately simple thrust model.

    CR-2144 gives thrust incidence ``i_th`` = 2.5 deg to the fuselage reference
    line and a moment arm ``l_th`` = 10 ft, so thrust is modelled as a force along
    that inclined line acting at that offset. Engine dynamics, spool-up lag and
    thrust lapse with altitude and Mach are all omitted — at the fidelity of a
    derivative-based model they would be false precision, and trim only needs the
    steady value.
    """

    max_thrust_n: float
    incidence_rad: float
    moment_arm_m: float

    def force_moment(self, throttle: float) -> tuple[np.ndarray, np.ndarray]:
        """Body-axis force and moment from a throttle setting."""
        t = self.max_thrust_n * throttle
        force = np.array([t * math.cos(self.incidence_rad), 0.0, -t * math.sin(self.incidence_rad)])
        # Positive arm below the c.g. gives a nose-up moment for positive thrust.
        moment = np.array([0.0, t * self.moment_arm_m, 0.0])
        return force, moment


class AeroModel:
    """Linear-derivative aerodynamics for one flight condition.

    ``alpha_ref`` and the trim coefficients set the expansion point. The zero-alpha
    intercepts are recovered from the published trim values so the model reproduces
    the reference condition exactly::

        CL0 = CL_trim - CL_alpha * alpha_trim

    That back-out is a modelling choice, not data: it assumes the lift slope holds
    linearly all the way back to zero incidence, which for a flapped approach
    configuration it does not. It is acceptable here because everything of
    interest happens within a few degrees of trim, and it is recorded in
    ASSUMPTIONS.md.
    """

    def __init__(
        self,
        geometry: Geometry,
        longitudinal: LongitudinalDerivatives,
        lateral: LateralDerivatives,
        alpha_trim_rad: float,
        propulsion: Propulsion,
        mach_trim: float = 0.0,
        speed_of_sound_m_s: float = 340.29,
        warn_outside_envelope: bool = True,
    ) -> None:
        self.geom = geometry
        self.mach_trim = mach_trim
        self.a_sound = speed_of_sound_m_s
        self.lon = longitudinal
        self.lat = lateral
        self.alpha_trim = alpha_trim_rad
        self.prop = propulsion
        self.warn = warn_outside_envelope

        self.CL0 = longitudinal.CL - longitudinal.CL_alpha * alpha_trim_rad
        self.CD0 = longitudinal.CD - longitudinal.CD_alpha * alpha_trim_rad
        # Trimmed flight means zero net pitching moment at the trim incidence
        # with zero elevator, so the intercept follows from Cm_alpha.
        self.Cm0 = -longitudinal.Cm_alpha * alpha_trim_rad

    # -- coefficient build-up --------------------------------------------------

    def _check_envelope(self, alpha: float, beta: float) -> None:
        if not self.warn:
            return
        if abs(alpha) > ALPHA_VALID_RAD:
            warnings.warn(
                f"alpha = {math.degrees(alpha):.1f} deg is outside the linear-derivative "
                f"validity band (+/-{math.degrees(ALPHA_VALID_RAD):.0f} deg); "
                "lift is being extrapolated through stall",
                RuntimeWarning,
                stacklevel=3,
            )
        if abs(beta) > BETA_VALID_RAD:
            warnings.warn(
                f"beta = {math.degrees(beta):.1f} deg is outside the validity band",
                RuntimeWarning,
                stacklevel=3,
            )

    def coefficients(
        self, v_body: np.ndarray, omega: np.ndarray, controls: Controls
    ) -> dict[str, float]:
        """Non-dimensional force and moment coefficients.

        Rate derivatives use the standard non-dimensionalisation from CR-2144
        Appendix A: ``qhat = q*c/(2V)`` for longitudinal and ``phat = p*b/(2V)``,
        ``rhat = r*b/(2V)`` for lateral. Getting these wrong scales the damping
        terms by ``2V/c``, which at 85 m/s is a factor of about twenty.
        """
        V, alpha, beta = airdata(v_body)
        if V < 1.0:
            return dict.fromkeys(("CL", "CD", "CY", "Cl", "Cm", "Cn"), 0.0)

        self._check_envelope(alpha, beta)
        p, q, r = omega
        c, b = self.geom.mean_chord_m, self.geom.span_m

        qhat = q * c / (2.0 * V)
        phat = p * b / (2.0 * V)
        rhat = r * b / (2.0 * V)

        # Compressibility. Omitting these leaves every speed derivative wrong -
        # Xu, Zu and Mu all depend on how the coefficients change with Mach, and
        # dropping them showed up as a 20-60% error confined to the u column of
        # the Jacobian while every w and q term agreed to under 3%. A clean
        # example of verification locating a missing term that validation missed.
        dM = V / self.a_sound - self.mach_trim

        CL = (
            self.CL0
            + self.lon.CL_alpha * alpha
            + self.lon.CL_q * qhat
            + self.lon.CL_de * controls.elevator
            + self.lon.CL_M * dM
        )
        CD = self.CD0 + self.lon.CD_alpha * alpha
        Cm = (
            self.Cm0
            + self.lon.Cm_alpha * alpha
            + self.lon.Cm_q * qhat
            + self.lon.Cm_de * controls.elevator
            + self.lon.Cm_M * dM
        )
        CY = self.lat.Cy_beta * beta + self.lat.Cy_dr * controls.rudder
        Cl = (
            self.lat.Cl_beta * beta
            + self.lat.Cl_p * phat
            + self.lat.Cl_r * rhat
            + self.lat.Cl_da * controls.aileron
            + self.lat.Cl_dr * controls.rudder
        )
        Cn = (
            self.lat.Cn_beta * beta
            + self.lat.Cn_p * phat
            + self.lat.Cn_r * rhat
            + self.lat.Cn_da * controls.aileron
            + self.lat.Cn_dr * controls.rudder
        )
        return {"CL": CL, "CD": CD, "CY": CY, "Cl": Cl, "Cm": Cm, "Cn": Cn}

    # -- dimensional forces ----------------------------------------------------

    def forces_moments(
        self, v_body: np.ndarray, omega: np.ndarray, controls: Controls, density: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Total body-axis aerodynamic + propulsive force (N) and moment (N m)."""
        V, alpha, beta = airdata(v_body)
        c = self.coefficients(v_body, omega, controls)

        q_dyn = 0.5 * density * V**2
        S, span, chord = self.geom.wing_area_m2, self.geom.span_m, self.geom.mean_chord_m

        lift = c["CL"] * q_dyn * S
        drag = c["CD"] * q_dyn * S
        side = c["CY"] * q_dyn * S

        # Wind axes -> body axes. Drag opposes the relative wind, lift is
        # perpendicular to it in the plane of symmetry.
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)
        fx = -drag * ca * cb - side * ca * sb + lift * sa
        fy = -drag * sb + side * cb
        fz = -drag * sa * cb - side * sa * sb - lift * ca

        moment = np.array([
            c["Cl"] * q_dyn * S * span,
            c["Cm"] * q_dyn * S * chord,
            c["Cn"] * q_dyn * S * span,
        ])

        thrust_f, thrust_m = self.prop.force_moment(controls.throttle)
        return np.array([fx, fy, fz]) + thrust_f, moment + thrust_m
