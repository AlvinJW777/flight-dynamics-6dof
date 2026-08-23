"""Boeing 747 data from NASA CR-2144, Section IX.

Heffley, R.K. and Jewell, W.F., *Aircraft Handling Qualities Data*, NASA CR-2144,
December 1972. Section IX covers the B-747; original data source is Boeing
simulator description D6-30643.

WHY FLIGHT CONDITION 2
----------------------
CR-2144 tabulates ten flight conditions, but only F/C 1 and 2 carry
*non-dimensional* derivatives in tabulated form (Tables IX-1 and IX-2); the other
eight are given only as plots against Mach, which cannot be read precisely. F/C 2
therefore uniquely gives the full chain from independent tables:

    Table IX-2  non-dimensional derivatives  -> input to the nonlinear model
    Table IX-3  mass, inertia, geometry      -> input
    Table IX-4  dimensional derivatives      -> VERIFICATION target
    Table IX-5  transfer function factors    -> VALIDATION target

Verification and validation come from different tables, so agreement is
meaningful rather than circular.

CONVENTIONS, from Appendix A
----------------------------
* All aerodynamic derivatives are **per radian** (stated explicitly in the tables).
* Longitudinal rate derivatives are normalised by ``c/2V``, lateral by ``b/2V``
  (Appendix A, pp. A-14/A-15).
* Moments of inertia are **body axis**, slug-ft^2; weight in lb (Appendix A, p. A-8).
* Tables IX-4 and IX-5 are explicitly headed "BODY AXIS SYSTEM".
* The lateral rolling and yawing derivatives use the *same symbols* for body and
  stability axes but differ numerically, so the axis system must be tracked
  (Appendix A, p. A-15).

TRANSCRIPTION WARNING
---------------------
CR-2144 is a 1972 scanned document. Its text layer is unusable, so these values
were read from page images. **Every number here must be checked against the PDF
by eye before any result is published.** A mistyped derivative produces a model
that trims perfectly and has entirely wrong modes — the failure is silent. Page
references are given on every block so checking is quick.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..dynamics import RigidBody
from ..units import (
    RAD_TO_DEG,
    ft2_to_m2,
    ft_s_to_m_s,
    ft_to_m,
    slug_ft2_to_kg_m2,
    slug_to_kg,
)

#: Standard gravity in ft/s^2, for converting weight to mass in the source units.
G_FT_S2 = 32.174


@dataclass(frozen=True)
class Geometry:
    """Reference geometry. CR-2144 Table IX-3 header, p. 229 (PDF p. 233)."""

    wing_area_m2: float
    span_m: float
    mean_chord_m: float


@dataclass(frozen=True)
class FlightCondition:
    """One trimmed operating point. CR-2144 Table IX-3."""

    name: str
    altitude_m: float
    mach: float
    true_airspeed_m_s: float
    mass_kg: float
    alpha_trim_rad: float
    gamma_trim_rad: float
    dynamic_pressure_pa: float
    cg_fraction_mgc: float


@dataclass(frozen=True)
class LongitudinalDerivatives:
    """Non-dimensional longitudinal derivatives, per radian. Table IX-2, p. 217."""

    CL: float
    CD: float
    CL_alpha: float
    CD_alpha: float
    Cm_alpha: float
    CL_alphadot: float
    Cm_alphadot: float
    CL_q: float
    Cm_q: float
    CL_M: float
    Cm_M: float
    CL_de: float
    Cm_de: float


@dataclass(frozen=True)
class LateralDerivatives:
    """Non-dimensional lateral-directional derivatives, per radian. Table IX-2."""

    Cy_beta: float
    Cl_beta: float
    Cn_beta: float
    Cl_p: float
    Cn_p: float
    Cl_r: float
    Cn_r: float
    Cl_da: float
    Cn_da: float
    Cy_dr: float
    Cl_dr: float
    Cn_dr: float


@dataclass(frozen=True)
class Mode:
    """A second-order mode: damping ratio and undamped natural frequency."""

    zeta: float
    omega_n_rad_s: float

    @property
    def period_s(self) -> float:
        """Damped period. Infinite if critically damped or worse."""
        if self.zeta >= 1.0:
            return float("inf")
        return 2.0 * 3.141592653589793 / (self.omega_n_rad_s * (1.0 - self.zeta**2) ** 0.5)

    @property
    def time_to_half_s(self) -> float:
        """Time for the amplitude to halve. Negative means divergent."""
        sigma = self.zeta * self.omega_n_rad_s
        return float("inf") if sigma == 0 else 0.693147 / sigma


# ---------------------------------------------------------------------------
# Flight Condition 2 — Power Approach, sea level, 165 KTAS
# ---------------------------------------------------------------------------

GEOMETRY = Geometry(
    wing_area_m2=ft2_to_m2(5500.0),
    span_m=ft_to_m(195.68),
    mean_chord_m=ft_to_m(27.31),
)

#: Table IX-3, column 2 (PDF p. 233).
FC2 = FlightCondition(
    name="FC2 power approach, sea level, M=0.249",
    altitude_m=0.0,
    mach=0.249,
    true_airspeed_m_s=ft_s_to_m_s(278.0),
    mass_kg=slug_to_kg(564032.0 / G_FT_S2),
    alpha_trim_rad=5.70 / RAD_TO_DEG,
    gamma_trim_rad=0.0,
    dynamic_pressure_pa=92.2 * 4.4482216152605 / 0.3048**2,
    cg_fraction_mgc=0.250,
)

#: Table IX-3, column 2. Body axis, slug-ft^2 in the source.
FC2_BODY = RigidBody(
    mass=FC2.mass_kg,
    Ixx=slug_ft2_to_kg_m2(0.142e8),
    Iyy=slug_ft2_to_kg_m2(0.323e8),
    Izz=slug_ft2_to_kg_m2(0.454e8),
    Ixz=slug_ft2_to_kg_m2(870050.0),
)

#: Table IX-2 (PDF p. 221). Per radian.
FC2_LONGITUDINAL = LongitudinalDerivatives(
    CL=1.11,
    CD=0.102,
    CL_alpha=5.70,
    CD_alpha=0.66,
    Cm_alpha=-1.26,
    CL_alphadot=-6.7,
    Cm_alphadot=-3.2,
    CL_q=5.4,
    Cm_q=-20.8,
    CL_M=-0.81,
    Cm_M=0.27,
    CL_de=0.338,
    Cm_de=-1.34,
)

#: Table IX-2 (PDF p. 221). Per radian.
#: Signs on Cn_r and Cn_dr were ambiguous in the scan (the minus renders faintly);
#: both are set negative because yaw damping and rudder yawing moment must be, and
#: the landing-configuration table (IX-1) shows them unambiguously negative.
#: FLAG FOR VISUAL CHECK.
FC2_LATERAL = LateralDerivatives(
    Cy_beta=-0.96,
    Cl_beta=-0.221,
    Cn_beta=0.150,
    Cl_p=-0.45,
    Cn_p=-0.121,
    Cl_r=0.101,
    Cn_r=-0.30,
    Cl_da=0.0461,
    Cn_da=0.0064,
    Cy_dr=0.175,
    Cl_dr=0.007,
    Cn_dr=-0.109,
)

#: Table IX-4, column 2 (PDF p. 234). Body axis, source units (ft, slug, s).
#: The VERIFICATION target: our numerically linearised A matrix must reproduce
#: the state matrix these imply.
FC2_DIMENSIONAL_FT: dict[str, float] = {
    "Xu": -0.0108,      # 1/s
    "Zu": -0.150,       # 1/s
    "Mu": 0.000181,     # 1/(ft s)
    "Xw": 0.106,        # 1/s
    "Zw": -0.613,       # 1/s
    "Mw": -0.00193,     # 1/(ft s)
    "Zwdot": 0.0338,    # dimensionless
    "Zq": -7.58,        # ft/s
    "Mwdot": -0.000240, # 1/ft
    "Mq": -0.437,       # 1/s
    "Xde": 0.971,       # ft/s^2
    "Zde": -9.73,       # ft/s^2
    "Mde": -0.574,      # 1/s^2
}

#: Table IX-5, column 2 (PDF p. 235), denominator factors of the bare-airframe
#: elevator transfer function. The VALIDATION target.
FC2_MODES: dict[str, Mode] = {
    "phugoid": Mode(zeta=0.0228, omega_n_rad_s=0.127),
    "short_period": Mode(zeta=0.629, omega_n_rad_s=0.910),
}


# ---------------------------------------------------------------------------
# Flight Condition 1 — Landing, sea level, 131 KTAS. Second complete case.
# ---------------------------------------------------------------------------

FC1 = FlightCondition(
    name="FC1 landing, sea level, M=0.198",
    altitude_m=0.0,
    mach=0.198,
    true_airspeed_m_s=ft_s_to_m_s(221.0),
    mass_kg=slug_to_kg(564032.0 / G_FT_S2),
    alpha_trim_rad=8.50 / RAD_TO_DEG,
    gamma_trim_rad=0.0,
    dynamic_pressure_pa=58.1 * 4.4482216152605 / 0.3048**2,
    cg_fraction_mgc=0.250,
)

FC1_BODY = FC2_BODY  # identical mass and inertia at this weight (Table IX-3)

#: Table IX-1 (PDF p. 220). Per radian.
FC1_LONGITUDINAL = LongitudinalDerivatives(
    CL=1.76,
    CD=0.263,
    CL_alpha=5.67,
    CD_alpha=1.13,
    Cm_alpha=-1.45,
    CL_alphadot=-6.7,
    Cm_alphadot=-3.3,
    CL_q=5.65,
    Cm_q=-21.4,
    CL_M=-1.1,
    Cm_M=0.36,
    CL_de=0.396,
    Cm_de=-1.40,
)

FC1_MODES: dict[str, Mode] = {
    "phugoid": Mode(zeta=0.0417, omega_n_rad_s=0.152),
    "short_period": Mode(zeta=0.616, omega_n_rad_s=0.771),
}
