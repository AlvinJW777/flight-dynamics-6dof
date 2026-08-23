"""Unit conversion at the data boundary.

NASA CR-2144 (1972) publishes in US customary units: feet, slugs, pounds-force,
ft/s, slug-ft^2. Everything inside this package works in **SI and radians**. This
module is the single place that conversion happens, so there is exactly one place
to get it wrong and it is tested.

Every factor below is *exact by definition*, not measured — the international
foot, avoirdupois pound and standard gravity are all defined constants, so these
conversions carry no uncertainty. Where a factor is derived from others the
derivation is shown, because a conversion you cannot re-derive is a conversion you
cannot check.

THE ONE THAT WILL CATCH YOU
---------------------------
Aerodynamic derivatives are quoted **per radian** or **per degree** depending on
the source, and the two differ by 57.3. A model built with per-degree derivatives
read as per-radian will still trim happily and produce completely wrong modes.
CR-2144 states its convention — read it, do not assume. :func:`per_degree_to_per_radian`
exists so that when you apply it, you do so deliberately and visibly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- exact defining constants -------------------------------------------------

#: International foot, defined as exactly 0.3048 m since 1959.
FOOT_TO_METRE = 0.3048

#: Avoirdupois pound, defined as exactly 0.45359237 kg.
POUND_MASS_TO_KG = 0.45359237

#: Standard gravity, defined as exactly 9.80665 m/s^2 (CGPM 1901).
STANDARD_GRAVITY = 9.80665

#: Pound-force = 1 lbm under standard gravity. Exact by definition.
POUND_FORCE_TO_NEWTON = POUND_MASS_TO_KG * STANDARD_GRAVITY  # 4.4482216152605

#: A slug is the mass accelerated at 1 ft/s^2 by 1 lbf, so slug = lbf*s^2/ft.
SLUG_TO_KG = POUND_FORCE_TO_NEWTON / FOOT_TO_METRE  # 14.593902937206364

#: Moment of inertia: slug*ft^2 -> kg*m^2.
SLUG_FT2_TO_KG_M2 = SLUG_TO_KG * FOOT_TO_METRE**2  # 1.3558179483314004

#: Torque: lbf*ft -> N*m. Numerically identical to the inertia factor, which is
#: not a coincidence — both are (force x length) in the two systems.
LBF_FT_TO_N_M = POUND_FORCE_TO_NEWTON * FOOT_TO_METRE

#: Knot -> m/s. 1 nautical mile = 1852 m exactly.
KNOT_TO_M_S = 1852.0 / 3600.0

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


# --- scalar conversions -------------------------------------------------------


def ft_to_m(x: float) -> float:
    """Length: feet to metres."""
    return x * FOOT_TO_METRE


def m_to_ft(x: float) -> float:
    return x / FOOT_TO_METRE


def ft_s_to_m_s(x: float) -> float:
    """Velocity: ft/s to m/s. Same factor as length — velocity is length/time and
    the second is common to both systems."""
    return x * FOOT_TO_METRE


def slug_to_kg(x: float) -> float:
    """Mass: slugs to kilograms."""
    return x * SLUG_TO_KG


def slug_ft2_to_kg_m2(x: float) -> float:
    """Moment of inertia: slug*ft^2 to kg*m^2."""
    return x * SLUG_FT2_TO_KG_M2


def lbf_to_n(x: float) -> float:
    """Force: pounds-force to newtons."""
    return x * POUND_FORCE_TO_NEWTON


def lbf_ft_to_n_m(x: float) -> float:
    """Moment: lbf*ft to N*m."""
    return x * LBF_FT_TO_N_M


def ft2_to_m2(x: float) -> float:
    """Area: ft^2 to m^2 — for wing reference area."""
    return x * FOOT_TO_METRE**2


def slug_ft3_to_kg_m3(x: float) -> float:
    """Density: slug/ft^3 to kg/m^3."""
    return x * SLUG_TO_KG / FOOT_TO_METRE**3


def psf_to_pa(x: float) -> float:
    """Pressure: lbf/ft^2 to pascals."""
    return x * POUND_FORCE_TO_NEWTON / FOOT_TO_METRE**2


def per_degree_to_per_radian(x: float) -> float:
    """Convert a derivative quoted per degree into per radian.

    Apply this **only** after confirming the source's convention. It is a factor
    of 57.3 — large enough to be obvious in a plot, small enough to pass a trim
    solve and quietly ruin your eigenvalues.
    """
    return x * RAD_TO_DEG


def per_radian_to_per_degree(x: float) -> float:
    return x * DEG_TO_RAD


# --- reporting back in source units -------------------------------------------


@dataclass(frozen=True)
class SourceUnitsReport:
    """Renders an SI quantity back into CR-2144's units for eyeball comparison.

    Verification means checking your numbers against the report. Doing that
    requires seeing them in the report's units, so this exists to make that easy
    without ever storing anything in imperial.
    """

    @staticmethod
    def length(m: float) -> str:
        return f"{m:,.4g} m  ({m_to_ft(m):,.4g} ft)"

    @staticmethod
    def speed(m_s: float) -> str:
        return f"{m_s:,.4g} m/s  ({m_s / FOOT_TO_METRE:,.4g} ft/s, {m_s / KNOT_TO_M_S:,.4g} kt)"

    @staticmethod
    def mass(kg: float) -> str:
        return f"{kg:,.6g} kg  ({kg / SLUG_TO_KG:,.6g} slug, {kg / POUND_MASS_TO_KG:,.6g} lbm)"

    @staticmethod
    def inertia(kg_m2: float) -> str:
        return f"{kg_m2:,.6g} kg m^2  ({kg_m2 / SLUG_FT2_TO_KG_M2:,.6g} slug ft^2)"

    @staticmethod
    def force(n: float) -> str:
        return f"{n:,.6g} N  ({n / POUND_FORCE_TO_NEWTON:,.6g} lbf)"

    @staticmethod
    def area(m2: float) -> str:
        return f"{m2:,.5g} m^2  ({m2 / FOOT_TO_METRE**2:,.5g} ft^2)"
