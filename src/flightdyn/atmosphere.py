"""International Standard Atmosphere, sea level to 20 km.

    dp/dh = -rho*g          hydrostatic balance
    p     = rho*R*T         ideal gas
    T(h)  = defined by ISA  linear lapse to 11 km, then isothermal

Combining the first three gives pressure in closed form. In the troposphere the
lapse rate is non-zero so

    p/p0 = (T/T0) ** (g / (R*L))

and in the isothermal layer the lapse rate is zero, that exponent is undefined,
and the same derivation instead yields

    p/p1 = exp(-g*(h - h1) / (R*T1))

Two different formulae because the *integral changes form* when L = 0 — not
because the physics changes. That is why the code branches at 11 km rather than
using one expression with a fudge.

Only the troposphere and lower stratosphere are implemented. Aircraft do not fly
above 20 km, and pretending to model higher would be unvalidated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- ISA defining constants ---------------------------------------------------
T0 = 288.15           # K, sea-level temperature
P0 = 101325.0         # Pa, sea-level pressure
RHO0 = 1.225          # kg/m^3, sea-level density
G0 = 9.80665          # m/s^2, standard gravity
R_AIR = 287.0528      # J/(kg K), specific gas constant for dry air
GAMMA = 1.4           # ratio of specific heats

LAPSE_RATE = 0.0065   # K/m, temperature fall per metre in the troposphere
H_TROPOPAUSE = 11000.0
T_TROPOPAUSE = T0 - LAPSE_RATE * H_TROPOPAUSE      # 216.65 K
H_MAX = 20000.0

# Sutherland's law, for viscosity
MU_REF = 1.716e-5
T_REF_SUTH = 273.15
S_SUTH = 110.4


@dataclass(frozen=True)
class AtmosphereState:
    """Air properties at one altitude. SI throughout."""

    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float
    viscosity_pa_s: float

    @property
    def kinematic_viscosity_m2_s(self) -> float:
        return self.viscosity_pa_s / self.density_kg_m3

    def mach(self, tas_m_s: float) -> float:
        """Mach number for a true airspeed."""
        return tas_m_s / self.speed_of_sound_m_s

    def dynamic_pressure(self, tas_m_s: float) -> float:
        """q = 1/2 rho V^2 — the scale that non-dimensionalises every aero force."""
        return 0.5 * self.density_kg_m3 * tas_m_s**2


def isa(altitude_m: float) -> AtmosphereState:
    """ISA properties at geopotential altitude.

    Raises above 20 km rather than extrapolating. An aircraft model asked for
    conditions outside its validated envelope should complain, not quietly return
    a plausible number — that is exactly how a wrong result survives review.
    """
    h = float(altitude_m)
    if h < -1000.0 or h > H_MAX:
        raise ValueError(
            f"altitude {h:.0f} m is outside the implemented range (-1000 to {H_MAX:.0f} m)"
        )

    if h <= H_TROPOPAUSE:
        temperature = T0 - LAPSE_RATE * h
        pressure = P0 * (temperature / T0) ** (G0 / (R_AIR * LAPSE_RATE))
    else:
        temperature = T_TROPOPAUSE
        p_tropopause = P0 * (T_TROPOPAUSE / T0) ** (G0 / (R_AIR * LAPSE_RATE))
        pressure = p_tropopause * math.exp(-G0 * (h - H_TROPOPAUSE) / (R_AIR * T_TROPOPAUSE))

    density = pressure / (R_AIR * temperature)
    a = math.sqrt(GAMMA * R_AIR * temperature)
    mu = MU_REF * (temperature / T_REF_SUTH) ** 1.5 * (T_REF_SUTH + S_SUTH) / (temperature + S_SUTH)

    return AtmosphereState(h, temperature, pressure, density, a, mu)


def altitude_from_pressure(pressure_pa: float) -> float:
    """Pressure altitude — the inverse of :func:`isa`.

    What an altimeter actually reports: it measures pressure and converts using
    exactly this relation, which is why altimeters need a subscale setting when
    the real atmosphere is not standard.
    """
    if pressure_pa <= 0:
        raise ValueError(f"pressure must be positive, got {pressure_pa}")

    p_tropopause = P0 * (T_TROPOPAUSE / T0) ** (G0 / (R_AIR * LAPSE_RATE))
    if pressure_pa >= p_tropopause:
        return (T0 / LAPSE_RATE) * (1.0 - (pressure_pa / P0) ** (R_AIR * LAPSE_RATE / G0))
    return H_TROPOPAUSE - (R_AIR * T_TROPOPAUSE / G0) * math.log(pressure_pa / p_tropopause)
