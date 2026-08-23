"""ISA, checked against the published standard table.

The table values below are the ISA definition itself, not this code's output, so
a wrong constant or a wrong exponent fails against the standard rather than
agreeing with itself.
"""

from __future__ import annotations

import math

import pytest

from flightdyn.atmosphere import (
    GAMMA,
    H_TROPOPAUSE,
    R_AIR,
    altitude_from_pressure,
    isa,
)

# Published ISA at **geopotential** altitude: h (m), T (K), p (Pa), rho (kg/m^3)
#
# The distinction matters and cost five failing tests to find. ISA is defined in
# geopotential altitude, which accounts for gravity weakening with height. Many
# tables list *geometric* altitude instead, and the two diverge with height:
# 8000 m geometric is 7989.9 m geopotential, a 10 m difference worth 52 Pa, or
# 0.14% in pressure. Two rows here were originally copied from a geometric table
# and disagreed by exactly that amount while every other row matched.
ISA_TABLE = [
    (0,     288.15, 101325.0, 1.22500),
    (1000,  281.65,  89874.6, 1.11164),
    (2000,  275.15,  79495.2, 1.00649),
    (5000,  255.65,  54019.9, 0.73612),
    (8000,  236.15,  35599.8, 0.52517),
    (11000, 216.65,  22632.0, 0.36392),
    (15000, 216.65,  12044.5, 0.19367),
    (20000, 216.65,   5474.9, 0.088035),
]

#: Earth's effective radius for the geopotential conversion, m (ISA definition).
EARTH_RADIUS_ISA = 6356766.0


def geometric_to_geopotential(z_m: float) -> float:
    """Geometric altitude to geopotential. h = R*z / (R + z)."""
    return EARTH_RADIUS_ISA * z_m / (EARTH_RADIUS_ISA + z_m)


class TestAgainstPublishedTable:
    @pytest.mark.parametrize("h, t_k, p_pa, rho", ISA_TABLE)
    def test_temperature(self, h, t_k, p_pa, rho):
        assert isa(h).temperature_k == pytest.approx(t_k, abs=0.01)

    @pytest.mark.parametrize("h, t_k, p_pa, rho", ISA_TABLE)
    def test_pressure(self, h, t_k, p_pa, rho):
        assert isa(h).pressure_pa == pytest.approx(p_pa, rel=2e-5)

    @pytest.mark.parametrize("h, t_k, p_pa, rho", ISA_TABLE)
    def test_density(self, h, t_k, p_pa, rho):
        assert isa(h).density_kg_m3 == pytest.approx(rho, rel=2e-5)

    def test_sea_level_speed_of_sound(self):
        # Published: 340.29 m/s
        assert isa(0).speed_of_sound_m_s == pytest.approx(340.29, rel=1e-4)

    def test_sea_level_viscosity(self):
        # Published: 1.789e-5 Pa s
        assert isa(0).viscosity_pa_s == pytest.approx(1.789e-5, rel=2e-3)


class TestGeopotentialAltitude:
    """Pin the geopotential/geometric distinction, since mixing them silently
    shifts every pressure by a tenth of a percent."""

    def test_they_agree_at_sea_level(self):
        assert geometric_to_geopotential(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_geopotential_is_always_the_smaller(self):
        for z in (1000, 5000, 8000, 11000, 20000):
            assert geometric_to_geopotential(z) < z

    def test_the_8km_offset_is_about_ten_metres(self):
        assert 8000.0 - geometric_to_geopotential(8000.0) == pytest.approx(10.1, abs=0.2)

    def test_ignoring_it_shifts_pressure_by_a_tenth_of_a_percent(self):
        """Quantifies the error that mixing conventions introduces."""
        p_geopotential = isa(geometric_to_geopotential(8000.0)).pressure_pa
        p_naive = isa(8000.0).pressure_pa
        assert 0.0010 < (p_geopotential - p_naive) / p_naive < 0.0020


class TestPhysicalConsistency:
    """Relations that must hold at every altitude, independent of the table."""

    ALTITUDES = [0, 500, 3000, 7500, 10999, 11001, 14000, 19000]

    @pytest.mark.parametrize("h", ALTITUDES)
    def test_ideal_gas_law_closes(self, h):
        s = isa(h)
        assert s.pressure_pa == pytest.approx(s.density_kg_m3 * R_AIR * s.temperature_k, rel=1e-12)

    @pytest.mark.parametrize("h", ALTITUDES)
    def test_speed_of_sound_depends_only_on_temperature(self, h):
        s = isa(h)
        assert s.speed_of_sound_m_s == pytest.approx(math.sqrt(GAMMA * R_AIR * s.temperature_k), rel=1e-12)

    @pytest.mark.parametrize("h", ALTITUDES)
    def test_hydrostatic_balance_holds(self, h):
        """dp/dh must equal −ρg. Checks the *derivation*, not just the endpoints —
        a wrong exponent would still hit the table at 0 and 11 km but fail here."""
        eps = 0.5
        s = isa(h)
        dpdh = (isa(h + eps).pressure_pa - isa(h - eps).pressure_pa) / (2 * eps)
        assert dpdh == pytest.approx(-s.density_kg_m3 * 9.80665, rel=1e-5)

    def test_properties_fall_monotonically(self):
        prev_p, prev_rho = float("inf"), float("inf")
        for h in range(0, 20001, 250):
            s = isa(h)
            assert s.pressure_pa < prev_p
            assert s.density_kg_m3 < prev_rho
            prev_p, prev_rho = s.pressure_pa, s.density_kg_m3

    def test_continuous_across_the_tropopause(self):
        """Two different formulae meet here. They must agree, or there is a step
        in density that would show up as a discontinuity in every aero force."""
        below, above = isa(H_TROPOPAUSE - 1e-6), isa(H_TROPOPAUSE + 1e-6)
        assert below.pressure_pa == pytest.approx(above.pressure_pa, rel=1e-9)
        assert below.density_kg_m3 == pytest.approx(above.density_kg_m3, rel=1e-9)
        assert below.temperature_k == pytest.approx(above.temperature_k, rel=1e-9)

    def test_lapse_rate_is_6_5_k_per_km(self):
        assert isa(0).temperature_k - isa(1000).temperature_k == pytest.approx(6.5, abs=1e-9)

    def test_stratosphere_is_isothermal(self):
        assert isa(12000).temperature_k == pytest.approx(isa(19000).temperature_k, rel=1e-12)


class TestPressureAltitude:
    @pytest.mark.parametrize("h", [0, 1500, 6000, 11000, 16000, 20000])
    def test_round_trip(self, h):
        assert altitude_from_pressure(isa(h).pressure_pa) == pytest.approx(h, abs=1e-6)

    def test_rejects_non_positive_pressure(self):
        with pytest.raises(ValueError):
            altitude_from_pressure(0.0)


class TestDerivedQuantities:
    def test_mach_at_sea_level(self):
        assert isa(0).mach(340.29) == pytest.approx(1.0, rel=1e-4)

    def test_dynamic_pressure_at_sea_level(self):
        # q = 0.5 * 1.225 * 100^2 = 6125 Pa
        assert isa(0).dynamic_pressure(100.0) == pytest.approx(6125.0, rel=1e-3)

    def test_same_true_airspeed_gives_less_dynamic_pressure_at_altitude(self):
        """Why aircraft climb: less density means less drag for the same TAS."""
        assert isa(11000).dynamic_pressure(250.0) < isa(0).dynamic_pressure(250.0)


class TestEnvelope:
    @pytest.mark.parametrize("h", [-2000, 25000, 100000])
    def test_rejects_altitudes_outside_the_implemented_range(self, h):
        with pytest.raises(ValueError, match="outside the implemented range"):
            isa(h)
