"""Unit conversions, checked against externally published values.

The discipline that matters here: a test asserting my conversion factor equals my
conversion factor proves nothing. Every check below anchors to a number published
independently — the ISA sea-level constants, the defined foot, the nautical mile —
so a wrong factor fails against the outside world rather than agreeing with itself.
"""

from __future__ import annotations

import math

import pytest

from flightdyn.units import (
    DEG_TO_RAD,
    FOOT_TO_METRE,
    KNOT_TO_M_S,
    LBF_FT_TO_N_M,
    POUND_FORCE_TO_NEWTON,
    RAD_TO_DEG,
    SLUG_FT2_TO_KG_M2,
    SLUG_TO_KG,
    ft2_to_m2,
    ft_s_to_m_s,
    ft_to_m,
    lbf_to_n,
    m_to_ft,
    per_degree_to_per_radian,
    per_radian_to_per_degree,
    psf_to_pa,
    slug_ft2_to_kg_m2,
    slug_ft3_to_kg_m3,
    slug_to_kg,
)


class TestDefiningConstants:
    """These are defined exactly, so they must match to full precision."""

    def test_foot_is_exactly_0_3048_m(self):
        assert FOOT_TO_METRE == 0.3048

    def test_pound_force_matches_published_value(self):
        # 1 lbf = 4.448 221 615 260 5 N, exact by definition
        assert POUND_FORCE_TO_NEWTON == pytest.approx(4.4482216152605, rel=1e-15)

    def test_slug_matches_published_value(self):
        # Widely tabulated as 14.593 90 kg
        assert SLUG_TO_KG == pytest.approx(14.5939029372, rel=1e-11)

    def test_slug_ft2_matches_published_value(self):
        # Widely tabulated as 1.355 818 kg m^2
        assert SLUG_FT2_TO_KG_M2 == pytest.approx(1.3558179483, rel=1e-10)

    def test_knot_is_one_nautical_mile_per_hour(self):
        assert KNOT_TO_M_S == pytest.approx(0.5144444444, rel=1e-9)

    def test_torque_and_inertia_factors_coincide(self):
        """Both are (force x length); the equality is structural, not luck."""
        assert LBF_FT_TO_N_M == pytest.approx(SLUG_FT2_TO_KG_M2, rel=1e-15)


class TestAgainstStandardAtmosphere:
    """The strongest checks available: ISA sea-level values are published in both
    unit systems independently, so converting one must reproduce the other."""

    def test_sea_level_density(self):
        # ISA: 0.0023769 slug/ft^3  and  1.225 kg/m^3
        assert slug_ft3_to_kg_m3(0.00237690) == pytest.approx(1.2250, rel=1e-4)

    def test_sea_level_pressure(self):
        # ISA: 2116.22 lbf/ft^2  and  101325 Pa
        assert psf_to_pa(2116.22) == pytest.approx(101325.0, rel=1e-5)

    def test_sea_level_speed_of_sound(self):
        # ISA: 1116.45 ft/s  and  340.29 m/s
        assert ft_s_to_m_s(1116.45) == pytest.approx(340.29, rel=1e-4)


class TestRoundTrips:
    @pytest.mark.parametrize("value", [0.0, 1.0, -3.5, 1.0e6, 1.0e-9])
    def test_length_round_trip(self, value):
        assert m_to_ft(ft_to_m(value)) == pytest.approx(value, rel=1e-14, abs=1e-18)

    def test_conversions_are_linear(self):
        """Scaling the input must scale the output identically — catches any
        stray offset, which would be a units-vs-scale confusion."""
        for fn in (ft_to_m, slug_to_kg, lbf_to_n, slug_ft2_to_kg_m2):
            assert fn(0.0) == 0.0
            assert fn(2.0) == pytest.approx(2.0 * fn(1.0), rel=1e-15)


class TestAngleConversions:
    def test_half_turn(self):
        assert 180.0 * DEG_TO_RAD == pytest.approx(math.pi, rel=1e-15)

    def test_round_trip(self):
        assert per_radian_to_per_degree(per_degree_to_per_radian(0.1)) == pytest.approx(0.1, rel=1e-15)

    def test_the_factor_is_57_3(self):
        """The one that silently ruins eigenvalues. Pin it explicitly."""
        assert RAD_TO_DEG == pytest.approx(57.29577951, rel=1e-8)

    def test_a_typical_lift_slope_converts_sensibly(self):
        """CL_alpha of about 0.1 per degree is roughly 5.7 per radian, and thin
        aerofoil theory says 2*pi = 6.28. If a converted derivative lands nowhere
        near that, the convention was misread."""
        cl_alpha_per_rad = per_degree_to_per_radian(0.1)
        assert 5.0 < cl_alpha_per_rad < 6.5


class TestAircraftScaleSanity:
    """Order-of-magnitude checks using published 747 figures. These would catch a
    factor-of-ten or a slug/lbm confusion, which the exact tests would not if the
    constant itself were wrong in a self-consistent way."""

    def test_747_max_takeoff_mass(self):
        # 747-100 MTOW ~735,000 lb -> ~333,000 kg
        mass_kg = slug_to_kg(735000.0 / 32.174)  # lbf / g_0 in ft/s^2 -> slugs
        assert 330_000 < mass_kg < 336_000

    def test_747_wing_area(self):
        # 747-100 reference wing area 5500 ft^2 -> ~511 m^2
        assert ft2_to_m2(5500.0) == pytest.approx(511.0, rel=0.01)

    def test_747_wingspan(self):
        # 747-100 span 195.7 ft -> ~59.6 m
        assert ft_to_m(195.7) == pytest.approx(59.6, rel=0.01)

    def test_cruise_speed_is_plausible(self):
        # 747 cruise ~ 871 ft/s at altitude -> ~265 m/s
        assert ft_s_to_m_s(871.0) == pytest.approx(265.5, rel=0.01)
