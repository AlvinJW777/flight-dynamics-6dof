"""Sanity checks on the CR-2144 Boeing 747 transcription.

These values were read from scanned page images, so the realistic failure mode is
a mistyped digit or a lost minus sign — and a bad derivative produces a model that
trims perfectly and has entirely wrong modes. Nothing about that failure is
visible in a plot.

So every check below is against **physics or an independent entry in the source**,
never against the transcription itself. Several are redundant on purpose: the
lift-equals-weight check simultaneously exercises CL, dynamic pressure, wing area
and weight, so one wrong digit in any of the four breaks it.
"""

from __future__ import annotations

import math

import pytest

from flightdyn.aircraft.b747 import (
    FC1,
    FC1_LONGITUDINAL,
    FC1_MODES,
    FC2,
    FC2_BODY,
    FC2_DIMENSIONAL_FT,
    FC2_LATERAL,
    FC2_LONGITUDINAL,
    FC2_MODES,
    GEOMETRY,
)
from flightdyn.atmosphere import isa
from flightdyn.units import ft2_to_m2, ft_to_m

G = 9.80665


class TestGeometry:
    def test_wing_area(self):
        # 5500 sq ft -> 511 m^2, published 747-100 figure
        assert GEOMETRY.wing_area_m2 == pytest.approx(511.0, rel=0.01)

    def test_span(self):
        assert GEOMETRY.span_m == pytest.approx(59.64, rel=0.01)

    def test_aspect_ratio_is_plausible_for_a_wide_body(self):
        ar = GEOMETRY.span_m**2 / GEOMETRY.wing_area_m2
        assert 6.5 < ar < 7.5, f"aspect ratio {ar:.2f} — 747 is about 7.0"

    def test_mean_chord_is_consistent_with_area_and_span(self):
        """S/b is the *geometric* mean chord; the MGC differs on a tapered swept
        wing, but not by more than about 15%. A transposed digit would."""
        implied = GEOMETRY.wing_area_m2 / GEOMETRY.span_m
        assert abs(GEOMETRY.mean_chord_m - implied) / implied < 0.15


class TestMassProperties:
    def test_mass_matches_the_quoted_weight(self):
        # 564,032 lb -> about 255,800 kg
        assert FC2.mass_kg == pytest.approx(255_800.0, rel=0.01)

    def test_mass_is_below_maximum_takeoff_weight(self):
        assert FC2.mass_kg < 333_400.0  # 747-100 MTOW

    def test_inertia_ordering(self):
        """Izz > Iyy > Ixx for a conventional aircraft: the span-wise mass
        distribution makes yaw the hardest axis to rotate, roll the easiest."""
        b = FC2_BODY
        assert b.Izz > b.Iyy > b.Ixx

    def test_product_of_inertia_is_small_relative_to_the_moments(self):
        """Ixz is non-zero but should be a couple of percent of Ixx, not
        comparable to it."""
        assert 0.0 < FC2_BODY.Ixz / FC2_BODY.Ixx < 0.10

    def test_inertias_satisfy_the_triangle_inequality(self):
        b = FC2_BODY
        assert b.Ixx + b.Iyy >= b.Izz


class TestFlightConditionSelfConsistency:
    """CR-2144 quotes Mach, true airspeed and dynamic pressure independently.
    They must agree with each other through the atmosphere model — three separate
    table entries reconciled by physics none of them contains."""

    def test_mach_matches_airspeed_and_altitude(self):
        a = isa(FC2.altitude_m).speed_of_sound_m_s
        assert FC2.true_airspeed_m_s / a == pytest.approx(FC2.mach, rel=0.005)

    def test_dynamic_pressure_matches_half_rho_v_squared(self):
        rho = isa(FC2.altitude_m).density_kg_m3
        q = 0.5 * rho * FC2.true_airspeed_m_s**2
        assert q == pytest.approx(FC2.dynamic_pressure_pa, rel=0.01)

    def test_fc1_mach_also_reconciles(self):
        a = isa(FC1.altitude_m).speed_of_sound_m_s
        assert FC1.true_airspeed_m_s / a == pytest.approx(FC1.mach, rel=0.01)

    def test_trim_angle_of_attack_is_plausible(self):
        """Power approach with flaps down: a few degrees, positive, well short of
        stall."""
        assert 0.0 < math.degrees(FC2.alpha_trim_rad) < 12.0


class TestLiftEqualsWeight:
    """The strongest single check available. In steady level flight L = W, and
    that one equation ties together CL, dynamic pressure, wing area and mass —
    four values from three different tables. A wrong digit in any breaks it."""

    def test_fc2_lift_balances_weight(self):
        lift = FC2_LONGITUDINAL.CL * FC2.dynamic_pressure_pa * GEOMETRY.wing_area_m2
        weight = FC2.mass_kg * G
        assert lift / weight == pytest.approx(1.0, rel=0.03), (
            f"lift {lift:,.0f} N vs weight {weight:,.0f} N — "
            "check CL, q, S and mass against CR-2144"
        )

    def test_fc1_lift_balances_weight(self):
        lift = FC1_LONGITUDINAL.CL * FC1.dynamic_pressure_pa * GEOMETRY.wing_area_m2
        weight = FC1.mass_kg * G
        assert lift / weight == pytest.approx(1.0, rel=0.03)

    def test_the_slower_condition_needs_more_lift_coefficient(self):
        """FC1 is slower than FC2 at the same weight, so it must fly at a higher
        CL. An internal consistency check across two tables."""
        assert FC1_LONGITUDINAL.CL > FC2_LONGITUDINAL.CL
        assert FC1.true_airspeed_m_s < FC2.true_airspeed_m_s


class TestSignConventions:
    """Every one of these signs is forced by physics. A lost minus in
    transcription is the single most likely error, and these catch it."""

    def test_pitch_stiffness_is_negative(self):
        """Cm_alpha < 0 is *the* condition for longitudinal static stability:
        pitch up, get a nose-down moment back."""
        assert FC2_LONGITUDINAL.Cm_alpha < 0
        assert FC1_LONGITUDINAL.Cm_alpha < 0

    def test_pitch_damping_is_negative(self):
        assert FC2_LONGITUDINAL.Cm_q < 0

    def test_elevator_produces_nose_down_moment_for_positive_deflection(self):
        assert FC2_LONGITUDINAL.Cm_de < 0

    def test_lift_slope_is_positive_and_physically_sized(self):
        """Thin aerofoil theory gives 2*pi per radian for a 2D section; a finite
        swept wing is lower, flaps raise it. Anything outside 3-7 is wrong."""
        assert 3.0 < FC2_LONGITUDINAL.CL_alpha < 7.0

    def test_drag_rises_with_incidence(self):
        assert FC2_LONGITUDINAL.CD_alpha > 0

    def test_sideforce_opposes_sideslip(self):
        assert FC2_LATERAL.Cy_beta < 0

    def test_dihedral_effect_is_negative(self):
        """Cl_beta < 0: sideslip right rolls left. This is what makes the dutch
        roll oscillatory rather than a pure yaw."""
        assert FC2_LATERAL.Cl_beta < 0

    def test_weathercock_stability_is_positive(self):
        """Cn_beta > 0: sideslip right yaws right, into the wind."""
        assert FC2_LATERAL.Cn_beta > 0

    def test_roll_damping_is_negative(self):
        assert FC2_LATERAL.Cl_p < 0

    def test_yaw_damping_is_negative(self):
        """Flagged in the source module: the minus sign was faint in the scan."""
        assert FC2_LATERAL.Cn_r < 0

    def test_aileron_rolls_the_aircraft(self):
        assert FC2_LATERAL.Cl_da != 0

    def test_adverse_yaw_from_aileron(self):
        """Cn_da for a conventional aircraft is small; its sign varies with
        configuration, so only the magnitude is checked."""
        assert abs(FC2_LATERAL.Cn_da) < abs(FC2_LATERAL.Cl_da)


class TestDimensionalDerivatives:
    """Table IX-4. Verification target, so its signs matter as much as the
    non-dimensional ones."""

    def test_speed_damping_is_negative(self):
        """Xu < 0: fly faster, get more drag."""
        assert FC2_DIMENSIONAL_FT["Xu"] < 0

    def test_heave_damping_is_negative(self):
        """Zw < 0: this is essentially -(CL_alpha + CD) * qS / (m V), the
        dominant term in short-period damping."""
        assert FC2_DIMENSIONAL_FT["Zw"] < 0

    def test_pitch_stiffness_and_damping_are_negative(self):
        assert FC2_DIMENSIONAL_FT["Mw"] < 0
        assert FC2_DIMENSIONAL_FT["Mq"] < 0

    def test_zw_is_consistent_with_the_nondimensional_derivatives(self):
        """Independent cross-check between Table IX-2 and Table IX-4:

            Zw ~ -(CL_alpha + CD) * q * S / (m * V)

        Different tables, one physical relation. Agreement to ~20% is expected
        since the exact definition carries extra terms."""
        q = FC2.dynamic_pressure_pa
        S = GEOMETRY.wing_area_m2
        m = FC2.mass_kg
        V = FC2.true_airspeed_m_s
        predicted = -(FC2_LONGITUDINAL.CL_alpha + FC2_LONGITUDINAL.CD) * q * S / (m * V)
        actual = FC2_DIMENSIONAL_FT["Zw"]  # 1/s, same in both unit systems
        assert abs(predicted - actual) / abs(actual) < 0.25, (
            f"Zw from Table IX-4 is {actual:.4f} but Table IX-2 implies {predicted:.4f}"
        )


class TestValidationTargets:
    """The published modes. Checked for physical character, since these are what
    the eigenvalues will be compared against."""

    def test_short_period_is_faster_than_phugoid(self):
        """The defining distinction: short period is a fast pitch rotation,
        phugoid a slow energy exchange. Typically an order of magnitude apart."""
        assert FC2_MODES["short_period"].omega_n_rad_s > 5 * FC2_MODES["phugoid"].omega_n_rad_s

    def test_phugoid_is_lightly_damped(self):
        assert 0.0 < FC2_MODES["phugoid"].zeta < 0.15

    def test_short_period_is_well_damped(self):
        assert 0.3 < FC2_MODES["short_period"].zeta < 1.0

    def test_both_modes_are_stable(self):
        for m in FC2_MODES.values():
            assert m.zeta > 0

    def test_phugoid_period_is_of_order_a_minute(self):
        assert 30.0 < FC2_MODES["phugoid"].period_s < 120.0

    def test_short_period_period_is_of_order_seconds(self):
        assert 2.0 < FC2_MODES["short_period"].period_s < 15.0

    def test_phugoid_frequency_matches_lanchester_within_a_factor(self):
        """Lanchester's approximation, omega_n = sqrt(2)*g/V, comes from energy
        exchange alone — no derivatives, no matrix. It is crude at low speed
        where drag matters, so a factor of two is the honest tolerance; the point
        is that an order-of-magnitude error would show."""
        lanchester = math.sqrt(2.0) * G / FC2.true_airspeed_m_s
        ratio = lanchester / FC2_MODES["phugoid"].omega_n_rad_s
        assert 0.5 < ratio < 2.0, f"Lanchester {lanchester:.4f} vs published {FC2_MODES['phugoid'].omega_n_rad_s}"

    def test_fc1_modes_are_also_physical(self):
        assert FC1_MODES["phugoid"].zeta < FC1_MODES["short_period"].zeta
        assert FC1_MODES["short_period"].omega_n_rad_s > FC1_MODES["phugoid"].omega_n_rad_s
