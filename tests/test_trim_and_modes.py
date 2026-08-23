"""Trim, linearisation and modal analysis against NASA CR-2144.

The two kinds of evidence are kept apart on purpose:

* **Verification** — the numerically linearised state matrix must reproduce the
  dimensional derivatives of Table IX-4. Same aircraft, different table, and it
  answers "did I implement the maths correctly?"
* **Validation** — the eigenvalues must match the transfer-function factors of
  Table IX-5. It answers "does the model describe a real aircraft?"

Having both separates two failures that otherwise look identical. That is not
theoretical here: the first run agreed on every w and q derivative to under 3%
while every u derivative was 14-60% out, which located a missing set of Mach
derivatives that the mode comparison alone would have shrugged off.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flightdyn.aerodynamics import AeroModel, Controls, Propulsion
from flightdyn.aircraft.b747 import (
    FC2,
    FC2_BODY,
    FC2_DIMENSIONAL_FT,
    FC2_LATERAL,
    FC2_LONGITUDINAL,
    FC2_MODES,
    GEOMETRY,
)
from flightdyn.analysis.linear import (
    LATERAL,
    LONGITUDINAL,
    coupling_ratio,
    euler_derivative,
    jacobian,
    lateral_modes,
    longitudinal_modes,
    perturbation_sweep,
    quat_state_to_euler,
    submatrix,
)
from flightdyn.atmosphere import isa
from flightdyn.dynamics import IDX_POS, IDX_RATE, IDX_VEL, propagate
from flightdyn.frames import airdata, quat_to_euler
from flightdyn.trim import trim_straight_level, trimmed_derivative
from flightdyn.units import FOOT_TO_METRE, ft_to_m, lbf_to_n

# 747-100: four engines at roughly 46,000 lbf each.
PROPULSION = Propulsion(
    max_thrust_n=lbf_to_n(4 * 46000.0),
    incidence_rad=math.radians(2.5),
    moment_arm_m=ft_to_m(10.0),
)


@pytest.fixture(scope="module")
def model():
    return AeroModel(
        GEOMETRY, FC2_LONGITUDINAL, FC2_LATERAL, FC2.alpha_trim_rad, PROPULSION,
        mach_trim=FC2.mach,
        speed_of_sound_m_s=isa(FC2.altitude_m).speed_of_sound_m_s,
    )


@pytest.fixture(scope="module")
def trimmed(model):
    return trim_straight_level(model, FC2_BODY, FC2.true_airspeed_m_s, FC2.altitude_m)


@pytest.fixture(scope="module")
def A(model, trimmed):
    d = trimmed_derivative(model, FC2_BODY, FC2.altitude_m, trimmed.controls)
    return jacobian(euler_derivative(d), quat_state_to_euler(trimmed.state))


class TestTrim:
    def test_converges(self, trimmed):
        assert trimmed.converged, trimmed.message

    def test_residuals_are_at_machine_precision(self, trimmed):
        """Not "small" — converged. A trim that is merely nearly an equilibrium
        gives a linearisation about a point that does not exist."""
        assert trimmed.max_residual < 1e-10

    def test_alpha_matches_the_published_trim(self, trimmed):
        """CR-2144 Table IX-3 gives 5.70 deg for this condition, arrived at
        independently of our solve."""
        assert math.degrees(trimmed.alpha_rad) == pytest.approx(5.70, abs=0.6)

    def test_theta_equals_alpha_for_level_flight(self, trimmed):
        """gamma = theta - alpha = 0 is the constraint that closes the problem."""
        assert trimmed.theta_rad == pytest.approx(trimmed.alpha_rad, abs=1e-12)

    def test_throttle_and_elevator_are_physically_sensible(self, trimmed):
        assert 0.0 < trimmed.controls.throttle < 1.0
        assert abs(math.degrees(trimmed.controls.elevator)) < 15.0

    def test_lift_balances_weight_at_the_solution(self, model, trimmed):
        density = isa(FC2.altitude_m).density_kg_m3
        f, _ = model.forces_moments(
            trimmed.state[IDX_VEL], trimmed.state[IDX_RATE], trimmed.controls, density
        )
        # Rotate the body-axis force back to NED; vertical component must cancel weight.
        from flightdyn.frames import body_to_ned

        f_ned = body_to_ned(trimmed.state[6:10], f)
        assert f_ned[2] == pytest.approx(-FC2_BODY.mass * 9.80665, rel=0.02)


class TestTrimHolds:
    """The real proof. A trim that will not hold is not a trim."""

    def test_state_is_unchanged_after_sixty_seconds(self, model, trimmed):
        d = trimmed_derivative(model, FC2_BODY, FC2.altitude_m, trimmed.controls)
        traj = propagate(d, trimmed.state, dt=0.01, n_steps=6000)
        v0, a0, _ = airdata(traj[0][IDX_VEL])
        v1, a1, _ = airdata(traj[-1][IDX_VEL])
        assert abs(v1 - v0) < 1e-6, f"speed drifted {v1 - v0:+.3e} m/s"
        assert abs(a1 - a0) < 1e-9
        assert abs(traj[-1][IDX_POS][2] - traj[0][IDX_POS][2]) < 1e-3

    def test_it_flies_level_and_forward(self, model, trimmed):
        """Positive control: it must actually be flying, not merely static."""
        d = trimmed_derivative(model, FC2_BODY, FC2.altitude_m, trimmed.controls)
        traj = propagate(d, trimmed.state, dt=0.01, n_steps=1000)  # 10 s
        assert traj[-1][IDX_POS][0] == pytest.approx(FC2.true_airspeed_m_s * 10.0, rel=1e-6)
        assert abs(traj[-1][IDX_POS][1]) < 1e-6

    def test_an_untrimmed_state_does_drift(self, model, trimmed):
        """Negative control. Without this, the hold test could pass on a model
        that simply never moves."""
        d = trimmed_derivative(model, FC2_BODY, FC2.altitude_m, trimmed.controls)
        bad = trimmed.state.copy()
        bad[IDX_VEL] = bad[IDX_VEL] * 1.10  # 10% fast
        traj = propagate(d, bad, dt=0.01, n_steps=3000)
        v0, _, _ = airdata(traj[0][IDX_VEL])
        v1, _, _ = airdata(traj[-1][IDX_VEL])
        assert abs(v1 - v0) > 1.0, "an untrimmed state should not hold"


class TestVerificationAgainstTableIX4:
    """Element-by-element against the published dimensional derivatives."""

    TOL = 0.05  # 5%

    def _lon(self, A):
        return submatrix(A, LONGITUDINAL)  # u, w, q, theta

    @pytest.mark.parametrize(
        "name, row, col, si_factor",
        [
            ("Xu", 0, 0, 1.0),
            ("Xw", 0, 1, 1.0),
            ("Zu", 1, 0, 1.0),
            ("Zw", 1, 1, 1.0),
            ("Mu", 2, 0, 1.0 / FOOT_TO_METRE),
            ("Mw", 2, 1, 1.0 / FOOT_TO_METRE),
            ("Mq", 2, 2, 1.0),
        ],
    )
    def test_derivative_matches(self, A, name, row, col, si_factor):
        mine = self._lon(A)[row, col]
        theirs = FC2_DIMENSIONAL_FT[name] * si_factor
        assert mine == pytest.approx(theirs, rel=self.TOL), (
            f"{name}: numerical {mine:.6f} vs CR-2144 {theirs:.6f}"
        )

    def test_theta_row_is_exactly_the_kinematic_identity(self, A):
        """dtheta/dq = 1 exactly. Not a modelling result — a definition."""
        assert self._lon(A)[3, 2] == pytest.approx(1.0, abs=1e-9)

    def test_longitudinal_and_lateral_decouple(self, A):
        """Symmetric aircraft in symmetric flight. A non-zero value would mean an
        asymmetry that should not exist."""
        assert coupling_ratio(A) < 1e-6


class TestValidationAgainstTableIX5:
    """Eigenvalues against the published transfer-function denominators."""

    def test_short_period_frequency(self, A):
        got = longitudinal_modes(submatrix(A, LONGITUDINAL))["short_period"]
        assert got.omega_n == pytest.approx(FC2_MODES["short_period"].omega_n_rad_s, rel=0.05)

    def test_short_period_damping(self, A):
        got = longitudinal_modes(submatrix(A, LONGITUDINAL))["short_period"]
        assert got.zeta == pytest.approx(FC2_MODES["short_period"].zeta, rel=0.12)

    def test_phugoid_frequency(self, A):
        got = longitudinal_modes(submatrix(A, LONGITUDINAL))["phugoid"]
        assert got.omega_n == pytest.approx(FC2_MODES["phugoid"].omega_n_rad_s, rel=0.05)

    def test_phugoid_damping_agrees_only_loosely(self, A):
        """Deliberately the loosest tolerance in the suite, and honest about why.

        Phugoid damping is roughly CD/(sqrt(2)*CL) — set almost entirely by drag.
        Our CD carries no Mach term (CR-2144 does not tabulate CD_M here) and
        thrust is speed-independent, so a few percent on drag moves this a lot.
        The absolute error is about 0.006 on a quantity of 0.023.
        """
        got = longitudinal_modes(submatrix(A, LONGITUDINAL))["phugoid"]
        assert abs(got.zeta - FC2_MODES["phugoid"].zeta) < 0.015

    def test_both_longitudinal_modes_are_stable(self, A):
        for m in longitudinal_modes(submatrix(A, LONGITUDINAL)).values():
            assert m.zeta > 0

    def test_short_period_is_much_faster_than_phugoid(self, A):
        lon = longitudinal_modes(submatrix(A, LONGITUDINAL))
        assert lon["short_period"].omega_n > 5 * lon["phugoid"].omega_n


class TestLateralModes:
    """CR-2144's lateral transfer functions were not transcribed, so these check
    physical character rather than published numbers. Stated as such."""

    def test_all_three_modes_are_present(self, A):
        modes = lateral_modes(submatrix(A, LATERAL))
        assert set(modes) == {"dutch_roll", "roll_subsidence", "spiral"}

    def test_dutch_roll_is_oscillatory_and_lightly_damped(self, A):
        dr = lateral_modes(submatrix(A, LATERAL))["dutch_roll"]
        assert dr.is_oscillatory
        assert 0.02 < dr.zeta < 0.35

    def test_roll_subsidence_is_a_fast_stable_real_root(self, A):
        roll = lateral_modes(submatrix(A, LATERAL))["roll_subsidence"]
        assert not roll.is_oscillatory
        assert roll.eigenvalue.real < 0
        assert 0.2 < roll.time_constant_s < 3.0

    def test_spiral_is_slow(self, A):
        """Often slightly unstable on a real aircraft; either sign is acceptable,
        but it must be slow."""
        spiral = lateral_modes(submatrix(A, LATERAL))["spiral"]
        assert not spiral.is_oscillatory
        assert abs(spiral.eigenvalue.real) < 0.2

    def test_roll_is_the_fastest_lateral_mode(self, A):
        modes = lateral_modes(submatrix(A, LATERAL))
        assert abs(modes["roll_subsidence"].eigenvalue.real) > abs(modes["spiral"].eigenvalue.real)


class TestPerturbationSize:
    """The trade-off that governs every numerical Jacobian."""

    def test_optimum_lands_where_theory_predicts(self, model, trimmed, A):
        """Optimum should sit near eps^(1/3) times the state's own scale.

        For u that scale is about 84 m/s, so theory gives 5.1e-4. Measured on a
        decade grid the minimum falls at 1e-3 — within a factor of two, which is
        as close as a decade grid can resolve.
        """
        d = euler_derivative(trimmed_derivative(model, FC2_BODY, FC2.altitude_m, trimmed.controls))
        y0 = quat_state_to_euler(trimmed.state)
        steps, errors = perturbation_sweep(d, y0, index=0, reference_column=A[:, 0])

        best = steps[int(np.argmin(errors))]
        predicted = np.finfo(float).eps ** (1 / 3) * abs(y0[0])
        assert 0.1 < best / predicted < 10.0, (
            f"optimum {best:.1e} is far from the predicted {predicted:.1e}"
        )

    def test_error_rises_at_both_extremes_for_a_nonlinear_state(self, model, trimmed, A):
        """The V-curve. Round-off dominates at small h, truncation at large h.

        Tested on the u column specifically, because the dynamics are genuinely
        nonlinear in u — dynamic pressure goes as V^2 and the Mach terms add more.
        """
        d = euler_derivative(trimmed_derivative(model, FC2_BODY, FC2.altitude_m, trimmed.controls))
        y0 = quat_state_to_euler(trimmed.state)
        steps, errors = perturbation_sweep(d, y0, index=0, reference_column=A[:, 0])
        i = int(np.argmin(errors))
        assert errors[0] / errors[i] > 1e6, "round-off should dominate at the smallest step"
        assert errors[-1] / errors[i] > 100, "truncation should dominate at the largest step"

    def test_a_nearly_linear_state_shows_no_truncation_branch(self, model, trimmed, A):
        """Contrast case, and the reason the V is not universal.

        The dynamics are almost exactly linear in the pitch rate q, so there is
        essentially no truncation error to grow at large h. The curve therefore
        falls and then flattens rather than turning back up. Expecting a V here
        and "fixing" the code when it does not appear would be chasing a
        phantom — the shape depends on the nonlinearity of the function, not on
        the quality of the implementation.
        """
        d = euler_derivative(trimmed_derivative(model, FC2_BODY, FC2.altitude_m, trimmed.controls))
        y0 = quat_state_to_euler(trimmed.state)
        steps, errors = perturbation_sweep(d, y0, index=4, reference_column=A[:, 4])
        i = int(np.argmin(errors))
        assert errors[0] / errors[i] > 1e6, "round-off still dominates at small h"
        assert errors[-1] / errors[i] < 10.0, "no significant truncation branch expected"
