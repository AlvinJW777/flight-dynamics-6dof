"""Rigid-body equations of motion.

Every test here checks physics that must hold regardless of how the equations are
coded — conserved quantities, closed-form motions, and one qualitative result
(the intermediate-axis instability) that is impossible to fake. None of them
compare against a stored output, so they stay meaningful if the implementation is
rewritten.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from flightdyn.dynamics import (
    IDX_POS,
    IDX_QUAT,
    IDX_RATE,
    IDX_VEL,
    RigidBody,
    angular_momentum_ned,
    gravity_body,
    kinetic_energy,
    make_state,
    propagate,
    quaternion_drift,
    rigid_body_derivative,
    rk4_step,
)
from flightdyn.frames import quat_to_euler

NO_FORCE = np.zeros(3)

# A deliberately asymmetric body: distinct principal moments and a non-zero
# product of inertia, so roll-yaw coupling is exercised rather than accidentally
# absent.
BODY = RigidBody(mass=1000.0, Ixx=4000.0, Iyy=8000.0, Izz=11000.0, Ixz=500.0)


def free_derivative(body=BODY):
    """Torque-free, force-free motion."""
    return lambda x: rigid_body_derivative(x, NO_FORCE, NO_FORCE, body)


class TestRigidBodyValidation:
    def test_rejects_non_positive_mass(self):
        with pytest.raises(ValueError, match="mass"):
            RigidBody(mass=0.0, Ixx=1, Iyy=1, Izz=1)

    def test_rejects_impossible_inertia(self):
        """Principal moments must satisfy the triangle inequality — no rigid mass
        distribution can violate it, so accepting one hides a data-entry error."""
        with pytest.raises(ValueError, match="triangle inequality"):
            RigidBody(mass=1.0, Ixx=1.0, Iyy=1.0, Izz=10.0)

    def test_inertia_tensor_is_symmetric(self):
        I = BODY.inertia
        np.testing.assert_allclose(I, I.T, atol=1e-15)

    def test_products_of_inertia_enter_negatively(self):
        assert BODY.inertia[0, 2] == -BODY.Ixz
        assert BODY.inertia[2, 0] == -BODY.Ixz

    def test_inertia_inverse_round_trips(self):
        np.testing.assert_allclose(BODY.inertia @ BODY.inertia_inverse, np.eye(3), atol=1e-12)


class TestNewtonsFirstLaw:
    def test_no_force_no_moment_means_nothing_changes(self):
        x0 = make_state(velocity=(100.0, 0.0, 0.0))
        traj = propagate(free_derivative(), x0, dt=0.01, n_steps=5_000)
        np.testing.assert_allclose(traj[-1][IDX_VEL], [100.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(traj[-1][IDX_RATE], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(traj[-1][IDX_QUAT], [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_straight_line_travel_matches_distance_equals_speed_times_time(self):
        x0 = make_state(velocity=(100.0, 0.0, 0.0))
        traj = propagate(free_derivative(), x0, dt=0.01, n_steps=1000)  # 10 s
        np.testing.assert_allclose(traj[-1][IDX_POS], [1000.0, 0.0, 0.0], atol=1e-8)

    def test_sideways_velocity_translates_sideways(self):
        x0 = make_state(velocity=(0.0, 50.0, 0.0))
        traj = propagate(free_derivative(), x0, dt=0.01, n_steps=200)  # 2 s
        np.testing.assert_allclose(traj[-1][IDX_POS], [0.0, 100.0, 0.0], atol=1e-9)


class TestGravity:
    def test_free_fall_accelerates_at_g(self):
        g = 9.80665
        body = RigidBody(mass=500.0, Ixx=100.0, Iyy=100.0, Izz=100.0)

        def deriv(x):
            return rigid_body_derivative(x, gravity_body(x[IDX_QUAT], body.mass, g), NO_FORCE, body)

        x0 = make_state()
        t = 3.0
        traj = propagate(deriv, x0, dt=1e-3, n_steps=int(t / 1e-3))
        # w = g t  and  fallen distance = 1/2 g t^2
        assert traj[-1][IDX_VEL][2] == pytest.approx(g * t, rel=1e-10)
        assert traj[-1][IDX_POS][2] == pytest.approx(0.5 * g * t**2, rel=1e-9)

    def test_free_fall_does_not_rotate(self):
        body = RigidBody(mass=500.0, Ixx=100.0, Iyy=200.0, Izz=300.0)

        def deriv(x):
            return rigid_body_derivative(x, gravity_body(x[IDX_QUAT], body.mass), NO_FORCE, body)

        traj = propagate(deriv, make_state(), dt=1e-3, n_steps=3000)
        np.testing.assert_allclose(traj[-1][IDX_RATE], np.zeros(3), atol=1e-14)

    def test_gravity_rotates_into_body_axes(self):
        """Pitched 90° nose-up, weight acts along the body −x axis."""
        from flightdyn.frames import euler_to_quat

        q = euler_to_quat(0.0, np.pi / 2, 0.0)
        f = gravity_body(q, mass=1000.0, g=9.80665)
        np.testing.assert_allclose(f, [-9806.65, 0.0, 0.0], atol=1e-9)

    def test_inverted_flight_reverses_the_weight_vector(self):
        from flightdyn.frames import euler_to_quat

        q = euler_to_quat(np.pi, 0.0, 0.0)
        f = gravity_body(q, mass=1000.0, g=9.80665)
        np.testing.assert_allclose(f, [0.0, 0.0, -9806.65], atol=1e-9)


class TestConservation:
    """Torque-free motion. These are the sharpest tests of the ω × Iω term."""

    X0 = make_state(velocity=(80.0, 5.0, -3.0), rates=(0.3, -0.2, 0.15))

    def test_kinetic_energy_is_conserved(self):
        traj = propagate(free_derivative(), self.X0, dt=1e-3, n_steps=8_000)
        e0 = kinetic_energy(traj[0], BODY)
        drift = abs(kinetic_energy(traj[-1], BODY) - e0) / e0
        assert drift < 1e-11, f"energy drifted {drift:.3e} — suspect the omega x I omega sign"

    def test_angular_momentum_is_conserved_in_the_inertial_frame(self):
        traj = propagate(free_derivative(), self.X0, dt=1e-3, n_steps=8_000)
        h0 = angular_momentum_ned(traj[0], BODY)
        h1 = angular_momentum_ned(traj[-1], BODY)
        np.testing.assert_allclose(h1, h0, rtol=1e-9)

    def test_body_frame_angular_momentum_is_not_constant(self):
        """A negative control. |H| is conserved but its body-frame components are
        not — if they were, Euler's equation is not being applied at all."""
        traj = propagate(free_derivative(), self.X0, dt=1e-3, n_steps=5000)
        h_body_0 = BODY.inertia @ traj[0][IDX_RATE]
        h_body_1 = BODY.inertia @ traj[-1][IDX_RATE]
        assert not np.allclose(h_body_0, h_body_1, atol=1e-3)
        assert np.linalg.norm(h_body_1) == pytest.approx(np.linalg.norm(h_body_0), rel=1e-9)

    def test_quaternion_drift_stays_negligible(self):
        traj = propagate(free_derivative(), self.X0, dt=1e-3, n_steps=8_000)
        assert quaternion_drift(traj[-1]) < 1e-12

    def test_spherical_inertia_gives_constant_body_rates(self):
        """With I = kI the ω × Iω term vanishes identically, so the rates must be
        frozen. Isolates that term from everything else."""
        sphere = RigidBody(mass=100.0, Ixx=500.0, Iyy=500.0, Izz=500.0)
        x0 = make_state(rates=(0.4, -0.3, 0.2))
        traj = propagate(free_derivative(sphere), x0, dt=1e-3, n_steps=5_000)
        np.testing.assert_allclose(traj[-1][IDX_RATE], [0.4, -0.3, 0.2], atol=1e-12)


class TestIntermediateAxis:
    """The Dzhanibekov effect: torque-free rotation about the intermediate
    principal axis is unstable, about the other two it is stable.

    This is qualitative and impossible to fake — it emerges only from a correctly
    signed ω × Iω with distinct principal moments. A sign error, a dropped term,
    or an inertia tensor mistake all destroy it.
    """

    PLAIN = RigidBody(mass=10.0, Ixx=1.0, Iyy=2.0, Izz=3.0)  # no product of inertia

    # 5 s, not an arbitrary "long enough". The analytic growth rate for the
    # unstable case is sigma = w*sqrt(|(I2-I3)(I1-I2)|/(I1*I3)) = 2.887 /s, an
    # e-folding time of 0.35 s, so a 1e-4 perturbation reaches unity in
    # ln(1e4)/sigma = 3.2 s. Measured: 5 s and 60 s give identical results, and
    # 5 s is 11x faster.
    def _max_off_axis_rate(self, spin_axis: int, seconds: float = 5.0) -> float:
        rates = [1e-4, 1e-4, 1e-4]
        rates[spin_axis] = 5.0
        traj = propagate(free_derivative(self.PLAIN), make_state(rates=rates),
                         dt=1e-3, n_steps=int(seconds / 1e-3))
        others = [i for i in range(3) if i != spin_axis]
        return float(np.max(np.abs(traj[:, IDX_RATE][:, others])))

    def test_minor_axis_spin_is_stable(self):
        assert self._max_off_axis_rate(0) < 1e-2

    def test_major_axis_spin_is_stable(self):
        assert self._max_off_axis_rate(2) < 1e-2

    def test_intermediate_axis_spin_is_unstable(self):
        """The perturbation grows by orders of magnitude and the body tumbles."""
        assert self._max_off_axis_rate(1) > 1.0


class TestIntegrator:
    def test_rk4_is_fourth_order(self):
        """Halving the step must cut global error ~16x.

        **Step sizes are chosen deliberately coarse.** Error is measured against a
        very fine reference, and that reference has an error of its own — about
        1.2e-14 here, measured by comparing 32k against 64k steps. Once the
        truncation error falls to that level the ratio stops measuring convergence
        and starts measuring floating-point noise: at n = 800 the observed ratio
        collapses to 4.3, then 0.8, purely because the signal has gone under the
        floor.

        So the range n = 25..200 is not arbitrary. It keeps the error between
        1.4e-9 and 3.6e-13, comfortably above the floor, where the measurement
        means something. This is the same trade-off that governs the perturbation
        size when linearising numerically — truncation falls, round-off rises, and
        the useful window sits between them.
        """
        x0 = make_state(velocity=(50.0, 0.0, 0.0), rates=(0.5, 0.3, -0.2))
        t_end = 2.0
        deriv = free_derivative()

        # 8000 steps, not 64000. Counterintuitively the cheaper reference is the
        # *more* accurate one: at this step size truncation is already below
        # round-off, so extra steps only accumulate more rounding. Measured
        # floors: 3.9e-15 at n=8000 versus 1.2e-14 at n=64000, and 8x faster.
        reference = propagate(deriv, x0, dt=t_end / 8000, n_steps=8000)[-1]

        # Establish the floor so the assertion below is justified, not assumed.
        coarse_ref = propagate(deriv, x0, dt=t_end / 4000, n_steps=4000)[-1]
        floor = np.linalg.norm(coarse_ref[IDX_RATE] - reference[IDX_RATE])

        errors = []
        for n in (25, 50, 100, 200):
            end = propagate(deriv, x0, dt=t_end / n, n_steps=n)[-1]
            errors.append(np.linalg.norm(end[IDX_RATE] - reference[IDX_RATE]))

        assert min(errors) > 10 * floor, (
            f"smallest error {min(errors):.2e} is not clear of the reference's own "
            f"error {floor:.2e} — the convergence ratio would be measuring noise"
        )

        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        for r in ratios:
            assert 14.0 < r < 18.0, f"expected ~16x per halving, got {ratios}"

    def test_matches_scipy_solve_ivp(self):
        """External reference, same role scipy played for the rotations."""
        x0 = make_state(velocity=(90.0, 4.0, -2.0), rates=(0.2, -0.15, 0.1))
        deriv = free_derivative()
        t_end = 2.0

        ours = propagate(deriv, x0, dt=5e-4, n_steps=int(t_end / 5e-4))[-1]
        sol = solve_ivp(lambda t, x: deriv(x), (0.0, t_end), x0,
                        method="DOP853", rtol=1e-12, atol=1e-12)
        theirs = sol.y[:, -1]

        np.testing.assert_allclose(ours[IDX_VEL], theirs[IDX_VEL], rtol=1e-7)
        np.testing.assert_allclose(ours[IDX_RATE], theirs[IDX_RATE], rtol=1e-7)
        np.testing.assert_allclose(ours[IDX_POS], theirs[IDX_POS], rtol=1e-6, atol=1e-6)

    def test_zero_step_is_a_no_op(self):
        x0 = make_state(velocity=(10.0, 1.0, 2.0), rates=(0.1, 0.2, 0.3))
        np.testing.assert_allclose(rk4_step(free_derivative(), x0, 0.0), x0, atol=1e-15)


class TestCoupling:
    def test_pure_yaw_rate_with_forward_speed_produces_sideways_acceleration(self):
        """The transport term in action: turning while moving forward pushes the
        body sideways. Directly probes the rv − qw block."""
        x0 = make_state(velocity=(100.0, 0.0, 0.0), rates=(0.0, 0.0, 0.5))
        dx = rigid_body_derivative(x0, NO_FORCE, NO_FORCE, BODY)
        # v̇ = pw − ru = −0.5 × 100
        assert dx[IDX_VEL][1] == pytest.approx(-50.0, rel=1e-12)
        assert dx[IDX_VEL][0] == pytest.approx(0.0, abs=1e-12)

    def test_pure_pitch_rate_with_forward_speed_produces_vertical_acceleration(self):
        x0 = make_state(velocity=(100.0, 0.0, 0.0), rates=(0.0, 0.4, 0.0))
        dx = rigid_body_derivative(x0, NO_FORCE, NO_FORCE, BODY)
        # ẇ = qu − pv = 0.4 × 100
        assert dx[IDX_VEL][2] == pytest.approx(40.0, rel=1e-12)

    def test_a_yaw_rate_yaws_the_aircraft(self):
        x0 = make_state(velocity=(100.0, 0.0, 0.0), rates=(0.0, 0.0, 0.2))
        traj = propagate(free_derivative(RigidBody(1000.0, 5000.0, 5000.0, 5000.0)),
                         x0, dt=1e-3, n_steps=5000)  # 5 s
        _, _, psi = quat_to_euler(traj[-1][IDX_QUAT])
        assert psi == pytest.approx(1.0, abs=1e-6)  # 0.2 rad/s x 5 s
