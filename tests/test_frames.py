"""Attitude representation and rotational kinematics.

Three kinds of check here, in increasing strength:

1. **Structural** — orthogonality, determinant, norm. Properties any valid
   rotation must have, independent of which rotation it is.
2. **Known geometry** — hand-computable cases where the answer is obvious if you
   picture the aircraft. These catch sign and transpose errors that structural
   tests sail straight past, because a transposed rotation is still orthogonal.
3. **External reference** — ``scipy.spatial.transform``, an independently written
   and widely validated implementation. Same role CR-2144 plays for the dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flightdyn.frames import (
    airdata,
    body_to_ned,
    dcm_to_quat,
    euler_to_quat,
    ned_to_body,
    normalise,
    quat_conjugate,
    quat_kinematics,
    quat_multiply,
    quat_to_dcm,
    quat_to_euler,
)

IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


def random_quats(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield normalise(rng.normal(size=4))


class TestStructural:
    """Properties every rotation matrix must satisfy."""

    def test_dcm_is_orthogonal(self):
        for q in random_quats(200):
            C = quat_to_dcm(q)
            np.testing.assert_allclose(C @ C.T, np.eye(3), atol=1e-13)

    def test_determinant_is_plus_one(self):
        """+1 is a rotation. −1 would be a reflection — a mirrored aircraft, which
        still passes the orthogonality test."""
        for q in random_quats(200):
            assert np.linalg.det(quat_to_dcm(q)) == pytest.approx(1.0, abs=1e-13)

    def test_rotation_preserves_length(self):
        rng = np.random.default_rng(3)
        for q in random_quats(50, seed=4):
            v = rng.normal(size=3)
            assert np.linalg.norm(ned_to_body(q, v)) == pytest.approx(np.linalg.norm(v), rel=1e-13)

    def test_identity_quaternion_is_identity_matrix(self):
        np.testing.assert_allclose(quat_to_dcm(IDENTITY_Q), np.eye(3), atol=1e-15)

    def test_conjugate_undoes_rotation(self):
        rng = np.random.default_rng(5)
        for q in random_quats(50, seed=6):
            v = rng.normal(size=3)
            back = quat_to_dcm(quat_conjugate(q)) @ (quat_to_dcm(q) @ v)
            np.testing.assert_allclose(back, v, atol=1e-13)

    def test_normalise_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            normalise(np.zeros(4))


class TestRoundTrips:
    def test_quat_dcm_quat(self):
        for q in random_quats(200, seed=7):
            if q[0] < 0:
                q = -q  # canonical sign, since q and -q are the same rotation
            np.testing.assert_allclose(dcm_to_quat(quat_to_dcm(q)), q, atol=1e-12)

    def test_euler_quat_euler(self):
        rng = np.random.default_rng(8)
        for _ in range(200):
            phi = rng.uniform(-np.pi, np.pi)
            theta = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)  # away from gimbal lock
            psi = rng.uniform(-np.pi, np.pi)
            got = quat_to_euler(euler_to_quat(phi, theta, psi))
            np.testing.assert_allclose(got, (phi, theta, psi), atol=1e-11)

    def test_shepperd_handles_180_degree_rotations(self):
        """Where the naive extraction divides by sqrt(1 + trace) → 0."""
        for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
            q = np.concatenate(([np.cos(np.pi / 2)], np.sin(np.pi / 2) * axis))
            C = quat_to_dcm(q)
            np.testing.assert_allclose(quat_to_dcm(dcm_to_quat(C)), C, atol=1e-12)


class TestKnownGeometry:
    """Hand-computable cases. These catch transposes; structural tests do not."""

    def test_yaw_90_puts_north_on_the_left_wing(self):
        """Nose East. North is then out of the aircraft's left, i.e. body −y."""
        q = euler_to_quat(0.0, 0.0, np.pi / 2)
        np.testing.assert_allclose(ned_to_body(q, [1, 0, 0]), [0, -1, 0], atol=1e-14)

    def test_pitch_90_puts_north_below_the_aircraft(self):
        """Nose vertical. Body x points up, so North lies along body +z."""
        q = euler_to_quat(0.0, np.pi / 2, 0.0)
        np.testing.assert_allclose(ned_to_body(q, [1, 0, 0]), [0, 0, 1], atol=1e-14)

    def test_pitch_90_puts_down_along_the_nose_reversed(self):
        q = euler_to_quat(0.0, np.pi / 2, 0.0)
        np.testing.assert_allclose(ned_to_body(q, [0, 0, 1]), [-1, 0, 0], atol=1e-14)

    def test_roll_90_puts_down_along_the_right_wing(self):
        """Rolled right through 90°: the Earth's down direction now lies along
        body +y, out of the right wing."""
        q = euler_to_quat(np.pi / 2, 0.0, 0.0)
        np.testing.assert_allclose(ned_to_body(q, [0, 0, 1]), [0, 1, 0], atol=1e-14)

    def test_level_flight_gravity_is_purely_down_in_body(self):
        g_ned = np.array([0.0, 0.0, 9.80665])
        np.testing.assert_allclose(ned_to_body(IDENTITY_Q, g_ned), g_ned, atol=1e-15)

    def test_body_to_ned_is_the_inverse(self):
        rng = np.random.default_rng(11)
        for q in random_quats(50, seed=12):
            v = rng.normal(size=3)
            np.testing.assert_allclose(body_to_ned(q, ned_to_body(q, v)), v, atol=1e-13)


class TestAgainstScipy:
    """Independent implementation as an external reference.

    scipy uses scalar-LAST ordering, and ``as_matrix()`` returns the *active*
    rotation. Ours is scalar-first and *passive*, so the comparison needs both a
    reorder and a transpose. Stating that explicitly is the point — an unexamined
    convention mismatch is exactly the bug this test exists to catch.
    """

    @staticmethod
    def to_scipy(q):
        q0, q1, q2, q3 = q
        return Rotation.from_quat([q1, q2, q3, q0])

    def test_dcm_matches_scipy_transposed(self):
        for q in random_quats(200, seed=13):
            expected = self.to_scipy(q).as_matrix().T   # active -> passive
            np.testing.assert_allclose(quat_to_dcm(q), expected, atol=1e-13)

    def test_euler_matches_scipy(self):
        rng = np.random.default_rng(14)
        for _ in range(200):
            phi = rng.uniform(-np.pi, np.pi)
            theta = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)
            psi = rng.uniform(-np.pi, np.pi)
            q = euler_to_quat(phi, theta, psi)
            # scipy 'ZYX' extrinsic order returns (yaw, pitch, roll)
            sp_psi, sp_theta, sp_phi = self.to_scipy(q).as_euler("ZYX")
            np.testing.assert_allclose(quat_to_euler(q), (sp_phi, sp_theta, sp_psi), atol=1e-11)

    def test_multiplication_matches_scipy_composition(self):
        for a, b in zip(random_quats(50, seed=15), random_quats(50, seed=16)):
            ours = quat_to_dcm(quat_multiply(a, b))
            theirs = (self.to_scipy(a) * self.to_scipy(b)).as_matrix().T
            np.testing.assert_allclose(ours, theirs, atol=1e-13)


class TestKinematics:
    def test_omega_matrix_is_skew_symmetric(self):
        """Skew-symmetry is what conserves the norm analytically. If Ω is not
        skew, the quaternion will drift no matter how small the timestep."""
        q = IDENTITY_Q
        for omega in ([0.1, 0, 0], [0, 0.2, 0], [0, 0, -0.3], [0.1, -0.2, 0.3]):
            # reconstruct Omega column by column from the linear operator
            Omega = np.column_stack([
                2.0 * quat_kinematics(basis, omega) for basis in np.eye(4)
            ])
            np.testing.assert_allclose(Omega, -Omega.T, atol=1e-15)

    def test_zero_rate_gives_zero_derivative(self):
        np.testing.assert_allclose(quat_kinematics(IDENTITY_Q, [0, 0, 0]), np.zeros(4), atol=1e-15)

    def test_norm_is_preserved_under_integration(self):
        """RK4 a constant body rate for a full revolution and check drift."""
        q = IDENTITY_Q.copy()
        omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s yaw
        dt = 1e-3
        for _ in range(int(2 * np.pi / dt)):
            k1 = quat_kinematics(q, omega)
            k2 = quat_kinematics(q + 0.5 * dt * k1, omega)
            k3 = quat_kinematics(q + 0.5 * dt * k2, omega)
            k4 = quat_kinematics(q + dt * k3, omega)
            q = q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12

    def test_constant_yaw_rate_integrates_to_the_right_angle(self):
        """1 rad/s for 1 s must give exactly 1 radian of yaw — an analytic result."""
        q = IDENTITY_Q.copy()
        omega = np.array([0.0, 0.0, 1.0])
        dt = 1e-4
        for _ in range(int(1.0 / dt)):
            k1 = quat_kinematics(q, omega)
            k2 = quat_kinematics(q + 0.5 * dt * k1, omega)
            k3 = quat_kinematics(q + 0.5 * dt * k2, omega)
            k4 = quat_kinematics(q + dt * k3, omega)
            q = normalise(q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        _, _, psi = quat_to_euler(q)
        assert psi == pytest.approx(1.0, abs=1e-9)


class TestGimbalLock:
    def test_quaternion_survives_vertical_pitch(self):
        """The case that breaks an Euler-angle state. The quaternion is finite and
        the DCM is still a valid rotation."""
        q = euler_to_quat(0.0, np.pi / 2, 0.0)
        assert np.all(np.isfinite(q))
        C = quat_to_dcm(q)
        np.testing.assert_allclose(C @ C.T, np.eye(3), atol=1e-14)
        _, theta, _ = quat_to_euler(q)
        assert theta == pytest.approx(np.pi / 2, abs=1e-12)

    def test_euler_extraction_does_not_return_nan_at_the_singularity(self):
        """Roll and yaw are genuinely undefined here, but pitch must still be
        correct and nothing may be NaN."""
        for sign in (+1, -1):
            q = euler_to_quat(0.0, sign * np.pi / 2, 0.0)
            phi, theta, psi = quat_to_euler(q)
            assert np.isfinite([phi, theta, psi]).all()
            assert theta == pytest.approx(sign * np.pi / 2, abs=1e-9)


class TestAirdata:
    def test_straight_and_level(self):
        V, alpha, beta = airdata([100.0, 0.0, 0.0])
        assert (V, alpha, beta) == pytest.approx((100.0, 0.0, 0.0))

    def test_positive_w_gives_positive_alpha(self):
        """Nose-up relative wind: w > 0 means the flow comes from below."""
        _, alpha, _ = airdata([100.0, 0.0, 10.0])
        assert alpha == pytest.approx(np.arctan2(10.0, 100.0))
        assert alpha > 0

    def test_positive_v_gives_positive_beta(self):
        _, _, beta = airdata([100.0, 10.0, 0.0])
        assert beta > 0
        assert beta == pytest.approx(np.arcsin(10.0 / np.linalg.norm([100.0, 10.0, 0.0])))

    def test_speed_matches_the_vector_norm(self):
        v = [80.0, -6.0, 12.0]
        V, _, _ = airdata(v)
        assert V == pytest.approx(np.linalg.norm(v))

    def test_zero_velocity_does_not_divide_by_zero(self):
        assert airdata([0.0, 0.0, 0.0]) == (0.0, 0.0, 0.0)

    def test_alpha_is_correct_in_backwards_flight(self):
        """atan2 rather than atan — this returns ~180°, not 0°."""
        _, alpha, _ = airdata([-100.0, 0.0, 0.0])
        assert abs(alpha) == pytest.approx(np.pi, abs=1e-12)
