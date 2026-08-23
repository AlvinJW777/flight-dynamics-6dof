"""Numerical linearisation and modal analysis.

WHY EULER COORDINATES HERE
--------------------------
The simulator carries a quaternion because it must not break at vertical pitch.
But a quaternion has four components under one norm constraint, so linearising in
it yields a redundant direction and a spurious zero eigenvalue that has to be
identified and discarded.

For linearisation we are sitting at a trim point far from any singularity, so the
Euler chart is perfectly well conditioned and gives exactly nine dynamic states.
**The nonlinear physics is unchanged** — this is a change of coordinates for
taking a derivative, not a change of model.

CHOOSING THE PERTURBATION SIZE
------------------------------
Central differences carry truncation error O(h^2) and round-off error O(eps/h).
Minimising their sum puts the optimum near

    h ~ eps^(1/3) ~ 6e-6   times the natural scale of the state

Too large and you capture the nonlinearity you are trying to linearise away; too
small and floating-point cancellation dominates. Perturbations are therefore
scaled per state - a step appropriate for a velocity in m/s is meaningless for an
angle in radians. :func:`perturbation_sweep` demonstrates the trade-off rather
than asserting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..frames import euler_to_quat, quat_to_euler
from ..dynamics import IDX_POS, IDX_QUAT, IDX_RATE, IDX_VEL, N_STATES

#: Euler-chart state ordering used for linearisation.
EULER_STATE_NAMES = ("u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "pn", "pe", "pd")
N_EULER = 12

#: Classical decoupled subsets, as indices into the Euler state.
LONGITUDINAL = (0, 2, 4, 7)   # u, w, q, theta
LATERAL = (1, 3, 5, 6)        # v, p, r, phi

#: Natural scale of each state, for sizing perturbations.
STATE_SCALE = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.0, 1.0, 1.0])

CUBE_ROOT_EPS = np.finfo(float).eps ** (1.0 / 3.0)


def quat_state_to_euler(x: np.ndarray) -> np.ndarray:
    """13-state (quaternion) -> 12-state (Euler)."""
    phi, theta, psi = quat_to_euler(x[IDX_QUAT])
    out = np.empty(N_EULER)
    out[0:3] = x[IDX_VEL]
    out[3:6] = x[IDX_RATE]
    out[6:9] = (phi, theta, psi)
    out[9:12] = x[IDX_POS]
    return out


def euler_state_to_quat(y: np.ndarray) -> np.ndarray:
    """12-state (Euler) -> 13-state (quaternion)."""
    out = np.empty(N_STATES)
    out[IDX_VEL] = y[0:3]
    out[IDX_RATE] = y[3:6]
    out[IDX_QUAT] = euler_to_quat(y[6], y[7], y[8])
    out[IDX_POS] = y[9:12]
    return out


def euler_derivative(quat_derivative):
    """Wrap a 13-state derivative so it acts on the 12-state Euler chart.

    Euler rates come from the body rates through the standard kinematic relation

        phi_dot   = p + (q sin phi + r cos phi) tan theta
        theta_dot = q cos phi - r sin phi
        psi_dot   = (q sin phi + r cos phi) / cos theta

    which is exactly the relation that blows up at theta = +/-90 degrees. It is
    used here only because trim is far from that, and never for integration.
    """

    def derivative(y: np.ndarray) -> np.ndarray:
        x = euler_state_to_quat(y)
        dx = quat_derivative(x)

        phi, theta = y[6], y[7]
        p, q, r = y[3:6]
        if abs(math.cos(theta)) < 1e-8:
            raise ValueError(
                f"theta = {math.degrees(theta):.1f} deg is at the Euler singularity; "
                "linearise closer to level flight"
            )
        s, c, t = math.sin(phi), math.cos(phi), math.tan(theta)
        common = q * s + r * c

        out = np.empty(N_EULER)
        out[0:3] = dx[IDX_VEL]
        out[3:6] = dx[IDX_RATE]
        out[6] = p + common * t
        out[7] = q * c - r * s
        out[8] = common / math.cos(theta)
        out[9:12] = dx[IDX_POS]
        return out

    return derivative


def jacobian(derivative, y0: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Central-difference Jacobian of a vector function.

    Each column is perturbed independently at a step sized to that state's own
    magnitude, because one absolute step cannot suit both a velocity in m/s and
    an angle in radians.
    """
    scale = STATE_SCALE if scale is None else scale
    n = y0.size
    A = np.empty((n, n))
    for j in range(n):
        h = CUBE_ROOT_EPS * max(abs(y0[j]), scale[j])
        yp, ym = y0.copy(), y0.copy()
        yp[j] += h
        ym[j] -= h
        A[:, j] = (derivative(yp) - derivative(ym)) / (2.0 * h)
    return A


def perturbation_sweep(derivative, y0: np.ndarray, index: int, reference_column: np.ndarray):
    """Error in one Jacobian column against perturbation size.

    Produces the V-shaped curve that justifies the step-size choice: truncation
    error falling as h^2 on one side, round-off rising as 1/h on the other, with a
    minimum near eps^(1/3). Demonstrating this is worth more than asserting it.
    """
    steps, errors = [], []
    for exponent in range(-14, 0):
        h = 10.0**exponent
        yp, ym = y0.copy(), y0.copy()
        yp[index] += h
        ym[index] -= h
        col = (derivative(yp) - derivative(ym)) / (2.0 * h)
        steps.append(h)
        errors.append(float(np.linalg.norm(col - reference_column)))
    return np.array(steps), np.array(errors)


@dataclass(frozen=True)
class ModeEstimate:
    """One eigenvalue interpreted as a physical mode."""

    name: str
    eigenvalue: complex

    @property
    def is_oscillatory(self) -> bool:
        return abs(self.eigenvalue.imag) > 1e-9

    @property
    def omega_n(self) -> float:
        return float(abs(self.eigenvalue))

    @property
    def zeta(self) -> float:
        """Damping ratio. Positive means stable; negative means divergent."""
        w = self.omega_n
        return float(-self.eigenvalue.real / w) if w > 1e-12 else 0.0

    @property
    def period_s(self) -> float:
        wd = abs(self.eigenvalue.imag)
        return float("inf") if wd < 1e-12 else 2.0 * math.pi / wd

    @property
    def time_constant_s(self) -> float:
        """For a real root: how long to decay by 1/e. Negative means divergent."""
        return float("inf") if abs(self.eigenvalue.real) < 1e-12 else float(-1.0 / self.eigenvalue.real)

    def __str__(self) -> str:
        if self.is_oscillatory:
            return (f"{self.name:<16} zeta={self.zeta:+.4f}  omega_n={self.omega_n:.4f} rad/s  "
                    f"T={self.period_s:.2f} s")
        return f"{self.name:<16} real root {self.eigenvalue.real:+.5f} /s  tau={self.time_constant_s:.2f} s"


def longitudinal_modes(A_lon: np.ndarray) -> dict[str, ModeEstimate]:
    """Identify short period and phugoid from a 4x4 longitudinal state matrix.

    Both are complex pairs for a conventional aircraft. They are told apart by
    frequency: the short period is a fast pitch rotation at roughly constant
    speed, the phugoid a slow exchange of kinetic and potential energy, and they
    typically differ by an order of magnitude.
    """
    eigs = np.linalg.eigvals(A_lon)
    pairs = sorted({complex(round(e.real, 12), round(abs(e.imag), 12)) for e in eigs},
                   key=abs, reverse=True)
    out: dict[str, ModeEstimate] = {}
    if len(pairs) >= 1:
        out["short_period"] = ModeEstimate("short period", pairs[0])
    if len(pairs) >= 2:
        out["phugoid"] = ModeEstimate("phugoid", pairs[1])
    return out


def lateral_modes(A_lat: np.ndarray) -> dict[str, ModeEstimate]:
    """Identify dutch roll, roll subsidence and spiral from a 4x4 lateral matrix.

    Dutch roll is the complex pair. Of the two real roots, the fast one is roll
    subsidence and the slow one is the spiral - which is frequently slightly
    unstable, and that is normal rather than a bug.
    """
    eigs = np.linalg.eigvals(A_lat)
    complex_roots = [e for e in eigs if abs(e.imag) > 1e-9]
    real_roots = sorted((e.real for e in eigs if abs(e.imag) <= 1e-9), key=abs, reverse=True)

    out: dict[str, ModeEstimate] = {}
    if complex_roots:
        out["dutch_roll"] = ModeEstimate("dutch roll", max(complex_roots, key=lambda e: e.imag))
    if len(real_roots) >= 1:
        out["roll_subsidence"] = ModeEstimate("roll subsidence", complex(real_roots[0], 0.0))
    if len(real_roots) >= 2:
        out["spiral"] = ModeEstimate("spiral", complex(real_roots[-1], 0.0))
    return out


def submatrix(A: np.ndarray, indices) -> np.ndarray:
    """Extract a decoupled subsystem by index."""
    idx = np.array(indices)
    return A[np.ix_(idx, idx)]


def coupling_ratio(A: np.ndarray) -> float:
    """How nearly the longitudinal and lateral dynamics decouple.

    For a symmetric aircraft in symmetric flight they separate exactly in theory.
    The residual measures how well the implementation respects that symmetry, so
    a large value points at an asymmetry that should not be there.
    """
    lon, lat = np.array(LONGITUDINAL), np.array(LATERAL)
    cross = np.concatenate([A[np.ix_(lon, lat)].ravel(), A[np.ix_(lat, lon)].ravel()])
    block = np.concatenate([A[np.ix_(lon, lon)].ravel(), A[np.ix_(lat, lat)].ravel()])
    return float(np.max(np.abs(cross)) / np.max(np.abs(block)))
