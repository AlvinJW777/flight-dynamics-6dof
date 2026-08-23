"""Generate the figures embedded in README.md.

Purpose-built: each one carries a specific claim made in the text. Regenerate with

    python scripts/make_readme_figures.py

A solid light background is set explicitly, because GitHub renders READMEs in
both themes and a transparent background leaves dark axis text invisible.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flightdyn.aerodynamics import AeroModel, Propulsion  # noqa: E402
from flightdyn.aircraft.b747 import (  # noqa: E402
    FC2,
    FC2_BODY,
    FC2_LATERAL,
    FC2_LONGITUDINAL,
    FC2_MODES,
    GEOMETRY,
)
from flightdyn.analysis.linear import (  # noqa: E402
    LATERAL,
    LONGITUDINAL,
    euler_derivative,
    jacobian,
    lateral_modes,
    longitudinal_modes,
    perturbation_sweep,
    quat_state_to_euler,
    submatrix,
)
from flightdyn.atmosphere import isa  # noqa: E402
from flightdyn.dynamics import IDX_VEL, propagate  # noqa: E402
from flightdyn.frames import quat_to_euler  # noqa: E402
from flightdyn.trim import trim_straight_level, trimmed_derivative  # noqa: E402
from flightdyn.units import ft_to_m, lbf_to_n  # noqa: E402

OUT = REPO / "figures" / "readme"
BG, FG, GRID = "#ffffff", "#1a1a1a", "#d8d8d8"
BLUE, RED, GREEN, AMBER = "#1f4e8c", "#c0392b", "#2f7d55", "#c47f17"


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=FG, labelsize=9)
    if title:
        ax.set_title(title, color=FG, fontsize=11, pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=FG, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG, fontsize=10)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote figures/readme/{name}")


def build():
    prop = Propulsion(lbf_to_n(4 * 46000.0), math.radians(2.5), ft_to_m(10.0))
    model = AeroModel(
        GEOMETRY, FC2_LONGITUDINAL, FC2_LATERAL, FC2.alpha_trim_rad, prop,
        mach_trim=FC2.mach, speed_of_sound_m_s=isa(FC2.altitude_m).speed_of_sound_m_s,
    )
    tr = trim_straight_level(model, FC2_BODY, FC2.true_airspeed_m_s, FC2.altitude_m)
    d13 = trimmed_derivative(model, FC2_BODY, FC2.altitude_m, tr.controls)
    A = jacobian(euler_derivative(d13), quat_state_to_euler(tr.state))
    return model, tr, d13, A


def figure_poles(A):
    """Computed eigenvalues against CR-2144's published modes."""
    lon = longitudinal_modes(submatrix(A, LONGITUDINAL))
    lat = lateral_modes(submatrix(A, LATERAL))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=BG)

    # --- longitudinal, with the published reference overlaid
    for key, colour in (("short_period", BLUE), ("phugoid", RED)):
        e = lon[key].eigenvalue
        ax1.plot([e.real, e.real], [e.imag, -e.imag], "o", color=colour, ms=9,
                 label=f"{key.replace('_', ' ')} (computed)")
        ref = FC2_MODES[key]
        s = -ref.zeta * ref.omega_n_rad_s
        w = ref.omega_n_rad_s * math.sqrt(max(0.0, 1 - ref.zeta**2))
        ax1.plot([s, s], [w, -w], "x", color=colour, ms=11, mew=2.2,
                 label=f"{key.replace('_', ' ')} (CR-2144)")

    ax1.axhline(0, color=FG, lw=0.8, alpha=0.4)
    ax1.axvline(0, color=FG, lw=0.8, alpha=0.4)
    ax1.margins(0.18)
    style(ax1, "Longitudinal poles vs NASA CR-2144", "real part (1/s)", "imaginary part (rad/s)")
    leg = ax1.legend(fontsize=8, framealpha=0.95, edgecolor=GRID, loc="upper left")
    for t in leg.get_texts():
        t.set_color(FG)

    # --- lateral, character only (no transcribed reference)
    for key, colour in (("dutch_roll", GREEN), ("roll_subsidence", AMBER), ("spiral", "#7a4fa3")):
        e = lat[key].eigenvalue
        ax2.plot([e.real, e.real], [e.imag, -e.imag], "o", color=colour, ms=9,
                 label=key.replace("_", " "))
    ax2.axhline(0, color=FG, lw=0.8, alpha=0.4)
    ax2.axvline(0, color=FG, lw=0.8, alpha=0.4)
    ax2.margins(0.18)
    style(ax2, "Lateral poles (character only — no reference transcribed)",
          "real part (1/s)", "imaginary part (rad/s)")
    leg = ax2.legend(fontsize=8, framealpha=0.95, edgecolor=GRID, loc="upper left")
    for t in leg.get_texts():
        t.set_color(FG)

    fig.text(0.5, -0.04,
             "Circles computed from the numerically linearised nonlinear model; crosses are "
             "CR-2144 Table IX-5. Everything is in the left half-plane, so every mode is stable.",
             ha="center", color=FG, fontsize=8.5, alpha=0.8)
    save(fig, "01_pole_map.png")


def figure_perturbation(model, tr, d13, A):
    """The V-curve, and the case where there isn't one."""
    d = euler_derivative(d13)
    y0 = quat_state_to_euler(tr.state)
    eps13 = np.finfo(float).eps ** (1 / 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG, sharey=True)
    for ax, idx, label, scale, colour in (
        (ax1, 0, "u  — nonlinear in this state", abs(y0[0]), BLUE),
        (ax2, 4, "q  — nearly linear in this state", 0.01, GREEN),
    ):
        steps, errors = perturbation_sweep(d, y0, index=idx, reference_column=A[:, idx])
        ax.loglog(steps, errors, "o-", color=colour, lw=1.6, ms=5)
        predicted = eps13 * scale
        ax.axvline(predicted, color=FG, ls="--", lw=1.2, alpha=0.7)
        ax.text(predicted * 1.4, max(errors) * 0.2, f"$\\epsilon^{{1/3}}\\times$scale\n{predicted:.1e}",
                color=FG, fontsize=8.5, alpha=0.85)
        style(ax, label, "perturbation size $h$")
    ax1.set_ylabel("error vs converged Jacobian column", color=FG, fontsize=10)

    fig.text(0.5, -0.06,
             "Left: truncation error falls as h² until round-off takes over, giving the classic V with its "
             "minimum where theory predicts. Right: the dynamics are nearly linear in q, so there is almost no "
             "truncation error to grow — the curve flattens instead of turning up.",
             ha="center", color=FG, fontsize=8.5, alpha=0.8, wrap=True)
    save(fig, "02_perturbation_size.png")


def figure_modes(model, tr, d13, A):
    """Time responses showing each mode's character."""
    lon = longitudinal_modes(submatrix(A, LONGITUDINAL))

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), facecolor=BG, sharex=False)

    # short period: pitch-rate doublet, watch alpha
    x0 = tr.state.copy()
    x0[3 + 1] = 0.03  # q perturbation
    traj = propagate(d13, x0, dt=0.01, n_steps=3000)
    t = np.arange(traj.shape[0]) * 0.01
    theta = np.array([quat_to_euler(s[6:10])[1] for s in traj])
    axes[0].plot(t, np.degrees(theta - theta[0]), color=BLUE, lw=1.5)
    style(axes[0], f"Short period — pitch rate perturbation "
                   f"($\\zeta$={lon['short_period'].zeta:.3f}, T={lon['short_period'].period_s:.1f} s)",
          ylabel="$\\Delta\\theta$ (deg)")

    # phugoid: speed perturbation, watch speed over a long window
    x0 = tr.state.copy()
    x0[IDX_VEL] = x0[IDX_VEL] * 1.03
    traj = propagate(d13, x0, dt=0.02, n_steps=15000)
    t = np.arange(traj.shape[0]) * 0.02
    speed = np.linalg.norm(traj[:, IDX_VEL], axis=1)
    axes[1].plot(t, speed - speed[0], color=RED, lw=1.5)
    style(axes[1], f"Phugoid — 3% speed perturbation "
                   f"($\\zeta$={lon['phugoid'].zeta:.4f}, T={lon['phugoid'].period_s:.0f} s)",
          xlabel="time (s)", ylabel="$\\Delta V$ (m/s)")

    fig.text(0.5, -0.03,
             "Two timescales an order of magnitude apart: pitch rotation settling in seconds, "
             "energy exchange persisting for minutes.",
             ha="center", color=FG, fontsize=8.5, alpha=0.8)
    fig.tight_layout()
    save(fig, "03_mode_responses.png")


def main() -> int:
    print("Generating README figures...")
    model, tr, d13, A = build()
    figure_poles(A)
    figure_perturbation(model, tr, d13, A)
    figure_modes(model, tr, d13, A)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
