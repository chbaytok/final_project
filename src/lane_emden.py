"""
Lane–Emden equation solver.

We solve:
  (1/xi^2) d/dxi (xi^2 dtheta/dxi) + theta^n = 0
with theta(0)=1, theta'(0)=0.

We integrate from a small epsilon > 0 using the regular series expansion:
  theta(xi)  = 1 - xi^2/6 + (n xi^4)/120 + ...
  theta'(xi) = -xi/3 + (n xi^3)/30 + ...

Returns the first zero xi1 (surface), and the mass coefficient -theta'(xi1)/xi1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class LaneEmdenSolution:
    n: float
    xi: np.ndarray
    theta: np.ndarray
    dtheta: np.ndarray
    xi1: float
    mass_coeff: float  # -theta'(xi1)/xi1


def _init_series(n: float, eps: float) -> Tuple[float, float]:
    """Regular center expansion up to O(eps^4) for theta and O(eps^3) for theta'."""
    theta = 1.0 - (eps**2) / 6.0 + (n * eps**4) / 120.0
    dtheta = -eps / 3.0 + (n * eps**3) / 30.0
    return theta, dtheta


def solve_lane_emden(
    n: float,
    *,
    eps: float = 1e-6,
    xi_max: float = 50.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float | None = None,
) -> LaneEmdenSolution:
    """
    Solve Lane–Emden for given polytropic index n until the first zero of theta.

    Notes:
    - We stop at the first crossing theta=0 (stellar surface).
    - For non-integer n, theta^n is undefined for negative theta, so we clamp
      theta to >=0 inside the RHS. The event will terminate near the first zero
      before negative values matter.
    """
    if n <= 0:
        raise ValueError("Lane–Emden index n should be positive for this project use-case.")
    if eps <= 0:
        raise ValueError("eps must be > 0 to avoid the xi=0 singular term.")

    theta0, dtheta0 = _init_series(n, eps)

    def rhs(xi: float, y: np.ndarray) -> np.ndarray:
        theta, dtheta = float(y[0]), float(y[1])
        theta_n = max(theta, 0.0) ** n
        ddtheta = -(2.0 / xi) * dtheta - theta_n
        return np.array([dtheta, ddtheta], dtype=float)

    def event_surface(xi: float, y: np.ndarray) -> float:
        return float(y[0])

    event_surface.terminal = True
    event_surface.direction = -1

    solve_kwargs = dict(
        fun=rhs,
        t_span=(eps, xi_max),
        y0=np.array([theta0, dtheta0], dtype=float),
        events=event_surface,
        dense_output=True,
        rtol=rtol,
        atol=atol,
    )
    if max_step is not None:
        solve_kwargs["max_step"] = float(max_step)

    sol = solve_ivp(**solve_kwargs)

    if sol.status != 1 or sol.t_events is None or len(sol.t_events[0]) == 0:
        raise RuntimeError(
            "Lane–Emden integration did not hit theta=0. "
            "Try increasing xi_max or adjusting tolerances."
        )

    xi1 = float(sol.t_events[0][0])

    # Evaluate solution at the root using dense output for better accuracy
    y1 = sol.sol(xi1)  # shape (2,)
    theta1, dtheta1 = float(y1[0]), float(y1[1])

    # Append endpoint for convenient plotting
    xi_arr = np.append(sol.t, xi1)
    theta_arr = np.append(sol.y[0], theta1)
    dtheta_arr = np.append(sol.y[1], dtheta1)

    mass_coeff = -dtheta1 / xi1

    return LaneEmdenSolution(
        n=float(n),
        xi=xi_arr,
        theta=theta_arr,
        dtheta=dtheta_arr,
        xi1=xi1,
        mass_coeff=mass_coeff,
    )

