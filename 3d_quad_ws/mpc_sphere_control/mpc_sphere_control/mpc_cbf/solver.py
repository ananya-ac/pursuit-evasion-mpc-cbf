"""
Shared CBF/MPC math used by the pursuit-evasion solver.

CBF formulation — Cheng et al., CDC 2020, eq. (5) / Definition 1:

  Safety function (degree-1 and degree-2 multi-agent):
      h(x) = (Δp^T Δv) / ‖Δp‖  +  √(a_max · max(0, ‖Δp‖ − D_s))
             ↑ closing speed          ↑ braking allowance

  Discrete-time Control Barrier Condition (CBC, Definition 1):
      h(x_{t+1}(u_t)) + (η − 1)·h(x_t) ≥ 0
      i.e.  h_{t+1}(u) ≥ (1 − η)·h_t        η ∈ (0, 1]
"""
import numpy as np


def braking_cbf_h(dp: np.ndarray, dv: np.ndarray, a_max: float, D_safe: float, eps: float = 1e-3) -> float:
    """
    Cheng et al. (CDC 2020) eq. 5 braking-distance safety function:

        h = (Δp^T Δv)/‖Δp‖  +  √(a_max · max(0, ‖Δp‖ − D_safe))
              closing speed          braking allowance

    dp, dv are (this agent) − (other agent/obstacle) position/velocity.
    Returns negative when unsafe (approaching faster than braking capacity
    a_max allows); more negative = more urgent.
    """
    dp = np.asarray(dp, dtype=float).reshape(3)
    dv = np.asarray(dv, dtype=float).reshape(3)
    dist = np.linalg.norm(dp)
    if dist < eps:
        return -1.0
    closing_speed     = np.dot(dp, dv) / dist
    braking_allowance = np.sqrt(max(a_max, 0.0) * max(0.0, dist - D_safe))
    return float(closing_speed + braking_allowance)


def _condense_linear_horizon(A_d, B_d, c_d, N):
    """
    Condense a frozen (LTI, i.e. linearized once and held across the whole
    horizon) discrete affine model

        x_{k+1} = A_d @ x_k + B_d @ u_k + c_d

    into a single affine map from the stacked control sequence to the
    stacked future-state sequence:

        X = S_x @ x0 + S_u @ U + S_c

    where X = [x_1; x_2; ...; x_N] (excludes x0, which the caller supplies
    separately) and U = [u_0; u_1; ...; u_{N-1}].

    Returns
    -------
    S_x : (N*nx, nx)
    S_u : (N*nx, N*nu)   block lower-triangular
    S_c : (N*nx,)
    """
    nx = A_d.shape[0]
    nu = B_d.shape[1]

    A_pow = [np.eye(nx)]
    for _ in range(N):
        A_pow.append(A_pow[-1] @ A_d)

    S_x = np.zeros((N * nx, nx))
    S_u = np.zeros((N * nx, N * nu))
    S_c = np.zeros(N * nx)

    for k in range(1, N + 1):
        S_x[(k - 1) * nx:k * nx, :] = A_pow[k]
        for j in range(k):
            S_u[(k - 1) * nx:k * nx, j * nu:(j + 1) * nu] = A_pow[k - 1 - j] @ B_d
        c_sum = np.zeros(nx)
        for j in range(k):
            c_sum += A_pow[j] @ c_d
        S_c[(k - 1) * nx:k * nx] = c_sum

    return S_x, S_u, S_c
