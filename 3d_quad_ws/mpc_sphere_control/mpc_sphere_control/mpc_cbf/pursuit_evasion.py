"""
Pursuit-evasion minimax MPC for Quadrotor entities, using the manifold-reduced
12-dim [p, v, δθ, w] linearization (frozen once per agent per horizon, as in
MPCCBFController's real-time-iteration pattern).

Each solve() is one best-response round: forecast the evader under hover
control, solve pursuers to minimize distance to that forecast (convex, OSQP),
then solve the evader to maximize distance from the pursuers' solution
(non-convex, IPOPT).
"""
import casadi as ca
import numpy as np
import osqp
import scipy.sparse as sparse

from .solver import _condense_linear_horizon, braking_cbf_h

# Reduced state layout Quadrotor.linearize_discrete()/reduce_state() use:
# [p(0:3), v(3:6), dtheta(6:9), w(9:12)]
_NX = 12
_NU = 4


class QuadrotorPursuitEvasionSolver:
    """Minimax pursuit-evasion MPC: N_p pursuers (convex, OSQP) vs. one
    evader (non-convex objective — maximizing distance — needs IPOPT)."""

    def __init__(
        self,
        pursuers,               # list[Quadrotor]
        evader,                 # Quadrotor
        horizon,
        radial_weight=1.0,
        radial_cap=100.0,
        cutoff_weight=1.0,
        cutoff_cap=100.0,
        altitude_weight=5.0,
        vel_weight=0.05,
        omega_weight=10.0,
        pursuer_control_weight=1e-3,
        evader_control_weight=1e-3,
        pursuer_state_slack_weight=1e4,
        evader_state_slack_weight=1e4,
        omega_slack_weight=1e3,
        workspace_half_extent=12.0,
        workspace_slack_weight=1e3,
        workspace_pull_weight=3.0,
        z_min=0.0,
        omega_xy_max=4.0,
        omega_z_max=3.0,
        enable_pursuer_pursuer_cbf=True,
        enable_pursuer_evader_cbf=True,
        enable_evader_cbf=True,
        D_safe_pursuer=1.5,
        D_safe_pursuer_evader=1.5,
        D_safe_evader=1.5,
        gamma_pursuer_cbf=1.0,
        gamma_evader_cbf=0.8,
        pursuer_pursuer_cbf_slack_weight=1e5,
        pursuer_evader_cbf_slack_weight=1e5,
        evader_cbf_slack_weight=1e5,
    ):
        if any(p.control_dim != _NU or p.state_dim != 13 for p in pursuers):
            raise ValueError("Pursuit-evasion solver expects Quadrotor pursuers.")
        if evader.control_dim != _NU or evader.state_dim != 13:
            raise ValueError("Pursuit-evasion solver expects a Quadrotor evader.")
        if any(abs(p.dt - evader.dt) > 1e-12 for p in pursuers):
            raise ValueError("All pursuers and the evader must share the same dt.")

        self.pursuers = list(pursuers)
        self.evader   = evader
        self.N_p      = len(self.pursuers)
        self.dt       = float(evader.dt)
        self.N        = int(horizon)

        self.w_radial = float(radial_weight)
        # Caps the radial cost's growth (see _saturating_sumsqr) so it can't dwarf the
        # fixed-scale regularization terms once agents are well-separated.
        self.radial_cap = float(radial_cap)
        # "Geometric cutoff": penalizes the component of pursuer-evader relative
        # velocity perpendicular to the current line-of-sight (LOS) — a collision-
        # course/parallel-navigation intercept term, not just closing distance. The
        # pursuer minimizing it wants zero transverse drift relative to the LOS
        # (aim ahead, not tail-chase); the evader maximizing the same term (its
        # Opti negates J like it does for the radial term) wants to juke sideways
        # and invalidate that aim point. Off by default (0.0) — opt-in, since it's
        # an additional shaping term beyond what's been tuned/tested so far.
        # cutoff_cap saturates it the same way radial_cap does, for the same reason
        # (an evader-maximized term must stay bounded or it dwarfs regularization).
        self.w_cutoff    = float(cutoff_weight)
        self.cutoff_cap  = float(cutoff_cap)
        self.cutoff_eps  = 1e-6
        # Penalizes deviation from altitude at construction time, so the evader can't
        # "farm" distance by diving under gravity instead of evading horizontally.
        self.altitude_weight = float(altitude_weight)
        self.pursuer_nominal_z = [float(p.position[2]) for p in self.pursuers]
        self.evader_nominal_z  = float(evader.position[2])
        self.vel_weight     = float(vel_weight)
        # Much stronger than vel_weight: sustained high omega breaks linearize_discrete's
        # validity (feeds dynamics, forecasts, and CBFs), and the per-horizon bound alone
        # doesn't stop it from compounding across ticks.
        self.omega_weight   = float(omega_weight)
        self.w_up = float(pursuer_control_weight)
        self.w_ue = float(evader_control_weight)
        self.pursuer_state_slack_weight = float(pursuer_state_slack_weight)
        self.evader_state_slack_weight  = float(evader_state_slack_weight)
        # Kept separate from pursuer/evader_state_slack_weight (z_min) so violating one
        # doesn't silently relax the other; weighted higher as the real omega backstop.
        self.omega_slack_weight = float(omega_slack_weight)

        # Soft framing box around the scene for RViz legibility (not a safety
        # constraint), centered on the evader's start position and backed by slack.
        self.workspace_half_extent  = float(workspace_half_extent)
        self.workspace_slack_weight = float(workspace_slack_weight)
        # Running cost pulling x,y toward workspace_center; the slack-backed box alone
        # let drift balloon over a long chase (tested empirically), so this holds it.
        self.workspace_pull_weight  = float(workspace_pull_weight)
        self.workspace_center = np.asarray(evader.position, dtype=float).reshape(3).copy()

        self.z_min        = float(z_min)
        self.omega_xy_max = float(omega_xy_max)
        self.omega_z_max  = float(omega_z_max)

        self.enable_pursuer_pursuer_cbf = bool(enable_pursuer_pursuer_cbf)
        self.enable_pursuer_evader_cbf  = bool(enable_pursuer_evader_cbf)
        self.enable_evader_cbf          = bool(enable_evader_cbf)
        self.D_safe_pursuer        = float(D_safe_pursuer)
        self.D_safe_pursuer_evader = float(D_safe_pursuer_evader)
        self.D_safe_evader         = float(D_safe_evader)
        self.gamma_pursuer_cbf = float(gamma_pursuer_cbf)
        self.gamma_evader_cbf  = float(gamma_evader_cbf)
        self.pursuer_pursuer_cbf_slack_weight = float(pursuer_pursuer_cbf_slack_weight)
        self.pursuer_evader_cbf_slack_weight  = float(pursuer_evader_cbf_slack_weight)
        self.evader_cbf_slack_weight          = float(evader_cbf_slack_weight)

        # Effective braking accel per agent (thrust_max/mass - gravity, entities.py
        # A_MAX), used by the one-step CBF filters below, not the Opti minimax solve.
        self.pursuer_a_eff = [float(p.A_MAX) for p in self.pursuers]
        self.evader_a_eff  = float(evader.A_MAX)

        self.pursuer_opti, self.p_vars, self.p_params = self._build_pursuer_solver()
        self.evader_opti,  self.e_vars, self.e_params = self._build_evader_solver()

    # ── hover reference control / per-agent frozen linearization ───────────

    @staticmethod
    def _hover_control(entity) -> np.ndarray:
        return np.array([entity.params.hover_thrust, 0.0, 0.0, 0.0])

    def _linearize_agent(self, entity, state: np.ndarray):
        """Freeze one linearization for `entity` at its current state,
        about hover control — reused across the whole horizon."""
        u_ref = self._hover_control(entity)
        A_d, B_d, c_d = entity.linearize_discrete(state, u_ref)
        x0 = entity.reduce_state(state)
        return A_d, B_d, c_d, x0

    # ── CasADi Opti model-building helpers ──────────────────────────────────

    def _add_state_bounds(self, opti, x_vec, z_slack, omega_slack):
        opti.subject_to(x_vec[2] >= self.z_min - z_slack)
        opti.subject_to(opti.bounded(-self.omega_xy_max - omega_slack, x_vec[9],  self.omega_xy_max + omega_slack))
        opti.subject_to(opti.bounded(-self.omega_xy_max - omega_slack, x_vec[10], self.omega_xy_max + omega_slack))
        opti.subject_to(opti.bounded(-self.omega_z_max  - omega_slack, x_vec[11], self.omega_z_max  + omega_slack))

    def _add_workspace_bounds(self, opti, pos_vec, ws_slack):
        """Soft box constraint keeping [x,y,z] within workspace_half_extent
        of the scene's fixed center — see workspace_center's __init__ note."""
        for axis in range(3):
            c = float(self.workspace_center[axis])
            opti.subject_to(opti.bounded(
                c - self.workspace_half_extent - ws_slack,
                pos_vec[axis],
                c + self.workspace_half_extent + ws_slack,
            ))

    @staticmethod
    def _add_control_bounds(opti, u_vec, entity):
        lo, hi = entity.control_bounds()
        for j in range(_NU):
            opti.subject_to(opti.bounded(float(lo[j]), u_vec[j], float(hi[j])))

    def _saturating_sumsqr(self, delta, cap=None):
        """Bounded sumsqr(delta), saturating toward `cap` (default radial_cap) as
        ‖delta‖ grows. Evader-only: its IPOPT solve handles the nonlinearity; the
        pursuer's OSQP solve needs a plain quadratic and doesn't need saturation."""
        cap = self.radial_cap if cap is None else cap
        return cap * (1 - ca.exp(-ca.sumsqr(delta) / cap))

    @staticmethod
    def _frozen_perp_projectors(pursuer_pos_forecasts, evader_pos_forecast, eps=1e-6):
        """(9*N_p, N+1) flattened LOS-perpendicular projectors P_perp = I - r̂r̂ᵀ,
        one per pursuer per horizon stage — r̂ is the unit line-of-sight vector
        from pursuer i to the evader. Computed from FIXED (already-forecasted,
        not decision-variable) positions so the resulting cutoff cost term stays
        exactly quadratic in the Opti's decision variables (required for the
        pursuer's OSQP solve; harmless for the evader's IPOPT solve too)."""
        N_p = len(pursuer_pos_forecasts)
        N_plus_1 = evader_pos_forecast.shape[1]
        P_perp = np.zeros((9 * N_p, N_plus_1))
        for k in range(N_plus_1):
            p_e = evader_pos_forecast[:, k]
            for i in range(N_p):
                p_i = pursuer_pos_forecasts[i][:, k]
                los = p_e - p_i
                los_norm = max(np.linalg.norm(los), eps)
                r_hat = los / los_norm
                proj = np.eye(3) - np.outer(r_hat, r_hat)
                P_perp[9 * i:9 * i + 9, k] = proj.reshape(-1, order="F")
        return P_perp

    def _build_pursuer_solver(self):
        opti = ca.Opti("conic")
        nx, nu, N_p, N = _NX, _NU, self.N_p, self.N

        X           = opti.variable(nx * N_p, N + 1)
        U           = opti.variable(nu * N_p, N)
        slack       = opti.variable(N_p, N)   # z_min
        omega_slack = opti.variable(N_p, N)   # omega bound — separate, weighted far higher (see __init__)
        ws_slack    = opti.variable(N_p, N)   # workspace framing box — see workspace_center note

        x0_param    = opti.parameter(nx * N_p)
        Ad_param    = opti.parameter(N_p, nx * nx)
        Bd_param    = opti.parameter(N_p, nx * nu)
        cd_param    = opti.parameter(N_p, nx)
        z_ref_param = opti.parameter(N_p)        # nominal altitude (fixed at construction), per pursuer
        evader_traj = opti.parameter(3, N + 1)   # fixed evader position forecast

        # Geometric-cutoff params — only added to the graph if actually used, so
        # this is a no-op (byte-for-byte prior behavior) when cutoff_weight <= 0.
        if self.w_cutoff > 0.0:
            evader_vel_traj = opti.parameter(3, N + 1)         # fixed evader velocity forecast
            P_perp_param    = opti.parameter(9 * N_p, N + 1)   # fixed LOS-perp projectors (see _frozen_perp_projectors)

        opti.subject_to(X[:, 0] == x0_param)
        opti.subject_to(ca.vec(slack) >= 0)
        opti.subject_to(ca.vec(omega_slack) >= 0)
        opti.subject_to(ca.vec(ws_slack) >= 0)

        J = 0
        for i in range(N_p):
            base_x, base_u = i * nx, i * nu
            A_i = ca.reshape(Ad_param[i, :], nx, nx)
            B_i = ca.reshape(Bd_param[i, :], nx, nu)
            c_i = cd_param[i, :].T
            u_eq = self._hover_control(self.pursuers[i])
            for k in range(N):
                opti.subject_to(
                    X[base_x:base_x + nx, k + 1]
                    == ca.mtimes(A_i, X[base_x:base_x + nx, k])
                    + ca.mtimes(B_i, U[base_u:base_u + nu, k])
                    + c_i
                )
                self._add_state_bounds(opti, X[base_x:base_x + nx, k + 1],
                                       slack[i, k], omega_slack[i, k])
                self._add_control_bounds(opti, U[base_u:base_u + nu, k], self.pursuers[i])

                pos_k   = X[base_x:base_x + 3, k + 1]
                vel_k   = X[base_x + 3:base_x + 6, k + 1]
                omega_k = X[base_x + 9:base_x + 12, k + 1]
                self._add_workspace_bounds(opti, pos_k, ws_slack[i, k])
                J += self.w_radial * ca.sumsqr(pos_k - evader_traj[:, k + 1])
                if self.w_cutoff > 0.0:
                    P_perp_ik = ca.reshape(P_perp_param[9 * i:9 * i + 9, k + 1], 3, 3)
                    v_err_ik  = ca.mtimes(P_perp_ik, vel_k - evader_vel_traj[:, k + 1])
                    J += self.w_cutoff * ca.sumsqr(v_err_ik)
                J += self.altitude_weight * ca.sumsqr(pos_k[2] - z_ref_param[i])
                J += self.workspace_pull_weight * ca.sumsqr(pos_k[0:2] - self.workspace_center[0:2])
                J += self.vel_weight * ca.sumsqr(vel_k)
                J += self.omega_weight * ca.sumsqr(omega_k)
                J += self.w_up * ca.sumsqr(U[base_u:base_u + nu, k] - u_eq)
            J += self.pursuer_state_slack_weight * ca.sumsqr(slack[i, :])
            J += self.omega_slack_weight * ca.sumsqr(omega_slack[i, :])
            J += self.workspace_slack_weight * ca.sumsqr(ws_slack[i, :])

        opti.minimize(J)
        opti.solver("osqp", {"verbose": False, "osqp": {"verbose": False}})

        params = {
            "x0": x0_param, "A": Ad_param, "B": Bd_param, "c": cd_param,
            "z_ref": z_ref_param, "evader_traj": evader_traj,
        }
        if self.w_cutoff > 0.0:
            params["evader_vel_traj"] = evader_vel_traj
            params["P_perp"] = P_perp_param

        return (
            opti,
            {"X": X, "U": U, "slack": slack, "omega_slack": omega_slack, "ws_slack": ws_slack},
            params,
        )

    def _build_evader_solver(self):
        opti = ca.Opti()
        nx, nu, N = _NX, _NU, self.N

        X           = opti.variable(nx, N + 1)
        U           = opti.variable(nu, N)
        slack       = opti.variable(1, N)   # z_min
        omega_slack = opti.variable(1, N)   # omega bound — separate, weighted far higher (see __init__)
        ws_slack    = opti.variable(1, N)   # workspace framing box — see workspace_center note

        x0_param     = opti.parameter(nx)
        A_param      = opti.parameter(nx, nx)
        B_param      = opti.parameter(nx, nu)
        c_param      = opti.parameter(nx)
        z_ref_param  = opti.parameter(1)                     # nominal altitude (fixed at construction)
        pursuer_traj = opti.parameter(3 * self.N_p, N + 1)   # fixed pursuer position forecasts

        if self.w_cutoff > 0.0:
            pursuer_vel_traj = opti.parameter(3 * self.N_p, N + 1)   # fixed pursuer velocity forecasts
            P_perp_param     = opti.parameter(9 * self.N_p, N + 1)   # fixed LOS-perp projectors

        opti.subject_to(X[:, 0] == x0_param)
        opti.subject_to(ca.vec(slack) >= 0)
        opti.subject_to(ca.vec(omega_slack) >= 0)
        opti.subject_to(ca.vec(ws_slack) >= 0)

        u_eq = self._hover_control(self.evader)
        J_radial = 0   # to be MAXIMIZED (evader wants distance) — saturating, see _saturating_sumsqr
        J_reg    = 0   # to be minimized normally (altitude/control/vel/omega/slack)
        for k in range(N):
            opti.subject_to(
                X[:, k + 1] == ca.mtimes(A_param, X[:, k]) + ca.mtimes(B_param, U[:, k]) + c_param
            )
            self._add_state_bounds(opti, X[:, k + 1], slack[0, k], omega_slack[0, k])
            self._add_control_bounds(opti, U[:, k], self.evader)
            self._add_workspace_bounds(opti, X[0:3, k + 1], ws_slack[0, k])

            for i in range(self.N_p):
                p_i = pursuer_traj[3 * i:3 * i + 3, k + 1]
                J_radial += self.w_radial * self._saturating_sumsqr(X[0:3, k + 1] - p_i)
                if self.w_cutoff > 0.0:
                    v_i = pursuer_vel_traj[3 * i:3 * i + 3, k + 1]
                    P_perp_ik = ca.reshape(P_perp_param[9 * i:9 * i + 9, k + 1], 3, 3)
                    v_err_ik  = ca.mtimes(P_perp_ik, v_i - X[3:6, k + 1])
                    J_radial += self.w_cutoff * self._saturating_sumsqr(v_err_ik, cap=self.cutoff_cap)

            # Altitude/workspace-pull terms stay in J_reg (not negated) so the evader
            # can't dive/run to farm distance instead of genuinely evading.
            J_reg += self.altitude_weight * ca.sumsqr(X[2, k + 1] - z_ref_param)
            J_reg += self.workspace_pull_weight * ca.sumsqr(X[0:2, k + 1] - self.workspace_center[0:2])
            J_reg += self.w_ue * ca.sumsqr(U[:, k] - u_eq)
            J_reg += self.vel_weight   * ca.sumsqr(X[3:6, k + 1])
            J_reg += self.omega_weight * ca.sumsqr(X[9:12, k + 1])
        J_reg += self.evader_state_slack_weight * ca.sumsqr(slack)
        J_reg += self.omega_slack_weight * ca.sumsqr(omega_slack)
        J_reg += self.workspace_slack_weight * ca.sumsqr(ws_slack)

        opti.minimize(-J_radial + J_reg)
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {"max_iter": 500, "print_level": 0, "sb": "yes"},
        )

        e_params = {
            "x0": x0_param, "A": A_param, "B": B_param, "c": c_param,
            "z_ref": z_ref_param, "pursuer_traj": pursuer_traj,
        }
        if self.w_cutoff > 0.0:
            e_params["pursuer_vel_traj"] = pursuer_vel_traj
            e_params["P_perp"] = P_perp_param

        return (
            opti,
            {"X": X, "U": U, "slack": slack, "omega_slack": omega_slack, "ws_slack": ws_slack},
            e_params,
        )

    # ── main entry point ────────────────────────────────────────────────────

    def solve(self, pursuer_states, evader_state):
        """
        One best-response round: solve the pursuers' Opti (minimize distance
        to a forecast of the evader), then solve the evader's Opti (maximize
        distance from the just-solved pursuer trajectory).

        Args:
            pursuer_states : list of raw 13-dim states, one per pursuer.
            evader_state   : raw 13-dim evader state.

        Returns:
            U_p0 : (N_p, 4) first control for each pursuer.
            u_e0 : (4,) first control for the evader.
            info : dict with solved trajectories (reduced-state, for
                   logging/visualization), or None on solver failure.
        """
        pursuer_states = [np.asarray(s, dtype=float).reshape(13) for s in pursuer_states]
        evader_state    = np.asarray(evader_state, dtype=float).reshape(13)

        # 1. Freeze one linearization per agent, at its current actual state.
        A_p, B_p, c_p, x0_p = [], [], [], []
        for entity, state in zip(self.pursuers, pursuer_states):
            A_i, B_i, c_i, x0_i = self._linearize_agent(entity, state)
            A_p.append(A_i); B_p.append(B_i); c_p.append(c_i); x0_p.append(x0_i)
        A_e, B_e, c_e, x0_e = self._linearize_agent(self.evader, evader_state)

        # 2. Evader forecast (for the pursuers' solve): propagate the
        #    evader's own frozen model under hover control across the
        #    horizon, via the same condensation MPCCBFController uses.
        S_x, S_u, S_c = _condense_linear_horizon(A_e, B_e, c_e, self.N)
        U_e_hover = np.tile(self._hover_control(self.evader), self.N)
        X_e_forecast = np.concatenate([x0_e, S_x @ x0_e + S_u @ U_e_hover + S_c]).reshape(self.N + 1, _NX).T
        evader_pos_forecast = X_e_forecast[0:3, :]   # (3, N+1)
        evader_vel_forecast = X_e_forecast[3:6, :]   # (3, N+1)

        # 2b. Geometric-cutoff projectors for the pursuers' solve: each pursuer's
        #     own position is the decision variable being solved for, so — same
        #     idea as the evader forecast above — forecast it under hover control
        #     from its own already-frozen (A,B,c) instead. Only computed when the
        #     cutoff term is enabled.
        if self.w_cutoff > 0.0:
            pursuer_pos_forecast_pre = []
            for i in range(self.N_p):
                Sx_i, Su_i, Sc_i = _condense_linear_horizon(A_p[i], B_p[i], c_p[i], self.N)
                U_hover_i = np.tile(self._hover_control(self.pursuers[i]), self.N)
                X_i = np.concatenate(
                    [x0_p[i], Sx_i @ x0_p[i] + Su_i @ U_hover_i + Sc_i]
                ).reshape(self.N + 1, _NX).T
                pursuer_pos_forecast_pre.append(X_i[0:3, :])
            P_perp_pre = self._frozen_perp_projectors(
                pursuer_pos_forecast_pre, evader_pos_forecast, self.cutoff_eps
            )

        # 3. Solve pursuers.
        self.pursuer_opti.set_value(self.p_params["x0"], np.concatenate(x0_p))
        self.pursuer_opti.set_value(
            self.p_params["A"], np.stack([A_i.reshape(-1, order="F") for A_i in A_p])
        )
        self.pursuer_opti.set_value(
            self.p_params["B"], np.stack([B_i.reshape(-1, order="F") for B_i in B_p])
        )
        self.pursuer_opti.set_value(self.p_params["c"], np.stack(c_p))
        self.pursuer_opti.set_value(self.p_params["z_ref"], np.array(self.pursuer_nominal_z))
        self.pursuer_opti.set_value(self.p_params["evader_traj"], evader_pos_forecast)
        if self.w_cutoff > 0.0:
            self.pursuer_opti.set_value(self.p_params["evader_vel_traj"], evader_vel_forecast)
            self.pursuer_opti.set_value(self.p_params["P_perp"], P_perp_pre)
        try:
            sol_p = self.pursuer_opti.solve()
        except RuntimeError as exc:
            print(f"Pursuit-evasion pursuer solve failed: {exc}")
            return None, None, None
        X_p_opt = sol_p.value(self.p_vars["X"])
        U_p_opt = sol_p.value(self.p_vars["U"])
        if self.N_p == 1:
            X_p_opt = X_p_opt.reshape(_NX, self.N + 1)
            U_p_opt = U_p_opt.reshape(_NU, self.N)

        # 4. Pursuer forecast (for the evader's solve): the just-solved
        #    pursuer trajectory's positions.
        pursuer_pos_forecast = np.zeros((3 * self.N_p, self.N + 1))
        for i in range(self.N_p):
            pursuer_pos_forecast[3 * i:3 * i + 3, :] = X_p_opt[i * _NX:i * _NX + 3, :]

        # 4b. Geometric-cutoff projectors for the evader's solve — recomputed
        #     from the pursuers' just-solved (actual, not forecasted) trajectory,
        #     same asymmetry pursuer_pos_forecast/pursuer_traj above already has.
        if self.w_cutoff > 0.0:
            pursuer_vel_forecast = np.zeros((3 * self.N_p, self.N + 1))
            pursuer_pos_forecast_post = []
            for i in range(self.N_p):
                pursuer_vel_forecast[3 * i:3 * i + 3, :] = X_p_opt[i * _NX + 3:i * _NX + 6, :]
                pursuer_pos_forecast_post.append(X_p_opt[i * _NX:i * _NX + 3, :])
            P_perp_post = self._frozen_perp_projectors(
                pursuer_pos_forecast_post, evader_pos_forecast, self.cutoff_eps
            )

        # 5. Solve evader.
        self.evader_opti.set_value(self.e_params["x0"], x0_e)
        self.evader_opti.set_value(self.e_params["A"], A_e)
        self.evader_opti.set_value(self.e_params["B"], B_e)
        self.evader_opti.set_value(self.e_params["c"], c_e)
        self.evader_opti.set_value(self.e_params["z_ref"], self.evader_nominal_z)
        if self.w_cutoff > 0.0:
            self.evader_opti.set_value(self.e_params["pursuer_vel_traj"], pursuer_vel_forecast)
            self.evader_opti.set_value(self.e_params["P_perp"], P_perp_post)
        self.evader_opti.set_value(self.e_params["pursuer_traj"], pursuer_pos_forecast)
        try:
            sol_e = self.evader_opti.solve()
        except RuntimeError as exc:
            print(f"Pursuit-evasion evader solve failed: {exc}")
            return None, None, None
        X_e_opt = sol_e.value(self.e_vars["X"])
        U_e_opt = sol_e.value(self.e_vars["U"])

        U_p0 = np.array([U_p_opt[i * _NU:(i + 1) * _NU, 0] for i in range(self.N_p)])
        u_e0 = U_e_opt[:, 0]
        # Nonzero workspace slack flags an agent leaving the framing box (not a safety fault).
        ws_slack_p = float(np.max(sol_p.value(self.p_vars["ws_slack"])))
        ws_slack_e = float(np.max(sol_e.value(self.e_vars["ws_slack"])))
        info = dict(
            X_p=X_p_opt, U_p=U_p_opt, X_e=X_e_opt, U_e=U_e_opt,
            ws_slack_p=ws_slack_p, ws_slack_e=ws_slack_e,
        )
        return U_p0, u_e0, info

    # ── ONE-STEP CBF FILTERS ──────────────────────────────────────────────
    # Safety post-filter independent of the Opti minimax solve: linearizes once about
    # the current state/control, then projects into {u : ḣ + γh ≥ 0} via a small OSQP QP.

    @staticmethod
    def _finite_difference_gradient(func, u_ref, eps: float = 1e-5):
        u_ref = np.asarray(u_ref, dtype=float).reshape(-1)
        f0 = float(func(u_ref))
        grad = np.zeros_like(u_ref)
        for idx in range(u_ref.size):
            du = np.zeros_like(u_ref)
            du[idx] = eps
            grad[idx] = (float(func(u_ref + du)) - float(func(u_ref - du))) / (2.0 * eps)
        return f0, grad

    def _linearized_next_state_affine(self, entity, state, u_ref):
        """One-step linearized prediction of the REDUCED next state:
        x_next_reduced(u) = const + B_d @ u, expanded about (state, u_ref)."""
        state = np.asarray(state, dtype=float).reshape(13)
        A_d, B_d, c_d = entity.linearize_discrete(state, u_ref)
        const = A_d @ entity.reduce_state(state) + c_d
        return const, B_d

    @staticmethod
    def _solve_cbf_qp(P, q, A, l, u):
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False, adaptive_rho=True, polish=True)
        return prob.solve()

    def _build_pursuer_evader_cbf_constraints(self, pursuer_states, evader_state, u_des_list, n_dec, slack_idx):
        N_p = self.N_p
        a_rows = sparse.lil_matrix((N_p, n_dec))
        l_rows = np.full(N_p, -np.inf)
        u_rows = np.zeros(N_p)

        u_e_hover = self._hover_control(self.evader)
        e_const, e_B = self._linearized_next_state_affine(self.evader, evader_state, u_e_hover)
        e_next = e_const + e_B @ u_e_hover
        p_e_next, v_e_next = e_next[0:3], e_next[3:6]

        for i, (entity, state, u_ref) in enumerate(zip(self.pursuers, pursuer_states, u_des_list)):
            p_const, p_B = self._linearized_next_state_affine(entity, state, u_ref)

            def h_next(u, p_const=p_const, p_B=p_B):
                x_next = p_const + p_B @ u
                return braking_cbf_h(
                    x_next[0:3] - p_e_next, x_next[3:6] - v_e_next,
                    self.pursuer_a_eff[i], self.D_safe_pursuer_evader,
                )

            h_nom, grad = self._finite_difference_gradient(h_next, u_ref)
            p_pos = entity.position_from_state(state)
            p_vel = entity.velocity_from_state(state)
            h_now = braking_cbf_h(
                p_pos - evader_state[0:3], p_vel - evader_state[3:6],
                self.pursuer_a_eff[i], self.D_safe_pursuer_evader,
            )
            rhs = (1.0 - self.gamma_evader_cbf) * h_now - h_nom + grad @ u_ref

            a_rows[i, i * _NU:(i + 1) * _NU] = -grad
            a_rows[i, slack_idx] = -1.0
            u_rows[i] = -rhs

        return a_rows.tocsc(), l_rows, u_rows

    def _build_pursuer_pursuer_cbf_constraints(self, pursuer_states, u_des_list, n_dec, slack_idx):
        N_p = self.N_p
        num_pairs = N_p * (N_p - 1) // 2
        if num_pairs <= 0:
            return None

        a_rows = sparse.lil_matrix((num_pairs, n_dec))
        l_rows = np.full(num_pairs, -np.inf)
        u_rows = np.zeros(num_pairs)

        pair_idx = 0
        for i in range(N_p):
            p_i_const, p_i_B = self._linearized_next_state_affine(
                self.pursuers[i], pursuer_states[i], u_des_list[i]
            )
            for j in range(i + 1, N_p):
                p_j_const, p_j_B = self._linearized_next_state_affine(
                    self.pursuers[j], pursuer_states[j], u_des_list[j]
                )
                u_pair_ref = np.hstack([u_des_list[i], u_des_list[j]])

                def h_next(u_pair, p_i_const=p_i_const, p_i_B=p_i_B,
                           p_j_const=p_j_const, p_j_B=p_j_B):
                    x_i_next = p_i_const + p_i_B @ u_pair[0:_NU]
                    x_j_next = p_j_const + p_j_B @ u_pair[_NU:2 * _NU]
                    return braking_cbf_h(
                        x_i_next[0:3] - x_j_next[0:3], x_i_next[3:6] - x_j_next[3:6],
                        self.pursuer_a_eff[i], self.D_safe_pursuer,
                    )

                h_nom, grad = self._finite_difference_gradient(h_next, u_pair_ref)
                p_i_pos = self.pursuers[i].position_from_state(pursuer_states[i])
                p_i_vel = self.pursuers[i].velocity_from_state(pursuer_states[i])
                p_j_pos = self.pursuers[j].position_from_state(pursuer_states[j])
                p_j_vel = self.pursuers[j].velocity_from_state(pursuer_states[j])
                h_now = braking_cbf_h(
                    p_i_pos - p_j_pos, p_i_vel - p_j_vel,
                    self.pursuer_a_eff[i], self.D_safe_pursuer,
                )
                rhs = (1.0 - self.gamma_pursuer_cbf) * h_now - h_nom + grad @ u_pair_ref

                a_rows[pair_idx, i * _NU:(i + 1) * _NU] = -grad[0:_NU]
                a_rows[pair_idx, j * _NU:(j + 1) * _NU] = -grad[_NU:2 * _NU]
                a_rows[pair_idx, slack_idx] = -1.0
                u_rows[pair_idx] = -rhs
                pair_idx += 1

        return a_rows.tocsc(), l_rows, u_rows

    def one_step_pursuer_cbf_filter(self, pursuer_states, evader_state, u_des_list):
        """
        Project each pursuer's desired one-step control to keep the braking
        CBF safe against (a) the evader and (b) other pursuers, via a
        single OSQP QP with one slack variable per enabled constraint type.

        Returns: (u_out, max_slack) — u_out is a list of (4,) controls (one
        per pursuer); max_slack > 0 flags a relaxed/violated constraint.
        """
        if not (self.enable_pursuer_evader_cbf or self.enable_pursuer_pursuer_cbf):
            return [np.asarray(u, dtype=float).reshape(_NU) for u in u_des_list], 0.0

        pursuer_states = [np.asarray(s, dtype=float).reshape(13) for s in pursuer_states]
        evader_state    = np.asarray(evader_state, dtype=float).reshape(13)
        u_des_list      = [np.asarray(u, dtype=float).reshape(_NU) for u in u_des_list]

        nu_total = _NU * self.N_p
        slack_indices = {}
        n_dec = nu_total
        if self.enable_pursuer_evader_cbf:
            slack_indices["pe"] = n_dec
            n_dec += 1
        if self.enable_pursuer_pursuer_cbf:
            slack_indices["pp"] = n_dec
            n_dec += 1

        p_diag = np.ones(n_dec)
        if "pe" in slack_indices:
            p_diag[slack_indices["pe"]] = self.pursuer_evader_cbf_slack_weight
        if "pp" in slack_indices:
            p_diag[slack_indices["pp"]] = self.pursuer_pursuer_cbf_slack_weight
        P = sparse.diags(p_diag).tocsc()
        q = np.hstack([-np.concatenate(u_des_list), np.zeros(n_dec - nu_total)])

        a_rows, l_rows, u_rows = [], [], []

        a_box = sparse.eye(n_dec).tolil()
        l_box = np.full(n_dec, -np.inf)
        u_box = np.full(n_dec, np.inf)
        for i, entity in enumerate(self.pursuers):
            lo, hi = entity.control_bounds()
            l_box[i * _NU:(i + 1) * _NU] = lo
            u_box[i * _NU:(i + 1) * _NU] = hi
        for idx in slack_indices.values():
            l_box[idx] = 0.0
        a_rows.append(a_box.tocsc()); l_rows.append(l_box); u_rows.append(u_box)

        if self.enable_pursuer_evader_cbf:
            a_pe, l_pe, u_pe = self._build_pursuer_evader_cbf_constraints(
                pursuer_states, evader_state, u_des_list, n_dec, slack_indices["pe"]
            )
            a_rows.append(a_pe); l_rows.append(l_pe); u_rows.append(u_pe)

        if self.enable_pursuer_pursuer_cbf:
            pp_block = self._build_pursuer_pursuer_cbf_constraints(
                pursuer_states, u_des_list, n_dec, slack_indices["pp"]
            )
            if pp_block is not None:
                a_pp, l_pp, u_pp = pp_block
                a_rows.append(a_pp); l_rows.append(l_pp); u_rows.append(u_pp)

        A = sparse.vstack(a_rows).tocsc()
        l = np.concatenate(l_rows)
        u = np.concatenate(u_rows)
        res = self._solve_cbf_qp(P, q, A, l, u)
        if res.info.status not in ("solved", "solved inaccurate") or res.x is None:
            return u_des_list, np.nan

        slack_vals = [res.x[idx] for idx in slack_indices.values()] if slack_indices else [0.0]
        u_out = [res.x[i * _NU:(i + 1) * _NU] for i in range(self.N_p)]
        return u_out, float(np.max(slack_vals))

    def one_step_evader_cbf_filter(self, pursuer_states, evader_state, u_des, pursuer_controls):
        """
        Project the evader's desired one-step control to keep the braking
        CBF safe against each pursuer (assumed to apply pursuer_controls —
        e.g. their own just-computed desired/filtered controls).

        Returns: (u_out, slack) — u_out is a (4,) control; slack > 0 flags
        a relaxed/violated constraint.
        """
        if not self.enable_evader_cbf:
            return np.asarray(u_des, dtype=float).reshape(_NU), 0.0

        pursuer_states    = [np.asarray(s, dtype=float).reshape(13) for s in pursuer_states]
        evader_state      = np.asarray(evader_state, dtype=float).reshape(13)
        u_des             = np.asarray(u_des, dtype=float).reshape(_NU)
        pursuer_controls  = [np.asarray(u, dtype=float).reshape(_NU) for u in pursuer_controls]

        n_dec = _NU + 1
        slack_idx = _NU
        p_diag = np.ones(n_dec)
        p_diag[slack_idx] = self.evader_cbf_slack_weight
        P = sparse.diags(p_diag).tocsc()
        q = np.hstack([-u_des, 0.0])

        lo, hi = self.evader.control_bounds()
        l_box = np.concatenate([lo, [0.0]])
        u_box = np.concatenate([hi, [np.inf]])

        e_const, e_B = self._linearized_next_state_affine(self.evader, evader_state, u_des)

        a_rows = sparse.lil_matrix((self.N_p, n_dec))
        l_rows = np.full(self.N_p, -np.inf)
        u_rows = np.zeros(self.N_p)

        for i, (entity, p_state, u_ref) in enumerate(zip(self.pursuers, pursuer_states, pursuer_controls)):
            p_const, p_B = self._linearized_next_state_affine(entity, p_state, u_ref)
            p_next = p_const + p_B @ u_ref

            def h_next(u, p_next=p_next):
                e_next = e_const + e_B @ u
                return braking_cbf_h(
                    e_next[0:3] - p_next[0:3], e_next[3:6] - p_next[3:6],
                    self.evader_a_eff, self.D_safe_evader,
                )

            h_nom, grad = self._finite_difference_gradient(h_next, u_des)
            e_pos, e_vel = evader_state[0:3], evader_state[3:6]
            p_pos = entity.position_from_state(p_state)
            p_vel = entity.velocity_from_state(p_state)
            h_now = braking_cbf_h(e_pos - p_pos, e_vel - p_vel, self.evader_a_eff, self.D_safe_evader)
            rhs = (1.0 - self.gamma_evader_cbf) * h_now - h_nom + grad @ u_des

            a_rows[i, 0:_NU] = -grad
            a_rows[i, slack_idx] = -1.0
            u_rows[i] = -rhs

        A = sparse.vstack([sparse.eye(n_dec).tocsc(), a_rows.tocsc()]).tocsc()
        l = np.concatenate([l_box, l_rows])
        u = np.concatenate([u_box, u_rows])
        res = self._solve_cbf_qp(P, q, A, l, u)
        if res.info.status not in ("solved", "solved inaccurate") or res.x is None:
            return u_des, np.nan
        return res.x[:_NU], float(res.x[slack_idx])
