"""Zero-sum minimax solver implementations for the current codebase."""

import casadi as ca
import numpy as np
import osqp
import scipy.sparse as sparse


class BaseMinimaxSolver:
    """Shared boilerplate for minimax solvers with the current agent setup."""

    supports_one_step_cbf = False

    def __init__(
        self,
        pursuer_agent,
        evader_agent,
        horizon,
        capture_radius,
        v_max=None,
        a_max=None,
        enable_pursuer_pursuer_cbf=True,
        enable_pursuer_evader_cbf=True,
        enable_convex_hull_containment=False,
        D_safe_pursuer=3.0,
        D_safe_pursuer_evader=3.0,
        D_safe_evader=3.0,
        gamma_pursuer_cbf=1.,
        gamma_evader_cbf=0.8,
        pursuer_pursuer_cbf_slack_weight=1e5,
        pursuer_evader_cbf_slack_weight=1e5,
        convex_hull_slack_weight=1e5,
        evader_cbf_slack_weight=1e5,
        cutoff_mode="geometric",
        **_ignored_kwargs,
    ):
        if abs(float(pursuer_agent.dt) - float(evader_agent.dt)) > 1e-12:
            raise ValueError(
                "Minimax solvers expect pursuer and evader to share the same dt."
            )

        self.pursuer = pursuer_agent
        self.evader = evader_agent
        self.dt = float(pursuer_agent.dt)
        self.N = int(horizon)
        self.N_p = int(pursuer_agent.count)

        self.v_max = float(pursuer_agent.v_max if v_max is None else v_max)
        self.a_max = float(pursuer_agent.a_max if a_max is None else a_max)
        self.r_cap = float(capture_radius)
        self.enable_pursuer_pursuer_cbf = bool(enable_pursuer_pursuer_cbf)
        self.enable_pursuer_evader_cbf = bool(enable_pursuer_evader_cbf)
        self.enable_convex_hull_containment = bool(enable_convex_hull_containment)
        self.D_safe_pursuer = float(D_safe_pursuer)
        self.D_safe_pursuer_evader = float(D_safe_pursuer_evader)
        self.D_safe_evader = float(D_safe_evader)
        self.gamma_pursuer_cbf = float(gamma_pursuer_cbf)
        self.gamma_evader_cbf = float(gamma_evader_cbf)
        self.pursuer_pursuer_cbf_slack_weight = float(pursuer_pursuer_cbf_slack_weight)
        self.pursuer_evader_cbf_slack_weight = float(pursuer_evader_cbf_slack_weight)
        self.convex_hull_slack_weight = float(convex_hull_slack_weight)
        self.evader_cbf_slack_weight = float(evader_cbf_slack_weight)
        self.evader_v_max = float(evader_agent.v_max)
        self.evader_a_max = float(evader_agent.a_max)
        self.prev_X_p = None
        self.prev_U_p = None
        self.prev_X_e = None
        self.prev_U_e = None
        self.fixed_hull_order = None

    def reset_warm_starts(self):
        self.prev_X_p = None
        self.prev_U_p = None
        self.prev_X_e = None
        self.prev_U_e = None

    def set_fixed_adjacency_from_state(self, x_state, evader_state=None):
        """Freeze a polygon ordering around the evader at simulation start."""
        if self.N_p < 3:
            self.fixed_hull_order = None
            return

        x_state = np.asarray(x_state).reshape(4 * self.N_p)
        if evader_state is None:
            center = np.mean(
                [x_state[i * 4 : i * 4 + 2] for i in range(self.N_p)], axis=0
            )
        else:
            center = np.asarray(evader_state).reshape(4)[0:2]

        pursuer_positions = np.array(
            [x_state[i * 4 : i * 4 + 2] for i in range(self.N_p)]
        )
        angles = np.arctan2(
            pursuer_positions[:, 1] - center[1], pursuer_positions[:, 0] - center[0]
        )
        self.fixed_hull_order = list(np.argsort(angles))

    def step_pursuer_dynamics(self, x_current, u_apply):
        return self.pursuer.Ad.dot(x_current) + self.pursuer.Bd.dot(u_apply)

    def _build_shared_cost(self, X_p, U_p, X_e, U_e):
        del X_p, U_p, X_e, U_e
        raise NotImplementedError

    def _build_pursuer_solver(self):
        raise NotImplementedError

    def _build_evader_solver(self):
        raise NotImplementedError

    def _prepare_shared_aux_data(self, X_p_guess, X_e_guess):
        del X_p_guess, X_e_guess
        return None

    def _set_pursuer_solver_params(
        self, x_current, X_e_guess, U_e_guess, aux_data
    ):
        del aux_data
        self.pursuer_opti.set_value(self.p_params["x0"], x_current)
        self.pursuer_opti.set_value(self.p_params["X_e"], X_e_guess)
        self.pursuer_opti.set_value(self.p_params["U_e"], U_e_guess)

    def _set_evader_solver_params(
        self, evader_state, X_p_guess, U_p_guess, aux_data
    ):
        del aux_data
        self.evader_opti.set_value(self.e_params["x0"], evader_state)
        self.evader_opti.set_value(self.e_params["X_p"], X_p_guess)
        self.evader_opti.set_value(self.e_params["U_p"], U_p_guess)

    def _build_initial_evader_guess(self, evader_state):
        X_e_guess = np.zeros((4, self.N + 1))
        for k in range(self.N + 1):
            X_e_guess[0, k] = evader_state[0] + k * self.dt * evader_state[2]
            X_e_guess[1, k] = evader_state[1] + k * self.dt * evader_state[3]
        X_e_guess[2:4, :] = np.array(evader_state[2:4])[:, None]
        U_e_guess = np.zeros((2, self.N))
        return X_e_guess, U_e_guess

    def _build_initial_pursuer_guess(self, x_current):
        X_p_guess = np.tile(x_current[:, None], (1, self.N + 1))
        U_p_guess = np.zeros((2 * self.N_p, self.N))
        return X_p_guess, U_p_guess

    def _initialize_cbf_qp(self, u_des):
        u_des = np.asarray(u_des).flatten()
        slack_specs = []
        if self.enable_pursuer_evader_cbf:
            slack_specs.append(("pursuer_evader", self.pursuer_evader_cbf_slack_weight))
        if self.enable_pursuer_pursuer_cbf:
            slack_specs.append(("pursuer_pursuer", self.pursuer_pursuer_cbf_slack_weight))
        if self.enable_convex_hull_containment:
            slack_specs.append(("convex_hull", self.convex_hull_slack_weight))

        slack_indices = {}
        n_dec = self.pursuer.nu + len(slack_specs)
        p_diag = np.ones(n_dec)
        next_idx = self.pursuer.nu
        for slack_name, slack_weight in slack_specs:
            slack_indices[slack_name] = next_idx
            p_diag[next_idx] = float(slack_weight)
            next_idx += 1
        P = sparse.diags(p_diag).tocsc()
        q = np.hstack([-u_des, np.zeros(n_dec - self.pursuer.nu)])
        return u_des, n_dec, slack_indices, P, q

    def _build_cbf_box_constraints(self, n_dec):
        a_box = sparse.eye(n_dec).tolil()
        num_slacks = n_dec - self.pursuer.nu
        l_box = np.hstack([np.tile([-self.a_max, -self.a_max], self.N_p), np.zeros(num_slacks)])
        u_box = np.hstack([np.tile([self.a_max, self.a_max], self.N_p), np.full(num_slacks, np.inf)])
        return a_box.tocsc(), l_box, u_box

    def _build_cbf_velocity_constraints(self, x_current, n_dec):
        a_vel = sparse.lil_matrix((self.pursuer.nu, n_dec))
        l_vel = np.zeros(self.pursuer.nu)
        u_vel = np.zeros(self.pursuer.nu)
        for i in range(self.N_p):
            ctrl_idx = i * 2
            v_i = x_current[i * 4 + 2 : i * 4 + 4]
            a_vel[ctrl_idx, ctrl_idx] = self.dt
            a_vel[ctrl_idx + 1, ctrl_idx + 1] = self.dt
            l_vel[ctrl_idx : ctrl_idx + 2] = -self.v_max - v_i
            u_vel[ctrl_idx : ctrl_idx + 2] = self.v_max - v_i
        return a_vel.tocsc(), l_vel, u_vel

    def _manual_barrier_row_terms(self, dp, dv, a_max, d_safe, gamma):
        """Hand-derived primal barrier approximation used in one-step OSQP filters."""
        eps = 1e-5
        pd = np.asarray(dp, dtype=float).reshape(2)
        vd = np.asarray(dv, dtype=float).reshape(2)
        c = pd + vd * self.dt
        c_norm = max(np.linalg.norm(c), eps)
        pd_norm = max(np.linalg.norm(pd), eps)

        h_next = (
            np.dot(c, vd) / c_norm
            + np.sqrt(abs(a_max) * max(c_norm - d_safe, 0.0))
        )
        h_now = (
            np.dot(pd, vd) / pd_norm
            + np.sqrt(abs(a_max) * max(pd_norm - d_safe, 0.0))
        )
        h_const = h_next - (1.0 - gamma) * h_now
        h_u = c * self.dt / c_norm
        return float(h_const), np.asarray(h_u, dtype=float)

    def _build_pursuer_evader_cbf_constraints(self, x_current, evader_state, n_dec, slack_idx):
        a_pe = sparse.lil_matrix((self.N_p, n_dec))
        l_pe = np.full(self.N_p, -np.inf)
        u_pe = np.zeros(self.N_p)

        p_e = evader_state[0:2]
        v_e = evader_state[2:4]
        for i in range(self.N_p):
            ctrl_idx = i * 2
            p_i = x_current[i * 4 : i * 4 + 2]
            v_i = x_current[i * 4 + 2 : i * 4 + 4]

            dp = p_i - p_e
            dv = v_i - v_e
            h_const, h_u = self._manual_barrier_row_terms(
                dp,
                dv,
                self.a_max,
                self.D_safe_pursuer_evader,
                self.gamma_evader_cbf,
            )

            a_pe[i, ctrl_idx : ctrl_idx + 2] = -h_u
            a_pe[i, slack_idx] = -1.0
            u_pe[i] = h_const

        return a_pe.tocsc(), l_pe, u_pe

    def _build_pursuer_pursuer_cbf_constraints(self, x_current, n_dec, slack_idx):
        num_pairs = self.N_p * (self.N_p - 1) // 2
        if num_pairs <= 0:
            return None

        a_cbf = sparse.lil_matrix((num_pairs, n_dec))
        l_cbf = np.full(num_pairs, -np.inf)
        u_cbf = np.zeros(num_pairs)

        pair_idx = 0
        for i in range(self.N_p):
            p_i = x_current[i * 4 : i * 4 + 2]
            v_i = x_current[i * 4 + 2 : i * 4 + 4]
            for j in range(i + 1, self.N_p):
                p_j = x_current[j * 4 : j * 4 + 2]
                v_j = x_current[j * 4 + 2 : j * 4 + 4]

                dp = p_i - p_j
                dv = v_i - v_j
                h_const, h_u = self._manual_barrier_row_terms(
                    dp,
                    dv,
                    self.a_max,
                    self.D_safe_pursuer,
                    self.gamma_pursuer_cbf,
                )

                a_cbf[pair_idx, i * 2 : i * 2 + 2] = -h_u
                a_cbf[pair_idx, j * 2 : j * 2 + 2] = h_u
                a_cbf[pair_idx, slack_idx] = -1.0
                u_cbf[pair_idx] = h_const
                pair_idx += 1

        return a_cbf.tocsc(), l_cbf, u_cbf

    def _build_convex_hull_containment_constraints(
        self, x_current, evader_state, n_dec, hull_slack_idx
    ):
        if not self.enable_convex_hull_containment or self.N_p < 3:
            return None
        if not self.fixed_hull_order or len(self.fixed_hull_order) < 3:
            return None

        x_current = np.asarray(x_current).reshape(4 * self.N_p)
        evader_state = np.asarray(evader_state).reshape(4)
        p_e = evader_state[0:2]

        num_edges = len(self.fixed_hull_order)
        a_hull = sparse.lil_matrix((num_edges, n_dec))
        l_hull = np.full(num_edges, -np.inf)
        u_hull = np.zeros(num_edges)

        u_sym = ca.SX.sym("u", self.pursuer.nu)
        gamma_hull = self.gamma_pursuer_cbf

        for edge_idx, i in enumerate(self.fixed_hull_order):
            j = self.fixed_hull_order[(edge_idx + 1) % num_edges]

            p_i = x_current[i * 4 : i * 4 + 2]
            v_i = x_current[i * 4 + 2 : i * 4 + 4]
            p_j = x_current[j * 4 : j * 4 + 2]
            v_j = x_current[j * 4 + 2 : j * 4 + 4]

            u_i = u_sym[i * 2 : i * 2 + 2]
            u_j = u_sym[j * 2 : j * 2 + 2]

            p_i_next = ca.DM(p_i + self.dt * v_i) + (self.dt ** 2) * u_i
            p_j_next = ca.DM(p_j + self.dt * v_j) + (self.dt ** 2) * u_j
            p_e_dm = ca.DM(p_e)

            edge_vec = p_j_next - p_i_next
            rel_vec = p_e_dm - p_i_next
            h_next_expr = edge_vec[0] * rel_vec[1] - edge_vec[1] * rel_vec[0]
            h_fun = ca.Function(
                f"h_edge_{edge_idx}",
                [u_sym],
                [h_next_expr, ca.jacobian(h_next_expr, u_sym)],
            )

            h_now_vec = p_j - p_i
            rel_now_vec = p_e - p_i
            h_now = h_now_vec[0] * rel_now_vec[1] - h_now_vec[1] * rel_now_vec[0]

            h_next_nom, grad_nom = h_fun(np.zeros(self.pursuer.nu))
            h_next_nom = float(h_next_nom)
            grad_nom = np.asarray(grad_nom).reshape(-1)

            h_const = h_next_nom - (1.0 - gamma_hull) * h_now
            a_hull[edge_idx, : self.pursuer.nu] = -grad_nom
            a_hull[edge_idx, hull_slack_idx] = -1.0
            u_hull[edge_idx] = h_const

        return a_hull.tocsc(), l_hull, u_hull

    def _solve_cbf_qp(self, P, q, a_rows, l_rows, u_rows):
        A = sparse.vstack(a_rows).tocsc()
        l = np.hstack(l_rows)
        u = np.hstack(u_rows)

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False, adaptive_rho=True)
        return prob.solve()

    def _initialize_evader_cbf_qp(self, u_des):
        u_des = np.asarray(u_des).flatten()
        n_dec = self.evader.nu_single + 1
        slack_idx = self.evader.nu_single

        p_diag = np.ones(n_dec)
        p_diag[slack_idx] = self.evader_cbf_slack_weight
        P = sparse.diags(p_diag).tocsc()
        q = np.hstack([-u_des, 0.0])
        return u_des, n_dec, slack_idx, P, q

    def one_step_evader_cbf_filter(self, x_current, evader_state, u_des):
        """One-step evader safety filter applied after the nominal evader solve."""

        x_current = np.asarray(x_current).reshape(4 * self.N_p)
        evader_state = np.asarray(evader_state).reshape(4)
        u_des, n_dec, slack_idx, P, q = self._initialize_evader_cbf_qp(u_des)

        a_rows = []
        l_rows = []
        u_rows = []

        a_box = sparse.eye(n_dec).tolil()
        l_box = np.array([-self.evader_a_max, -self.evader_a_max, 0.0])
        u_box = np.array([self.evader_a_max, self.evader_a_max, np.inf])
        a_rows.append(a_box.tocsc())
        l_rows.append(l_box)
        u_rows.append(u_box)

        a_vel = sparse.lil_matrix((2, n_dec))
        a_vel[0, 0] = self.dt
        a_vel[1, 1] = self.dt
        l_vel = np.array(
            [-self.evader_v_max - evader_state[2], -self.evader_v_max - evader_state[3]]
        )
        u_vel = np.array(
            [self.evader_v_max - evader_state[2], self.evader_v_max - evader_state[3]]
        )
        a_rows.append(a_vel.tocsc())
        l_rows.append(l_vel)
        u_rows.append(u_vel)

        a_cbf = sparse.lil_matrix((self.N_p, n_dec))
        l_cbf = np.full(self.N_p, -np.inf)
        u_cbf = np.zeros(self.N_p)

        p_e = evader_state[0:2]
        v_e = evader_state[2:4]
        for i in range(self.N_p):
            p_i = x_current[i * 4 : i * 4 + 2]
            v_i = x_current[i * 4 + 2 : i * 4 + 4]
            dp_now = p_e - p_i
            dv_now = v_e - v_i
            h_const, h_u = self._manual_barrier_row_terms(
                dp_now,
                dv_now,
                self.evader_a_max,
                self.D_safe_evader,
                self.gamma_evader_cbf,
            )

            a_cbf[i, 0:2] = -h_u
            a_cbf[i, slack_idx] = -1.0
            u_cbf[i] = h_const

        a_rows.append(a_cbf.tocsc())
        l_rows.append(l_cbf)
        u_rows.append(u_cbf)

        res = self._solve_cbf_qp(P, q, a_rows, l_rows, u_rows)
        if res.info.status not in ("solved", "solved inaccurate") or res.x is None:
            return np.clip(u_des, -self.evader_a_max, self.evader_a_max), np.nan

        return res.x[: self.evader.nu_single], res.x[slack_idx]

    def one_step_cbf_filter(self, x_current, evader_state, u_des):
        """Outer pursuer safety filter with optional pursuer-pursuer and pursuer-evader CBFs."""

        if not (
            self.enable_pursuer_pursuer_cbf
            or self.enable_pursuer_evader_cbf
            or self.enable_convex_hull_containment
        ):
            u_clip = np.clip(np.asarray(u_des), -self.a_max, self.a_max)
            return u_clip, 0.0

        evader_state = np.asarray(evader_state).reshape(4)
        u_des, n_dec, slack_indices, P, q = self._initialize_cbf_qp(u_des)

        a_rows = []
        l_rows = []
        u_rows = []

        a_box, l_box, u_box = self._build_cbf_box_constraints(n_dec)
        a_rows.append(a_box.tocsc())
        l_rows.append(l_box)
        u_rows.append(u_box)

        a_vel, l_vel, u_vel = self._build_cbf_velocity_constraints(x_current, n_dec)
        a_rows.append(a_vel)
        l_rows.append(l_vel)
        u_rows.append(u_vel)

        if self.enable_pursuer_evader_cbf:
            a_pe, l_pe, u_pe = self._build_pursuer_evader_cbf_constraints(
                x_current, evader_state, n_dec, slack_indices["pursuer_evader"]
            )
            a_rows.append(a_pe)
            l_rows.append(l_pe)
            u_rows.append(u_pe)

        if self.enable_pursuer_pursuer_cbf:
            pursuer_cbf_block = self._build_pursuer_pursuer_cbf_constraints(
                x_current, n_dec, slack_indices["pursuer_pursuer"]
            )
            if pursuer_cbf_block is not None:
                a_cbf, l_cbf, u_cbf = pursuer_cbf_block
                a_rows.append(a_cbf)
                l_rows.append(l_cbf)
                u_rows.append(u_cbf)

        if self.enable_convex_hull_containment:
            hull_block = self._build_convex_hull_containment_constraints(
                x_current, evader_state, n_dec, slack_indices["convex_hull"]
            )
            if hull_block is not None:
                a_hull, l_hull, u_hull = hull_block
                a_rows.append(a_hull)
                l_rows.append(l_hull)
                u_rows.append(u_hull)

        res = self._solve_cbf_qp(P, q, a_rows, l_rows, u_rows)

        if res.info.status not in ("solved", "solved inaccurate") or res.x is None:
            return np.clip(u_des, -self.a_max, self.a_max), np.nan

        slack_values = [res.x[idx] for idx in slack_indices.values()]
        if not slack_values:
            return res.x[: self.pursuer.nu], 0.0
        return res.x[: self.pursuer.nu], float(np.max(slack_values))

    def _debug_pairwise_distances(self, x_current, evader_state):
        x_current = np.asarray(x_current).reshape(4 * self.N_p)
        evader_state = np.asarray(evader_state).reshape(4)

        pursuer_positions = [
            x_current[i * 4 : i * 4 + 2] for i in range(self.N_p)
        ]
        evader_position = evader_state[0:2]

        min_pp = np.inf
        for i in range(self.N_p):
            for j in range(i + 1, self.N_p):
                dist = np.linalg.norm(pursuer_positions[i] - pursuer_positions[j])
                min_pp = min(min_pp, dist)

        min_pe = np.inf
        for i in range(self.N_p):
            dist = np.linalg.norm(pursuer_positions[i] - evader_position)
            min_pe = min(min_pe, dist)

        if not np.isfinite(min_pp):
            min_pp = np.nan
        if not np.isfinite(min_pe):
            min_pe = np.nan

        print(f"  Min pursuer-pursuer distance: {min_pp:.4f}")
        print(f"  Min pursuer-evader distance: {min_pe:.4f}")

    def _debug_evader_cbf_margins(self, X_p_guess, X_e_guess):
        if X_p_guess is None or X_e_guess is None:
            print("  Evader CBF margins unavailable: missing trajectory guess.")
            return

        min_h = np.inf
        for k in range(self.N + 1):
            p_e = X_e_guess[0:2, k]
            for i in range(self.N_p):
                p_i = X_p_guess[i * 4 : i * 4 + 2, k]
                h = np.dot(p_e - p_i, p_e - p_i) - self.D_safe_evader ** 2
                min_h = min(min_h, h)

        print(f"  Min evader CBF h value over guess: {min_h:.6f}")

    def solve_minimax_turn(self, x_current, evader_state, iters=3):
        x_current = np.asarray(x_current).reshape(4 * self.N_p)
        evader_state = np.asarray(evader_state).reshape(4)

        if self.prev_X_e is not None:
            X_e_guess = np.roll(self.prev_X_e, -1, axis=1)
            X_e_guess[:, -1] = X_e_guess[:, -2]
            U_e_guess = np.roll(self.prev_U_e, -1, axis=1)
        else:
            X_e_guess, U_e_guess = self._build_initial_evader_guess(evader_state)

        if self.prev_X_p is not None:
            X_p_guess = self.prev_X_p
        else:
            X_p_guess, _ = self._build_initial_pursuer_guess(x_current)

        if self.prev_U_p is not None:
            U_p_guess = self.prev_U_p
        else:
            _, U_p_guess = self._build_initial_pursuer_guess(x_current)

        try:
            for iteration in range(iters):
                aux_data = self._prepare_shared_aux_data(X_p_guess, X_e_guess)

                self._set_pursuer_solver_params(
                    x_current=x_current,
                    X_e_guess=X_e_guess,
                    U_e_guess=U_e_guess,
                    aux_data=aux_data,
                )
                self.pursuer_opti.set_initial(self.p_vars["X"], X_p_guess)
                self.pursuer_opti.set_initial(self.p_vars["U"], U_p_guess)

                try:
                    sol_p = self.pursuer_opti.solve()
                except RuntimeError as exc:
                    print(f"Pursuer solve failed at best-response iteration {iteration}.")
                    print(f"  Exception: {exc}")
                    self._debug_pairwise_distances(x_current, evader_state)
                    print(
                        "  Current pursuer guess first state:",
                        np.array2string(X_p_guess[:, 0], precision=3),
                    )
                    print(
                        "  Current evader guess first state:",
                        np.array2string(X_e_guess[:, 0], precision=3),
                    )
                    try:
                        x_debug = self.pursuer_opti.debug.value(self.p_vars["X"])
                        u_debug = self.pursuer_opti.debug.value(self.p_vars["U"])
                        print(
                            "  Pursuer debug X first column:",
                            np.array2string(x_debug[:, 0], precision=3),
                        )
                        print(
                            "  Pursuer debug U first column:",
                            np.array2string(u_debug[:, 0], precision=3),
                        )
                    except Exception as debug_exc:
                        print(f"  Unable to read pursuer debug values: {debug_exc}")
                    raise
                X_p_guess = sol_p.value(self.p_vars["X"])
                U_p_guess = sol_p.value(self.p_vars["U"])

                self._set_evader_solver_params(
                    evader_state=evader_state,
                    X_p_guess=X_p_guess,
                    U_p_guess=U_p_guess,
                    aux_data=aux_data,
                )
                self.evader_opti.set_initial(self.e_vars["X"], X_e_guess)
                self.evader_opti.set_initial(self.e_vars["U"], U_e_guess)

                try:
                    sol_e = self.evader_opti.solve()
                except RuntimeError as exc:
                    print(f"Evader solve failed at best-response iteration {iteration}.")
                    print(f"  Exception: {exc}")
                    self._debug_pairwise_distances(x_current, evader_state)
                    self._debug_evader_cbf_margins(X_p_guess, X_e_guess)
                    print(
                        "  Current evader guess first state:",
                        np.array2string(X_e_guess[:, 0], precision=3),
                    )
                    print(
                        "  Current pursuer plan first state:",
                        np.array2string(X_p_guess[:, 0], precision=3),
                    )
                    try:
                        x_debug = self.evader_opti.debug.value(self.e_vars["X"])
                        u_debug = self.evader_opti.debug.value(self.e_vars["U"])
                        print(
                            "  Evader debug X first column:",
                            np.array2string(x_debug[:, 0], precision=3),
                        )
                        print(
                            "  Evader debug U first column:",
                            np.array2string(u_debug[:, 0], precision=3),
                        )
                    except Exception as debug_exc:
                        print(f"  Unable to read evader debug values: {debug_exc}")
                    raise
                X_e_guess = sol_e.value(self.e_vars["X"])
                U_e_guess = sol_e.value(self.e_vars["U"])
        except RuntimeError as exc:
            print(f"Solver failed to converge in Nash Iteration: {exc}")
            return None, None, None, None

        self.prev_X_p, self.prev_U_p = X_p_guess, U_p_guess
        self.prev_X_e, self.prev_U_e = X_e_guess, U_e_guess

        return X_p_guess, U_p_guess, X_e_guess, U_e_guess

    def solve_minimax(self, x_current, evader_state, num_best_response_iters=3):
        X_p_guess, U_p_guess, X_e_guess, U_e_guess = self.solve_minimax_turn(
            x_current=x_current,
            evader_state=evader_state,
            iters=num_best_response_iters,
        )
        if X_p_guess is None:
            return None, None, None, None

        return X_p_guess.T, U_p_guess.T, X_e_guess.T, U_e_guess.T


class PursuitEvasionMinimaxSolver(BaseMinimaxSolver):
    """Pure zero-sum shared-cost minimax solver for pursuit-evasion."""

    supports_one_step_cbf = True

    def __init__(
        self,
        pursuer_agent,
        evader_agent,
        horizon,
        capture_radius,
        v_max=None,
        a_max=None,
        enable_pursuer_pursuer_cbf=True,
        enable_pursuer_evader_cbf=True,
        enable_convex_hull_containment=False,
        D_safe_pursuer=3.0,
        D_safe_pursuer_evader=3.0,
        D_safe_evader=3.0,
        gamma_pursuer_cbf=1.0,
        gamma_evader_cbf=0.8,
        pursuer_pursuer_cbf_slack_weight=1e5,
        pursuer_evader_cbf_slack_weight=1e5,
        evader_cbf_slack_weight=1e5,
        cutoff_mode="geometric",
        **kwargs,
    ):
        super().__init__(
            pursuer_agent=pursuer_agent,
            evader_agent=evader_agent,
            horizon=horizon,
            capture_radius=capture_radius,
            v_max=v_max,
            a_max=a_max,
            enable_pursuer_pursuer_cbf=enable_pursuer_pursuer_cbf,
            enable_pursuer_evader_cbf=enable_pursuer_evader_cbf,
            enable_convex_hull_containment=enable_convex_hull_containment,
            D_safe_pursuer=D_safe_pursuer,
            D_safe_pursuer_evader=D_safe_pursuer_evader,
            D_safe_evader=D_safe_evader,
            gamma_pursuer_cbf=gamma_pursuer_cbf,
            gamma_evader_cbf=gamma_evader_cbf,
            pursuer_pursuer_cbf_slack_weight=pursuer_pursuer_cbf_slack_weight,
            pursuer_evader_cbf_slack_weight=pursuer_evader_cbf_slack_weight,
            evader_cbf_slack_weight=evader_cbf_slack_weight,
            **kwargs,
        )
        self.cutoff_mode = str(cutoff_mode)
        if self.cutoff_mode not in ("linear", "geometric"):
            raise ValueError(
                f"Unsupported cutoff_mode '{self.cutoff_mode}'. "
                "Expected 'linear' or 'geometric'."
            )
        self.cutoff_eps = 1e-6
        self.pursuer_qp_solver = "osqp"

        # Shared zero-sum payoff weights.
        self.w_e = 5.0
        self.w_c = 10.0
        self.k_p = 1.0
        self.w_up = 0.5
        self.w_ue = 0.5

        self.pursuer_opti, self.p_vars, self.p_params = self._build_pursuer_solver()
        self.evader_opti, self.e_vars, self.e_params = self._build_evader_solver()

    def _projector_matrix_from_param(self, P_perp_param, agent_idx, k):
        """Read a frozen 2x2 transverse projector for one pursuer/stage."""
        base = 4 * agent_idx
        return ca.vertcat(
            ca.horzcat(P_perp_param[base + 0, k], P_perp_param[base + 1, k]),
            ca.horzcat(P_perp_param[base + 2, k], P_perp_param[base + 3, k]),
        )

    def _compute_frozen_projectors(self, X_p_bar, X_e_bar):
        """Freeze line-of-sight projectors from the previous trajectory guess."""
        X_p_bar = np.asarray(X_p_bar)
        X_e_bar = np.asarray(X_e_bar)
        P_perp = np.zeros((4 * self.N_p, self.N + 1))

        for k in range(self.N + 1):
            p_e = X_e_bar[0:2, k]
            for i in range(self.N_p):
                p_i = X_p_bar[i * 4 : i * 4 + 2, k]
                los = p_e - p_i
                los_norm = max(np.linalg.norm(los), self.cutoff_eps)
                r_hat = los / los_norm
                proj = np.eye(2) - np.outer(r_hat, r_hat)

                base = 4 * i
                P_perp[base + 0, k] = proj[0, 0]
                P_perp[base + 1, k] = proj[0, 1]
                P_perp[base + 2, k] = proj[1, 0]
                P_perp[base + 3, k] = proj[1, 1]

        return P_perp

    def _shared_cutoff_target(self, p_i, p_e, v_e):
        return v_e + self.k_p * (p_e - p_i)

    def _prepare_shared_aux_data(self, X_p_guess, X_e_guess):
        return self._compute_frozen_projectors(X_p_guess, X_e_guess)

    def _set_pursuer_solver_params(
        self, x_current, X_e_guess, U_e_guess, aux_data
    ):
        super()._set_pursuer_solver_params(x_current, X_e_guess, U_e_guess, aux_data)
        self.pursuer_opti.set_value(self.p_params["P_perp"], aux_data)

    def _set_evader_solver_params(
        self, evader_state, X_p_guess, U_p_guess, aux_data
    ):
        super()._set_evader_solver_params(evader_state, X_p_guess, U_p_guess, aux_data)
        self.evader_opti.set_value(self.e_params["P_perp"], aux_data)

    def _build_shared_cost(self, X_p, U_p, X_e, U_e, P_perp_param=None):
        """Strictly shared minimax cost: pursuer minimizes, evader maximizes."""
        J = 0
        for k in range(self.N):
            J += self.w_up * ca.sumsqr(U_p[:, k])
            J -= self.w_ue * ca.sumsqr(U_e[:, k])

        for k in range(1, self.N + 1):
            p_e = X_e[0:2, k]
            v_e = X_e[2:4, k]

            for i in range(self.N_p):
                p_i = X_p[i * 4 : i * 4 + 2, k]
                v_i = X_p[i * 4 + 2 : i * 4 + 4, k]

                J += self.w_e * ca.sumsqr(p_i - p_e)

                if self.cutoff_mode == "geometric":
                    if P_perp_param is None:
                        raise ValueError(
                            "Geometric cutoff mode requires frozen projector parameters."
                        )
                    P_perp = self._projector_matrix_from_param(P_perp_param, i, k)
                    v_error = ca.mtimes(P_perp, (v_i - v_e))
                    J += self.w_c * ca.sumsqr(v_error)
                else:
                    v_cutoff = self._shared_cutoff_target(p_i, p_e, v_e)
                    J += self.w_c * ca.sumsqr(v_i - v_cutoff)

        return J

    def _build_pursuer_solver(self):
        opti = ca.Opti("conic")

        X_p = opti.variable(4 * self.N_p, self.N + 1)
        U_p = opti.variable(2 * self.N_p, self.N)

        X_e_param = opti.parameter(4, self.N + 1)
        U_e_param = opti.parameter(2, self.N)
        P_perp_param = opti.parameter(4 * self.N_p, self.N + 1)
        x0_p_param = opti.parameter(4 * self.N_p)

        opti.subject_to(X_p[:, 0] == x0_p_param)

        for k in range(self.N):
            for i in range(self.N_p):
                idx_x = i * 4
                idx_u = i * 2

                px, py = X_p[idx_x, k], X_p[idx_x + 1, k]
                vx, vy = X_p[idx_x + 2, k], X_p[idx_x + 3, k]
                ax, ay = U_p[idx_u, k], U_p[idx_u + 1, k]

                vx_next = vx + self.dt * ax
                vy_next = vy + self.dt * ay
                px_next = px + self.dt * vx_next
                py_next = py + self.dt * vy_next

                opti.subject_to(
                    X_p[idx_x : idx_x + 4, k + 1]
                    == ca.vertcat(px_next, py_next, vx_next, vy_next)
                )
                opti.subject_to(opti.bounded(-self.a_max, ax, self.a_max))
                opti.subject_to(opti.bounded(-self.a_max, ay, self.a_max))
                opti.subject_to(opti.bounded(-self.v_max, vx_next, self.v_max))
                opti.subject_to(opti.bounded(-self.v_max, vy_next, self.v_max))

        J = self._build_shared_cost(X_p, U_p, X_e_param, U_e_param, P_perp_param)
        opti.minimize(J)

        qpsol_opts = {"verbose": False}
        opti.solver(self.pursuer_qp_solver, qpsol_opts)

        return (
            opti,
            {"X": X_p, "U": U_p},
            {
                "X_e": X_e_param,
                "U_e": U_e_param,
                "P_perp": P_perp_param,
                "x0": x0_p_param,
            },
        )

    def _build_evader_solver(self):
        opti = ca.Opti()

        X_e = opti.variable(4, self.N + 1)
        U_e = opti.variable(2, self.N)

        X_p_param = opti.parameter(4 * self.N_p, self.N + 1)
        U_p_param = opti.parameter(2 * self.N_p, self.N)
        P_perp_param = opti.parameter(4 * self.N_p, self.N + 1)
        x0_e_param = opti.parameter(4)

        opti.subject_to(X_e[:, 0] == x0_e_param)

        for k in range(self.N):
            px, py = X_e[0, k], X_e[1, k]
            vx, vy = X_e[2, k], X_e[3, k]
            ax, ay = U_e[0, k], U_e[1, k]

            vx_next = vx + self.dt * ax
            vy_next = vy + self.dt * ay
            px_next = px + self.dt * vx_next
            py_next = py + self.dt * vy_next

            opti.subject_to(
                X_e[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next)
            )
            opti.subject_to(opti.bounded(-self.evader_a_max, ax, self.evader_a_max))
            opti.subject_to(opti.bounded(-self.evader_a_max, ay, self.evader_a_max))
            opti.subject_to(opti.bounded(-self.evader_v_max, vx_next, self.evader_v_max))
            opti.subject_to(opti.bounded(-self.evader_v_max, vy_next, self.evader_v_max))

        J = self._build_shared_cost(X_p_param, U_p_param, X_e, U_e, P_perp_param)
        opti.minimize(-J)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 500, "print_level": 0, "sb": "yes"}
        opti.solver("ipopt", p_opts, s_opts)

        return (
            opti,
            {"X": X_e, "U": U_e},
            {
                "X_p": X_p_param,
                "U_p": U_p_param,
                "P_perp": P_perp_param,
                "x0": x0_e_param,
            },
        )

class PerimeterDefenseMinimaxSolver(BaseMinimaxSolver):
    """Zero-sum minimax solver for a simple perimeter-defense touchdown game."""

    supports_one_step_cbf = True

    def __init__(
        self,
        pursuer_agent,
        evader_agent,
        horizon,
        capture_radius,
        defense_center=None,
        defense_polygon=None,
        defended_shape_points=None,
        enable_convex_hull_containment=False,
        **kwargs,
    ):
        self.defense_center = (
            None if defense_center is None else np.asarray(defense_center, dtype=float).reshape(2)
        )
        self.defense_polygon = (
            None
            if defense_polygon is None
            else np.asarray(defense_polygon, dtype=float).reshape(-1, 2)
        )
        self.defended_shape_points = (
            None
            if defended_shape_points is None
            else np.asarray(defended_shape_points, dtype=float).reshape(-1, 2)
        )

        # Shared zero-sum payoff weights for touchdown denial.
        self.w_progress = 5.0
        self.w_terminal_progress = 40.0
        self.w_line = 6.0
        self.w_terminal_line = 12.0
        self.w_block = 10.0
        self.w_terminal_block = 20.0

        super().__init__(
            pursuer_agent=pursuer_agent,
            evader_agent=evader_agent,
            horizon=horizon,
            capture_radius=capture_radius,
            cutoff_mode="linear",
            enable_convex_hull_containment=enable_convex_hull_containment,
            **kwargs,
        )
        self.w_up = 0.5
        self.w_ue = 0.5

        self.pursuer_opti, self.p_vars, self.p_params = self._build_pursuer_solver()
        self.evader_opti, self.e_vars, self.e_params = self._build_evader_solver()

    @staticmethod
    def _closest_point_on_segment(point, start, end):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return start.copy()
        alpha = np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0)
        return start + alpha * segment

    def _polygon_perimeter_data(self):
        polygon = np.asarray(self.defense_polygon, dtype=float)
        closed_polygon = np.vstack([polygon, polygon[0]])
        edge_vectors = np.diff(closed_polygon, axis=0)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        cumulative_lengths = np.concatenate([[0.0], np.cumsum(edge_lengths)])
        perimeter = float(cumulative_lengths[-1])
        return polygon, closed_polygon, edge_vectors, edge_lengths, cumulative_lengths, perimeter

    def _point_at_polygon_arclength(
        self, distance_along, polygon, edge_vectors, edge_lengths, cumulative_lengths, perimeter
    ):
        wrapped_distance = np.mod(distance_along, perimeter)
        edge_idx = np.searchsorted(cumulative_lengths[1:], wrapped_distance, side="right")
        edge_start = polygon[edge_idx]
        edge_length = max(edge_lengths[edge_idx], 1e-12)
        offset = wrapped_distance - cumulative_lengths[edge_idx]
        return edge_start + (offset / edge_length) * edge_vectors[edge_idx]

    def _nearest_points_on_curve(self, X_e_guess, curve_points):
        curve = np.asarray(curve_points, dtype=float)
        closed_curve = np.vstack([curve, curve[0]])
        nearest_points = np.zeros((2, self.N + 1), dtype=float)
        for k in range(self.N + 1):
            point = np.asarray(X_e_guess[0:2, k], dtype=float)
            best_point = curve[0]
            best_dist_sq = np.inf
            for edge_idx in range(len(curve)):
                candidate = self._closest_point_on_segment(
                    point,
                    closed_curve[edge_idx],
                    closed_curve[edge_idx + 1],
                )
                dist_sq = float(np.sum((candidate - point) ** 2))
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_point = candidate
            nearest_points[:, k] = best_point
        return nearest_points

    def _nearest_boundary_target_sets(self, X_e_guess):
        polygon = np.asarray(self.defense_polygon, dtype=float)
        (
            _,
            closed_polygon,
            edge_vectors,
            edge_lengths,
            cumulative_lengths,
            perimeter,
        ) = self._polygon_perimeter_data()
        boundary_targets = np.zeros((2 * self.N_p, self.N + 1), dtype=float)
        spacing = perimeter / max(16 * self.N_p, 1)
        centered_offsets = spacing * (
            np.arange(self.N_p, dtype=float) - 0.5 * (self.N_p - 1)
        )

        for k in range(self.N + 1):
            point = np.asarray(X_e_guess[0:2, k], dtype=float)
            best_point = polygon[0]
            best_dist_sq = np.inf
            best_distance_along = 0.0
            for edge_idx in range(len(polygon)):
                candidate = self._closest_point_on_segment(
                    point,
                    closed_polygon[edge_idx],
                    closed_polygon[edge_idx + 1],
                )
                dist_sq = float(np.sum((candidate - point) ** 2))
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_point = candidate
                    edge_offset = float(
                        np.linalg.norm(candidate - closed_polygon[edge_idx])
                    )
                    best_distance_along = cumulative_lengths[edge_idx] + edge_offset

            for i, offset in enumerate(centered_offsets):
                target = self._point_at_polygon_arclength(
                    best_distance_along + offset,
                    polygon,
                    edge_vectors,
                    edge_lengths,
                    cumulative_lengths,
                    perimeter,
                )
                boundary_targets[2 * i : 2 * i + 2, k] = target
        return boundary_targets

    def _prepare_shared_aux_data(self, X_p_guess, X_e_guess):
        del X_p_guess
        return {
            "boundary_targets": self._nearest_boundary_target_sets(X_e_guess),
            "progress_targets": self._nearest_points_on_curve(
                X_e_guess,
                self.defended_shape_points,
            ),
        }

    def _set_pursuer_solver_params(
        self, x_current, X_e_guess, U_e_guess, aux_data
    ):
        super()._set_pursuer_solver_params(x_current, X_e_guess, U_e_guess, aux_data)
        self.pursuer_opti.set_value(
            self.p_params["boundary_targets"],
            aux_data["boundary_targets"],
        )
        self.pursuer_opti.set_value(
            self.p_params["progress_targets"],
            aux_data["progress_targets"],
        )

    def _set_evader_solver_params(
        self, evader_state, X_p_guess, U_p_guess, aux_data
    ):
        super()._set_evader_solver_params(evader_state, X_p_guess, U_p_guess, aux_data)
        self.evader_opti.set_value(
            self.e_params["boundary_targets"],
            aux_data["boundary_targets"],
        )
        self.evader_opti.set_value(
            self.e_params["progress_targets"],
            aux_data["progress_targets"],
        )

    def _build_shared_cost(
        self,
        X_p,
        U_p,
        X_e,
        U_e,
        boundary_targets,
        progress_targets,
        P_perp_param=None,
    ):
        del P_perp_param
        J = 0

        for k in range(self.N):
            J += self.w_up * ca.sumsqr(U_p[:, k])
            J -= self.w_ue * ca.sumsqr(U_e[:, k])
            p_e = X_e[0:2, k]
            J -= self.w_progress * ca.sumsqr(p_e - progress_targets[:, k])

            for i in range(self.N_p):
                p_i = X_p[i * 4 : i * 4 + 2, k]
                target = boundary_targets[2 * i : 2 * i + 2, k]
                J += self.w_line * ca.sumsqr(p_i - target)

        p_e_terminal = X_e[0:2, self.N]
        J -= self.w_terminal_progress * ca.sumsqr(
            p_e_terminal - progress_targets[:, self.N]
        )
        for i in range(self.N_p):
            p_i_terminal = X_p[i * 4 : i * 4 + 2, self.N]
            target = boundary_targets[2 * i : 2 * i + 2, self.N]
            J += self.w_terminal_line * ca.sumsqr(p_i_terminal - target)

        return J

    def _build_pursuer_solver(self):
        opti = ca.Opti("conic")

        X_p = opti.variable(4 * self.N_p, self.N + 1)
        U_p = opti.variable(2 * self.N_p, self.N)

        X_e_param = opti.parameter(4, self.N + 1)
        U_e_param = opti.parameter(2, self.N)
        boundary_targets_param = opti.parameter(2 * self.N_p, self.N + 1)
        progress_targets_param = opti.parameter(2, self.N + 1)
        x0_p_param = opti.parameter(4 * self.N_p)

        opti.subject_to(X_p[:, 0] == x0_p_param)

        for k in range(self.N):
            for i in range(self.N_p):
                idx_x = i * 4
                idx_u = i * 2

                px, py = X_p[idx_x, k], X_p[idx_x + 1, k]
                vx, vy = X_p[idx_x + 2, k], X_p[idx_x + 3, k]
                ax, ay = U_p[idx_u, k], U_p[idx_u + 1, k]

                vx_next = vx + self.dt * ax
                vy_next = vy + self.dt * ay
                px_next = px + self.dt * vx_next
                py_next = py + self.dt * vy_next

                opti.subject_to(
                    X_p[idx_x : idx_x + 4, k + 1]
                    == ca.vertcat(px_next, py_next, vx_next, vy_next)
                )
                opti.subject_to(opti.bounded(-self.a_max, ax, self.a_max))
                opti.subject_to(opti.bounded(-self.a_max, ay, self.a_max))
                opti.subject_to(opti.bounded(-self.v_max, vx_next, self.v_max))
                opti.subject_to(opti.bounded(-self.v_max, vy_next, self.v_max))

        J = self._build_shared_cost(
            X_p,
            U_p,
            X_e_param,
            U_e_param,
            boundary_targets_param,
            progress_targets_param,
        )
        opti.minimize(J)

        qpsol_opts = {"verbose": False}
        opti.solver("osqp", qpsol_opts)

        return (
            opti,
            {"X": X_p, "U": U_p},
            {
                "X_e": X_e_param,
                "U_e": U_e_param,
                "boundary_targets": boundary_targets_param,
                "progress_targets": progress_targets_param,
                "x0": x0_p_param,
            },
        )

    def _build_evader_solver(self):
        opti = ca.Opti()

        X_e = opti.variable(4, self.N + 1)
        U_e = opti.variable(2, self.N)

        X_p_param = opti.parameter(4 * self.N_p, self.N + 1)
        U_p_param = opti.parameter(2 * self.N_p, self.N)
        boundary_targets_param = opti.parameter(2 * self.N_p, self.N + 1)
        progress_targets_param = opti.parameter(2, self.N + 1)
        x0_e_param = opti.parameter(4)

        opti.subject_to(X_e[:, 0] == x0_e_param)

        for k in range(self.N):
            px, py = X_e[0, k], X_e[1, k]
            vx, vy = X_e[2, k], X_e[3, k]
            ax, ay = U_e[0, k], U_e[1, k]

            vx_next = vx + self.dt * ax
            vy_next = vy + self.dt * ay
            px_next = px + self.dt * vx_next
            py_next = py + self.dt * vy_next

            opti.subject_to(
                X_e[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next)
            )
            opti.subject_to(opti.bounded(-self.a_max, ax, self.a_max))
            opti.subject_to(opti.bounded(-self.a_max, ay, self.a_max))
            opti.subject_to(opti.bounded(-self.v_max, vx_next, self.v_max))
            opti.subject_to(opti.bounded(-self.v_max, vy_next, self.v_max))

        J = self._build_shared_cost(
            X_p_param,
            U_p_param,
            X_e,
            U_e,
            boundary_targets_param,
            progress_targets_param,
        )
        opti.minimize(-J)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 500, "print_level": 0, "sb": "yes"}
        opti.solver("ipopt", p_opts, s_opts)

        return (
            opti,
            {"X": X_e, "U": U_e},
            {
                "X_p": X_p_param,
                "U_p": U_p_param,
                "boundary_targets": boundary_targets_param,
                "progress_targets": progress_targets_param,
                "x0": x0_e_param,
            },
        )


Solver = PursuitEvasionMinimaxSolver
SolverPursuitEvasionMinimax = PursuitEvasionMinimaxSolver
SolverPerimeterDefenseMinimax = PerimeterDefenseMinimaxSolver
SolverPureMinimaxSharedCost = PursuitEvasionMinimaxSolver
