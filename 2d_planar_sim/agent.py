import casadi as ca
import numpy as np
import scipy.sparse as sparse
from scipy.integrate import solve_ivp


class Agent:
    PURSUER = "pursuer"
    EVADER = "evader"

    def __init__(
        self,
        agent_type,
        dt,
        count=1,
        v_max=4.0,
        a_max=3.0,
        use_random_policy=False,
    ):
        if agent_type not in (self.PURSUER, self.EVADER):
            raise ValueError(f"Unsupported agent type: {agent_type}")

        self.agent_type = agent_type
        self.dt = dt
        self.count = count
        self.v_max = v_max
        self.a_max = a_max
        self.use_random_policy = use_random_policy

        self.nx_single = 4
        self.nu_single = 2
        self.nx = self.nx_single * count
        self.nu = self.nu_single * count

        ad_single = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        bd_single = np.array(
            [
                [dt * dt, 0],
                [0, dt * dt],
                [dt, 0],
                [0, dt],
            ]
        )

        self.ad_single = ad_single
        self.bd_single = bd_single
        self.Ad = sparse.block_diag([ad_single] * count).tocsc()
        self.Bd = sparse.block_diag([bd_single] * count).tocsc()

    def clip_velocity(self, velocity):
        speed = np.hypot(velocity[0], velocity[1])
        if speed <= self.v_max or speed == 0.0:
            return velocity
        return (velocity / speed) * self.v_max

    def step_single(self, state, control):
        px, py, vx, vy = map(float, state)
        ax, ay = map(float, control)

        vx += ax * self.dt
        vy += ay * self.dt
        px += vx * self.dt
        py += vy * self.dt
        return np.array([px, py, vx, vy])

    def rollout_single(self, initial_state, controls):
        controls = np.asarray(controls).reshape((-1, 2))
        states = np.zeros((controls.shape[0] + 1, self.nx_single))
        traj = np.zeros((2, controls.shape[0] + 1))
        states[0, :] = np.asarray(initial_state).reshape(self.nx_single)
        traj[:, 0] = states[0, 0:2]

        current_state = states[0, :].copy()
        for k, control in enumerate(controls):
            current_state = self.step_single(current_state, control)
            states[k + 1, :] = current_state
            traj[:, k + 1] = current_state[0:2]

        return traj, states

    def constant_velocity_traj(self, state, horizon):
        px, py, vx, vy = np.asarray(state).reshape(self.nx_single)
        traj = np.zeros((2, horizon + 1))
        for k in range(horizon + 1):
            traj[0, k] = px + k * self.dt * vx
            traj[1, k] = py + k * self.dt * vy
        return traj

    def sample_random_controls(self, horizon):
        controls = np.random.uniform(-self.a_max, self.a_max, size=(horizon, self.nu_single))
        norms = np.linalg.norm(controls, axis=1)
        mask = norms > self.a_max
        if np.any(mask):
            controls[mask] = controls[mask] / norms[mask][:, None] * self.a_max
        return controls

    def random_policy_plan(self, state, horizon):
        controls = self.sample_random_controls(horizon)
        traj, states = self.rollout_single(state, controls)
        return traj, states, controls


class QuadcopterAgent:
    """Nonlinear 3D quadcopter model with quaternion attitude and local linearization.

    State x = [p(3), v(3), q(4), w(3)], control u = [collective_thrust, tau_x, tau_y, tau_z].
    """

    PURSUER = "pursuer"
    EVADER = "evader"
    GENERIC = "generic"

    def __init__(
        self,
        dt,
        agent_type=GENERIC,
        count=1,
        mass=1.0,
        gravity=9.81,
        inertia_diag=(0.02, 0.02, 0.04),
        thrust_min=0.0,
        thrust_max=20.0,
        torque_max=(1.0, 1.0, 1.0),
    ):
        if agent_type not in (self.PURSUER, self.EVADER, self.GENERIC):
            raise ValueError(f"Unsupported quadcopter agent type: {agent_type}")

        self.agent_type = agent_type
        self.dt = float(dt)
        self.count = int(count)
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.inertia = np.diag(np.asarray(inertia_diag, dtype=float))
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)
        self.torque_max = np.asarray(torque_max, dtype=float).reshape(3)

        self.nx_single = 13
        self.nu_single = 4
        self.nx = self.nx_single * self.count
        self.nu = self.nu_single * self.count

        self.hover_thrust = self.mass * self.gravity
        self._casadi_linearize_fun = self._build_casadi_linearize_function()

    @staticmethod
    def normalize_quaternion(q):
        q = np.asarray(q, dtype=float).reshape(4)
        norm_q = np.linalg.norm(q)
        if norm_q <= 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm_q

    @staticmethod
    def quaternion_product(q, r):
        qw, qx, qy, qz = q
        rw, rx, ry, rz = r
        return np.array(
            [
                qw * rw - qx * rx - qy * ry - qz * rz,
                qw * rx + qx * rw + qy * rz - qz * ry,
                qw * ry - qx * rz + qy * rw + qz * rx,
                qw * rz + qx * ry - qy * rx + qz * rw,
            ],
            dtype=float,
        )

    @classmethod
    def quaternion_to_rotation_matrix(cls, q):
        qw, qx, qy, qz = cls.normalize_quaternion(q)
        return np.array(
            [
                [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
            ],
            dtype=float,
        )

    @classmethod
    def quaternion_derivative(cls, q, omega_body):
        omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]], dtype=float)
        return 0.5 * cls.quaternion_product(q, omega_quat)

    @staticmethod
    def _casadi_normalize_quaternion(q):
        return q / ca.sqrt(ca.sumsqr(q) + 1e-12)

    @staticmethod
    def _casadi_quaternion_product(q, r):
        return ca.vertcat(
            q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
            q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
            q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
            q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0],
        )

    @classmethod
    def _casadi_quaternion_to_rotation_matrix(cls, q):
        q = cls._casadi_normalize_quaternion(q)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return ca.vertcat(
            ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
            ca.horzcat(2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)),
            ca.horzcat(2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)),
        )

    @classmethod
    def _casadi_quaternion_derivative(cls, q, omega_body):
        omega_quat = ca.vertcat(0.0, omega_body[0], omega_body[1], omega_body[2])
        return 0.5 * cls._casadi_quaternion_product(q, omega_quat)

    @staticmethod
    def _casadi_cross(a, b):
        return ca.vertcat(
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _casadi_continuous_dynamics(self, state, control):
        v = state[3:6]
        q = state[6:10]
        omega = state[10:13]
        thrust = control[0]
        tau = control[1:4]

        rotation = self._casadi_quaternion_to_rotation_matrix(q)
        e3_body = ca.vertcat(0.0, 0.0, 1.0)
        gravity_world = ca.vertcat(0.0, 0.0, -self.gravity)
        inertia_dm = ca.DM(self.inertia)
        inertia_inv_dm = ca.DM(self.inertia_inv)

        p_dot = v
        v_dot = gravity_world + (thrust / self.mass) * (rotation @ e3_body)
        q_dot = self._casadi_quaternion_derivative(q, omega)
        omega_dot = inertia_inv_dm @ (tau - self._casadi_cross(omega, inertia_dm @ omega))
        return ca.vertcat(p_dot, v_dot, q_dot, omega_dot)

    def _build_casadi_linearize_function(self):
        x = ca.SX.sym("x", self.nx_single)
        u = ca.SX.sym("u", self.nu_single)
        f = self._casadi_continuous_dynamics(x, u)
        A = ca.jacobian(f, x)
        B = ca.jacobian(f, u)
        return ca.Function("quad_linearize", [x, u], [f, A, B])

    def clip_control(self, control):
        control = np.asarray(control, dtype=float).reshape(self.nu_single)
        clipped = control.copy()
        clipped[0] = np.clip(clipped[0], self.thrust_min, self.thrust_max)
        clipped[1:] = np.clip(clipped[1:], -self.torque_max, self.torque_max)
        return clipped

    def continuous_dynamics(self, state, control):
        state = np.asarray(state, dtype=float).reshape(self.nx_single)
        control = self.clip_control(control)

        p = state[0:3]
        v = state[3:6]
        q = self.normalize_quaternion(state[6:10])
        omega = state[10:13]
        thrust = control[0]
        tau = control[1:4]

        del p  # position does not enter the continuous RHS directly

        R = self.quaternion_to_rotation_matrix(q)
        e3_body = np.array([0.0, 0.0, 1.0])
        gravity_world = np.array([0.0, 0.0, -self.gravity])

        p_dot = v
        v_dot = gravity_world + (thrust / self.mass) * (R @ e3_body)
        q_dot = self.quaternion_derivative(q, omega)
        omega_dot = self.inertia_inv @ (tau - np.cross(omega, self.inertia @ omega))

        return np.hstack([p_dot, v_dot, q_dot, omega_dot])

    def step_single(self, state, control):
        state = np.asarray(state, dtype=float).reshape(self.nx_single)
        control = self.clip_control(control)

        def rhs(_, x):
            return self.continuous_dynamics(x, control)

        sol = solve_ivp(
            rhs,
            (0.0, self.dt),
            state,
            method="RK45",
            t_eval=[self.dt],
            vectorized=False,
        )
        if not sol.success:
            raise RuntimeError(f"Quadcopter integration failed: {sol.message}")

        next_state = np.asarray(sol.y[:, -1], dtype=float)
        next_state[6:10] = self.normalize_quaternion(next_state[6:10])
        return next_state

    def rollout_single(self, initial_state, controls):
        controls = np.asarray(controls, dtype=float).reshape((-1, self.nu_single))
        states = np.zeros((controls.shape[0] + 1, self.nx_single))
        states[0, :] = np.asarray(initial_state, dtype=float).reshape(self.nx_single)

        current_state = states[0, :].copy()
        for k, control in enumerate(controls):
            current_state = self.step_single(current_state, control)
            states[k + 1, :] = current_state

        return states

    def linearize(self, state_ref, control_ref, eps=1e-5):
        """Numerically evaluate CasADi Jacobians of x_dot = f(x,u) about (x_ref, u_ref).

        Returns continuous-time affine dynamics:
            x_dot ≈ A_c x + B_c u + c_c
        """
        del eps
        x_ref = np.asarray(state_ref, dtype=float).reshape(self.nx_single)
        u_ref = self.clip_control(control_ref)
        x_ref[6:10] = self.normalize_quaternion(x_ref[6:10])
        f_ref, A_c, B_c = self._casadi_linearize_fun(x_ref, u_ref)
        f_ref = np.asarray(f_ref, dtype=float).reshape(self.nx_single)
        A_c = np.asarray(A_c, dtype=float).reshape(self.nx_single, self.nx_single)
        B_c = np.asarray(B_c, dtype=float).reshape(self.nx_single, self.nu_single)
        c_c = f_ref - A_c @ x_ref - B_c @ u_ref
        return A_c, B_c, c_c

    def linearize_discrete(self, state_ref, control_ref, eps=1e-5):
        """RK4 discretization of the local affine linearization.

        For the continuous local model
            x_dot = A_c x + B_c u + c_c,
        with piecewise-constant control over one sample, augment
            z = [x; u; 1]
        so that z_dot = M z and apply one RK4 step to that linear system.
        The resulting discrete affine map is extracted as
            x_{k+1} = A_d x_k + B_d u_k + c_d.
        """
        A_c, B_c, c_c = self.linearize(state_ref, control_ref, eps=eps)
        nx = self.nx_single
        nu = self.nu_single

        aug_dim = nx + nu + 1
        M = np.zeros((aug_dim, aug_dim), dtype=float)
        M[:nx, :nx] = A_c
        M[:nx, nx : nx + nu] = B_c
        M[:nx, -1] = c_c

        hM = self.dt * M
        hM2 = hM @ hM
        hM3 = hM2 @ hM
        hM4 = hM3 @ hM
        Phi = (
            np.eye(aug_dim)
            + hM
            + 0.5 * hM2
            + (1.0 / 6.0) * hM3
            + (1.0 / 24.0) * hM4
        )

        A_d = Phi[:nx, :nx]
        B_d = Phi[:nx, nx : nx + nu]
        c_d = Phi[:nx, -1]
        return A_d, B_d, c_d
