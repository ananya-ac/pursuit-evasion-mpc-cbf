"""
Entities — simulation entity library, containing Quadrotor (plus QuadrotorParams
and the Entity base class it inherits from), the only entity the pursuit-evasion
path uses.

Quadrotor owns its physical state, a dynamics model, a path history, AND
the generic CBF/MPC contract that lets the CBF/MPC solvers stay
vehicle-agnostic:

    relative_degree                 -> int   (2: force/torque-actuated)
    state_dim, control_dim          -> int, int
    control_bounds()                -> (lo, hi) arrays, length control_dim
    position_from_state(x)          -> (3,) world position, for ANY state size
    velocity_from_state(x)          -> (3,) world-frame translational velocity (ṗ)
    accel_affine_terms(x)           -> (a_drift, A_control)   [degree-2 only]
                                        p̈ = a_drift(x) + A_control(x) @ u
"""
import casadi as ca
import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
# DYNAMICS MODELS
# Signature: f(state, control, dt) -> next_state
# ═══════════════════════════════════════════════════════════════════════════

def quadrotor_dynamics(state: np.ndarray, control: np.ndarray, dt: float,
                        params: 'QuadrotorParams', max_substep: float = 0.02) -> np.ndarray:
    """
    Nonlinear 3D quadcopter dynamics, quaternion attitude, RK4 integration.

    State   = [p(3), v(3), q(4), w(3)]   (13-dim)
        p       world-frame position (m)
        v       world-frame translational velocity (m/s)
        q       body-to-world unit quaternion [qw, qx, qy, qz]
        w       body-frame angular velocity (rad/s)
    Control = [thrust, tau_x, tau_y, tau_z]   (4-dim, body-frame collective
              thrust + torques)

    Internal substepping (max_substep) keeps attitude/rate coupling stable at the
    MPC's coarse dt.
    """
    n_sub = max(1, int(np.ceil(dt / max_substep)))
    h = dt / n_sub

    def xdot(x):
        return _quadrotor_continuous_dynamics(x, control, params)

    s = state.copy()
    for _ in range(n_sub):
        k1 = xdot(s)
        k2 = xdot(s + 0.5 * h * k1)
        k3 = xdot(s + 0.5 * h * k2)
        k4 = xdot(s + h * k3)
        s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    s[6:10] = _normalize_quaternion(s[6:10])
    return s


# ── Quadrotor math helpers ──────────────────────────────────────────────────

def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm_q = np.linalg.norm(q)
    if norm_q <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm_q


def _quaternion_product(q: np.ndarray, r: np.ndarray) -> np.ndarray:
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


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Body-to-world rotation matrix from unit quaternion [qw, qx, qy, qz]."""
    qw, qx, qy, qz = _normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
        ],
        dtype=float,
    )


def _quaternion_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """q_dot = 0.5 * q ⊗ [0, omega_body]."""
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]], dtype=float)
    return 0.5 * _quaternion_product(q, omega_quat)


def _skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix [v]_x such that [v]_x @ w == v × w."""
    vx, vy, vz = v
    return np.array([
        [0.0, -vz,  vy],
        [vz,  0.0, -vx],
        [-vy,  vx, 0.0],
    ])


def _quaternion_insertion_jacobian(q_bar: np.ndarray) -> np.ndarray:
    """
    D1(q̄) = ∂(q̄ ⊗ Exp(δθ)) / ∂δθ |_{δθ=0}   (4×3).

    Maps a small attitude-error tangent vector δθ into the raw quaternion
    tangent space at q̄ — the "insertion" half of the manifold sandwich used
    to build a reduced (12-dim) linearization from the raw (13-dim) one.
    """
    w_bar, v_bar = q_bar[0], q_bar[1:4]
    D1 = np.zeros((4, 3))
    D1[0, :] = -0.5 * v_bar
    D1[1:4, :] = 0.5 * (w_bar * np.eye(3) + _skew(v_bar))
    return D1


def _quaternion_reduction_jacobian(q_bar: np.ndarray) -> np.ndarray:
    """
    D2(q̄): linearization of δθ ≈ 2·vec(Conj(q̄) ⊗ q) around q=q̄   (3×4).

    Maps a raw quaternion-space quantity (e.g. q̇) back into the 3-dim
    attitude-error tangent space — the "reduction" half of the sandwich.
    Satisfies D2(q̄) @ D1(q̄) == I_3 exactly (uses |q̄| == 1).
    """
    w_bar, v_bar = q_bar[0], q_bar[1:4]
    D2 = np.zeros((3, 4))
    D2[:, 0]  = -2.0 * v_bar
    D2[:, 1:4] = 2.0 * (w_bar * np.eye(3) - _skew(v_bar))
    return D2


def _quadrotor_continuous_dynamics(state: np.ndarray, control: np.ndarray,
                                    params: 'QuadrotorParams') -> np.ndarray:
    """x_dot = f(x, u) for the 13-dim quadrotor state [p, v, q, w]."""
    v = state[3:6]
    q = _normalize_quaternion(state[6:10])
    omega = state[10:13]
    thrust = control[0]
    tau = control[1:4]

    R = _quaternion_to_rotation_matrix(q)
    e3_body = np.array([0.0, 0.0, 1.0])
    gravity_world = np.array([0.0, 0.0, -params.gravity])

    p_dot = v
    v_dot = gravity_world + (thrust / params.mass) * (R @ e3_body)
    q_dot = _quaternion_derivative(q, omega)
    omega_dot = params.inertia_inv @ (tau - np.cross(omega, params.inertia @ omega))

    return np.concatenate([p_dot, v_dot, q_dot, omega_dot])


# ── Quadrotor CasADi symbolic dynamics (for linearize()'s autodiff Jacobian) ─
# Mirrors _quadrotor_continuous_dynamics, built with CasADi ops for ca.jacobian().
# q is not normalized here since linearize() always evaluates at a normalized q̄.

def _casadi_quaternion_product(q, r):
    return ca.vertcat(
        q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
        q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
        q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
        q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0],
    )


def _casadi_quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        ca.horzcat(2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)),
        ca.horzcat(2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)),
    )


def _casadi_quaternion_derivative(q, omega_body):
    omega_quat = ca.vertcat(0.0, omega_body[0], omega_body[1], omega_body[2])
    return 0.5 * _casadi_quaternion_product(q, omega_quat)


def _casadi_cross(a, b):
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _casadi_continuous_dynamics(state, control, params: 'QuadrotorParams'):
    """Symbolic x_dot = f(x, u) for the raw 13-dim [p, v, q, w] state."""
    v = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    thrust = control[0]
    tau = control[1:4]

    R = _casadi_quaternion_to_rotation_matrix(q)
    e3_body = ca.vertcat(0.0, 0.0, 1.0)
    gravity_world = ca.vertcat(0.0, 0.0, -params.gravity)
    inertia_dm = ca.DM(params.inertia)
    inertia_inv_dm = ca.DM(params.inertia_inv)

    p_dot = v
    v_dot = gravity_world + (thrust / params.mass) * (R @ e3_body)
    q_dot = _casadi_quaternion_derivative(q, omega)
    omega_dot = inertia_inv_dm @ (tau - _casadi_cross(omega, inertia_dm @ omega))

    return ca.vertcat(p_dot, v_dot, q_dot, omega_dot)


def _build_quadrotor_casadi_linearize_fun(params: 'QuadrotorParams'):
    """Build (once per Quadrotor instance) the CasADi Function evaluating
    f(x,u) and its raw 13x13/13x4 Jacobians A_full=df/dx, B_full=df/du."""
    x = ca.SX.sym('x', 13)
    u = ca.SX.sym('u', 4)
    f = _casadi_continuous_dynamics(x, u, params)
    A = ca.jacobian(f, x)
    B = ca.jacobian(f, u)
    return ca.Function('quadrotor_linearize', [x, u], [f, A, B])


def _discretize_affine_rk4(A_c: np.ndarray, B_c: np.ndarray, c_c: np.ndarray, dt: float):
    """
    RK4 discretization (a 4th-order Taylor series of exp(dt·M)) of a
    continuous affine model ẋ = A_c@x + B_c@u + c_c, for piecewise-constant
    control over one sample. Augments z = [x; u; 1] so that ż = M@z and
    applies one RK4 step to that linear system, extracting the discrete
    affine map x_{k+1} = A_d@x_k + B_d@u_k + c_d.
    """
    nx = A_c.shape[0]
    nu = B_c.shape[1]

    aug_dim = nx + nu + 1
    M = np.zeros((aug_dim, aug_dim))
    M[:nx, :nx]        = A_c
    M[:nx, nx:nx + nu] = B_c
    M[:nx, -1]         = c_c

    hM  = dt * M
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
    B_d = Phi[:nx, nx:nx + nu]
    c_d = Phi[:nx, -1]
    return A_d, B_d, c_d


# ═══════════════════════════════════════════════════════════════════════════
# ENTITY BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════

class Entity:
    """
    Base class for all simulation entities.

    Owns:
    - Physical state (position, velocity), dynamics model, path history
    - Generic CBF/MPC contract. Base implements the contract for
      relative-degree-1 (kinematic) entities; degree-2 entities override
      accel_affine_terms() and set relative_degree = 2.

    Does NOT own:
    - ROS publishers (the node manages those)
    - Control logic or task data
    """

    relative_degree = 1
    state_dim       = 3
    control_dim     = 3

    def __init__(
        self,
        entity_id:   str,
        start_pose:  list,
        start_vel:   list,
        dynamics_fn,
        dt: float = 0.1,
        start_orientation: list = None,
        start_omega:       list = None,
    ):
        self.id          = entity_id
        self.dt          = dt
        self.dynamics_fn = dynamics_fn

        self.position    = np.array(start_pose, dtype=float)
        self.velocity    = np.array(start_vel,  dtype=float)
        self.orientation = np.array(start_orientation if start_orientation is not None
                                     else [1.0, 0.0, 0.0, 0.0], dtype=float)
        self.omega       = np.array(start_omega if start_omega is not None
                                     else [0.0, 0.0, 0.0], dtype=float)
        self.history     = deque([self.position.copy()], maxlen=3000)

    # ── Accessors ──────────────────────────────────────────────────────────

    def get_current_position(self) -> np.ndarray:
        return self.position.copy()

    def get_current_velocity(self) -> np.ndarray:
        return self.velocity.copy()

    def get_path_traversed(self) -> np.ndarray:
        return np.array(self.history)

    def full_state(self) -> np.ndarray:
        """Raw state vector fed to DYNAMICS / the CBF/MPC contract methods.
        Default: [position, velocity]. Override if the entity tracks more
        (e.g. Quadrotor also folds in orientation/omega)."""
        return np.concatenate([self.position, self.velocity])

    # ── SITL sync ──────────────────────────────────────────────────────────

    def sync_external_state(self, position: np.ndarray, velocity: np.ndarray = None):
        """Override state with ground-truth from Gazebo/MAVROS."""
        self.position = np.array(position, dtype=float)
        if velocity is not None:
            self.velocity = np.array(velocity, dtype=float)
        self.history.append(self.position.copy())

    # ── Physics step ───────────────────────────────────────────────────────

    def update_step(self, control_input=None):
        """Advance state one dt. Called only in simulation mode."""
        if control_input is not None:
            self.velocity = np.array(control_input, dtype=float)
        self.position = self.dynamics_fn(self.position, self.velocity, self.dt)
        self.history.append(self.position.copy())

    # ── Generic CBF/MPC contract (relative-degree-1 default) ───────────────

    def control_bounds(self):
        """(lo, hi) arrays, length control_dim. Override per vehicle."""
        v_max = getattr(self, 'V_MAX', np.inf)
        lo = -np.ones(self.control_dim) * v_max
        hi =  np.ones(self.control_dim) * v_max
        return lo, hi

    def position_from_state(self, state: np.ndarray) -> np.ndarray:
        """Extract world position (3,) from a state vector of any size."""
        return state[:3]

    def velocity_from_state(self, state: np.ndarray) -> np.ndarray:
        """World-frame translational velocity (ṗ) from state. Default (degree-1
        kinematic): instantaneous control IS the velocity."""
        return self.velocity.copy()

    # accel_affine_terms() is intentionally absent from the base class:
    # only entities with relative_degree == 2 implement it.


# ═══════════════════════════════════════════════════════════════════════════
# QUADROTOR PHYSICAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

class QuadrotorParams:
    """Physical parameters for the nonlinear 3D quadcopter model (quaternion attitude)."""

    def __init__(
        self,
        mass:         float = 1.0,
        gravity:      float = 9.81,
        inertia_diag: tuple = (0.02, 0.02, 0.04),
        thrust_min:   float = 0.0,
        thrust_max:   float = 20.0,
        torque_max:   tuple = (1.0, 1.0, 1.0),
    ):
        self.mass         = float(mass)
        self.gravity      = float(gravity)
        self.inertia      = np.diag(np.asarray(inertia_diag, dtype=float))
        self.inertia_inv  = np.linalg.inv(self.inertia)
        self.thrust_min   = float(thrust_min)
        self.thrust_max   = float(thrust_max)
        self.torque_max   = np.asarray(torque_max, dtype=float).reshape(3)
        self.hover_thrust = self.mass * self.gravity


# ═══════════════════════════════════════════════════════════════════════════
# QUADROTOR  —  nonlinear 3D quadcopter, quaternion attitude  (relative degree 2)
# ═══════════════════════════════════════════════════════════════════════════

class Quadrotor(Entity):
    """
    Quadrotor — nonlinear 3D quadcopter with quaternion attitude.

    State   = [p(3), v(3), q(4), w(3)]   (13-dim; split across
              self.position=p, self.velocity=v, self.orientation=q,
              self.omega=w — see Entity base)
    Control = [thrust, tau_x, tau_y, tau_z]   (4-dim)
    """

    relative_degree   = 2
    state_dim         = 13
    control_dim       = 4
    reduced_state_dim = 12    # [p, v, δθ, w] — used by linearize()/linearize_discrete()
    V_MAX             = 5.0  # physical speed limit [m/s], informational

    def __init__(self, entity_id: str, start_pose: list, dt: float = 0.1,
                 params: QuadrotorParams = None):
        self.params = params if params is not None else QuadrotorParams()
        # Max vertical accel margin above hover — braking capacity for the
        # degree-2 braking CBF.
        self.A_MAX  = (self.params.thrust_max / self.params.mass) - self.params.gravity

        def dynamics_fn(state, control, dt_):
            return quadrotor_dynamics(state, control, dt_, self.params)

        self.DYNAMICS = dynamics_fn
        self._casadi_linearize_fun = _build_quadrotor_casadi_linearize_fun(self.params)

        super().__init__(
            entity_id   = entity_id,
            start_pose  = list(start_pose),
            start_vel   = [0.0, 0.0, 0.0],
            dynamics_fn = dynamics_fn,
            dt          = dt,
        )

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_current_position(self) -> np.ndarray:
        return self.position.copy()

    def get_current_orientation(self) -> np.ndarray:
        return self.orientation.copy()

    def get_current_omega(self) -> np.ndarray:
        return self.omega.copy()

    def full_state(self) -> np.ndarray:
        """[p(3), v(3), q(4), w(3)] — matches DYNAMICS/position_from_state/
        velocity_from_state/accel_affine_terms ordering."""
        return np.concatenate([self.position, self.velocity, self.orientation, self.omega])

    # ── Physics step (override to handle split position/velocity/orientation/omega) ─

    def update_step(self, control_input=None):
        """Advance quadrotor state one dt. control_input = [thrust, tau_x, tau_y, tau_z]."""
        control = np.array(control_input if control_input is not None else np.zeros(4))
        full_state = np.concatenate([self.position, self.velocity, self.orientation, self.omega])
        next_state = self.DYNAMICS(full_state, control, self.dt)
        self.position    = next_state[0:3]
        self.velocity    = next_state[3:6]
        self.orientation = next_state[6:10]
        self.omega       = next_state[10:13]
        self.history.append(self.position.copy())

    # ── SITL sync ────────────────────────────────────────────────────────────

    def sync_external_state(self, position: np.ndarray, velocity: np.ndarray = None,
                             orientation: np.ndarray = None, omega: np.ndarray = None):
        """
        Override state with ground-truth from Gazebo/an external estimator.

        position    : [x,y,z] world position
        velocity    : world-frame translational velocity (m/s)
        orientation : body-to-world unit quaternion [qw,qx,qy,qz]
        omega       : body-frame angular velocity (rad/s)
        """
        self.position = np.array(position, dtype=float)
        if velocity is not None:
            self.velocity = np.array(velocity, dtype=float)
        if orientation is not None:
            self.orientation = _normalize_quaternion(np.array(orientation, dtype=float))
        if omega is not None:
            self.omega = np.array(omega, dtype=float)
        self.history.append(self.position.copy())

    # ── Generic CBF/MPC contract (relative degree 2) ────────────────────────

    def control_bounds(self):
        lo = np.concatenate([[self.params.thrust_min], -self.params.torque_max])
        hi = np.concatenate([[self.params.thrust_max],  self.params.torque_max])
        return lo, hi

    def position_from_state(self, state: np.ndarray) -> np.ndarray:
        """state = [p(3), v(3), q(4), w(3)] -> world position (3,)."""
        return state[:3]

    def velocity_from_state(self, state: np.ndarray) -> np.ndarray:
        """World-frame translational velocity ṗ = v (already world-frame, no rotation needed)."""
        return state[3:6]

    def omega_from_state(self, state: np.ndarray) -> np.ndarray:
        """Body-frame angular velocity w."""
        return state[10:13]

    def accel_affine_terms(self, state: np.ndarray):
        """
        Drift/control-affine split of p̈ = v̇ (needed for the braking-distance
        degree-2 CBF): thrust enters v̇ affinely through the current (frozen)
        attitude; torque does not enter v̇ at all — it only drives attitude,
        one relative degree further out.

            v̇ = a_drift(state) + A_control(state) @ [thrust, tau_x, tau_y, tau_z]

        Returns
        -------
        a_drift   : (3,)    v̇ contribution independent of u
        A_control : (3, 4)  v̇ = a_drift + A_control @ u
        """
        q = state[6:10]
        R = _quaternion_to_rotation_matrix(q)

        a_drift = np.array([0.0, 0.0, -self.params.gravity])

        A_control = np.zeros((3, 4))
        A_control[:, 0] = R @ np.array([0.0, 0.0, 1.0]) / self.params.mass

        return a_drift, A_control

    def linearize(self, state: np.ndarray, control: np.ndarray):
        """
        Local error-state linearization about (state, control), for use in a
        linearized-QP (OSQP) MPC. Reduces the raw 13-dim [p,v,q,w] Jacobian
        (from CasADi autodiff) to a well-posed 12-dim [p,v,δθ,w] one via the
        insertion/reduction sandwich (_quaternion_insertion_jacobian /
        _quaternion_reduction_jacobian) — q's 4 components carry only 3 true
        attitude DOF, so differentiating w.r.t. raw q directly would be
        ill-posed and let a linear model drift off the unit sphere.

            ẋ_reduced ≈ A_c @ x_reduced + B_c @ u + c_c     (continuous-time affine)
            x_reduced = [p, v, δθ, w]   (δθ = 0 at this state by construction)

        Returns
        -------
        A_c : (12, 12)
        B_c : (12, 4)
        c_c : (12,)
        """
        state   = np.asarray(state, dtype=float).reshape(13)
        control = np.asarray(control, dtype=float).reshape(4)
        q_bar   = _normalize_quaternion(state[6:10])

        f_full, A_full, B_full = self._casadi_linearize_fun(state, control)
        f_full = np.asarray(f_full).reshape(13)
        A_full = np.asarray(A_full).reshape(13, 13)
        B_full = np.asarray(B_full).reshape(13, 4)

        D1_att = _quaternion_insertion_jacobian(q_bar)
        D2_att = _quaternion_reduction_jacobian(q_bar)

        D1 = np.zeros((13, 12))
        D1[0:6, 0:6]    = np.eye(6)
        D1[6:10, 6:9]   = D1_att
        D1[10:13, 9:12] = np.eye(3)

        D2 = np.zeros((12, 13))
        D2[0:6, 0:6]    = np.eye(6)
        D2[6:9, 6:10]   = D2_att
        D2[9:12, 10:13] = np.eye(3)

        A_c = D2 @ A_full @ D1
        B_c = D2 @ B_full

        x_ref_reduced = np.concatenate([state[0:6], np.zeros(3), state[10:13]])
        c_c = D2 @ f_full - A_c @ x_ref_reduced - B_c @ control

        return A_c, B_c, c_c

    def linearize_discrete(self, state: np.ndarray, control: np.ndarray):
        """
        RK4 discretization of the reduced-state local affine linearization.

        For the continuous local model
            ẋ_reduced = A_c @ x_reduced + B_c @ u + c_c,
        with piecewise-constant control over one MPC sample, augment
            z = [x_reduced; u; 1]
        so that ż = M @ z and apply one RK4 step to that linear system
        (equivalently, a 4th-order Taylor series of exp(dt·M)). The resulting
        discrete affine map is extracted as
            x_reduced_{k+1} = A_d @ x_reduced_k + B_d @ u_k + c_d.

        Returns
        -------
        A_d : (12, 12)
        B_d : (12, 4)
        c_d : (12,)
        """
        A_c, B_c, c_c = self.linearize(state, control)
        nx = 12
        nu = self.control_dim

        aug_dim = nx + nu + 1
        M = np.zeros((aug_dim, aug_dim))
        M[:nx, :nx]        = A_c
        M[:nx, nx:nx + nu] = B_c
        M[:nx, -1]         = c_c

        hM  = self.dt * M
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
        B_d = Phi[:nx, nx:nx + nu]
        c_d = Phi[:nx, -1]
        return A_d, B_d, c_d

    def reduce_state(self, state: np.ndarray) -> np.ndarray:
        """This entity's raw 13-dim state in the 12-dim reduced [p, v, δθ, w]
        coordinates linearize()/linearize_discrete() use (δθ = 0 always)."""
        state = np.asarray(state, dtype=float).reshape(13)
        return np.concatenate([state[0:6], np.zeros(3), state[10:13]])
