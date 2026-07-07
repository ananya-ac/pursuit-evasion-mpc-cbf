import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import inspect

from agent import Agent
from solver import PerimeterDefenseMinimaxSolver, PursuitEvasionMinimaxSolver

class Simulation:
    def __init__(
        self,
        num_pursuers=4,
        dt=0.1,
        horizon=60,
        arena_size=100.0,
        capture_radius=5.0,
        D_safe_pursuer=4.1e-1,
        D_safe_pursuer_evader=3.0,
        D_safe_evader=3.0,
        solver_mode="perimeter_defense",
        sim_steps=600,
        config=3,
        num_best_response_iters=2,
        seed=130,
        pursuer_v_max=2.0,
        pursuer_a_max=2.0,
        evader_v_max=2.0,
        evader_a_max=2.0,
        evader_random_policy=False,
        enable_pursuer_pursuer_cbf=False,
        enable_pursuer_evader_cbf=False,
        enable_evader_cbf=False,
        enable_convex_hull_containment=False,
        pursuer_pursuer_cbf_slack_weight=1e7,
        pursuer_evader_cbf_slack_weight=1e-1,
        evader_cbf_slack_weight=1e7,
        convex_hull_slack_weight=1e7,
        contact_tolerance=4e-1,
    ):
        self.seed = seed
        np.random.seed(self.seed)

        self.num_pursuers = num_pursuers
        self.dt = dt
        self.horizon = horizon
        self.arena_size = float(arena_size)
        self.arena_min = 0.0
        self.arena_max = self.arena_size
        self.capture_radius = capture_radius
        self.D_safe_pursuer = D_safe_pursuer
        self.D_safe_pursuer_evader = D_safe_pursuer_evader
        self.D_safe_evader = D_safe_evader
        self.solver_mode = solver_mode
        self.sim_steps = sim_steps
        self.config = config
        self.num_best_response_iters = num_best_response_iters
        self.evader_random_policy = evader_random_policy
        self.enable_pursuer_pursuer_cbf = enable_pursuer_pursuer_cbf
        self.enable_pursuer_evader_cbf = enable_pursuer_evader_cbf
        self.enable_evader_cbf = enable_evader_cbf
        self.enable_convex_hull_containment = enable_convex_hull_containment
        self.pursuer_pursuer_cbf_slack_weight = float(pursuer_pursuer_cbf_slack_weight)
        self.pursuer_evader_cbf_slack_weight = float(pursuer_evader_cbf_slack_weight)
        self.evader_cbf_slack_weight = float(evader_cbf_slack_weight)
        self.convex_hull_slack_weight = convex_hull_slack_weight
        self.contact_tolerance = float(contact_tolerance)
        self.defended_shape_points = None
        self.defense_polygon = None
        self.defense_center = None
        self.defender_spawn_points = None

        self.pursuer_v_max = float(pursuer_v_max)
        self.pursuer_a_max = float(pursuer_a_max)
        self.evader_v_max = float(evader_v_max)
        self.evader_a_max = float(evader_a_max)

        self.pursuer = Agent(
            agent_type=Agent.PURSUER,
            dt=self.dt,
            count=self.num_pursuers,
            v_max=self.pursuer_v_max,
            a_max=self.pursuer_a_max,
        )
        self.evader = Agent(
            agent_type=Agent.EVADER,
            dt=self.dt,
            count=1,
            v_max=self.evader_v_max,
            a_max=self.evader_a_max,
            use_random_policy=self.evader_random_policy,
        )
        self._build_perimeter_geometry()
        self.solver = self._build_solver()
        self.solver_supports_cbf = bool(
            getattr(self.solver, "supports_one_step_cbf", False)
        )

        self.x_current = None
        self.evader_state = None
        self.history_pursuers = []
        self.history_evader = []
        self.history_cbf_slack = []
        self.captured = False
        self.capture_step = None
        self.capture_agent = None
        self.touchdown = False
        self.touchdown_step = None
        self.escaped = False
        self.escape_step = None
        self.collided = False
        self.collision_step = None
        self.collision_agents = None
        self.collision_kind = None
        self.defenders_won = False
        self.defenders_win_step = None
        self.defenders_win_agents = None
        self.solver_failed = False
        self.solver_failure_step = None

    def _build_solver(self):
        solver_cls = self._select_solver_class()
        candidate_kwargs = {
            "pursuer_agent": self.pursuer,
            "evader_agent": self.evader,
            "horizon": self.horizon,
            "capture_radius": self.capture_radius,
            "D_safe_pursuer": self.D_safe_pursuer,
            "D_safe_pursuer_evader": self.D_safe_pursuer_evader,
            "D_safe_evader": self.D_safe_evader,
            "enable_pursuer_pursuer_cbf": self.enable_pursuer_pursuer_cbf,
            "enable_pursuer_evader_cbf": self.enable_pursuer_evader_cbf,
            "enable_convex_hull_containment": self.enable_convex_hull_containment,
            "pursuer_pursuer_cbf_slack_weight": self.pursuer_pursuer_cbf_slack_weight,
            "pursuer_evader_cbf_slack_weight": self.pursuer_evader_cbf_slack_weight,
            "evader_cbf_slack_weight": self.evader_cbf_slack_weight,
            "convex_hull_slack_weight": self.convex_hull_slack_weight,
            "defense_center": self.defense_center,
            "defense_polygon": self.defense_polygon,
            "defended_shape_points": self.defended_shape_points,
        }
        solver_signature = inspect.signature(solver_cls.__init__)
        supported_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in solver_signature.parameters
        }
        return solver_cls(**supported_kwargs)

    def _select_solver_class(self):
        if self.solver_mode == "pursuit_evasion":
            return PursuitEvasionMinimaxSolver
        if self.solver_mode == "perimeter_defense":
            return PerimeterDefenseMinimaxSolver
        raise ValueError(
            "Unsupported solver_mode "
            f"'{self.solver_mode}'. Expected 'pursuit_evasion' or 'perimeter_defense'."
        )

    def _build_perimeter_geometry(self):
        if self.solver_mode != "perimeter_defense":
            return
        center = np.array([0.82 * self.arena_size, 0.82 * self.arena_size], dtype=float)
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        base_radius = 0.08 * self.arena_size
        radial_scale = 1.0 + 0.18 * np.cos(3.0 * angles) + 0.08 * np.sin(5.0 * angles)
        radii = base_radius * radial_scale
        shape = center + np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles)]
        )
        polygon_stride = max(1, len(shape) // 12)
        polygon = center + 1.15 * (shape[::polygon_stride] - center)
        self.defended_shape_points = shape
        self.defense_polygon = polygon
        self.defense_center = center
        self.defender_spawn_points = self._sample_polygon_perimeter(
            polygon,
            self.num_pursuers,
        )

    @staticmethod
    def _sample_polygon_perimeter(polygon, n_points):
        vertices = np.asarray(polygon, dtype=float)
        if n_points <= 0:
            return np.zeros((0, 2), dtype=float)
        closed_vertices = np.vstack([vertices, vertices[0]])
        edge_vectors = np.diff(closed_vertices, axis=0)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        perimeter = float(np.sum(edge_lengths))
        distances = np.linspace(0.0, perimeter, n_points, endpoint=False)
        anchors = []
        cumulative = np.concatenate([[0.0], np.cumsum(edge_lengths)])
        for distance_along in distances:
            edge_idx = np.searchsorted(cumulative[1:], distance_along, side="right")
            edge_start = vertices[edge_idx]
            edge_vector = edge_vectors[edge_idx]
            edge_length = max(edge_lengths[edge_idx], 1e-9)
            offset = distance_along - cumulative[edge_idx]
            anchors.append(edge_start + (offset / edge_length) * edge_vector)
        return np.asarray(anchors, dtype=float)

    @staticmethod
    def _point_in_polygon(point, polygon):
        x, y = point
        vertices = np.asarray(polygon, dtype=float)
        inside = False
        x_prev, y_prev = vertices[-1]
        for x_curr, y_curr in vertices:
            intersects = ((y_curr > y) != (y_prev > y)) and (
                x < (x_prev - x_curr) * (y - y_curr) / max(y_prev - y_curr, 1e-9) + x_curr
            )
            if intersects:
                inside = not inside
            x_prev, y_prev = x_curr, y_curr
        return inside

    def setup_agents(self):
        arena_mid = 0.5 * self.arena_size
        arena_margin = 0.5

        if self.solver_mode == "perimeter_defense":
            evader_pos = np.array(
                [
                    np.random.uniform(0.45 * self.arena_size, 0.65 * self.arena_size),
                    np.random.uniform(0.45 * self.arena_size, 0.65 * self.arena_size),
                ],
                dtype=float,
            )
            angle_e = np.random.rand() * 2 * np.pi
            vx_e = self.evader.v_max * np.cos(angle_e)
            vy_e = self.evader.v_max * np.sin(angle_e)
            x0_list = []
            for spawn_point in self.defender_spawn_points:
                outward_dir = np.asarray(spawn_point, dtype=float) - self.defense_center
                outward_norm = float(np.linalg.norm(outward_dir))
                if outward_norm > 1e-9:
                    outward_dir = outward_dir / outward_norm
                else:
                    outward_dir = np.array([1.0, 0.0], dtype=float)
                spawn_base = np.asarray(spawn_point, dtype=float) + 8.0 * outward_dir
                px = np.clip(
                    spawn_base[0] + np.random.uniform(-0.75, 0.75),
                    arena_margin,
                    self.arena_max - arena_margin,
                )
                py = np.clip(
                    spawn_base[1] + np.random.uniform(-0.75, 0.75),
                    arena_margin,
                    self.arena_max - arena_margin,
                )
                x0_list.append([px, py, 0.0, 0.0])
            self.x_current = np.asarray(x0_list, dtype=float).reshape(-1)
            self.evader_state = np.array([evader_pos[0], evader_pos[1], vx_e, vy_e], dtype=float)
            return

        if self.config == 1:
            px_e, py_e = arena_mid, arena_mid
        elif self.config == 4:
            px_e = np.random.uniform(3.0, 0.3 * self.arena_size)
            py_e = np.random.uniform(0.2 * self.arena_size, 0.8 * self.arena_size)
        else:
            px_e = np.random.uniform(0.25 * self.arena_size, 0.75 * self.arena_size)
            py_e = np.random.uniform(0.25 * self.arena_size, 0.75 * self.arena_size)

        if self.config == 4:
            vx_e = np.random.uniform(0.5 * self.evader.v_max, self.evader.v_max)
            vy_e = np.random.uniform(-0.3 * self.evader.v_max, 0.3 * self.evader.v_max)
        else:
            angle_e = np.random.rand() * 2 * np.pi
            vx_e = self.evader.v_max * np.cos(angle_e)
            vy_e = self.evader.v_max * np.sin(angle_e)

        x0_list = []

        if self.config == 1:
            spawn_radius = 20.0
            for i in range(self.num_pursuers):
                initial_angle = i * (2 * np.pi / self.num_pursuers)
                px = px_e + spawn_radius * np.cos(initial_angle)
                py = py_e + spawn_radius * np.sin(initial_angle)
                x0_list.append([px, py, 0.0, 0.0])
        elif self.config == 2:
            if self.num_pursuers != 4:
                raise ValueError("config=2 currently expects exactly 4 pursuers.")

            pair_half_spacing = 1.25
            min_evader_dist = 8.0
            cluster_radius = np.random.uniform(12.0, 20.0)
            axis_is_horizontal = np.random.rand() < 0.5
            side_jitter = np.random.uniform(-4.0, 4.0)

            if axis_is_horizontal:
                left_center = np.array([px_e - cluster_radius, py_e + side_jitter])
                right_center = np.array([px_e + cluster_radius, py_e - side_jitter])
                pair_offset = np.array([0.0, pair_half_spacing])
            else:
                lower_center = np.array([px_e + side_jitter, py_e - cluster_radius])
                upper_center = np.array([px_e - side_jitter, py_e + cluster_radius])
                left_center = lower_center
                right_center = upper_center
                pair_offset = np.array([pair_half_spacing, 0.0])

            candidate_positions = [
                left_center - pair_offset,
                left_center + pair_offset,
                right_center - pair_offset,
                right_center + pair_offset,
            ]

            clipped_positions = []
            for pos in candidate_positions:
                pos = np.clip(pos, arena_margin, self.arena_max - arena_margin)
                if np.hypot(pos[0] - px_e, pos[1] - py_e) < min_evader_dist:
                    direction = pos - np.array([px_e, py_e])
                    norm = max(np.linalg.norm(direction), 1e-6)
                    pos = np.array([px_e, py_e]) + direction / norm * min_evader_dist
                    pos = np.clip(pos, arena_margin, self.arena_max - arena_margin)
                clipped_positions.append([pos[0], pos[1], 0.0, 0.0])

            x0_list = clipped_positions

            initial_angles = [np.arctan2(p[1] - py_e, p[0] - px_e) for p in x0_list]
            sorted_indices = np.argsort(initial_angles)
            x0_list = [x0_list[idx] for idx in sorted_indices]
        elif self.config == 3:
            min_spawn_dist = 4.0

            for _ in range(self.num_pursuers):
                collision = True
                while collision:
                    px = np.random.uniform(self.arena_min, self.arena_max)
                    py = np.random.uniform(self.arena_min, self.arena_max)
                    collision = False

                    if np.hypot(px - px_e, py - py_e) < min_spawn_dist:
                        collision = True

                    for prev_x in x0_list:
                        if np.hypot(px - prev_x[0], py - prev_x[1]) < min_spawn_dist:
                            collision = True
                            break

                x0_list.append([px, py, 0.0, 0.0])

            initial_angles = [np.arctan2(p[1] - py_e, p[0] - px_e) for p in x0_list]
            sorted_indices = np.argsort(initial_angles)
            x0_list = [x0_list[idx] for idx in sorted_indices]
        elif self.config == 4:
            y_positions = np.linspace(
                0.15 * self.arena_size, 0.85 * self.arena_size, self.num_pursuers
            )
            x_center = 0.35 * self.arena_size
            x_band_half_width = 1.5

            for y in y_positions:
                px = np.clip(
                    x_center + np.random.uniform(-x_band_half_width, x_band_half_width),
                    arena_margin,
                    self.arena_max - arena_margin,
                )
                py = np.clip(
                    y + np.random.uniform(-1.0, 1.0),
                    arena_margin,
                    self.arena_max - arena_margin,
                )
                x0_list.append([px, py, 0.0, 0.0])
        else:
            raise ValueError(f"Unsupported config: {self.config}")

        self.x_current = np.array(x0_list).flatten()
        self.evader_state = np.array([px_e, py_e, vx_e, vy_e], dtype=float)

    def _capture_status(self, tot_pursuers=1):
        p_e = self.evader_state[0:2]
        distances = []

        for i in range(self.num_pursuers):
            p_i = self.x_current[i * 4 : i * 4 + 2]
            dist = np.linalg.norm(p_i - p_e)
            distances.append(dist)

        distances = np.asarray(distances)

        # Captured once at least tot_pursuers are within the capture radius.
        pursuers_in_circle = np.sum(distances <= self.capture_radius)
        is_captured = bool(pursuers_in_circle >= tot_pursuers)
        
        closest_agent = int(np.argmin(distances))
        closest_dist = float(distances[closest_agent])
        
        return is_captured, closest_dist, closest_agent

    def _touchdown_status(self):
        evader_pos = self.evader_state[0:2]
        return self._point_in_polygon(evader_pos, self.defended_shape_points), evader_pos

    def _escape_status(self):
        evader_x = float(self.evader_state[0])
        evader_y = float(self.evader_state[1])
        escaped = not (
            self.arena_min <= evader_x <= self.arena_max
            and self.arena_min <= evader_y <= self.arena_max
        )
        return escaped, evader_x, evader_y

    def _collision_status(self):
        p_e = self.evader_state[0:2]

        for i in range(self.num_pursuers):
            p_i = self.x_current[i * 4 : i * 4 + 2]
            dist_pe = float(np.linalg.norm(p_i - p_e))
            if dist_pe <= self.contact_tolerance:
                return True, "pursuer_evader", (i,), dist_pe

        for i in range(self.num_pursuers):
            p_i = self.x_current[i * 4 : i * 4 + 2]
            for j in range(i + 1, self.num_pursuers):
                p_j = self.x_current[j * 4 : j * 4 + 2]
                dist_pp = float(np.linalg.norm(p_i - p_j))
                if dist_pp <= self.contact_tolerance:
                    return True, "pursuer_pursuer", (i, j), dist_pp

        return False, None, None, None

    def run(self):
        self.setup_agents()
        self.solver.reset_warm_starts()
        self.solver.set_fixed_adjacency_from_state(self.x_current, self.evader_state)
        self.history_pursuers = [self.x_current.copy()]
        self.history_evader = [self.evader_state[0:2].copy()]
        self.history_cbf_slack = []
        self.captured = False
        self.capture_step = None
        self.capture_agent = None
        self.touchdown = False
        self.touchdown_step = None
        self.escaped = False
        self.escape_step = None
        self.collided = False
        self.collision_step = None
        self.collision_agents = None
        self.collision_kind = None
        self.defenders_won = False
        self.defenders_win_step = None
        self.defenders_win_agents = None
        self.solver_failed = False
        self.solver_failure_step = None

        print("Running minimax simulation...")
        for step in range(self.sim_steps):
            collided, collision_kind, collision_agents, collision_dist = self._collision_status()
            if collided:
                if (
                    self.solver_mode == "perimeter_defense"
                    and collision_kind == "pursuer_evader"
                ):
                    self.defenders_won = True
                    self.defenders_win_step = step
                    self.defenders_win_agents = collision_agents
                    print(
                        f"Defenders win before solve at step {step} "
                        f"(defender {collision_agents[0] + 1} contacted attacker; "
                        f"distance={collision_dist:.3f})."
                    )
                    break
                self.collided = True
                self.collision_step = step
                self.collision_agents = collision_agents
                self.collision_kind = collision_kind
                print(
                    f"Swarm lost before solve at step {step} "
                    f"(collision kind={collision_kind}, agents={collision_agents}, "
                    f"distance={collision_dist:.3f})."
                )
                break

            if self.solver_mode == "pursuit_evasion":
                escaped, evader_x, evader_y = self._escape_status()
                if escaped:
                    self.escaped = True
                    self.escape_step = step
                    print(
                        f"Swarm lost before solve at step {step} "
                        f"(evader escaped arena with position=({evader_x:.3f}, {evader_y:.3f}))."
                    )
                    break
                any_captured, closest_dist, closest_agent = self._capture_status()
                if any_captured:
                    self.captured = True
                    self.capture_step = step
                    self.capture_agent = closest_agent
                    print(
                        f"Evader already captured before solve at step {step} "
                        f"(pursuer {closest_agent + 1} within radius; "
                        f"distance={closest_dist:.3f})."
                    )
                    break
            elif self.solver_mode == "perimeter_defense":
                touchdown, evader_pos = self._touchdown_status()
                if touchdown:
                    self.touchdown = True
                    self.touchdown_step = step
                    print(
                        f"Swarm lost before solve at step {step} "
                        f"(evader entered defended region at "
                        f"({evader_pos[0]:.3f}, {evader_pos[1]:.3f}))."
                    )
                    break

            x_p_opt, u_p_opt, _, u_e_plan = self.solver.solve_minimax(
                x_current=self.x_current,
                evader_state=self.evader_state,
                num_best_response_iters=self.num_best_response_iters,
            )

            if x_p_opt is None or u_p_opt is None or u_e_plan is None:
                self.solver_failed = True
                self.solver_failure_step = step
                print(f"Minimax solve failed at step {step}")
                break

            if self.solver_supports_cbf and (
                self.enable_pursuer_pursuer_cbf
                or self.enable_pursuer_evader_cbf
                or self.enable_convex_hull_containment
            ):
                u_safe, eps_used = self.solver.one_step_cbf_filter(
                    x_current=self.x_current,
                    evader_state=self.evader_state,
                    u_des=u_p_opt[0, :],
                )
                self.x_current = self.solver.step_pursuer_dynamics(self.x_current, u_safe)
                self.history_cbf_slack.append(eps_used)
            else:
                self.x_current = x_p_opt[1, :]
            self.history_pursuers.append(self.x_current.copy())

            if self.solver_supports_cbf and self.enable_evader_cbf:
                u_e_safe, _ = self.solver.one_step_evader_cbf_filter(
                    x_current=self.x_current,
                    evader_state=self.evader_state,
                    u_des=u_e_plan[0, :],
                )
            else:
                u_e_safe = u_e_plan[0, :]
            self.evader_state = self.evader.step_single(self.evader_state, u_e_safe)
            self.history_evader.append(self.evader_state[0:2].copy())

            collided, collision_kind, collision_agents, collision_dist = self._collision_status()
            if collided:
                if (
                    self.solver_mode == "perimeter_defense"
                    and collision_kind == "pursuer_evader"
                ):
                    self.defenders_won = True
                    self.defenders_win_step = step
                    self.defenders_win_agents = collision_agents
                    print(
                        f"Defenders win at step {step} "
                        f"(defender {collision_agents[0] + 1} contacted attacker; "
                        f"distance={collision_dist:.3f})."
                    )
                    break
                self.collided = True
                self.collision_step = step
                self.collision_agents = collision_agents
                self.collision_kind = collision_kind
                print(
                    f"Swarm lost at step {step} "
                    f"(collision kind={collision_kind}, agents={collision_agents}, "
                    f"distance={collision_dist:.3f})."
                )
                break

            if self.solver_mode == "pursuit_evasion":
                escaped, evader_x, evader_y = self._escape_status()
                if escaped:
                    self.escaped = True
                    self.escape_step = step
                    print(
                        f"Swarm lost at step {step} "
                        f"(evader escaped arena with position=({evader_x:.3f}, {evader_y:.3f}))."
                    )
                    break
                any_captured, closest_dist, closest_agent = self._capture_status()
                if any_captured:
                    self.captured = True
                    self.capture_step = step
                    self.capture_agent = closest_agent
                    print(
                        f"Evader captured at step {step} "
                        f"(pursuer {closest_agent + 1} within radius; "
                        f"distance={closest_dist:.3f})."
                    )
                    break
            elif self.solver_mode == "perimeter_defense":
                touchdown, evader_pos = self._touchdown_status()
                if touchdown:
                    self.touchdown = True
                    self.touchdown_step = step
                    print(
                        f"Swarm lost at step {step} "
                        f"(evader entered defended region at "
                        f"({evader_pos[0]:.3f}, {evader_pos[1]:.3f}))."
                    )
                    break

        self.history_pursuers = np.array(self.history_pursuers)
        self.history_evader = np.array(self.history_evader)
        print("Simulation complete.")

    def generate_animation(self, video_filename=None):
        if len(self.history_pursuers) <= 1:
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        if self.solver_mode == "perimeter_defense":
            title_fontsize = 22
            label_fontsize = 18
            tick_fontsize = 15
            legend_fontsize = 16
        else:
            title_fontsize = 16
            label_fontsize = 12
            tick_fontsize = 10
            legend_fontsize = 10
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("x", fontsize=label_fontsize)
        ax.set_ylabel("y", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        plot_max = max(self.arena_max, 105.0)
        ax.set_xlim(self.arena_min, plot_max)
        ax.set_ylim(self.arena_min, plot_max)
        arena_border = None
        if self.solver_mode != "perimeter_defense":
            arena_border = plt.Rectangle(
                (self.arena_min, self.arena_min),
                self.arena_size,
                self.arena_size,
                fill=False,
                edgecolor="k",
                linewidth=2,
                linestyle="-",
                alpha=0.8,
            )
            ax.add_patch(arena_border)

        evader_line, = ax.plot([], [], "r--", linewidth=1.5, alpha=0.6)
        evader_label = "Attacker" if self.solver_mode == "perimeter_defense" else "Evader"
        evader_point, = ax.plot([], [], "rX", markersize=12, label=evader_label)
        capture_circle = None
        target_line = None
        true_shape_line = None
        if self.solver_mode == "perimeter_defense":
            polygon_closed = np.vstack([self.defense_polygon, self.defense_polygon[0]])
            target_line, = ax.plot(
                polygon_closed[:, 0],
                polygon_closed[:, 1],
                color="k",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="Defense Polygon",
            )
            shape_closed = np.vstack([self.defended_shape_points, self.defended_shape_points[0]])
            true_shape_line, = ax.plot(
                shape_closed[:, 0],
                shape_closed[:, 1],
                color="0.35",
                linestyle="-",
                linewidth=1.5,
                alpha=0.8,
                label="Defended Shape",
            )
        else:
            capture_circle = plt.Circle(
                (0, 0), self.capture_radius, color="r", fill=False, linestyle="--", alpha=0.5
            )
            ax.add_patch(capture_circle)

        pursuer_lines = []
        pursuer_points = []
        colors = ["b", "g", "c", "m"]

        for i in range(self.num_pursuers):
            color = colors[i % len(colors)]
            line_alpha = 0.35 if self.solver_mode == "perimeter_defense" else 0.7
            line, = ax.plot([], [], f"{color}-", linewidth=2, alpha=line_alpha)
            pursuer_label = (
                f"Defender {i + 1}"
                if self.solver_mode == "perimeter_defense"
                else f"Pursuer {i + 1}"
            )
            point, = ax.plot([], [], f"{color}o", markersize=8, label=pursuer_label)
            pursuer_lines.append(line)
            pursuer_points.append(point)

        legend_loc = "upper left" if self.solver_mode == "perimeter_defense" else "upper right"
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)

        def update(frame):
            evader_line.set_data(
                self.history_evader[:frame + 1, 0], self.history_evader[:frame + 1, 1]
            )
            evader_point.set_data(
                [self.history_evader[frame, 0]], [self.history_evader[frame, 1]]
            )
            if capture_circle is not None:
                capture_circle.set_center(
                    (self.history_evader[frame, 0], self.history_evader[frame, 1])
                )

            for i in range(self.num_pursuers):
                idx = i * 4
                pursuer_lines[i].set_data(
                    self.history_pursuers[:frame + 1, idx],
                    self.history_pursuers[:frame + 1, idx + 1],
                )
                pursuer_points[i].set_data(
                    [self.history_pursuers[frame, idx]],
                    [self.history_pursuers[frame, idx + 1]],
                )

            artists = [evader_line, evader_point] + pursuer_lines + pursuer_points
            if arena_border is not None:
                artists.append(arena_border)
            if capture_circle is not None:
                artists.append(capture_circle)
            if target_line is not None:
                artists.append(target_line)
            if true_shape_line is not None:
                artists.append(true_shape_line)
            return artists

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.history_evader),
            interval=50,
            blit=True,
        )
        if video_filename is None:
            video_filename = (
                f"pursuit_evasion_minimax_{self.num_pursuers}.gif"
                if self.solver_mode == "pursuit_evasion"
                else "perimeter_defense.gif"
            )
        print(f"Saving video to {video_filename}...")
        anim.save(video_filename, writer="pillow", fps=20)
        print(f"Video saved successfully: {video_filename}")

        base_filename = video_filename.rsplit(".", 1)[0]
        if self.solver_mode == "perimeter_defense":
            snapshot_frames = {
                "start": 0,
                "middle": len(self.history_evader) // 2,
                "end": len(self.history_evader) - 1,
            }
            original_ylim = ax.get_ylim()
            cropped_ymin = max(40.0, self.arena_min)
            for label, frame_idx in snapshot_frames.items():
                update(frame_idx)
                ax.set_ylim(cropped_ymin, plot_max)
                png_filename = f"{base_filename}_{label}.png"
                fig.savefig(png_filename, dpi=300, bbox_inches="tight")
                print(f"{label.capitalize()} frame saved successfully: {png_filename}")
            ax.set_ylim(original_ylim)
        else:
            final_frame_idx = len(self.history_evader) - 1
            update(final_frame_idx)
            png_filename = base_filename + ".png"
            fig.savefig(png_filename, dpi=300, bbox_inches="tight")
            print(f"Final frame saved successfully: {png_filename}")

        # Close the figure to free up memory
        plt.close(fig)
