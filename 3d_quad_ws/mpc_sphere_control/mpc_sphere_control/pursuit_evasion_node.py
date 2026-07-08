#!/usr/bin/env python3
"""
PursuitEvasionNode — standalone minimax pursuit-evasion demo, independent of
task.py/main_node.py's single-robot Task framework. Drives N_p pursuers and
one evader directly via QuadrotorPursuitEvasionSolver, reusing render_node.py's
scene visualization (pursuers as 'quadrotor' markers, evader as 'evader').

Synchronous loop each dt: read state -> minimax solve -> one-step CBF filters
-> step entities -> publish poses/paths/capture sphere -> check capture.

Parameters (--ros-args -p key:=value):
  num_pursuers               : int   (default: 1)
  horizon                    : int   (default: 6)
  enable_pursuer_evader_cbf  : bool  (default: true)
  enable_pursuer_pursuer_cbf : bool  (default: true)
  enable_evader_cbf          : bool  (default: true)
  playback_rate              : float (default: 1.0)  — scales only the wall-clock
                                timer period, not the physics/MPC timestep.
  encirclement_radius        : float (default: 22.0) — pursuers spawn scattered
                                within +-30% radius / +-30 deg elevation of this.
  d_safe_pursuer_evader       : float (default: 0.75) — braking-CBF safety margin.
  capture_radius              : float (default: 1.5)  — game-termination distance;
                                must be > d_safe_pursuer_evader.
  workspace_half_extent       : float (default: 32.0) — soft RViz framing box;
                                must be > 1.3 * encirclement_radius.
  cutoff_weight               : float (default: 0.0)  — weight on the "geometric
                                cutoff" term (penalizes pursuer-evader relative
                                velocity transverse to the line-of-sight, a
                                collision-course/parallel-navigation intercept
                                cue rather than pure tail-chase). 0.0 disables it
                                (byte-identical to before this was added).

Snapshots: record a rosbag (e.g. `ros2 bag record -a`) and play it back through
RViz, pausing at the init/middle/capture timestamps logged below.
"""
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from mpc_interfaces.srv import ConfigureScene

from mpc_sphere_control.entities import Quadrotor, QuadrotorParams
from mpc_sphere_control.mpc_cbf.pursuit_evasion import QuadrotorPursuitEvasionSolver


class BootStep:
    WAIT_FOR_SERVICES = 0
    CLEAR_SCENE       = 1
    WAIT_CLEAR        = 2
    CONFIGURE_SCENE   = 3
    WAIT_CONFIGURE    = 4
    SETUP_SCENE       = 5
    WAIT_SETUP        = 6
    START_CONTROL     = 7


class PursuitEvasionNode(Node):
    def __init__(self):
        super().__init__('pursuit_evasion_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('num_pursuers', 1)
        self.declare_parameter('horizon', 30)
        self.declare_parameter('enable_pursuer_evader_cbf', True)
        self.declare_parameter('enable_pursuer_pursuer_cbf', True)
        self.declare_parameter('enable_evader_cbf', True)
        self.declare_parameter('playback_rate', 1.0)
        self.declare_parameter('encirclement_radius', 22.0)
        self.declare_parameter('d_safe_pursuer_evader', 0.5)
        self.declare_parameter('capture_radius', 2.3)
        self.declare_parameter('workspace_half_extent', 32.0)
        self.declare_parameter('cutoff_weight', 1.0)

        self.num_pursuers = int(self.get_parameter('num_pursuers').value)
        self.horizon       = int(self.get_parameter('horizon').value)
        self.enable_pe     = bool(self.get_parameter('enable_pursuer_evader_cbf').value)
        self.enable_pp     = bool(self.get_parameter('enable_pursuer_pursuer_cbf').value)
        self.enable_ec     = bool(self.get_parameter('enable_evader_cbf').value)
        self.playback_rate = float(self.get_parameter('playback_rate').value)
        self.encirclement_radius  = float(self.get_parameter('encirclement_radius').value)
        self.d_safe_pursuer_evader = float(self.get_parameter('d_safe_pursuer_evader').value)
        self.capture_radius        = float(self.get_parameter('capture_radius').value)
        self.workspace_half_extent = float(self.get_parameter('workspace_half_extent').value)
        self.cutoff_weight        = float(self.get_parameter('cutoff_weight').value)

        if self.playback_rate <= 0.0:
            raise ValueError(f"playback_rate must be > 0, got {self.playback_rate!r}")
        if self.capture_radius <= self.d_safe_pursuer_evader:
            raise ValueError(
                f"capture_radius ({self.capture_radius}) must be > "
                f"d_safe_pursuer_evader ({self.d_safe_pursuer_evader}) — otherwise the "
                f"CBF's safety margin sits outside the capture radius and actively "
                f"resists ever reaching it."
            )
        max_spawn_dist = 1.3 * self.encirclement_radius
        if self.workspace_half_extent <= max_spawn_dist:
            raise ValueError(
                f"workspace_half_extent ({self.workspace_half_extent}) must be > the max "
                f"possible pursuer spawn distance ({max_spawn_dist} = 1.3 * "
                f"encirclement_radius) — otherwise pursuers start outside the framing box."
            )

        # Physics/MPC timestep, unaffected by playback_rate (only the wall-clock
        # timer period is scaled, so slow motion is a viewing speed, not fidelity change).
        self.dt = 0.1

        # ── Scene: evader in the center, N_p pursuers scattered around it ───
        params = QuadrotorParams()
        evader_start = [8.0, 8.0, 12.0]
        self.evader  = Quadrotor('evader_0', evader_start, self.dt, params=params)

        self.pursuer_ids = [f'pursuer_{i}' for i in range(self.num_pursuers)]
        # Scattered (random azimuth/elevation/radius around encirclement_radius),
        # not a fixed symmetric ring, for stress-testing from varied directions.
        elevation_max = np.pi / 6.0
        start_positions = []
        for _ in range(self.num_pursuers):
            azimuth   = np.random.uniform(0.0, 2.0 * np.pi)
            elevation = np.random.uniform(-elevation_max, elevation_max)
            radius    = np.random.uniform(0.7, 1.3) * self.encirclement_radius
            start_positions.append([
                evader_start[0] + radius * np.cos(elevation) * np.cos(azimuth),
                evader_start[1] + radius * np.cos(elevation) * np.sin(azimuth),
                max(10.0, evader_start[2] + radius * np.sin(elevation)),
            ])
        self.pursuers = [
            Quadrotor(pid, pos, self.dt, params=params)
            for pid, pos in zip(self.pursuer_ids, start_positions)
        ]

        self.solver = QuadrotorPursuitEvasionSolver(
            self.pursuers, self.evader, horizon=self.horizon,
            enable_pursuer_evader_cbf=self.enable_pe,
            enable_pursuer_pursuer_cbf=self.enable_pp,
            enable_evader_cbf=self.enable_ec,
            D_safe_pursuer_evader=self.d_safe_pursuer_evader,
            workspace_half_extent=self.workspace_half_extent,
            cutoff_weight=self.cutoff_weight,
        )

        # Capture-radius sphere: precomputed once as local offsets (Fibonacci/golden-spiral
        # sampling); re-centered on the evader each tick via the marker's own pose.
        self._capture_sphere_offsets = self._fibonacci_sphere_offsets(300, self.capture_radius)
        self._capture_sphere_points = [
            Point(x=float(o[0]), y=float(o[1]), z=float(o[2])) for o in self._capture_sphere_offsets
        ]
        self._capture_sphere_pub = self.create_publisher(Marker, '/pursuit_evasion/capture_sphere', 10)

        # ── Publishers ───────────────────────────────────────────────────────
        self._pursuer_pose_pubs = {
            pid: self.create_publisher(PoseStamped, f'/robot/{pid}/pose', 10)
            for pid in self.pursuer_ids
        }
        self._evader_pose_pub = self.create_publisher(PoseStamped, '/obstacle/obs_0/pose', 10)
        self._pursuer_path_pubs = {
            pid: self.create_publisher(Path, f'/pursuit_evasion/{pid}/path', 10)
            for pid in self.pursuer_ids
        }
        self._evader_path_pub = self.create_publisher(Path, '/pursuit_evasion/evader/path', 10)

        # ── RenderNode services (reuse the existing visualisation pipeline) ──
        self.clear_scene_cli     = self.create_client(Empty,          '/render_node/clear_scene')
        self.configure_scene_cli = self.create_client(ConfigureScene, '/render_node/configure_scene')
        self.setup_scene_cli     = self.create_client(Empty,          '/render_node/setup_scene')

        self.boot_step      = BootStep.WAIT_FOR_SERVICES
        self.pending_future = None
        self._log_tick      = 0

        self.startup_timer = self.create_timer(0.2, self._boot_sequence)
        self.get_logger().info(
            f'PursuitEvasionNode — {self.num_pursuers} pursuer(s) scattered around 1 evader '
            f'(r~{self.encirclement_radius}m) | N={self.horizon} | dt={self.dt}s | '
            f'playback_rate={self.playback_rate}x | '
            f'CBF: pe={self.enable_pe} pp={self.enable_pp} ec={self.enable_ec}'
        )

    # =========================================================================
    # BOOT SEQUENCE (mirrors main_node.py's, simplified — no waypoints/obstacles)
    # =========================================================================

    def _all_services_ready(self) -> bool:
        return (
            self.clear_scene_cli.service_is_ready()     and
            self.configure_scene_cli.service_is_ready() and
            self.setup_scene_cli.service_is_ready()
        )

    def _boot_sequence(self):
        if self.boot_step == BootStep.WAIT_FOR_SERVICES:
            if not self._all_services_ready():
                self.get_logger().info('Waiting for RenderNode...', throttle_duration_sec=2.0)
                return
            self.boot_step = BootStep.CLEAR_SCENE

        elif self.boot_step == BootStep.CLEAR_SCENE:
            self.get_logger().info('Boot [1/4] -> clear_scene')
            self.pending_future = self.clear_scene_cli.call_async(Empty.Request())
            self.boot_step = BootStep.WAIT_CLEAR

        elif self.boot_step == BootStep.WAIT_CLEAR:
            if not self.pending_future.done():
                return
            self.get_logger().info('Boot [1/4] ok')
            self.boot_step = BootStep.CONFIGURE_SCENE

        elif self.boot_step == BootStep.CONFIGURE_SCENE:
            self.get_logger().info('Boot [2/4] -> configure_scene')
            req = ConfigureScene.Request()
            req.num_obs         = 1              # evader, registered as obs_0
            req.robot_ids       = list(self.pursuer_ids)
            req.robot_style     = 'quadrotor'   # pursuers: drone mesh, red (see render_node.ENTITY_STYLES)
            req.obstacle_style  = 'evader'      # evader: drone mesh, blue
            self.pending_future = self.configure_scene_cli.call_async(req)
            self.boot_step = BootStep.WAIT_CONFIGURE

        elif self.boot_step == BootStep.WAIT_CONFIGURE:
            if not self.pending_future.done():
                return
            self.get_logger().info('Boot [2/4] ok')
            self.boot_step = BootStep.SETUP_SCENE

        elif self.boot_step == BootStep.SETUP_SCENE:
            self.get_logger().info('Boot [3/4] -> setup_scene')
            self.pending_future = self.setup_scene_cli.call_async(Empty.Request())
            self.boot_step = BootStep.WAIT_SETUP

        elif self.boot_step == BootStep.WAIT_SETUP:
            if not self.pending_future.done():
                return
            self.get_logger().info('Boot [3/4] ok')
            self.boot_step = BootStep.START_CONTROL

        elif self.boot_step == BootStep.START_CONTROL:
            self.get_logger().info('Boot [4/4] ok - pursuit-evasion loop starting.')
            self.startup_timer.cancel()
            self.control_timer = self.create_timer(self.dt / self.playback_rate, self._control_loop)

    # =========================================================================
    # CONTROL LOOP  (synchronous: read -> minimax solve -> CBF filter -> step -> publish)
    # =========================================================================

    def _control_loop(self):
        now = self.get_clock().now().to_msg()

        pursuer_states = [p.full_state() for p in self.pursuers]
        evader_state   = self.evader.full_state()

        if self._log_tick == 0:
            self.get_logger().info('INIT — tick 0, t=0.0s (rosbag: seek here for the "start" snapshot).')

        # 1. Minimax best-response round.
        U_p0, u_e0, _info = self.solver.solve(pursuer_states, evader_state)
        if U_p0 is None:
            self.get_logger().error('Pursuit-evasion solve failed — holding at current controls.')
            return
        U_p0 = list(U_p0)
        ws_slack = max(_info.get('ws_slack_p', 0.0), _info.get('ws_slack_e', 0.0))
        if ws_slack > 1e-3:
            self.get_logger().warn(
                f'Workspace framing box relaxed (slack={ws_slack:.3f}) — an agent is '
                f'pushing outside the fixed screenshot framing region.',
                throttle_duration_sec=1.0,
            )

        # 2. One-step CBF safety filters.
        if self.enable_pe or self.enable_pp:
            U_p0, slack_p = self.solver.one_step_pursuer_cbf_filter(pursuer_states, evader_state, U_p0)
            if slack_p > 1e-3:
                self.get_logger().warn(f'Pursuer CBF relaxed (slack={slack_p:.3f})',
                                       throttle_duration_sec=1.0)
        if self.enable_ec:
            u_e0, slack_e = self.solver.one_step_evader_cbf_filter(pursuer_states, evader_state, u_e0, U_p0)
            if slack_e > 1e-3:
                self.get_logger().warn(f'Evader CBF relaxed (slack={slack_e:.3f})',
                                       throttle_duration_sec=1.0)

        # 3. Step dynamics.
        for p, u in zip(self.pursuers, U_p0):
            p.update_step(u)
        self.evader.update_step(u_e0)

        # 4. Publish.
        for pid, p in zip(self.pursuer_ids, self.pursuers):
            self._publish_pose(p, self._pursuer_pose_pubs[pid], now)
            self._publish_path(p.get_path_traversed(), self._pursuer_path_pubs[pid])
        self._publish_pose(self.evader, self._evader_pose_pub, now)
        self._publish_path(self.evader.get_path_traversed(), self._evader_path_pub)
        self._publish_capture_sphere(now)

        self._log_tick += 1

        # 5. Capture check — terminate once any pursuer is within capture_radius
        # (a game-termination threshold, distinct from the CBF safety margin).
        dists = [np.linalg.norm(p.position - self.evader.position) for p in self.pursuers]
        min_dist = min(dists)
        if min_dist <= self.capture_radius:
            captor = self.pursuer_ids[int(np.argmin(dists))]
            mid_tick = self._log_tick // 2
            self.get_logger().info(
                f'CAPTURE — {captor} reached the evader at dist={min_dist:.3f}m '
                f'(tick {self._log_tick}, t={self._log_tick * self.dt:.1f}s). Stopping.'
            )
            self.get_logger().info(
                f'For rosbag playback snapshots — init: t=0.0s, '
                f'middle: tick {mid_tick} (t={mid_tick * self.dt:.1f}s), '
                f'capture: tick {self._log_tick} (t={self._log_tick * self.dt:.1f}s).'
            )
            self.control_timer.cancel()
            return

        # 6. Logging every 20 ticks.
        if self._log_tick % 20 == 0:
            self.get_logger().info(
                'distances to evader [m]: ' + ', '.join(f'{d:.2f}' for d in dists)
            )

    # ── Publish helpers ──────────────────────────────────────────────────────

    def _publish_pose(self, entity, publisher, stamp):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp    = stamp
        msg.pose.position.x = float(entity.position[0])
        msg.pose.position.y = float(entity.position[1])
        msg.pose.position.z = float(entity.position[2])
        qw, qx, qy, qz = entity.orientation
        msg.pose.orientation.w = float(qw)
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        publisher.publish(msg)

    def _publish_path(self, np_path_array, publisher):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        for point in np_path_array:
            pose = PoseStamped()
            pose.header              = path_msg.header
            pose.pose.position.x, \
            pose.pose.position.y, \
            pose.pose.position.z     = map(float, point[:3])
            path_msg.poses.append(pose)
        publisher.publish(path_msg)

    @staticmethod
    def _fibonacci_sphere_offsets(n: int, radius: float) -> np.ndarray:
        """(n, 3) local offsets spread evenly over a sphere surface via the
        golden-angle spiral construction (avoids pole-bunching from a naive grid)."""
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(n)
        y = 1.0 - (i / float(n - 1)) * 2.0   # from 1 to -1
        r_xy = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
        theta = golden_angle * i
        x = np.cos(theta) * r_xy
        z = np.sin(theta) * r_xy
        return np.stack([x, y, z], axis=1) * radius

    def _publish_capture_sphere(self, stamp):
        """Dotted sphere marking the capture radius; the point cloud is fixed
        local geometry from __init__, re-centered via `pose` each tick."""
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = stamp
        m.ns   = 'capture_radius'
        m.id   = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.pose.position.x = float(self.evader.position[0])
        m.pose.position.y = float(self.evader.position[1])
        m.pose.position.z = float(self.evader.position[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 0.08
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.4, 0.6
        m.points = self._capture_sphere_points
        self._capture_sphere_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = PursuitEvasionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
