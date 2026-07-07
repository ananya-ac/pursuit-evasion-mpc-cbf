#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Empty
from mpc_interfaces.srv import ConfigureScene
from visualization_msgs.msg import Marker, MarkerArray


# Entity style registry. 'mesh_resource' renders as a tinted MESH_RESOURCE marker;
# entities without it fall back to a plain SPHERE. Chosen via ConfigureScene's
# robot_style/obstacle_style (default 'quadrotor'/'evader').
ENTITY_STYLES = {
    'quadrotor': {'scale': 0.01, 'rgba': [1.0, 0.0, 0.0, 1.0],
                  'mesh_resource': 'package://mpc_sphere_control/meshes/drone_body.stl'},
    'evader':    {'scale': 0.01, 'rgba': [0.1, 0.3, 1.0, 1.0],
                  'mesh_resource': 'package://mpc_sphere_control/meshes/drone_body.stl'},
}

# Parking position for entities before their first pose update
_OFFSCREEN = [9999.0, 9999.0, 9999.0]


class RenderNode(Node):
    def __init__(self):
        super().__init__('render_node')

        # ── Internal state ────────────────────────────────────────────────
        # task_config: populated by configure_scene, consumed by setup_scene
        self._pending_config: dict = {}

        # entity_registry: maps entity_id → entity_type  (persists across clear)
        self._entity_registry: dict[str, str] = {}

        # marker_refs: maps entity_id → Marker object in marker_array
        # (rebuilt on setup_scene, lazily on pose_callback after clear)
        self._marker_refs: dict[str, Marker] = {}
        self._marker_array = MarkerArray()
        self._marker_id_seq = 0

        # Dynamic subscriptions created per setup_scene
        self._active_subscriptions: list = []

        # ── Services ──────────────────────────────────────────────────────
        self.create_service(Empty, '/render_node/clear_scene',     self._handle_clear_scene)
        self.create_service(ConfigureScene, '/render_node/configure_scene', self._handle_configure_scene)
        self.create_service(Empty, '/render_node/setup_scene',     self._handle_setup_scene)

        # ── Marker publisher ──────────────────────────────────────────────
        self._marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)

        # ── Render loop: republishes the complete scene matrix at 10 Hz ───
        self.create_timer(0.1, self._render_scene)

        self.get_logger().info('RenderNode online — waiting for PursuitEvasionNode.')


    # ═══════════════════════════════════════════════════════════════════════
    # SERVICE HANDLERS  (called in order by the node's boot sequence)
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_clear_scene(self, request, response):
        """Step 1 — wipe visual state from the previous run at startup (DELETEALL to
        RViz, reset registry/marker refs). Subscriptions are rebuilt in setup_scene."""
        self.get_logger().info('clear_scene → wiping previous run...')

        # Stop pose callbacks immediately so no stale updates arrive
        # between this clear and the next setup_scene call.
        for sub in self._active_subscriptions:
            self.destroy_subscription(sub)
        self._active_subscriptions = []

        # Send DELETEALL to RViz so ghost markers are immediately removed
        deleteall = Marker()
        deleteall.action = Marker.DELETEALL
        deleteall.header.frame_id = 'map'
        deleteall.header.stamp    = self.get_clock().now().to_msg()
        nuke = MarkerArray()
        nuke.markers.append(deleteall)
        self._marker_pub.publish(nuke)

        # Reset internal marker state
        self._marker_array    = MarkerArray()
        self._marker_refs     = {}
        self._entity_registry = {}
        self._marker_id_seq   = 0

        self.get_logger().info('clear_scene ✓')
        return response

    def _handle_configure_scene(self, request, response):
        self._pending_config = {
            'num_obs': request.num_obs,
            'robot_ids': list(request.robot_ids),
            'robot_style': request.robot_style or 'quadrotor',
            'obstacle_style': request.obstacle_style or 'evader',
        }

        response.success = True
        response.message = f'Configured: {request.num_obs} obs'
        return response

    def _handle_setup_scene(self, request, response):
        """Step 3 — build the entity registry, pre-allocate a marker per robot/obstacle
        (parked offscreen until first pose update), and subscribe to their pose topics."""
        cfg = self._pending_config
        if not cfg:
            self.get_logger().error('setup_scene called without a valid config. Aborting.')
            return response

        self.get_logger().info('setup_scene → building entity registry...')

        # Destroy old subscriptions before rebuilding
        for sub in self._active_subscriptions:
            self.destroy_subscription(sub)
        self._active_subscriptions = []

        # ── Register robots ───────────────────────────────────────────────
        robot_style = cfg.get('robot_style', 'quadrotor')
        for robot_id in cfg.get('robot_ids', []):
            self._register_entity(robot_id, robot_style)
            self._active_subscriptions.append(
                self.create_subscription(
                    PoseStamped, f'/robot/{robot_id}/pose',
                    lambda msg, eid=robot_id: self._on_pose(msg, eid), 10
                )
            )

        # ── Register obstacles ────────────────────────────────────────────
        obstacle_style = cfg.get('obstacle_style', 'evader')
        for i in range(cfg.get('num_obs', 0)):
            obs_id = f'obs_{i}'
            self._register_entity(obs_id, obstacle_style)
            self._active_subscriptions.append(
                self.create_subscription(
                    PoseStamped, f'/obstacle/{obs_id}/pose',
                    lambda msg, eid=obs_id: self._on_pose(msg, eid), 10
                )
            )

        total = len(self._entity_registry)
        self.get_logger().info(f'setup_scene ✓ — {total} entities registered.')
        return response

    # ═══════════════════════════════════════════════════════════════════════
    # POSE CALLBACK
    # ═══════════════════════════════════════════════════════════════════════

    def _on_pose(self, msg: PoseStamped, entity_id: str):
        """Update the marker position for a registered entity, reconstructing it
        lazily if it was cleared but the entity is still registered."""
        if entity_id not in self._entity_registry:
            self.get_logger().warn(f'Pose update for unregistered entity: {entity_id}', throttle_duration_sec=5.0)
            return

        # Lazy reconstruction after a clear_scene
        if entity_id not in self._marker_refs:
            entity_type = self._entity_registry[entity_id]
            self._register_entity(entity_id, entity_type, initial_pos=[
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])

        marker = self._marker_refs[entity_id]
        marker.header.stamp    = msg.header.stamp
        marker.pose.position   = msg.pose.position
        marker.pose.orientation = msg.pose.orientation

    # ═══════════════════════════════════════════════════════════════════════
    # RENDER LOOP
    # ═══════════════════════════════════════════════════════════════════════

    def _render_scene(self):
        """Publish the full scene at 10 Hz with refreshed timestamps, as a single
        MarkerArray so RViz doesn't expire markers or flood the queue."""
        now = self.get_clock().now().to_msg()
        for m in self._marker_array.markers:
            m.header.stamp = now
        self._marker_pub.publish(self._marker_array)

    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _register_entity(self, entity_id: str, entity_type: str, initial_pos=None):
        """
        Add entity to registry and create its marker slot.
        Idempotent: re-registering the same entity_id updates its marker.
        """
        if entity_type not in ENTITY_STYLES:
            self.get_logger().error(f'Unknown entity type: {entity_type}')
            return

        pos    = initial_pos if initial_pos is not None else _OFFSCREEN
        style  = ENTITY_STYLES[entity_type]

        m = Marker()
        m.header.frame_id  = 'map'
        m.ns               = entity_type
        m.id               = self._marker_id_seq
        if 'mesh_resource' in style:
            m.type          = Marker.MESH_RESOURCE
            m.mesh_resource = style['mesh_resource']
        else:
            m.type = Marker.SPHERE
        m.action           = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = float(style['scale'])
        m.color.r, m.color.g, m.color.b, m.color.a = map(float, style['rgba'])
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
        m.pose.orientation.w = 1.0

        self._marker_id_seq += 1
        self._entity_registry[entity_id] = entity_type
        self._marker_refs[entity_id]     = m
        self._marker_array.markers.append(m)


def main(args=None):
    rclpy.init(args=args)
    node = RenderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
