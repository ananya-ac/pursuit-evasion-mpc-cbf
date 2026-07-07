
# Implementation for Capture, Shield, or Neutralize: Engagement-Aware Pursuit-Evasion

Two implementations of a minimax pursuit-evasion game (pursuers minimize distance, evader maximizes it) with a Control Barrier Function safety filter on top.

- `2d_planar_sim/` — standalone Python, matplotlib animation, no ROS.
- `3d_quad_ws/` — ROS 2 (Jazzy) quadrotor implementation with RViz2 visualization.

## `2d_planar_sim`

```bash
cd 2d_planar_sim
python3 main.py
```

Parameter sweeps (produce outcome heatmaps):

```bash
python3 run_velocity_grid.py
python3 run_pursuer_count_grid.py
```

## `3d_quad_ws`

Build:

```bash
cd 3d_quad_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select mpc_interfaces mpc_sphere_control
source install/setup.bash
```

Run (terminal 1 — leave running):

```bash
ros2 run mpc_sphere_control render_node
```

Run (terminal 2):

```bash
ros2 run mpc_sphere_control pursuit_evasion_node
# e.g. with 4 pursuers:
ros2 run mpc_sphere_control pursuit_evasion_node --ros-args -p num_pursuers:=4
```

Visualize (terminal 3):

```bash
rviz2 -d $(ros2 pkg prefix mpc_sphere_control)/share/mpc_sphere_control/rviz/simple_mpc.rviz
```
