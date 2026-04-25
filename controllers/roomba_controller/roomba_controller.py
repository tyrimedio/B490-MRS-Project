"""
Roomba Cluster - E-puck Controller

Basic wall-following and obstacle-avoidance behavior for the e-puck robot.
This serves as the foundation for SLAM exploration and cleaning coverage.
Each robot drives forward and turns when it detects an obstacle, covering
its assigned area with a sweeping pattern.
"""

import json
import math

from controller import Robot

# E-puck proximity sensor names (8 sensors around the body)
SENSOR_NAMES = [
    "ps0", "ps1", "ps2", "ps3",
    "ps4", "ps5", "ps6", "ps7",
]

# Sensor layout (looking down at the robot, 0 is front-right):
#   ps7 ps0      <- front
#  ps6    ps1
#  ps5    ps2
#   ps4 ps3      <- rear

OBSTACLE_THRESHOLD = 80.0  # proximity value that counts as "something is close"
MAX_SPEED = 6.28           # max motor speed in rad/s for e-puck
CRUISE_SPEED = 0.9 * MAX_SPEED
LIDAR_STATUS_HEARTBEAT_STEPS = 240
LIDAR_VALUE_AT_1_METER = 1000.0
LIDAR_MAX_RANGE_M = 1.0
LIDAR_NO_HIT_MARGIN = 1.0
LIDAR_MIN_VALID_HIT_M = 0.03
LIDAR_MAX_VALID_HIT_M = 0.95
COMMUNICATION_SEND_INTERVAL_STEPS = 30
POSE_CORRECTION_DRIFT_LOG_M = 0.15
POSE_CORRECTION_HEADING_LOG_RAD = 0.35
POSE_CORRECTION_LOG_INTERVAL_STEPS = 120

# Initial Webots poses for the four robots in the shared world frame.
# These match the robot translations and rotations in worlds/roomba_cluster.wbt.
ROBOT_START_POSES = {
    "epuck_1": (-1.5, 0.0, 0.5 * math.pi),
    "epuck_2": (-0.5, 0.0, 0.5 * math.pi),
    "epuck_3": (0.5, 0.0, -0.5 * math.pi),
    "epuck_4": (1.5, 0.0, -0.5 * math.pi),
}

# Launch carries each robot to a small staging point inside the central hub.
# Robots wait there for the supervisor's real task assignment instead of
# drifting toward a room before MRTA has made that choice.
ROBOT_LAUNCH_WAYPOINTS = {
    "epuck_1": ((-0.35, 0.30),),
    "epuck_2": ((-0.12, 0.15),),
    "epuck_3": ((0.12, -0.15),),
    "epuck_4": ((0.35, -0.30),),
}
DEFAULT_START_POSE = (0.0, 0.0, 0.0)
DEFAULT_LAUNCH_WAYPOINTS = ()
LAUNCH_WAYPOINT_REACHED_M = 0.18
LAUNCH_ENABLE_MAPPING_AT_WAYPOINT = 1
LAUNCH_TIMEOUT_STEPS = 2500
LAUNCH_DISTANCE_GAIN = 4.0
LAUNCH_MIN_SPEED = 0.25 * MAX_SPEED
LAUNCH_MAX_SPEED = 0.70 * MAX_SPEED
LAUNCH_TURN_GAIN = 2.2
LAUNCH_TURN_LIMIT = 0.50 * MAX_SPEED
LAUNCH_SPIN_THRESHOLD_RAD = 1.05
LAUNCH_SPIN_SPEED = 0.45 * MAX_SPEED
ASSIGNMENT_TARGET_REACHED_M = 0.20
COVERAGE_WAYPOINT_REACHED_M = 0.05

# Corner recovery behavior
CORNER_PRESSURE_TRIGGER_STEPS = 6
ESCAPE_REVERSE_STEPS = 20
ESCAPE_TURN_STEPS = 24
ESCAPE_REVERSE_SPEED = -0.55 * MAX_SPEED
ESCAPE_TURN_SPEED = 0.65 * MAX_SPEED

# E-puck kinematics for simple dead-reckoning
WHEEL_RADIUS_M = 0.0205
AXLE_LENGTH_M = 0.053

# Occupancy-grid cell states
GRID_UNKNOWN = -1
GRID_FREE = 0
GRID_WALL = 1
FREE_EVIDENCE = -1
WALL_EVIDENCE = 3
MIN_OCCUPANCY_SCORE = -8
MAX_OCCUPANCY_SCORE = 8
FREE_SCORE_THRESHOLD = -2
WALL_SCORE_THRESHOLD = 3
CONTROLLER_BUILD = "2026-04-10-log-throttle-v4"


class OccupancyGrid:
    """
    Small 2D map stored as a grid (like graph paper).

    Each cell starts as UNKNOWN. Lidar readings add evidence to a score:
    negative means probably open floor, positive means probably wall.
    The public FREE/WALL/UNKNOWN state is recalculated from that score.
    """

    def __init__(self, world_size_m=6.0, cell_size_m=0.05):
        self.world_size_m = world_size_m
        self.cell_size_m = cell_size_m
        self.width = int(round(world_size_m / cell_size_m))
        self.height = int(round(world_size_m / cell_size_m))
        self.data = []
        self.scores = []
        self.wall_cell_count = 0
        self.free_cell_count = 0
        self.pending_free_cells = set()
        self.pending_wall_cells = set()
        self.pending_free_observations = {}
        self.pending_wall_observations = {}
        self.pending_observations = []
        self.reset()

    def reset(self):
        """Reset every cell to UNKNOWN."""
        self.data = [
            [GRID_UNKNOWN for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.scores = [
            [0 for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.wall_cell_count = 0
        self.free_cell_count = 0
        self.pending_free_cells = set()
        self.pending_wall_cells = set()
        self.pending_free_observations = {}
        self.pending_wall_observations = {}
        self.pending_observations = []

    def world_to_grid(self, x_m, y_m):
        """Convert local map coordinates in meters to grid cell indices."""
        half_size = 0.5 * self.world_size_m
        grid_x = int(math.floor((x_m + half_size) / self.cell_size_m))
        grid_y = int(math.floor((y_m + half_size) / self.cell_size_m))
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return grid_x, grid_y
        return None

    def mark_wall(self, x_m, y_m):
        """Mark a map cell as WALL using local map coordinates in meters."""
        grid_cell = self.world_to_grid(x_m, y_m)
        if grid_cell is None:
            return False

        grid_x, grid_y = grid_cell
        return self.mark_wall_cell(grid_x, grid_y)

    def mark_wall_cell(self, grid_x, grid_y):
        """Add wall evidence to a grid cell."""
        cell = (grid_x, grid_y)
        self.pending_wall_observations[cell] = (
            self.pending_wall_observations.get(cell, 0) + 1
        )
        self.append_pending_observation("wall", grid_x, grid_y)
        return self.add_cell_evidence(grid_x, grid_y, WALL_EVIDENCE)

    def mark_free_cell(self, grid_x, grid_y):
        """Add free-space evidence to a grid cell."""
        cell = (grid_x, grid_y)
        self.pending_free_observations[cell] = (
            self.pending_free_observations.get(cell, 0) + 1
        )
        self.append_pending_observation("free", grid_x, grid_y)
        return self.add_cell_evidence(grid_x, grid_y, FREE_EVIDENCE)

    def append_pending_observation(self, kind, grid_x, grid_y):
        """Remember evidence in the same order the robot observed it."""
        if self.pending_observations:
            last_kind, last_grid_x, last_grid_y, last_count = self.pending_observations[-1]
            if last_kind == kind and last_grid_x == grid_x and last_grid_y == grid_y:
                self.pending_observations[-1] = [
                    last_kind,
                    last_grid_x,
                    last_grid_y,
                    last_count + 1,
                ]
                return

        self.pending_observations.append([kind, grid_x, grid_y, 1])

    def add_cell_evidence(self, grid_x, grid_y, evidence):
        """Update one cell's score and public map state."""
        old_state = self.data[grid_y][grid_x]
        old_score = self.scores[grid_y][grid_x]
        new_score = clamp(
            old_score + evidence,
            MIN_OCCUPANCY_SCORE,
            MAX_OCCUPANCY_SCORE,
        )
        self.scores[grid_y][grid_x] = new_score

        if new_score >= WALL_SCORE_THRESHOLD:
            new_state = GRID_WALL
        elif new_score <= FREE_SCORE_THRESHOLD:
            new_state = GRID_FREE
        else:
            new_state = GRID_UNKNOWN

        if new_state == old_state:
            return False

        if old_state == GRID_FREE:
            self.free_cell_count -= 1
        elif old_state == GRID_WALL:
            self.wall_cell_count -= 1

        if new_state == GRID_FREE:
            self.free_cell_count += 1
        elif new_state == GRID_WALL:
            self.wall_cell_count += 1

        self.data[grid_y][grid_x] = new_state
        cell = (grid_x, grid_y)
        self.pending_free_cells.discard(cell)
        self.pending_wall_cells.discard(cell)
        if new_state == GRID_FREE:
            self.pending_free_cells.add(cell)
        elif new_state == GRID_WALL:
            self.pending_wall_cells.add(cell)
        return True

    def mark_free_line(self, start_x_m, start_y_m, end_x_m, end_y_m):
        """
        Mark open cells along a lidar beam.

        The robot is at the start point. The wall is at the end point. Every
        grid square before the wall is treated as open floor.
        """
        wall_cell = self.world_to_grid(end_x_m, end_y_m)
        distance_m = math.hypot(end_x_m - start_x_m, end_y_m - start_y_m)
        if distance_m <= 0.0:
            return 0

        sample_count = max(1, int(math.ceil(distance_m / (0.5 * self.cell_size_m))))
        changed_count = 0
        for sample_index in range(sample_count):
            ratio = sample_index / sample_count
            sample_x_m = start_x_m + ratio * (end_x_m - start_x_m)
            sample_y_m = start_y_m + ratio * (end_y_m - start_y_m)
            grid_cell = self.world_to_grid(sample_x_m, sample_y_m)
            if grid_cell is None or grid_cell == wall_cell:
                continue

            grid_x, grid_y = grid_cell
            if self.mark_free_cell(grid_x, grid_y):
                changed_count += 1

        return changed_count

    def drain_pending_updates(self):
        """Return changed cells and clear the pending update queue."""
        updates = {
            "free_cells": [list(cell) for cell in sorted(self.pending_free_cells)],
            "wall_cells": [list(cell) for cell in sorted(self.pending_wall_cells)],
            "free_observations": [
                [cell[0], cell[1], count]
                for cell, count in sorted(self.pending_free_observations.items())
            ],
            "wall_observations": [
                [cell[0], cell[1], count]
                for cell, count in sorted(self.pending_wall_observations.items())
            ],
            "observations": list(self.pending_observations),
        }
        self.pending_free_cells.clear()
        self.pending_wall_cells.clear()
        self.pending_free_observations.clear()
        self.pending_wall_observations.clear()
        self.pending_observations.clear()
        return updates


def normalize_angle(angle_rad):
    """Keep an angle in the range [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def clamp(value, low, high):
    """Clamp a number between low and high."""
    return max(low, min(high, value))


def get_optional_device(robot, device_name):
    """Return a Webots device if it exists on this robot."""
    try:
        return robot.getDevice(device_name)
    except Exception:
        return None


def drive_toward_target(robot_x_m, robot_y_m, robot_theta_rad, target_x_m, target_y_m):
    """Return wheel speeds that steer toward one world-frame target point."""
    delta_x_m = target_x_m - robot_x_m
    delta_y_m = target_y_m - robot_y_m
    target_distance_m = math.hypot(delta_x_m, delta_y_m)
    target_heading_rad = math.atan2(delta_y_m, delta_x_m)
    heading_error_rad = normalize_angle(target_heading_rad - robot_theta_rad)

    if abs(heading_error_rad) > LAUNCH_SPIN_THRESHOLD_RAD:
        spin_speed = LAUNCH_SPIN_SPEED if heading_error_rad > 0 else -LAUNCH_SPIN_SPEED
        return -spin_speed, spin_speed

    forward_speed = clamp(
        LAUNCH_DISTANCE_GAIN * target_distance_m,
        LAUNCH_MIN_SPEED,
        LAUNCH_MAX_SPEED,
    )
    turn_speed = clamp(
        LAUNCH_TURN_GAIN * heading_error_rad,
        -LAUNCH_TURN_LIMIT,
        LAUNCH_TURN_LIMIT,
    )
    left_speed = clamp(forward_speed - turn_speed, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(forward_speed + turn_speed, -MAX_SPEED, MAX_SPEED)
    return left_speed, right_speed


def should_hold_for_assignment(launch_finished, launch_timed_out, assigned_target):
    """Return whether the robot should wait at launch staging."""
    return (launch_finished or launch_timed_out) and assigned_target is None


def should_hold_after_task(assigned_target, assignment_target_reached, coverage_complete):
    """Return whether the robot should wait after its active plan is done."""
    return (
        assigned_target is not None
        and assignment_target_reached
        and coverage_complete
    )


def run():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    robot_name = robot.getName()
    print(f"[{robot_name}] Starting up ({CONTROLLER_BUILD})")
    launch_waypoints = ROBOT_LAUNCH_WAYPOINTS.get(
        robot_name,
        DEFAULT_LAUNCH_WAYPOINTS,
    )
    start_x_m, start_y_m, start_theta_rad = ROBOT_START_POSES.get(
        robot_name,
        DEFAULT_START_POSE,
    )
    if launch_waypoints:
        print(
            f"[{robot_name}] Launch target loaded: "
            f"{len(launch_waypoints)} waypoint(s)"
        )
    else:
        print(f"[{robot_name}] No launch target; mapping starts immediately")

    # Create local map storage (6m world / 5cm per cell = 120x120).
    occupancy_grid = OccupancyGrid(world_size_m=6.0, cell_size_m=0.05)
    occupancy_grid.reset()
    print(
        f"[{robot_name}] Occupancy grid ready: "
        f"{occupancy_grid.width}x{occupancy_grid.height} cells"
    )

    # Initialize proximity sensors
    sensors = []
    for name in SENSOR_NAMES:
        sensor = robot.getDevice(name)
        sensor.enable(timestep)
        sensors.append(sensor)

    # Initialize turret-mounted lidar sensor (if present)
    lidar = get_optional_device(robot, "lidar")

    if lidar is not None:
        lidar.enable(timestep)
        print(f"[{robot_name}] Lidar enabled")
    else:
        print(f"[{robot_name}] Lidar not found")

    emitter = get_optional_device(robot, "emitter")
    if emitter is not None:
        print(f"[{robot_name}] Communication emitter ready")
    else:
        print(f"[{robot_name}] Communication emitter not found")

    receiver = get_optional_device(robot, "receiver")
    if receiver is not None:
        receiver.enable(timestep)
        print(f"[{robot_name}] Communication receiver ready")
    else:
        print(f"[{robot_name}] Communication receiver not found")

    # Initialize motors
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    dt_s = timestep / 1000.0
    step_count = 0
    left_speed_cmd = 0.0
    right_speed_cmd = 0.0
    robot_x_m = start_x_m
    robot_y_m = start_y_m
    robot_theta_rad = start_theta_rad
    corner_pressure_steps = 0
    escape_reverse_steps = 0
    escape_turn_steps = 0
    escape_turn_left = True
    launch_waypoint_index = 0
    mapping_enabled = len(launch_waypoints) == 0
    launch_timed_out = False
    assigned_room = None
    assigned_target = None
    assignment_route = []
    assignment_route_index = 0
    assignment_target_reached = False
    coverage_room = None
    coverage_waypoints = []
    coverage_waypoint_index = 0
    coverage_complete = False
    last_wall_hit = None
    last_lidar_log_key = None
    last_lidar_log_step = -LIDAR_STATUS_HEARTBEAT_STEPS
    last_pose_correction_log_step = -POSE_CORRECTION_LOG_INTERVAL_STEPS

    def should_log_lidar(log_key):
        nonlocal last_lidar_log_key, last_lidar_log_step

        state_changed = log_key != last_lidar_log_key
        heartbeat_due = (step_count - last_lidar_log_step) >= LIDAR_STATUS_HEARTBEAT_STEPS
        if state_changed or heartbeat_due:
            last_lidar_log_key = log_key
            last_lidar_log_step = step_count
            return True
        return False

    while robot.step(timestep) != -1:
        step_count += 1

        launch_finished = launch_waypoint_index >= len(launch_waypoints)
        if not launch_finished and not launch_timed_out and step_count >= LAUNCH_TIMEOUT_STEPS:
            launch_timed_out = True
            print(f"[{robot_name}] Launch timeout reached")
            if not mapping_enabled:
                mapping_enabled = True
                print(f"[{robot_name}] Mapping enabled")

        # Update a simple odometry estimate from the previous wheel commands.
        left_linear_mps = left_speed_cmd * WHEEL_RADIUS_M
        right_linear_mps = right_speed_cmd * WHEEL_RADIUS_M
        forward_mps = 0.5 * (left_linear_mps + right_linear_mps)
        yaw_rate_rps = (right_linear_mps - left_linear_mps) / AXLE_LENGTH_M

        robot_theta_rad = normalize_angle(robot_theta_rad + yaw_rate_rps * dt_s)
        robot_x_m += forward_mps * math.cos(robot_theta_rad) * dt_s
        robot_y_m += forward_mps * math.sin(robot_theta_rad) * dt_s

        # Read all proximity sensor values
        values = [s.getValue() for s in sensors]

        if receiver is not None:
            while receiver.getQueueLength() > 0:
                command = receiver.getString()
                try:
                    command_message = json.loads(command)
                except json.JSONDecodeError:
                    print(f"[{robot_name}] Supervisor command received: {command}")
                    receiver.nextPacket()
                    continue

                if command_message.get("type") == "task_assignment":
                    target_robot = command_message.get("robot")
                    if target_robot in (robot_name, "all"):
                        assigned_room = command_message.get("room")
                        target = command_message.get("target")
                        if isinstance(target, list) and len(target) == 2:
                            assigned_target = (float(target[0]), float(target[1]))
                            assignment_route = []
                            for waypoint in command_message.get("route", []):
                                if isinstance(waypoint, list) and len(waypoint) == 2:
                                    assignment_route.append(
                                        (float(waypoint[0]), float(waypoint[1]))
                                    )
                            if not assignment_route:
                                assignment_route = [assigned_target]
                            assignment_route_index = 0
                            assignment_target_reached = False
                        else:
                            assigned_target = None
                            assignment_route = []
                            assignment_route_index = 0
                            assignment_target_reached = False
                        print(
                            f"[{robot_name}] Assigned to room "
                            f"{assigned_room} via {len(assignment_route)} "
                            "route waypoint(s)"
                        )
                elif command_message.get("type") == "idle":
                    target_robot = command_message.get("robot")
                    if target_robot in (robot_name, "all"):
                        assigned_room = None
                        assigned_target = None
                        assignment_route = []
                        assignment_route_index = 0
                        assignment_target_reached = False
                        coverage_room = None
                        coverage_waypoints = []
                        coverage_waypoint_index = 0
                        coverage_complete = True
                        print(f"[{robot_name}] Holding idle")
                elif command_message.get("type") == "coverage_plan":
                    target_robot = command_message.get("robot")
                    if target_robot in (robot_name, "all"):
                        coverage_room = command_message.get("room")
                        coverage_waypoints = []
                        for waypoint in command_message.get("waypoints", []):
                            if isinstance(waypoint, list) and len(waypoint) == 2:
                                coverage_waypoints.append(
                                    (float(waypoint[0]), float(waypoint[1]))
                                )
                        coverage_waypoint_index = 0
                        coverage_complete = len(coverage_waypoints) == 0
                        print(
                            f"[{robot_name}] Coverage plan loaded for "
                            f"{coverage_room}: {len(coverage_waypoints)} waypoint(s)"
                        )
                elif command_message.get("type") == "pose_correction":
                    target_robot = command_message.get("robot")
                    if target_robot in (robot_name, "all"):
                        pose = command_message.get("pose", {})
                        try:
                            corrected_x_m = float(pose["x_m"])
                            corrected_y_m = float(pose["y_m"])
                            corrected_theta_rad = normalize_angle(
                                float(pose["theta_rad"])
                            )
                        except (KeyError, TypeError, ValueError):
                            receiver.nextPacket()
                            continue

                        position_error_m = math.hypot(
                            corrected_x_m - robot_x_m,
                            corrected_y_m - robot_y_m,
                        )
                        heading_error_rad = abs(
                            normalize_angle(corrected_theta_rad - robot_theta_rad)
                        )
                        should_log_pose_correction = (
                            position_error_m >= POSE_CORRECTION_DRIFT_LOG_M
                            or heading_error_rad >= POSE_CORRECTION_HEADING_LOG_RAD
                        ) and (
                            step_count - last_pose_correction_log_step
                            >= POSE_CORRECTION_LOG_INTERVAL_STEPS
                        )

                        robot_x_m = corrected_x_m
                        robot_y_m = corrected_y_m
                        robot_theta_rad = corrected_theta_rad

                        if should_log_pose_correction:
                            last_pose_correction_log_step = step_count
                            print(
                                f"[{robot_name}] Pose corrected by supervisor: "
                                f"position_error={position_error_m:.2f}m "
                                f"heading_error={heading_error_rad:.2f}rad"
                            )
                else:
                    print(f"[{robot_name}] Supervisor command received: {command}")
                receiver.nextPacket()

        if lidar is not None:
            lidar_value = lidar.getValue()
            has_lidar_hit = lidar_value < (LIDAR_VALUE_AT_1_METER - LIDAR_NO_HIT_MARGIN)
            if has_lidar_hit:
                lidar_distance_m = min(
                    LIDAR_MAX_RANGE_M,
                    max(0.0, lidar_value / LIDAR_VALUE_AT_1_METER),
                )
                valid_hit = LIDAR_MIN_VALID_HIT_M <= lidar_distance_m <= LIDAR_MAX_VALID_HIT_M
                if mapping_enabled and valid_hit:
                    hit_x_m = robot_x_m + lidar_distance_m * math.cos(robot_theta_rad)
                    hit_y_m = robot_y_m + lidar_distance_m * math.sin(robot_theta_rad)
                    free_updates = occupancy_grid.mark_free_line(
                        robot_x_m,
                        robot_y_m,
                        hit_x_m,
                        hit_y_m,
                    )
                    marked = occupancy_grid.mark_wall(hit_x_m, hit_y_m)
                    last_wall_hit = {
                        "x_m": round(hit_x_m, 3),
                        "y_m": round(hit_y_m, 3),
                        "distance_m": round(lidar_distance_m, 3),
                    }
                    update_tag = "new" if marked else "repeat"
                    if should_log_lidar(f"wall:{update_tag}"):
                        print(
                            f"[{robot_name}] lidar={lidar_value:.1f} "
                            f"dist={lidar_distance_m:.3f}m "
                            f"wall={update_tag} free+={free_updates} "
                            f"free={occupancy_grid.free_cell_count} "
                            f"walls={occupancy_grid.wall_cell_count}"
                        )
                else:
                    hold_reason = "launch_hold" if not mapping_enabled else "out_of_range"
                    if should_log_lidar(hold_reason):
                        print(
                            f"[{robot_name}] lidar={lidar_value:.1f} "
                            f"dist={lidar_distance_m:.3f}m {hold_reason}"
                        )
            else:
                logged_clear = False
                if mapping_enabled:
                    clear_x_m = robot_x_m + LIDAR_MAX_VALID_HIT_M * math.cos(robot_theta_rad)
                    clear_y_m = robot_y_m + LIDAR_MAX_VALID_HIT_M * math.sin(robot_theta_rad)
                    free_updates = occupancy_grid.mark_free_line(
                        robot_x_m,
                        robot_y_m,
                        clear_x_m,
                        clear_y_m,
                    )
                    if free_updates > 0 and should_log_lidar("clear_free"):
                        print(
                            f"[{robot_name}] lidar={lidar_value:.1f} "
                            f"clear free+={free_updates} "
                            f"free={occupancy_grid.free_cell_count} "
                            f"walls={occupancy_grid.wall_cell_count}"
                        )
                        logged_clear = True
                if not logged_clear and should_log_lidar("no_hit"):
                    print(f"[{robot_name}] lidar={lidar_value:.1f} no_hit")

        launch_left_speed = None
        launch_right_speed = None
        if launch_waypoint_index < len(launch_waypoints) and not launch_timed_out:
            target_x_m, target_y_m = launch_waypoints[launch_waypoint_index]
            delta_x_m = target_x_m - robot_x_m
            delta_y_m = target_y_m - robot_y_m
            target_distance_m = math.hypot(delta_x_m, delta_y_m)

            if target_distance_m <= LAUNCH_WAYPOINT_REACHED_M:
                launch_waypoint_index += 1
                if (
                    not mapping_enabled
                    and launch_waypoint_index >= LAUNCH_ENABLE_MAPPING_AT_WAYPOINT
                ):
                    mapping_enabled = True
                    print(f"[{robot_name}] Mapping enabled")
                if launch_waypoint_index >= len(launch_waypoints):
                    print(f"[{robot_name}] Launch complete")
                else:
                    print(
                        f"[{robot_name}] Launch waypoint "
                        f"{launch_waypoint_index}/{len(launch_waypoints)} reached"
                    )
            else:
                launch_left_speed, launch_right_speed = drive_toward_target(
                    robot_x_m,
                    robot_y_m,
                    robot_theta_rad,
                    target_x_m,
                    target_y_m,
                )

        assignment_left_speed = None
        assignment_right_speed = None
        launch_finished = launch_waypoint_index >= len(launch_waypoints)
        if (
            assigned_target is not None
            and assignment_route_index < len(assignment_route)
            and not assignment_target_reached
            and (launch_finished or launch_timed_out)
        ):
            target_x_m, target_y_m = assignment_route[assignment_route_index]
            delta_x_m = target_x_m - robot_x_m
            delta_y_m = target_y_m - robot_y_m
            target_distance_m = math.hypot(delta_x_m, delta_y_m)

            if target_distance_m <= ASSIGNMENT_TARGET_REACHED_M:
                assignment_route_index += 1
                if assignment_route_index >= len(assignment_route):
                    assignment_target_reached = True
                    print(f"[{robot_name}] Reached assigned room {assigned_room}")
                else:
                    print(
                        f"[{robot_name}] Assignment route waypoint "
                        f"{assignment_route_index}/{len(assignment_route)} reached"
                    )
            else:
                assignment_left_speed, assignment_right_speed = drive_toward_target(
                    robot_x_m,
                    robot_y_m,
                    robot_theta_rad,
                    target_x_m,
                    target_y_m,
                )

        coverage_left_speed = None
        coverage_right_speed = None
        if (
            assignment_target_reached
            and coverage_waypoint_index < len(coverage_waypoints)
            and not coverage_complete
        ):
            target_x_m, target_y_m = coverage_waypoints[coverage_waypoint_index]
            target_distance_m = math.hypot(target_x_m - robot_x_m, target_y_m - robot_y_m)
            if target_distance_m <= COVERAGE_WAYPOINT_REACHED_M:
                coverage_waypoint_index += 1
                if coverage_waypoint_index >= len(coverage_waypoints):
                    coverage_complete = True
                    print(f"[{robot_name}] Coverage complete for {coverage_room}")
                else:
                    print(
                        f"[{robot_name}] Coverage waypoint "
                        f"{coverage_waypoint_index}/{len(coverage_waypoints)} reached"
                    )
            else:
                coverage_left_speed, coverage_right_speed = drive_toward_target(
                    robot_x_m,
                    robot_y_m,
                    robot_theta_rad,
                    target_x_m,
                    target_y_m,
                )

        # Check for obstacles on each side
        front_left = values[7] > OBSTACLE_THRESHOLD
        front_right = values[0] > OBSTACLE_THRESHOLD
        left_side = values[6] > OBSTACLE_THRESHOLD or values[5] > OBSTACLE_THRESHOLD
        right_side = values[1] > OBSTACLE_THRESHOLD or values[2] > OBSTACLE_THRESHOLD
        rear_blocked = values[3] > OBSTACLE_THRESHOLD or values[4] > OBSTACLE_THRESHOLD

        front_blocked = front_left or front_right
        left_pressure = values[7] + values[6] + values[5]
        right_pressure = values[0] + values[1] + values[2]
        corner_pressure = front_blocked and left_side and right_side

        if escape_reverse_steps > 0 and rear_blocked:
            escape_reverse_steps = 0

        if corner_pressure:
            corner_pressure_steps += 1
        else:
            corner_pressure_steps = 0

        # Decide motor speeds based on the highest-priority active behavior.
        if should_hold_for_assignment(launch_finished, launch_timed_out, assigned_target):
            # Hold position while waiting for the supervisor to assign a room.
            left_speed = 0.0
            right_speed = 0.0
        elif should_hold_after_task(
            assigned_target,
            assignment_target_reached,
            coverage_complete,
        ):
            # Hold position after finishing the current room until reassigned.
            left_speed = 0.0
            right_speed = 0.0
        elif escape_reverse_steps > 0 and not rear_blocked:
            left_speed = ESCAPE_REVERSE_SPEED
            right_speed = ESCAPE_REVERSE_SPEED
            escape_reverse_steps -= 1
        elif escape_turn_steps > 0:
            if escape_turn_left:
                left_speed = -ESCAPE_TURN_SPEED
                right_speed = ESCAPE_TURN_SPEED
            else:
                left_speed = ESCAPE_TURN_SPEED
                right_speed = -ESCAPE_TURN_SPEED
            escape_turn_steps -= 1
        elif corner_pressure_steps >= CORNER_PRESSURE_TRIGGER_STEPS:
            escape_turn_left = right_pressure >= left_pressure
            corner_pressure_steps = 0

            if rear_blocked:
                escape_reverse_steps = 0
                escape_turn_steps = max(0, ESCAPE_TURN_STEPS - 1)
                if escape_turn_left:
                    left_speed = -ESCAPE_TURN_SPEED
                    right_speed = ESCAPE_TURN_SPEED
                else:
                    left_speed = ESCAPE_TURN_SPEED
                    right_speed = -ESCAPE_TURN_SPEED
            else:
                escape_reverse_steps = max(0, ESCAPE_REVERSE_STEPS - 1)
                escape_turn_steps = ESCAPE_TURN_STEPS
                left_speed = ESCAPE_REVERSE_SPEED
                right_speed = ESCAPE_REVERSE_SPEED

            print(f"[{robot_name}] Corner recovery activated")
        elif front_blocked:
            # Turn away from whichever side has the closer obstacle
            if values[0] + values[1] > values[6] + values[7]:
                # Obstacle is more on the right, turn left
                left_speed = -0.5 * MAX_SPEED
                right_speed = 0.5 * MAX_SPEED
            else:
                # Obstacle is more on the left, turn right
                left_speed = 0.5 * MAX_SPEED
                right_speed = -0.5 * MAX_SPEED
        elif right_side:
            # Slight left turn to avoid right-side obstacle
            left_speed = 0.3 * MAX_SPEED
            right_speed = 0.8 * MAX_SPEED
        elif left_side:
            # Slight right turn to avoid left-side obstacle
            left_speed = 0.8 * MAX_SPEED
            right_speed = 0.3 * MAX_SPEED
        elif launch_left_speed is not None and launch_right_speed is not None:
            # Follow launch waypoints when there is no immediate obstacle pressure.
            left_speed = launch_left_speed
            right_speed = launch_right_speed
        elif assignment_left_speed is not None and assignment_right_speed is not None:
            # Drive toward the supervisor-assigned room after launch.
            left_speed = assignment_left_speed
            right_speed = assignment_right_speed
        elif coverage_left_speed is not None and coverage_right_speed is not None:
            # Sweep through the assigned room after reaching its entry target.
            left_speed = coverage_left_speed
            right_speed = coverage_right_speed
        else:
            # No obstacles, drive straight
            left_speed = CRUISE_SPEED
            right_speed = CRUISE_SPEED

        left_speed_cmd = left_speed
        right_speed_cmd = right_speed
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        if emitter is not None and step_count % COMMUNICATION_SEND_INTERVAL_STEPS == 0:
            map_update = occupancy_grid.drain_pending_updates()
            message = {
                "type": "robot_status",
                "robot": robot_name,
                "step": step_count,
                "pose": {
                    "x_m": round(robot_x_m, 3),
                    "y_m": round(robot_y_m, 3),
                    "theta_rad": round(robot_theta_rad, 3),
                },
                "launch": {
                    "waypoint_index": launch_waypoint_index,
                    "waypoint_count": len(launch_waypoints),
                    "complete": launch_waypoint_index >= len(launch_waypoints),
                    "timed_out": launch_timed_out,
                },
                "mapping_enabled": mapping_enabled,
                "assigned_room": assigned_room,
                "assignment_target_reached": assignment_target_reached,
                "assignment_route": {
                    "waypoint_index": assignment_route_index,
                    "waypoint_count": len(assignment_route),
                },
                "coverage": {
                    "room": coverage_room,
                    "waypoint_index": coverage_waypoint_index,
                    "waypoint_count": len(coverage_waypoints),
                    "complete": coverage_complete,
                },
                "free_cell_count": occupancy_grid.free_cell_count,
                "wall_cell_count": occupancy_grid.wall_cell_count,
                "last_wall_hit": last_wall_hit,
                "map_update": map_update,
            }
            emitter.send(json.dumps(message))


if __name__ == "__main__":
    run()
