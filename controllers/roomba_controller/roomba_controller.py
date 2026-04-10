"""
Roomba Cluster - E-puck Controller

Basic wall-following and obstacle-avoidance behavior for the e-puck robot.
This serves as the foundation for SLAM exploration and cleaning coverage.
Each robot drives forward and turns when it detects an obstacle, covering
its assigned area with a sweeping pattern.
"""

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
LIDAR_PRINT_EVERY_STEPS = 10
LIDAR_VALUE_AT_1_METER = 1000.0
LIDAR_MAX_RANGE_M = 1.0
LIDAR_NO_HIT_MARGIN = 1.0

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


class OccupancyGrid:
    """
    Small 2D map stored as a grid (like graph paper).

    Each cell starts as UNKNOWN and can later be marked FREE or WALL.
    This class currently handles grid allocation and reset.
    """

    def __init__(self, world_size_m=6.0, cell_size_m=0.05):
        self.world_size_m = world_size_m
        self.cell_size_m = cell_size_m
        self.width = int(round(world_size_m / cell_size_m))
        self.height = int(round(world_size_m / cell_size_m))
        self.data = []
        self.wall_cell_count = 0
        self.reset()

    def reset(self):
        """Reset every cell to UNKNOWN."""
        self.data = [
            [GRID_UNKNOWN for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.wall_cell_count = 0

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
        if self.data[grid_y][grid_x] != GRID_WALL:
            self.data[grid_y][grid_x] = GRID_WALL
            self.wall_cell_count += 1
            return True
        return False


def normalize_angle(angle_rad):
    """Keep an angle in the range [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def run():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    robot_name = robot.getName()
    print(f"[{robot_name}] Starting up")

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
    lidar = None
    try:
        lidar = robot.getDevice("lidar")
    except Exception:
        lidar = None

    if lidar is not None:
        lidar.enable(timestep)
        print(f"[{robot_name}] Lidar enabled")
    else:
        print(f"[{robot_name}] Lidar not found")

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
    robot_x_m = 0.0
    robot_y_m = 0.0
    robot_theta_rad = 0.0
    corner_pressure_steps = 0
    escape_reverse_steps = 0
    escape_turn_steps = 0
    escape_turn_left = True

    while robot.step(timestep) != -1:
        step_count += 1

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

        if lidar is not None:
            lidar_value = lidar.getValue()
            has_lidar_hit = lidar_value < (LIDAR_VALUE_AT_1_METER - LIDAR_NO_HIT_MARGIN)
            if has_lidar_hit:
                lidar_distance_m = min(
                    LIDAR_MAX_RANGE_M,
                    max(0.0, lidar_value / LIDAR_VALUE_AT_1_METER),
                )
                hit_x_m = robot_x_m + lidar_distance_m * math.cos(robot_theta_rad)
                hit_y_m = robot_y_m + lidar_distance_m * math.sin(robot_theta_rad)
                marked = occupancy_grid.mark_wall(hit_x_m, hit_y_m)

                if step_count % LIDAR_PRINT_EVERY_STEPS == 0:
                    update_tag = "new" if marked else "repeat"
                    print(
                        f"[{robot_name}] lidar={lidar_value:.1f} "
                        f"dist={lidar_distance_m:.3f}m "
                        f"wall={update_tag} total={occupancy_grid.wall_cell_count}"
                    )
            elif step_count % LIDAR_PRINT_EVERY_STEPS == 0:
                print(f"[{robot_name}] lidar={lidar_value:.1f} no_hit")

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

        # Decide motor speeds based on obstacles
        if escape_reverse_steps > 0 and not rear_blocked:
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
        else:
            # No obstacles, drive straight
            left_speed = CRUISE_SPEED
            right_speed = CRUISE_SPEED

        left_speed_cmd = left_speed
        right_speed_cmd = right_speed
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)


if __name__ == "__main__":
    run()
