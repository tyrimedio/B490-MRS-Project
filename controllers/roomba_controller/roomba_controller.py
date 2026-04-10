"""
Roomba Cluster - E-puck Controller

Basic wall-following and obstacle-avoidance behavior for the e-puck robot.
This serves as the foundation for SLAM exploration and cleaning coverage.
Each robot drives forward and turns when it detects an obstacle, covering
its assigned area with a sweeping pattern.
"""

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


def run():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    robot_name = robot.getName()
    print(f"[{robot_name}] Starting up")

    # Initialize proximity sensors
    sensors = []
    for name in SENSOR_NAMES:
        sensor = robot.getDevice(name)
        sensor.enable(timestep)
        sensors.append(sensor)

    # Initialize turret-mounted lidar sensor (if present)
    lidar = robot.getDevice("lidar")
    if lidar:
        lidar.enable(timestep)

    # Initialize motors
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    while robot.step(timestep) != -1:
        # Read all proximity sensor values
        values = [s.getValue() for s in sensors]

        # Check for obstacles on each side
        front_left = values[7] > OBSTACLE_THRESHOLD
        front_right = values[0] > OBSTACLE_THRESHOLD
        left_side = values[6] > OBSTACLE_THRESHOLD or values[5] > OBSTACLE_THRESHOLD
        right_side = values[1] > OBSTACLE_THRESHOLD or values[2] > OBSTACLE_THRESHOLD

        front_blocked = front_left or front_right

        # Decide motor speeds based on obstacles
        if front_blocked:
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
            left_speed = MAX_SPEED
            right_speed = MAX_SPEED

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)


if __name__ == "__main__":
    run()
