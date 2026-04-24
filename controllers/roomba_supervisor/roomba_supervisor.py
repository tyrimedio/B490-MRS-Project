"""
Roomba Cluster - Central Supervisor

Receives status messages from each robot. This is the first communication
layer for the future shared map and task assignment system.
"""

import json

from controller import Supervisor

COMMUNICATION_SUMMARY_INTERVAL_STEPS = 120
EXPECTED_ROBOTS = ("epuck_1", "epuck_2", "epuck_3", "epuck_4")


def run():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    receiver = supervisor.getDevice("receiver")
    receiver.enable(timestep)

    latest_robot_status = {}
    step_count = 0
    print("[supervisor] Starting central communication hub")

    while supervisor.step(timestep) != -1:
        step_count += 1

        while receiver.getQueueLength() > 0:
            raw_message = receiver.getString()
            receiver.nextPacket()

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                print(f"[supervisor] Ignoring unreadable message: {raw_message}")
                continue

            if message.get("type") != "robot_status":
                print(f"[supervisor] Ignoring unknown message: {message}")
                continue

            robot_name = message.get("robot", "unknown_robot")
            first_message = robot_name not in latest_robot_status
            latest_robot_status[robot_name] = message

            if first_message:
                print(f"[supervisor] Connected to {robot_name}")

        if step_count % COMMUNICATION_SUMMARY_INTERVAL_STEPS == 0:
            connected_count = len(latest_robot_status)
            print(
                f"[supervisor] Robot status: "
                f"{connected_count}/{len(EXPECTED_ROBOTS)} reporting"
            )
            for robot_name in EXPECTED_ROBOTS:
                status = latest_robot_status.get(robot_name)
                if status is None:
                    print(f"[supervisor]   {robot_name}: no message yet")
                    continue

                pose = status["pose"]
                launch = status["launch"]
                print(
                    f"[supervisor]   {robot_name}: "
                    f"pose=({pose['x_m']:.2f}, {pose['y_m']:.2f}) "
                    f"launch={launch['waypoint_index']}/{launch['waypoint_count']} "
                    f"map_cells={status['wall_cell_count']}"
                )


if __name__ == "__main__":
    run()
