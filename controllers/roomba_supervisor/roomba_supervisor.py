"""
Roomba Cluster - Central Supervisor

Receives status messages from each robot. This is the first communication
layer for the future shared map and task assignment system.
"""

import json

from controller import Supervisor

COMMUNICATION_SUMMARY_INTERVAL_STEPS = 120
EXPECTED_ROBOTS = ("epuck_1", "epuck_2", "epuck_3", "epuck_4")
GRID_UNKNOWN = -1
GRID_FREE = 0
GRID_WALL = 1


class GlobalOccupancyGrid:
    """Shared map built by merging the small updates sent by each robot."""

    def __init__(self, world_size_m=6.0, cell_size_m=0.05):
        self.world_size_m = world_size_m
        self.cell_size_m = cell_size_m
        self.width = int(round(world_size_m / cell_size_m))
        self.height = int(round(world_size_m / cell_size_m))
        self.data = [
            [GRID_UNKNOWN for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.free_cell_count = 0
        self.wall_cell_count = 0

    def is_valid_cell(self, cell):
        """Return True when a [x, y] cell index fits inside the map."""
        if not isinstance(cell, list) or len(cell) != 2:
            return False
        grid_x, grid_y = cell
        return (
            isinstance(grid_x, int)
            and isinstance(grid_y, int)
            and 0 <= grid_x < self.width
            and 0 <= grid_y < self.height
        )

    def mark_free(self, grid_x, grid_y):
        """Mark one cell as free unless it is already known to be a wall."""
        current_value = self.data[grid_y][grid_x]
        if current_value in (GRID_FREE, GRID_WALL):
            return False

        self.data[grid_y][grid_x] = GRID_FREE
        self.free_cell_count += 1
        return True

    def mark_wall(self, grid_x, grid_y):
        """Mark one cell as wall. Walls override previous free readings."""
        current_value = self.data[grid_y][grid_x]
        if current_value == GRID_WALL:
            return False

        if current_value == GRID_FREE:
            self.free_cell_count -= 1

        self.data[grid_y][grid_x] = GRID_WALL
        self.wall_cell_count += 1
        return True

    def merge_update(self, map_update):
        """Merge one robot's compact cell update into the global map."""
        if not isinstance(map_update, dict):
            return 0, 0

        new_free_cells = 0
        new_wall_cells = 0
        for cell in map_update.get("free_cells", []):
            if not self.is_valid_cell(cell):
                continue

            grid_x, grid_y = cell
            if self.mark_free(grid_x, grid_y):
                new_free_cells += 1

        for cell in map_update.get("wall_cells", []):
            if not self.is_valid_cell(cell):
                continue

            grid_x, grid_y = cell
            if self.mark_wall(grid_x, grid_y):
                new_wall_cells += 1

        return new_free_cells, new_wall_cells


def run():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    receiver = supervisor.getDevice("receiver")
    receiver.enable(timestep)

    global_grid = GlobalOccupancyGrid(world_size_m=6.0, cell_size_m=0.05)
    latest_robot_status = {}
    step_count = 0
    print("[supervisor] Starting central communication hub")
    print(
        f"[supervisor] Global map ready: "
        f"{global_grid.width}x{global_grid.height} cells"
    )

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
            new_free_cells, new_wall_cells = global_grid.merge_update(
                message.get("map_update", {})
            )

            if first_message:
                print(f"[supervisor] Connected to {robot_name}")
            if new_free_cells or new_wall_cells:
                print(
                    f"[supervisor] Map update from {robot_name}: "
                    f"free+={new_free_cells} wall+={new_wall_cells} "
                    f"global_free={global_grid.free_cell_count} "
                    f"global_walls={global_grid.wall_cell_count}"
                )

        if step_count % COMMUNICATION_SUMMARY_INTERVAL_STEPS == 0:
            connected_count = len(latest_robot_status)
            print(
                f"[supervisor] Robot status: "
                f"{connected_count}/{len(EXPECTED_ROBOTS)} reporting, "
                f"global_map free={global_grid.free_cell_count} "
                f"walls={global_grid.wall_cell_count}"
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
                    f"free={status.get('free_cell_count', 0)} "
                    f"walls={status['wall_cell_count']}"
                )


if __name__ == "__main__":
    run()
