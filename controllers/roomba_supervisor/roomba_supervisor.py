"""
Roomba Cluster - Central Supervisor

Receives status messages from each robot. This is the first communication
layer for the future shared map and task assignment system.
"""

import json
import math

from controller import Supervisor

COMMUNICATION_SUMMARY_INTERVAL_STEPS = 120
EXPECTED_ROBOTS = ("epuck_1", "epuck_2", "epuck_3", "epuck_4")
MRTA_ASSIGNMENT_DELAY_STEPS = 180
MRTA_DISTANCE_WEIGHT = 1.0
MRTA_AREA_WEIGHT = 0.05
GRID_UNKNOWN = -1
GRID_FREE = 0
GRID_WALL = 1

ROOM_TASKS = {
    "northwest": {
        "center": (-1.75, 1.75),
        "area_m2": 2.5 * 2.5,
    },
    "northeast": {
        "center": (1.75, 1.75),
        "area_m2": 2.5 * 2.5,
    },
    "southeast": {
        "center": (1.75, -1.75),
        "area_m2": 2.5 * 2.5,
    },
    "southwest": {
        "center": (-1.75, -1.75),
        "area_m2": 2.5 * 2.5,
    },
}


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


class TaskAllocator:
    """
    Assign robots to rooms using a small MRTA cost function.

    Cost means "how expensive is it for this robot to clean that room." For
    now it combines travel distance and room size.
    """

    def __init__(self, rooms, distance_weight, area_weight):
        self.rooms = rooms
        self.distance_weight = distance_weight
        self.area_weight = area_weight

    def assignment_cost(self, robot_status, room):
        """Return the cost of sending one robot to one room."""
        pose = robot_status["pose"]
        room_x_m, room_y_m = self.rooms[room]["center"]
        distance_m = math.hypot(
            room_x_m - pose["x_m"],
            room_y_m - pose["y_m"],
        )
        area_m2 = self.rooms[room]["area_m2"]
        return self.distance_weight * distance_m + self.area_weight * area_m2

    def assign(self, robot_statuses):
        """Return room assignments that minimize total cost for this small team."""
        robots = [robot for robot in EXPECTED_ROBOTS if robot in robot_statuses]
        rooms = list(self.rooms)
        if len(robots) != len(rooms):
            return {}

        best_assignments = None
        best_total_cost = None
        remaining_rooms = set(rooms)

        def search(robot_index, current_assignments, current_cost):
            nonlocal best_assignments, best_total_cost

            if robot_index >= len(robots):
                if best_total_cost is None or current_cost < best_total_cost:
                    best_total_cost = current_cost
                    best_assignments = dict(current_assignments)
                return

            robot = robots[robot_index]
            for room in sorted(remaining_rooms):
                room_cost = self.assignment_cost(robot_statuses[robot], room)
                total_cost = current_cost + room_cost
                if best_total_cost is not None and total_cost >= best_total_cost:
                    continue

                remaining_rooms.remove(room)
                current_assignments[robot] = {
                    "room": room,
                    "cost": round(room_cost, 3),
                    "target": self.rooms[room]["center"],
                }
                search(robot_index + 1, current_assignments, total_cost)
                current_assignments.pop(robot)
                remaining_rooms.add(room)

        search(0, {}, 0.0)
        return best_assignments or {}


def run():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    receiver = supervisor.getDevice("receiver")
    receiver.enable(timestep)
    emitter = supervisor.getDevice("emitter")

    global_grid = GlobalOccupancyGrid(world_size_m=6.0, cell_size_m=0.05)
    task_allocator = TaskAllocator(
        rooms=ROOM_TASKS,
        distance_weight=MRTA_DISTANCE_WEIGHT,
        area_weight=MRTA_AREA_WEIGHT,
    )
    latest_robot_status = {}
    room_assignments = {}
    assignments_sent = False
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

        ready_for_assignment = (
            not assignments_sent
            and step_count >= MRTA_ASSIGNMENT_DELAY_STEPS
            and all(robot in latest_robot_status for robot in EXPECTED_ROBOTS)
            and all(
                latest_robot_status[robot]["launch"]["complete"]
                or latest_robot_status[robot]["launch"].get("timed_out", False)
                for robot in EXPECTED_ROBOTS
            )
        )
        if ready_for_assignment:
            room_assignments = task_allocator.assign(latest_robot_status)
            if room_assignments:
                for robot_name, assignment in room_assignments.items():
                    command = {
                        "type": "task_assignment",
                        "robot": robot_name,
                        "room": assignment["room"],
                        "target": assignment["target"],
                        "cost": assignment["cost"],
                    }
                    emitter.send(json.dumps(command))
                    print(
                        f"[supervisor] Assigned {robot_name} -> "
                        f"{assignment['room']} cost={assignment['cost']:.3f}"
                    )
                assignments_sent = True

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
                assignment = room_assignments.get(robot_name)
                room = "unassigned"
                if assignment is not None:
                    room = assignment["room"]
                print(
                    f"[supervisor]   {robot_name}: "
                    f"pose=({pose['x_m']:.2f}, {pose['y_m']:.2f}) "
                    f"launch={launch['waypoint_index']}/{launch['waypoint_count']} "
                    f"room={room} "
                    f"free={status.get('free_cell_count', 0)} "
                    f"walls={status['wall_cell_count']}"
                )


if __name__ == "__main__":
    run()
