"""
Roomba Cluster - Central Supervisor

Receives status messages from each robot. This is the first communication
layer for the future shared map and task assignment system.
"""

import json
import math

from controller import Supervisor

COMMUNICATION_SUMMARY_INTERVAL_STEPS = 120
POSE_CORRECTION_INTERVAL_STEPS = 10
EXPECTED_ROBOTS = ("epuck_1", "epuck_2", "epuck_3", "epuck_4")
ROBOT_DEF_NAMES = {
    "epuck_1": "EPUCK_1",
    "epuck_2": "EPUCK_2",
    "epuck_3": "EPUCK_3",
    "epuck_4": "EPUCK_4",
}
ROOM_ASSIGNMENT_PREVIEW_DELAY_STEPS = 60
MRTA_ASSIGNMENT_DELAY_STEPS = 180
MRTA_DISTANCE_WEIGHT = 1.0
MRTA_AREA_WEIGHT = 0.05
COVERAGE_MARGIN_M = 0.125
COVERAGE_ROW_SPACING_M = 0.35
CLEAN_TILE_SIZE_M = 0.25
CLEAN_RADIUS_M = 0.18
CLEAN_TRAIL_SAMPLE_SPACING_M = 0.08
PREVIEW_TILE_COLOR = [0.34, 0.47, 0.66]
PREVIEW_TILE_TRANSPARENCY = 0.60
DIRTY_TILE_COLOR = [0.72, 0.56, 0.32]
DIRTY_TILE_TRANSPARENCY = 0.35
CLEAN_TILE_COLOR = [0.18, 0.62, 0.42]
CLEAN_TILE_TRANSPARENCY = 0.15
HIDDEN_TILE_TRANSPARENCY = 1.0
GRID_UNKNOWN = -1
GRID_FREE = 0
GRID_WALL = 1

ROOM_TASKS = {
    "northwest": {
        "center": (-1.75, 1.75),
        "bounds": (-3.0, -0.5, 0.5, 3.0),
        "area_m2": 2.5 * 2.5,
    },
    "northeast": {
        "center": (1.75, 1.75),
        "bounds": (0.5, 3.0, 0.5, 3.0),
        "area_m2": 2.5 * 2.5,
    },
    "southeast": {
        "center": (1.75, -1.75),
        "bounds": (0.5, 3.0, -3.0, -0.5),
        "area_m2": 2.5 * 2.5,
    },
    "southwest": {
        "center": (-1.75, -1.75),
        "bounds": (-3.0, -0.5, -3.0, -0.5),
        "area_m2": 2.5 * 2.5,
    },
}

ROOM_ASSIGNMENT_PREVIEW = {
    "epuck_1": "northwest",
    "epuck_2": "northeast",
    "epuck_3": "southeast",
    "epuck_4": "southwest",
}


def normalize_angle(angle_rad):
    """Keep an angle between -pi and pi."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


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


def generate_coverage_waypoints(room):
    """Create simple back-and-forth sweep waypoints for one room."""
    min_x, max_x, min_y, max_y = ROOM_TASKS[room]["bounds"]
    sweep_min_x = min_x + COVERAGE_MARGIN_M
    sweep_max_x = max_x - COVERAGE_MARGIN_M
    sweep_min_y = min_y + COVERAGE_MARGIN_M
    sweep_max_y = max_y - COVERAGE_MARGIN_M

    waypoints = []
    row_positions = []
    y_m = sweep_min_y
    while y_m <= sweep_max_y + 1e-6:
        row_positions.append(y_m)
        y_m += COVERAGE_ROW_SPACING_M
    if not row_positions or row_positions[-1] < sweep_max_y - 1e-6:
        row_positions.append(sweep_max_y)

    for row_index, y_m in enumerate(row_positions):
        if row_index % 2 == 0:
            waypoints.append([round(sweep_min_x, 3), round(y_m, 3)])
            waypoints.append([round(sweep_max_x, 3), round(y_m, 3)])
        else:
            waypoints.append([round(sweep_max_x, 3), round(y_m, 3)])
            waypoints.append([round(sweep_min_x, 3), round(y_m, 3)])

    return waypoints


def get_actual_robot_pose(supervisor, robot_name):
    """Read the robot's real Webots position from the supervisor."""
    def_name = ROBOT_DEF_NAMES.get(robot_name)
    if def_name is None:
        return None

    robot_node = supervisor.getFromDef(def_name)
    if robot_node is None:
        return None

    translation_field = robot_node.getField("translation")
    if translation_field is None:
        return None

    x_m, y_m, _ = translation_field.getSFVec3f()
    pose = {"x_m": x_m, "y_m": y_m}

    rotation_field = robot_node.getField("rotation")
    if rotation_field is not None:
        rotation = rotation_field.getSFRotation()
        z_axis = rotation[2]
        theta_rad = rotation[3] if z_axis >= 0.0 else -rotation[3]
        pose["theta_rad"] = normalize_angle(theta_rad)

    return pose


def interpolate_cleaning_path(start_pose, end_pose):
    """Return points close enough together that cleaning does not skip tiles."""
    if start_pose is None:
        return [(end_pose["x_m"], end_pose["y_m"])]

    start_x_m = start_pose["x_m"]
    start_y_m = start_pose["y_m"]
    end_x_m = end_pose["x_m"]
    end_y_m = end_pose["y_m"]
    distance_m = math.hypot(end_x_m - start_x_m, end_y_m - start_y_m)
    sample_count = max(1, int(math.ceil(distance_m / CLEAN_TRAIL_SAMPLE_SPACING_M)))

    points = []
    for sample_index in range(sample_count + 1):
        ratio = sample_index / sample_count
        points.append(
            (
                start_x_m + ratio * (end_x_m - start_x_m),
                start_y_m + ratio * (end_y_m - start_y_m),
            )
        )

    return points


def robot_can_mark_cleaning(robot_status):
    """Cleaning starts after the robot reaches its assigned room."""
    return bool(robot_status.get("assignment_target_reached", False))


class CleaningOverlay:
    """Draw floor tiles that change color as robots clean their rooms."""

    def __init__(self, supervisor, rooms):
        self.supervisor = supervisor
        self.rooms = rooms
        self.root_children = supervisor.getRoot().getField("children")
        self.tile_appearances = {}
        self.dirty_tiles = set()
        self.preview_rooms = set()
        self.active_rooms = set()
        self.enabled = True
        self.create_tiles()

    def create_tiles(self):
        """Create flat dirty tiles over each room floor."""
        for room, config in self.rooms.items():
            min_x, max_x, min_y, max_y = config["bounds"]
            row_index = 0
            y_m = min_y + 0.5 * CLEAN_TILE_SIZE_M
            while y_m < max_y:
                col_index = 0
                x_m = min_x + 0.5 * CLEAN_TILE_SIZE_M
                while x_m < max_x:
                    tile_key = (room, col_index, row_index)
                    def_name = f"CLEAN_TILE_{room}_{col_index}_{row_index}".upper()
                    tile_source = (
                        f"DEF {def_name} Solid {{ "
                        f"translation {x_m:.3f} {y_m:.3f} 0.004 "
                        "children [ Shape { "
                        "appearance PBRAppearance { "
                        "baseColor 0.72 0.56 0.32 "
                        f"transparency {HIDDEN_TILE_TRANSPARENCY:.1f} "
                        "roughness 0.9 "
                        "} "
                        f"geometry Box {{ size {CLEAN_TILE_SIZE_M * 0.82:.3f} "
                        f"{CLEAN_TILE_SIZE_M * 0.82:.3f} 0.004 }} "
                        "} ] "
                        "}"
                    )
                    try:
                        self.root_children.importMFNodeFromString(-1, tile_source)
                        node = self.supervisor.getFromDef(def_name)
                        shape = node.getField("children").getMFNode(0)
                        appearance = shape.getField("appearance").getSFNode()
                        self.tile_appearances[tile_key] = appearance
                        self.dirty_tiles.add(tile_key)
                    except Exception as exc:
                        self.enabled = False
                        print(f"[supervisor] Cleaning overlay disabled: {exc}")
                        return
                    x_m += CLEAN_TILE_SIZE_M
                    col_index += 1
                y_m += CLEAN_TILE_SIZE_M
                row_index += 1

        print(f"[supervisor] Cleaning overlay ready: {len(self.dirty_tiles)} dirty tiles")

    def show_preview_room(self, room):
        """Show one planned room without enabling cleaning marks yet."""
        if not self.enabled or room in self.preview_rooms or room in self.active_rooms:
            return

        for tile_key in self.dirty_tiles:
            tile_room, _, _ = tile_key
            if tile_room != room:
                continue

            appearance = self.tile_appearances[tile_key]
            appearance.getField("baseColor").setSFColor(PREVIEW_TILE_COLOR)
            appearance.getField("transparency").setSFFloat(PREVIEW_TILE_TRANSPARENCY)

        self.preview_rooms.add(room)

    def show_dirty_room(self, room):
        """Make one assigned room's dirty tiles visible."""
        if not self.enabled or room in self.active_rooms:
            return

        for tile_key in self.dirty_tiles:
            tile_room, _, _ = tile_key
            if tile_room != room:
                continue

            appearance = self.tile_appearances[tile_key]
            appearance.getField("baseColor").setSFColor(DIRTY_TILE_COLOR)
            appearance.getField("transparency").setSFFloat(DIRTY_TILE_TRANSPARENCY)

        self.preview_rooms.discard(room)
        self.active_rooms.add(room)

    def mark_clean_near(self, room, x_m, y_m):
        """Turn nearby dirty tiles green."""
        if not self.enabled or room not in self.active_rooms:
            return 0

        cleaned_count = 0
        for tile_key in list(self.dirty_tiles):
            tile_room, col_index, row_index = tile_key
            if tile_room != room:
                continue

            min_x, _, min_y, _ = self.rooms[room]["bounds"]
            tile_x_m = min_x + (col_index + 0.5) * CLEAN_TILE_SIZE_M
            tile_y_m = min_y + (row_index + 0.5) * CLEAN_TILE_SIZE_M
            if math.hypot(tile_x_m - x_m, tile_y_m - y_m) > CLEAN_RADIUS_M:
                continue

            appearance = self.tile_appearances[tile_key]
            appearance.getField("baseColor").setSFColor(CLEAN_TILE_COLOR)
            appearance.getField("transparency").setSFFloat(CLEAN_TILE_TRANSPARENCY)
            self.dirty_tiles.remove(tile_key)
            cleaned_count += 1

        return cleaned_count

    def mark_clean_trail(self, room, start_pose, end_pose):
        """Turn dirty tiles green along the robot's recent movement path."""
        cleaned_count = 0
        for x_m, y_m in interpolate_cleaning_path(start_pose, end_pose):
            cleaned_count += self.mark_clean_near(room, x_m, y_m)
        return cleaned_count


def run():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    receiver = supervisor.getDevice("receiver")
    receiver.enable(timestep)
    emitter = supervisor.getDevice("emitter")

    global_grid = GlobalOccupancyGrid(world_size_m=6.0, cell_size_m=0.05)
    cleaning_overlay = CleaningOverlay(supervisor, ROOM_TASKS)
    task_allocator = TaskAllocator(
        rooms=ROOM_TASKS,
        distance_weight=MRTA_DISTANCE_WEIGHT,
        area_weight=MRTA_AREA_WEIGHT,
    )
    latest_robot_status = {}
    last_cleaning_poses = {}
    room_assignments = {}
    coverage_plans = {}
    preview_sent = False
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

            assignment = room_assignments.get(robot_name)
            pose = get_actual_robot_pose(supervisor, robot_name)
            if pose is None:
                pose = message["pose"]

            if assignment is not None and robot_can_mark_cleaning(message):
                cleaned_count = cleaning_overlay.mark_clean_trail(
                    assignment["room"],
                    last_cleaning_poses.get(robot_name),
                    pose,
                )
                if cleaned_count:
                    print(
                        f"[supervisor] Cleaned {cleaned_count} tile(s) in "
                        f"{assignment['room']} with {robot_name}; "
                        f"dirty_remaining={len(cleaning_overlay.dirty_tiles)}"
                    )
                last_cleaning_poses[robot_name] = pose
            else:
                last_cleaning_poses.pop(robot_name, None)

        if step_count % POSE_CORRECTION_INTERVAL_STEPS == 0:
            for robot_name in EXPECTED_ROBOTS:
                pose = get_actual_robot_pose(supervisor, robot_name)
                if pose is None:
                    continue

                emitter.send(
                    json.dumps(
                        {
                            "type": "pose_correction",
                            "robot": robot_name,
                            "pose": {
                                "x_m": round(pose["x_m"], 4),
                                "y_m": round(pose["y_m"], 4),
                                "theta_rad": round(pose.get("theta_rad", 0.0), 4),
                            },
                        }
                    )
                )

        ready_for_preview = (
            not preview_sent
            and step_count >= ROOM_ASSIGNMENT_PREVIEW_DELAY_STEPS
            and all(robot in latest_robot_status for robot in EXPECTED_ROBOTS)
        )
        if ready_for_preview:
            for robot_name in EXPECTED_ROBOTS:
                room = ROOM_ASSIGNMENT_PREVIEW[robot_name]
                cleaning_overlay.show_preview_room(room)
                print(f"[supervisor] Preview assignment {robot_name} -> {room}")
            preview_sent = True

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
                    cleaning_overlay.show_dirty_room(assignment["room"])
                    command = {
                        "type": "task_assignment",
                        "robot": robot_name,
                        "room": assignment["room"],
                        "target": assignment["target"],
                        "cost": assignment["cost"],
                    }
                    emitter.send(json.dumps(command))
                    coverage_plans[robot_name] = generate_coverage_waypoints(
                        assignment["room"]
                    )
                    plan_command = {
                        "type": "coverage_plan",
                        "robot": robot_name,
                        "room": assignment["room"],
                        "waypoints": coverage_plans[robot_name],
                    }
                    emitter.send(json.dumps(plan_command))
                    print(
                        f"[supervisor] Assigned {robot_name} -> "
                        f"{assignment['room']} cost={assignment['cost']:.3f} "
                        f"coverage_waypoints={len(coverage_plans[robot_name])}"
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
                actual_pose = get_actual_robot_pose(supervisor, robot_name)
                launch = status["launch"]
                assignment = room_assignments.get(robot_name)
                room = "unassigned"
                if assignment is not None:
                    room = assignment["room"]
                coverage = status.get("coverage", {})
                pose_text = f"pose=({pose['x_m']:.2f}, {pose['y_m']:.2f}) "
                if actual_pose is not None:
                    pose_text += (
                        f"actual=({actual_pose['x_m']:.2f}, "
                        f"{actual_pose['y_m']:.2f}) "
                    )
                print(
                    f"[supervisor]   {robot_name}: "
                    f"{pose_text}"
                    f"launch={launch['waypoint_index']}/{launch['waypoint_count']} "
                    f"room={room} "
                    f"coverage={coverage.get('waypoint_index', 0)}/"
                    f"{coverage.get('waypoint_count', 0)} "
                    f"free={status.get('free_cell_count', 0)} "
                    f"walls={status['wall_cell_count']}"
                )


if __name__ == "__main__":
    run()
