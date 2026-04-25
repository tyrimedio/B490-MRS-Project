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
# Assignment is already gated on every robot finishing its launch waypoints,
# so we don't need an extra time floor — run MRTA the moment launches complete.
MRTA_ASSIGNMENT_DELAY_STEPS = 0
MRTA_DISTANCE_WEIGHT = 1.0
MRTA_AREA_WEIGHT = 0.05
COVERAGE_MARGIN_M = 0.125
COVERAGE_ROW_SPACING_M = 0.35
COVERAGE_COMPLETE_PERCENT = 95.0
CLEAN_TILE_SIZE_M = 0.25
CLEAN_RADIUS_M = 0.18
CLEAN_TRAIL_SAMPLE_SPACING_M = 0.08
DIRTY_TILE_COLOR = [0.72, 0.56, 0.32]
DIRTY_TILE_TRANSPARENCY = 0.35
CLEAN_TILE_COLOR = [0.18, 0.62, 0.42]
CLEAN_TILE_TRANSPARENCY = 0.15
HIDDEN_TILE_TRANSPARENCY = 1.0
GRID_UNKNOWN = -1
GRID_FREE = 0
GRID_WALL = 1

ROOM_TASKS = {
    "nw_small": {
        "center": (-2.25, 1.875),
        "bounds": (-3.0, -1.5, 0.75, 3.0),
        "area_m2": 1.5 * 2.25,
    },
    "n_medium": {
        "center": (-0.4, 1.875),
        "bounds": (-1.5, 0.7, 0.75, 3.0),
        "area_m2": 2.2 * 2.25,
    },
    "ne_large": {
        "center": (1.85, 1.875),
        "bounds": (0.7, 3.0, 0.75, 3.0),
        "area_m2": 2.3 * 2.25,
    },
    "sw_large": {
        "center": (-1.85, -1.875),
        "bounds": (-3.0, -0.7, -3.0, -0.75),
        "area_m2": 2.3 * 2.25,
    },
    "s_medium": {
        "center": (0.4, -1.875),
        "bounds": (-0.7, 1.5, -3.0, -0.75),
        "area_m2": 2.2 * 2.25,
    },
    "se_small": {
        "center": (2.25, -1.875),
        "bounds": (1.5, 3.0, -3.0, -0.75),
        "area_m2": 1.5 * 2.25,
    },
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
        if not robots or len(robots) > len(rooms):
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


def robot_can_mark_cleaning(pose, room):
    """Cleaning marks turn on once the robot is inside the assigned room's bounds."""
    if pose is None or room not in ROOM_TASKS:
        return False

    min_x, max_x, min_y, max_y = ROOM_TASKS[room]["bounds"]
    return (
        min_x <= pose["x_m"] <= max_x
        and min_y <= pose["y_m"] <= max_y
    )


class CleaningOverlay:
    """Draw floor tiles that change color as robots clean their rooms."""

    def __init__(self, supervisor, rooms):
        self.supervisor = supervisor
        self.rooms = rooms
        self.root_children = supervisor.getRoot().getField("children")
        self.tile_appearances = {}
        self.dirty_tiles = set()
        self.room_tile_counts = {room: 0 for room in rooms}
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
                        self.room_tile_counts[room] += 1
                    except Exception as exc:
                        self.enabled = False
                        print(f"[supervisor] Cleaning overlay disabled: {exc}")
                        return
                    x_m += CLEAN_TILE_SIZE_M
                    col_index += 1
                y_m += CLEAN_TILE_SIZE_M
                row_index += 1

        print(f"[supervisor] Cleaning overlay ready: {len(self.dirty_tiles)} dirty tiles")

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

    def dirty_tile_count(self, room):
        """Return how many cleaning tiles in one room are still dirty."""
        return sum(1 for tile_room, _, _ in self.dirty_tiles if tile_room == room)

    def cleaned_tile_count(self, room):
        """Return how many cleaning tiles in one room have been cleaned."""
        total_tiles = self.room_tile_counts.get(room, 0)
        return max(0, total_tiles - self.dirty_tile_count(room))

    def room_progress_percent(self, room):
        """Return how much of one room is clean, from 0 to 100."""
        total_tiles = self.room_tile_counts.get(room, 0)
        if total_tiles <= 0:
            return 0.0
        return 100.0 * self.cleaned_tile_count(room) / total_tiles


def room_reached_coverage_goal(cleaning_overlay, room):
    """Return True when enough visible cleaning tiles are clean."""
    return cleaning_overlay.room_progress_percent(room) >= COVERAGE_COMPLETE_PERCENT


def room_progress_snapshot(cleaning_overlay):
    """Return clean percentages for every room in a stable order."""
    return {
        room: round(cleaning_overlay.room_progress_percent(room), 1)
        for room in ROOM_TASKS
    }


def format_room_progress(snapshot):
    """Format room progress for one compact supervisor log line."""
    return " ".join(
        f"{room}={snapshot.get(room, 0.0):.1f}%"
        for room in ROOM_TASKS
    )


def select_reassignment_room(
    robot_name,
    room_assignments,
    completed_rooms,
    cleaning_overlay,
):
    """
    Pick the unfinished room that most needs help.

    This is a simple dynamic MRTA step: when one robot finishes, it helps the
    least-clean unfinished room instead of sitting idle.
    """
    current_assignment = room_assignments.get(robot_name)
    current_room = None
    if current_assignment is not None:
        current_room = current_assignment["room"]

    helper_rooms = {
        assignment["room"]
        for robot, assignment in room_assignments.items()
        if robot != robot_name and assignment.get("helper", False)
    }
    candidate_rooms = []
    for room in ROOM_TASKS:
        if room == current_room or room in completed_rooms or room in helper_rooms:
            continue
        progress_percent = cleaning_overlay.room_progress_percent(room)
        if progress_percent >= COVERAGE_COMPLETE_PERCENT:
            continue
        candidate_rooms.append((progress_percent, room))

    if not candidate_rooms:
        return None

    candidate_rooms.sort()
    return candidate_rooms[0][1]


def build_assignment(robot_status, room, helper=False):
    """Create one assignment dictionary from the robot pose to a room."""
    room_x_m, room_y_m = ROOM_TASKS[room]["center"]
    pose = robot_status["pose"]
    distance_m = math.hypot(room_x_m - pose["x_m"], room_y_m - pose["y_m"])
    area_m2 = ROOM_TASKS[room]["area_m2"]
    cost = MRTA_DISTANCE_WEIGHT * distance_m + MRTA_AREA_WEIGHT * area_m2
    return {
        "room": room,
        "cost": round(cost, 3),
        "target": ROOM_TASKS[room]["center"],
        "helper": helper,
    }


def send_assignment_commands(emitter, robot_name, assignment):
    """Send the room target and coverage path to one robot."""
    emitter.send(
        json.dumps(
            {
                "type": "task_assignment",
                "robot": robot_name,
                "room": assignment["room"],
                "target": assignment["target"],
                "cost": assignment["cost"],
            }
        )
    )
    coverage_plan = generate_coverage_waypoints(assignment["room"])
    emitter.send(
        json.dumps(
            {
                "type": "coverage_plan",
                "robot": robot_name,
                "room": assignment["room"],
                "waypoints": coverage_plan,
            }
        )
    )
    return coverage_plan


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
    completed_rooms = set()
    completed_robot_rooms = {}
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

            if assignment is not None and robot_can_mark_cleaning(pose, assignment["room"]):
                cleaned_count = cleaning_overlay.mark_clean_trail(
                    assignment["room"],
                    last_cleaning_poses.get(robot_name),
                    pose,
                )
                if cleaned_count:
                    room_progress = cleaning_overlay.room_progress_percent(
                        assignment["room"]
                    )
                    print(
                        f"[supervisor] Cleaned {cleaned_count} tile(s) in "
                        f"{assignment['room']} with {robot_name}; "
                        f"room_clean={room_progress:.1f}%"
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
                    coverage_plans[robot_name] = send_assignment_commands(
                        emitter,
                        robot_name,
                        assignment,
                    )
                    print(
                        f"[supervisor] Assigned {robot_name} -> "
                        f"{assignment['room']} cost={assignment['cost']:.3f} "
                        f"coverage_waypoints={len(coverage_plans[robot_name])}"
                    )
                assignments_sent = True

        if assignments_sent:
            for robot_name in EXPECTED_ROBOTS:
                status = latest_robot_status.get(robot_name)
                assignment = room_assignments.get(robot_name)
                if status is None or assignment is None:
                    continue

                room = assignment["room"]
                if not room_reached_coverage_goal(cleaning_overlay, room):
                    continue

                if room not in completed_rooms:
                    completed_rooms.add(room)
                    print(
                        f"[supervisor] Room {room} reached "
                        f"{cleaning_overlay.room_progress_percent(room):.1f}% clean"
                    )

                if completed_robot_rooms.get(robot_name) == room:
                    continue

                completed_robot_rooms[robot_name] = room
                next_room = select_reassignment_room(
                    robot_name,
                    room_assignments,
                    completed_rooms,
                    cleaning_overlay,
                )
                if next_room is None:
                    print(f"[supervisor] {robot_name} has no unfinished room to help")
                    continue

                next_assignment = build_assignment(status, next_room, helper=True)
                room_assignments[robot_name] = next_assignment
                cleaning_overlay.show_dirty_room(next_room)
                coverage_plans[robot_name] = send_assignment_commands(
                    emitter,
                    robot_name,
                    next_assignment,
                )
                last_cleaning_poses.pop(robot_name, None)
                print(
                    f"[supervisor] Reassigned {robot_name} to help {next_room}; "
                    f"room_clean={cleaning_overlay.room_progress_percent(next_room):.1f}% "
                    f"coverage_waypoints={len(coverage_plans[robot_name])}"
                )

        if step_count % COMMUNICATION_SUMMARY_INTERVAL_STEPS == 0:
            connected_count = len(latest_robot_status)
            progress_snapshot = room_progress_snapshot(cleaning_overlay)
            print(
                f"[supervisor] Robot status: "
                f"{connected_count}/{len(EXPECTED_ROBOTS)} reporting, "
                f"global_map free={global_grid.free_cell_count} "
                f"walls={global_grid.wall_cell_count} "
                f"room_progress {format_room_progress(progress_snapshot)}"
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
                    if assignment.get("helper", False):
                        room += " (helping)"
                coverage = status.get("coverage", {})
                room_progress = 0.0
                if assignment is not None:
                    room_progress = cleaning_overlay.room_progress_percent(
                        assignment["room"]
                    )
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
                    f"clean={room_progress:.1f}% "
                    f"free={status.get('free_cell_count', 0)} "
                    f"walls={status['wall_cell_count']}"
                )


if __name__ == "__main__":
    run()
