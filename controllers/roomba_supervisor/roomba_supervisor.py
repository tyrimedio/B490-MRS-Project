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
COVERAGE_MARGIN_M = 0.2
COVERAGE_ROW_SPACING_M = 0.35
COVERAGE_COMPLETE_PERCENT = 100.0
CLEAN_TILE_SIZE_M = 0.25
CLEAN_RADIUS_M = 0.18
CLEAN_TRAIL_SAMPLE_SPACING_M = 0.08
CLEANUP_TARGETS_PER_ROBOT = 8
FLOOR_MIN_X_M = -3.0
FLOOR_MIN_Y_M = -3.0
DIRTY_TILE_COLOR = [0.72, 0.56, 0.32]
DIRTY_TILE_TRANSPARENCY = 0.35
CLEAN_TILE_COLOR = [0.18, 0.62, 0.42]
CLEAN_TILE_TRANSPARENCY = 0.15
HIDDEN_TILE_TRANSPARENCY = 1.0
GRID_UNKNOWN = -1
GRID_FREE = 0
GRID_WALL = 1
FREE_EVIDENCE = -1
WALL_EVIDENCE = 3
MIN_OCCUPANCY_SCORE = -8
MAX_OCCUPANCY_SCORE = 8
FREE_SCORE_THRESHOLD = -2
WALL_SCORE_THRESHOLD = 3
HUB_ROUTE_WAYPOINT = [0.0, 0.0]
ROOM_DOORWAY_INSIDE_OFFSET_M = 0.18
ROOM_DOORWAY_HUB_OFFSET_M = 0.18

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


def room_for_pose(pose):
    """Return the room containing a pose, or None when the robot is outside rooms."""
    if pose is None:
        return None

    x_m = pose.get("x_m")
    y_m = pose.get("y_m")
    if not isinstance(x_m, (int, float)) or not isinstance(y_m, (int, float)):
        return None

    for room, config in ROOM_TASKS.items():
        min_x, max_x, min_y, max_y = config["bounds"]
        if min_x <= x_m <= max_x and min_y <= y_m <= max_y:
            return room
    return None


def room_doorway_waypoints(room):
    """
    Return doorway waypoints for entering or leaving a room.

    The first point is just inside the room. The second is just outside the
    room on the central hallway side. Driving through both keeps robots aimed
    at the doorway instead of at a wall.
    """
    center_x_m, center_y_m = ROOM_TASKS[room]["center"]
    _, _, min_y, max_y = ROOM_TASKS[room]["bounds"]
    if center_y_m > 0.0:
        wall_y_m = min_y
        inside_y_m = wall_y_m + ROOM_DOORWAY_INSIDE_OFFSET_M
        hub_y_m = wall_y_m - ROOM_DOORWAY_HUB_OFFSET_M
    else:
        wall_y_m = max_y
        inside_y_m = wall_y_m - ROOM_DOORWAY_INSIDE_OFFSET_M
        hub_y_m = wall_y_m + ROOM_DOORWAY_HUB_OFFSET_M

    return [round(center_x_m, 3), round(inside_y_m, 3)], [
        round(center_x_m, 3),
        round(hub_y_m, 3),
    ]


def append_route_waypoint(route, waypoint):
    """Append a route waypoint unless it repeats the previous point."""
    if route and route[-1] == waypoint:
        return
    route.append(waypoint)


def generate_assignment_route(pose, target_room):
    """Build a doorway-aware route from the robot's current room to a target room."""
    route = []
    current_room = room_for_pose(pose)
    target_x_m, target_y_m = ROOM_TASKS[target_room]["center"]
    target_center = [round(target_x_m, 3), round(target_y_m, 3)]

    if current_room == target_room:
        return [target_center]

    if current_room is not None and current_room != target_room:
        current_inside, current_hub = room_doorway_waypoints(current_room)
        append_route_waypoint(route, current_inside)
        append_route_waypoint(route, current_hub)

    append_route_waypoint(route, HUB_ROUTE_WAYPOINT)

    target_inside, target_hub = room_doorway_waypoints(target_room)
    append_route_waypoint(route, target_hub)
    append_route_waypoint(route, target_inside)
    append_route_waypoint(route, target_center)
    return route


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
        self.scores = [
            [0 for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.free_cell_count = 0
        self.wall_cell_count = 0

    def is_valid_cell(self, cell):
        """Return True when a [x, y] cell index fits inside the map."""
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            return False
        grid_x, grid_y = cell
        return (
            isinstance(grid_x, int)
            and isinstance(grid_y, int)
            and 0 <= grid_x < self.width
            and 0 <= grid_y < self.height
        )

    def is_valid_observation(self, observation):
        """Return True when a [x, y, count] observation is usable."""
        if not isinstance(observation, list) or len(observation) != 3:
            return False
        grid_x, grid_y, count = observation
        return (
            self.is_valid_cell([grid_x, grid_y])
            and isinstance(count, int)
            and count > 0
        )

    def is_valid_ordered_observation(self, observation):
        """Return True when a [kind, x, y, count] observation is usable."""
        if not isinstance(observation, list) or len(observation) != 4:
            return False
        kind, grid_x, grid_y, count = observation
        return (
            kind in ("free", "wall")
            and self.is_valid_cell([grid_x, grid_y])
            and isinstance(count, int)
            and count > 0
        )

    def mark_free(self, grid_x, grid_y):
        """Apply enough free evidence to support older map updates."""
        if self.data[grid_y][grid_x] == GRID_WALL:
            return False
        score = min(self.scores[grid_y][grid_x], FREE_SCORE_THRESHOLD)
        return self.set_cell_score(grid_x, grid_y, score)

    def mark_wall(self, grid_x, grid_y):
        """Apply enough wall evidence to support older map updates."""
        score = max(self.scores[grid_y][grid_x], WALL_SCORE_THRESHOLD)
        return self.set_cell_score(grid_x, grid_y, score)

    def add_cell_evidence(self, grid_x, grid_y, evidence):
        """Update one cell's score and public map state."""
        old_score = self.scores[grid_y][grid_x]
        new_score = max(
            MIN_OCCUPANCY_SCORE,
            min(MAX_OCCUPANCY_SCORE, old_score + evidence),
        )
        return self.set_cell_score(grid_x, grid_y, new_score)

    def set_cell_score(self, grid_x, grid_y, new_score):
        """Set one score directly and refresh its public map state."""
        old_state = self.data[grid_y][grid_x]
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
        return True

    def merge_update(self, map_update):
        """Merge one robot's compact cell update into the global map."""
        if not isinstance(map_update, dict):
            return 0, 0

        new_free_cells = 0
        new_wall_cells = 0
        if "observations" in map_update:
            for observation in map_update.get("observations", []):
                if not self.is_valid_ordered_observation(observation):
                    continue

                kind, grid_x, grid_y, count = observation
                if kind == "free":
                    evidence = FREE_EVIDENCE * count
                else:
                    evidence = WALL_EVIDENCE * count

                if self.add_cell_evidence(grid_x, grid_y, evidence):
                    if self.data[grid_y][grid_x] == GRID_FREE:
                        new_free_cells += 1
                    elif self.data[grid_y][grid_x] == GRID_WALL:
                        new_wall_cells += 1

            return new_free_cells, new_wall_cells

        has_observations = (
            "free_observations" in map_update
            or "wall_observations" in map_update
        )
        if has_observations:
            for observation in map_update.get("free_observations", []):
                if not self.is_valid_observation(observation):
                    continue

                grid_x, grid_y, count = observation
                evidence = FREE_EVIDENCE * count
                if self.add_cell_evidence(grid_x, grid_y, evidence):
                    if self.data[grid_y][grid_x] == GRID_FREE:
                        new_free_cells += 1
                    elif self.data[grid_y][grid_x] == GRID_WALL:
                        new_wall_cells += 1

            for observation in map_update.get("wall_observations", []):
                if not self.is_valid_observation(observation):
                    continue

                grid_x, grid_y, count = observation
                evidence = WALL_EVIDENCE * count
                if self.add_cell_evidence(grid_x, grid_y, evidence):
                    if self.data[grid_y][grid_x] == GRID_FREE:
                        new_free_cells += 1
                    elif self.data[grid_y][grid_x] == GRID_WALL:
                        new_wall_cells += 1

            return new_free_cells, new_wall_cells

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
                    "route": generate_assignment_route(
                        robot_statuses[robot]["pose"],
                        room,
                    ),
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


def first_aligned_tile_center(min_m, floor_min_m):
    """Return the first floor-grid tile center at or after a room edge."""
    offset_tiles = math.ceil(
        (min_m - floor_min_m - 0.5 * CLEAN_TILE_SIZE_M) / CLEAN_TILE_SIZE_M
        - 1e-9
    )
    return round(floor_min_m + (offset_tiles + 0.5) * CLEAN_TILE_SIZE_M, 3)


class CleaningOverlay:
    """Draw floor tiles that change color as robots clean their rooms."""

    def __init__(self, supervisor, rooms):
        self.supervisor = supervisor
        self.rooms = rooms
        self.root_children = supervisor.getRoot().getField("children")
        self.tile_appearances = {}
        self.tile_centers = {}
        self.dirty_tiles = set()
        self.tile_claims = {}
        self.room_tile_counts = {room: 0 for room in rooms}
        self.active_rooms = set()
        self.enabled = True
        self.create_tiles()

    def create_tiles(self):
        """Create flat dirty tiles over each room floor."""
        for room, config in self.rooms.items():
            min_x, max_x, min_y, max_y = config["bounds"]
            row_index = 0
            y_m = first_aligned_tile_center(min_y, FLOOR_MIN_Y_M)
            while y_m < max_y:
                col_index = 0
                x_m = first_aligned_tile_center(min_x, FLOOR_MIN_X_M)
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
                        self.tile_centers[tile_key] = [round(x_m, 3), round(y_m, 3)]
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

            tile_x_m, tile_y_m = self.tile_center(tile_key)
            if math.hypot(tile_x_m - x_m, tile_y_m - y_m) > CLEAN_RADIUS_M:
                continue

            appearance = self.tile_appearances[tile_key]
            appearance.getField("baseColor").setSFColor(CLEAN_TILE_COLOR)
            appearance.getField("transparency").setSFFloat(CLEAN_TILE_TRANSPARENCY)
            self.dirty_tiles.remove(tile_key)
            self.tile_claims.pop(tile_key, None)
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

    def tile_center(self, tile_key):
        """Return the world position of one tile center."""
        if hasattr(self, "tile_centers") and tile_key in self.tile_centers:
            return self.tile_centers[tile_key]

        room, col_index, row_index = tile_key
        min_x, _, min_y, _ = self.rooms[room]["bounds"]
        start_x_m = first_aligned_tile_center(min_x, FLOOR_MIN_X_M)
        start_y_m = first_aligned_tile_center(min_y, FLOOR_MIN_Y_M)
        return [
            round(start_x_m + col_index * CLEAN_TILE_SIZE_M, 3),
            round(start_y_m + row_index * CLEAN_TILE_SIZE_M, 3),
        ]

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

    def dirty_tile_centers(self, room):
        """Return center points for the dirty tiles that remain in one room."""
        if room not in self.rooms:
            return []

        centers = []
        for tile_key in sorted(self.dirty_tiles):
            tile_room, _, _ = tile_key
            if tile_room != room:
                continue
            centers.append(self.tile_center(tile_key))
        return centers

    def release_robot_claims(self, robot_name):
        """Forget dirty-tile claims owned by one robot."""
        self.tile_claims = {
            tile_key: owner
            for tile_key, owner in self.tile_claims.items()
            if owner != robot_name
        }

    def release_room_claims(self, room):
        """Forget all dirty-tile claims in one room."""
        self.tile_claims = {
            tile_key: owner
            for tile_key, owner in self.tile_claims.items()
            if tile_key[0] != room
        }

    def prune_cleaned_claims(self):
        """Drop claims for tiles that are no longer dirty."""
        self.tile_claims = {
            tile_key: owner
            for tile_key, owner in self.tile_claims.items()
            if tile_key in self.dirty_tiles
        }

    def claim_dirty_tile_centers(
        self,
        room,
        robot_name,
        robot_pose=None,
        max_tiles=CLEANUP_TARGETS_PER_ROBOT,
    ):
        """
        Reserve a small cleanup batch for one robot.

        A claim is like putting a robot's name on a dirty square, so another
        robot does not drive to the same square unless the claim is released.
        """
        if room not in self.rooms or max_tiles <= 0:
            return []

        self.prune_cleaned_claims()
        claimed_by_robot = [
            tile_key
            for tile_key, owner in self.tile_claims.items()
            if owner == robot_name
            and tile_key in self.dirty_tiles
            and tile_key[0] == room
        ]

        open_tiles = [
            tile_key
            for tile_key in self.dirty_tiles
            if tile_key[0] == room
            and self.tile_claims.get(tile_key, robot_name) == robot_name
            and tile_key not in claimed_by_robot
        ]

        def sort_key(tile_key):
            center_x_m, center_y_m = self.tile_center(tile_key)
            if robot_pose is None:
                distance_m = 0.0
            else:
                distance_m = math.hypot(
                    center_x_m - robot_pose["x_m"],
                    center_y_m - robot_pose["y_m"],
                )
            _, col_index, row_index = tile_key
            return (distance_m, row_index, col_index)

        selected_tiles = sorted(claimed_by_robot, key=sort_key)
        selected_tiles.extend(
            sorted(open_tiles, key=sort_key)[:max(0, max_tiles - len(selected_tiles))]
        )
        selected_tiles = selected_tiles[:max_tiles]

        for tile_key in selected_tiles:
            self.tile_claims[tile_key] = robot_name

        return [self.tile_center(tile_key) for tile_key in selected_tiles]


def room_reached_coverage_goal(cleaning_overlay, room):
    """Return True when no visible cleaning tiles remain dirty."""
    return cleaning_overlay.dirty_tile_count(room) == 0


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
    least-supported unfinished room instead of sitting idle. When two rooms
    have the same number of robots working on them, it picks the dirtier room.
    """
    current_assignment = room_assignments.get(robot_name)
    current_room = None
    if current_assignment is not None:
        current_room = current_assignment["room"]

    active_room_counts = {}
    for robot, assignment in room_assignments.items():
        if robot == robot_name or assignment is None:
            continue
        room = assignment["room"]
        active_room_counts[room] = active_room_counts.get(room, 0) + 1

    candidate_rooms = []
    for room in ROOM_TASKS:
        if room == current_room or room in completed_rooms:
            continue
        progress_percent = cleaning_overlay.room_progress_percent(room)
        if progress_percent >= COVERAGE_COMPLETE_PERCENT:
            continue
        candidate_rooms.append((
            active_room_counts.get(room, 0),
            progress_percent,
            room,
        ))

    if not candidate_rooms:
        return None

    candidate_rooms.sort()
    return candidate_rooms[0][2]


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
        "route": generate_assignment_route(pose, room),
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
                "route": assignment.get("route", [assignment["target"]]),
                "cost": assignment["cost"],
            }
        )
    )
    coverage_plan = generate_coverage_waypoints(assignment["room"])
    send_coverage_plan(emitter, robot_name, assignment["room"], coverage_plan)
    return coverage_plan


def send_coverage_plan(emitter, robot_name, room, coverage_plan):
    """Send coverage waypoints to one robot."""
    emitter.send(
        json.dumps(
            {
                "type": "coverage_plan",
                "robot": robot_name,
                "room": room,
                "waypoints": coverage_plan,
            }
        )
    )
    return coverage_plan


def send_idle_command(emitter, robot_name):
    """Tell one robot to stop and wait for a future assignment."""
    emitter.send(
        json.dumps(
            {
                "type": "idle",
                "robot": robot_name,
            }
        )
    )


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
    cleanup_plan_signatures = {}
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
                coverage = status.get("coverage", {})
                coverage_done_for_room = (
                    coverage.get("room") == room
                    and coverage.get("complete", False)
                )
                if (
                    coverage_done_for_room
                    and not room_reached_coverage_goal(cleaning_overlay, room)
                ):
                    pose = get_actual_robot_pose(supervisor, robot_name)
                    if pose is None:
                        pose = status["pose"]
                    cleanup_plan = cleaning_overlay.claim_dirty_tile_centers(
                        room,
                        robot_name,
                        pose,
                    )
                    cleanup_signature = (
                        room,
                        tuple(tuple(waypoint) for waypoint in cleanup_plan),
                    )
                    if (
                        cleanup_plan
                        and cleanup_plan_signatures.get(robot_name)
                        != cleanup_signature
                    ):
                        coverage_plans[robot_name] = send_coverage_plan(
                            emitter,
                            robot_name,
                            room,
                            cleanup_plan,
                        )
                        cleanup_plan_signatures[robot_name] = cleanup_signature
                        print(
                            f"[supervisor] Sent {robot_name} cleanup pass for "
                            f"{room}: {len(cleanup_plan)} claimed dirty tile(s)"
                        )
                    continue

                if not room_reached_coverage_goal(cleaning_overlay, room):
                    continue

                cleanup_plan_signatures.pop(robot_name, None)
                cleaning_overlay.release_robot_claims(robot_name)
                if room not in completed_rooms:
                    completed_rooms.add(room)
                    cleaning_overlay.release_room_claims(room)
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
                    send_idle_command(emitter, robot_name)
                    room_assignments[robot_name] = None
                    coverage_plans[robot_name] = []
                    cleanup_plan_signatures.pop(robot_name, None)
                    cleaning_overlay.release_robot_claims(robot_name)
                    last_cleaning_poses.pop(robot_name, None)
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
                cleanup_plan_signatures.pop(robot_name, None)
                cleaning_overlay.release_robot_claims(robot_name)
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
