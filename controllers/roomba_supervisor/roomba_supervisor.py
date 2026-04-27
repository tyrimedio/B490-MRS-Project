"""
Roomba Cluster - Central Supervisor

Receives status messages from each robot. This is the first communication
layer for the future shared map and task assignment system.
"""

import json
import math
from pathlib import Path

from controller import Supervisor

COMMUNICATION_SUMMARY_INTERVAL_STEPS = 120
MAP_UPDATE_LOG_INTERVAL_STEPS = 120
GROUND_TRUTH_POSE_CORRECTION_INTERVAL_STEPS = 30
SCAN_MATCH_LOG_INTERVAL_STEPS = 120
SCAN_MATCH_MAX_OFFSET_CELLS = 2
SCAN_MATCH_MIN_SCORE_GAIN = 3
SCAN_MATCH_MIN_INFORMATIVE_CELLS = 2
GROUND_TRUTH_DRIFT_CORRECTION_M = 0.45
GROUND_TRUTH_HEADING_CORRECTION_RAD = 0.75
LOCALIZATION_MIN_CONFIDENCE = 0.35
EXPECTED_ROBOTS = ("epuck_1", "epuck_2", "epuck_3", "epuck_4")
ROBOT_DEF_NAMES = {
    "epuck_1": "EPUCK_1",
    "epuck_2": "EPUCK_2",
    "epuck_3": "EPUCK_3",
    "epuck_4": "EPUCK_4",
}
ROBOT_START_POSES = {
    "epuck_1": (-1.5, 0.0, 0.5 * math.pi),
    "epuck_2": (-0.5, 0.0, 0.5 * math.pi),
    "epuck_3": (0.5, 0.0, -0.5 * math.pi),
    "epuck_4": (1.5, 0.0, -0.5 * math.pi),
}
ROBOT_START_Z_M = -0.000031
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
FINAL_CLEANUP_DIRTY_TILE_COUNT = 4
FINAL_CLEANUP_PROGRESS_PERCENT = 95.0
STALLED_CLEANUP_PROGRESS_PERCENT = 90.0
CLEANUP_RESEND_STEPS = 240
CLEANUP_EDGE_PRIORITY_MARGIN_M = 0.35
ROBOT_STUCK_WINDOW_STEPS = 300
ROBOT_STUCK_MIN_MOVE_M = 0.08
ROBOT_STUCK_ALERT_COOLDOWN_STEPS = 450
ROBOT_RECOVERY_REVERSE_STEPS = 18
ROBOT_RECOVERY_TURN_STEPS = 24
ROOM_PROGRESS_STALL_STEPS = 600
ROOM_PROGRESS_MIN_DELTA_PERCENT = 0.1
ROOM_PROGRESS_ALERT_COOLDOWN_STEPS = 600
ROOM_PROGRESS_RATE_WINDOW_STEPS = 300
MRTA_PROGRESS_RATE_WEIGHT = 0.35
MRTA_ROUTE_DISTANCE_WEIGHT = 0.18
MRTA_DIRTY_PERCENT_WEIGHT = 0.01
STUCK_ROUTE_RETRY_LIMIT = 1
SMALL_ROOM_MAX_PRIMARY_ROBOTS = 1
LARGE_ROOM_MAX_PRIMARY_ROBOTS = 2
TRAFFIC_RESOURCE_STAGGER_STEPS = 24
TRAFFIC_MAX_START_DELAY_STEPS = 120
OPERATOR_CONTROL_FILE = "operator_controls.json"
OPERATOR_STATE_FILE = "operator_state.json"
EVALUATION_METRICS_FILE = "evaluation_metrics.json"
SINGLE_ROBOT_BASELINE_FILE = "single_robot_baseline.json"
OPERATOR_CONTROL_POLL_STEPS = 30
OPERATOR_STATE_WRITE_STEPS = 15
OPERATOR_MAX_PRIORITY_ZONES = 6
OPERATOR_MAX_NO_GO_ZONES = 6
OPERATOR_ZONE_Z_M = 0.018
OPERATOR_NO_GO_ROUTE_MARGIN_M = 0.18
OPERATOR_PRIORITY_ROUTE_WEIGHT = 0.45
OPERATOR_PRIORITY_COLOR = [0.15, 0.35, 0.95]
OPERATOR_NO_GO_COLOR = [0.9, 0.08, 0.05]
OPERATOR_ZONE_TRANSPARENCY = 0.48
OPERATOR_ZONE_HIDDEN_TRANSPARENCY = 1.0
FLOOR_MIN_X_M = -3.0
FLOOR_MIN_Y_M = -3.0
DIRTY_TILE_COLOR = [0.72, 0.56, 0.32]
DIRTY_TILE_TRANSPARENCY = 0.35
CLEAN_TILE_COLOR = [0.18, 0.62, 0.42]
CLEAN_TILE_TRANSPARENCY = 0.15
HIDDEN_TILE_TRANSPARENCY = 1.0
EVALUATION_COVERAGE_TARGET_PERCENT = 95.0
INTER_ROBOT_COLLISION_DISTANCE_M = 0.08
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
ROOM_DOORWAY_ALIGNMENT_OFFSET_M = 0.45

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


def room_doorway_alignment_waypoints(room):
    """Return doorway centerline points that line a robot up before crossing."""
    center_x_m, center_y_m = ROOM_TASKS[room]["center"]
    _, _, min_y, max_y = ROOM_TASKS[room]["bounds"]
    if center_y_m > 0.0:
        wall_y_m = min_y
        inside_y_m = wall_y_m + ROOM_DOORWAY_ALIGNMENT_OFFSET_M
        hub_y_m = wall_y_m - ROOM_DOORWAY_ALIGNMENT_OFFSET_M
    else:
        wall_y_m = max_y
        inside_y_m = wall_y_m - ROOM_DOORWAY_ALIGNMENT_OFFSET_M
        hub_y_m = wall_y_m + ROOM_DOORWAY_ALIGNMENT_OFFSET_M

    return [round(center_x_m, 3), round(inside_y_m, 3)], [
        round(center_x_m, 3),
        round(hub_y_m, 3),
    ]


def append_route_waypoint(route, waypoint):
    """Append a route waypoint unless it repeats the previous point."""
    if route and route[-1] == waypoint:
        return
    route.append(waypoint)


def point_distance_m(first_point, second_point):
    """Return ground distance between two [x, y] points."""
    return math.hypot(
        second_point[0] - first_point[0],
        second_point[1] - first_point[1],
    )


def pose_point(pose):
    """Return a rounded [x, y] point for a pose dictionary."""
    if not isinstance(pose, dict):
        return None
    x_m = pose.get("x_m")
    y_m = pose.get("y_m")
    if not isinstance(x_m, (int, float)) or not isinstance(y_m, (int, float)):
        return None
    return [round(x_m, 3), round(y_m, 3)]


def route_distance_m(pose, route):
    """Return the planned distance from a pose through a route."""
    start_point = pose_point(pose)
    if start_point is None:
        return 0.0

    distance_m = 0.0
    previous_point = start_point
    for waypoint in route:
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) != 2:
            continue
        point = [float(waypoint[0]), float(waypoint[1])]
        distance_m += point_distance_m(previous_point, point)
        previous_point = point
    return distance_m


def room_hub_node(room):
    """Return the route graph node name for one room's hub doorway."""
    return f"hub:{room}"


def room_side(room):
    """Return which side of the central hub the room connects to."""
    _, center_y_m = ROOM_TASKS[room]["center"]
    return "north" if center_y_m > 0.0 else "south"


def hub_route_graph():
    """
    Build a tiny graph over room doorways on the central hub side.

    Every doorway can see every other doorway across the open hub, so the
    shortest graph path is usually direct. Keeping this as a graph gives the
    supervisor one clear place to add future hallway obstacles or no-go areas.
    """
    nodes = {
        room_hub_node(room): room_doorway_waypoints(room)[1]
        for room in ROOM_TASKS
    }
    edges = {node: {} for node in nodes}
    for first_node, first_point in nodes.items():
        for second_node, second_point in nodes.items():
            if first_node == second_node:
                continue
            edges[first_node][second_node] = point_distance_m(
                first_point,
                second_point,
            )
    return nodes, edges


def shortest_hub_route_nodes(source_room, target_room):
    """Return the shortest route graph node path between two room doorways."""
    if source_room == target_room:
        return [room_hub_node(target_room)]

    nodes, edges = hub_route_graph()
    source_node = room_hub_node(source_room)
    target_node = room_hub_node(target_room)
    if source_node not in nodes or target_node not in nodes:
        return [source_node, target_node]

    open_nodes = {source_node}
    distances = {source_node: 0.0}
    previous_nodes = {}
    visited_nodes = set()

    while open_nodes:
        current_node = min(open_nodes, key=lambda node: distances[node])
        open_nodes.remove(current_node)
        if current_node == target_node:
            break
        visited_nodes.add(current_node)

        for neighbor, edge_cost in edges[current_node].items():
            if neighbor in visited_nodes:
                continue
            candidate_distance = distances[current_node] + edge_cost
            if candidate_distance >= distances.get(neighbor, float("inf")):
                continue
            distances[neighbor] = candidate_distance
            previous_nodes[neighbor] = current_node
            open_nodes.add(neighbor)

    if target_node not in distances:
        return [source_node, target_node]

    path = [target_node]
    while path[-1] != source_node:
        path.append(previous_nodes[path[-1]])
    path.reverse()
    return path


def shortest_hub_route_waypoints(source_room, target_room):
    """Return hub-side doorway waypoints for the shortest room-to-room route."""
    nodes, _ = hub_route_graph()
    return [
        nodes[node_name]
        for node_name in shortest_hub_route_nodes(source_room, target_room)
        if node_name in nodes
    ]


def traffic_resource_keys(pose, target_room):
    """Return coarse doorway and hub resources used by an assignment route."""
    resources = []
    current_room = room_for_pose(pose)
    if current_room == target_room:
        return [f"room:{target_room}"]

    if current_room is not None:
        resources.append(f"doorway:{current_room}")
        source_side = room_side(current_room)
    else:
        source_side = None

    target_side = room_side(target_room)
    resources.append("hub:all")
    if source_side is None:
        resources.append(f"hub:{target_side}")
    elif source_side == target_side:
        resources.append(f"hub:{target_side}")
    else:
        resources.append("hub:cross")
    resources.append(f"doorway:{target_room}")
    return list(dict.fromkeys(resources))


def generate_assignment_route(pose, target_room, final_waypoint=None):
    """Build a doorway-aware route from the robot's current room to a target room."""
    route = []
    current_room = room_for_pose(pose)
    target_x_m, target_y_m = ROOM_TASKS[target_room]["center"]
    if final_waypoint is None:
        target_point = [round(target_x_m, 3), round(target_y_m, 3)]
    else:
        target_point = [round(final_waypoint[0], 3), round(final_waypoint[1], 3)]

    if current_room == target_room:
        return [target_point]

    if current_room is not None and current_room != target_room:
        current_inside, current_hub = room_doorway_waypoints(current_room)
        current_inside_approach, current_hub_approach = (
            room_doorway_alignment_waypoints(current_room)
        )
        append_route_waypoint(route, current_inside_approach)
        append_route_waypoint(route, current_inside)
        append_route_waypoint(route, current_hub)
        append_route_waypoint(route, current_hub_approach)
        hub_waypoints = shortest_hub_route_waypoints(current_room, target_room)[1:]
        for waypoint_index, waypoint in enumerate(hub_waypoints):
            if waypoint_index == len(hub_waypoints) - 1:
                _, target_hub_approach = room_doorway_alignment_waypoints(target_room)
                append_route_waypoint(route, target_hub_approach)
            append_route_waypoint(route, waypoint)
    else:
        _, target_hub = room_doorway_waypoints(target_room)
        _, target_hub_approach = room_doorway_alignment_waypoints(target_room)
        append_route_waypoint(route, target_hub_approach)
        append_route_waypoint(route, target_hub)

    target_inside, _ = room_doorway_waypoints(target_room)
    target_inside_approach, _ = room_doorway_alignment_waypoints(target_room)
    append_route_waypoint(route, target_inside)
    append_route_waypoint(route, target_inside_approach)
    append_route_waypoint(route, target_point)
    return route


def generate_route_to_room_hub(pose, target_room, final_waypoint=None):
    """Route to the hub side of a room doorway without entering that room."""
    target_hub = room_doorway_waypoints(target_room)[1]
    if final_waypoint is None:
        target_point = target_hub
    else:
        target_point = [round(final_waypoint[0], 3), round(final_waypoint[1], 3)]

    start_point = pose_point(pose)
    if start_point == target_point:
        return [target_point]

    route = []
    current_room = room_for_pose(pose)
    if current_room is not None and current_room != target_room:
        current_inside, current_hub = room_doorway_waypoints(current_room)
        current_inside_approach, current_hub_approach = (
            room_doorway_alignment_waypoints(current_room)
        )
        append_route_waypoint(route, current_inside_approach)
        append_route_waypoint(route, current_inside)
        append_route_waypoint(route, current_hub)
        append_route_waypoint(route, current_hub_approach)
        hub_waypoints = shortest_hub_route_waypoints(current_room, target_room)[1:]
        for waypoint_index, waypoint in enumerate(hub_waypoints):
            if waypoint_index == len(hub_waypoints) - 1:
                _, target_hub_approach = room_doorway_alignment_waypoints(target_room)
                append_route_waypoint(route, target_hub_approach)
            append_route_waypoint(route, waypoint)
    elif current_room is None:
        _, target_hub_approach = room_doorway_alignment_waypoints(target_room)
        append_route_waypoint(route, target_hub_approach)
        append_route_waypoint(route, target_hub)

    append_route_waypoint(route, target_point)
    return route


def normalized_bounds(value):
    """Return [min_x, max_x, min_y, max_y] when a raw bounds value is usable."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        first_x, second_x, first_y, second_y = [float(entry) for entry in value]
    except (TypeError, ValueError):
        return None

    min_x, max_x = sorted((first_x, second_x))
    min_y, max_y = sorted((first_y, second_y))
    if min_x == max_x or min_y == max_y:
        return None
    return [round(min_x, 3), round(max_x, 3), round(min_y, 3), round(max_y, 3)]


def expanded_bounds(bounds, margin_m):
    """Return a rectangle expanded by a small clearance margin."""
    min_x, max_x, min_y, max_y = bounds
    return [
        round(min_x - margin_m, 3),
        round(max_x + margin_m, 3),
        round(min_y - margin_m, 3),
        round(max_y + margin_m, 3),
    ]


def point_inside_bounds(point, bounds):
    """Return True when a [x, y] point is inside a rectangle."""
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return False
    min_x, max_x, min_y, max_y = bounds
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y


def bounds_overlap(first_bounds, second_bounds):
    """Return True when two rectangles share actual area, not only an edge."""
    first_min_x, first_max_x, first_min_y, first_max_y = first_bounds
    second_min_x, second_max_x, second_min_y, second_max_y = second_bounds
    return not (
        first_max_x <= second_min_x
        or second_max_x <= first_min_x
        or first_max_y <= second_min_y
        or second_max_y <= first_min_y
    )


def _orientation(first, second, third):
    """Return the turn direction for three points."""
    value = (
        (second[1] - first[1]) * (third[0] - second[0])
        - (second[0] - first[0]) * (third[1] - second[1])
    )
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0.0 else 2


def _point_on_segment(first, point, second):
    """Return True when point sits on the line segment from first to second."""
    return (
        min(first[0], second[0]) - 1e-9 <= point[0] <= max(first[0], second[0]) + 1e-9
        and min(first[1], second[1]) - 1e-9
        <= point[1]
        <= max(first[1], second[1]) + 1e-9
    )


def line_segments_intersect(first_start, first_end, second_start, second_end):
    """Return True when two line segments touch or cross."""
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)

    if (
        first_orientation != second_orientation
        and third_orientation != fourth_orientation
    ):
        return True

    if first_orientation == 0 and _point_on_segment(first_start, second_start, first_end):
        return True
    if second_orientation == 0 and _point_on_segment(first_start, second_end, first_end):
        return True
    if third_orientation == 0 and _point_on_segment(second_start, first_start, second_end):
        return True
    if fourth_orientation == 0 and _point_on_segment(second_start, first_end, second_end):
        return True
    return False


def segment_intersects_bounds(start_point, end_point, bounds):
    """Return True when a route segment enters or crosses a rectangular zone."""
    if point_inside_bounds(start_point, bounds) or point_inside_bounds(end_point, bounds):
        return True

    min_x, max_x, min_y, max_y = bounds
    edges = (
        ([min_x, min_y], [max_x, min_y]),
        ([max_x, min_y], [max_x, max_y]),
        ([max_x, max_y], [min_x, max_y]),
        ([min_x, max_y], [min_x, min_y]),
    )
    return any(
        line_segments_intersect(start_point, end_point, edge_start, edge_end)
        for edge_start, edge_end in edges
    )


def path_distance_between_points(points):
    """Return distance through a plain list of [x, y] points."""
    distance_m = 0.0
    for index in range(1, len(points)):
        distance_m += point_distance_m(points[index - 1], points[index])
    return distance_m


def detour_waypoints_around_bounds(start_point, end_point, bounds):
    """
    Return a short set of waypoints that steers one segment around a no-go box.

    The no-go box is like a taped-off square on the floor. If the straight
    path crosses the tape, the supervisor adds two corner waypoints just
    outside the box so the robot walks around it.
    """
    if not segment_intersects_bounds(start_point, end_point, bounds):
        return []

    min_x, max_x, min_y, max_y = expanded_bounds(
        bounds,
        OPERATOR_NO_GO_ROUTE_MARGIN_M,
    )
    candidates = [
        [[min_x, max_y], [max_x, max_y]],
        [[max_x, max_y], [min_x, max_y]],
        [[min_x, min_y], [max_x, min_y]],
        [[max_x, min_y], [min_x, min_y]],
        [[min_x, min_y], [min_x, max_y]],
        [[min_x, max_y], [min_x, min_y]],
        [[max_x, min_y], [max_x, max_y]],
        [[max_x, max_y], [max_x, min_y]],
    ]

    clear_candidates = []
    fallback_candidates = []
    for candidate in candidates:
        route_points = [start_point] + candidate + [end_point]
        route_distance = path_distance_between_points(route_points)
        fallback_candidates.append((route_distance, candidate))
        if any(
            segment_intersects_bounds(route_points[index], route_points[index + 1], bounds)
            for index in range(len(route_points) - 1)
        ):
            continue
        clear_candidates.append((route_distance, candidate))

    choices = clear_candidates or fallback_candidates
    choices.sort()
    return [[round(point[0], 3), round(point[1], 3)] for point in choices[0][1]]


def first_no_go_segment_hit(points, no_go_bounds):
    """Return the first segment and no-go bounds crossed by a route."""
    for index in range(len(points) - 1):
        for bounds in no_go_bounds:
            if segment_intersects_bounds(points[index], points[index + 1], bounds):
                return index, bounds
    return None, None


def point_inside_any_bounds(point, bounds_list):
    """Return True when a point is inside any rectangle in a bounds list."""
    return any(point_inside_bounds(point, bounds) for bounds in bounds_list)


def points_share_no_go_bounds(first_point, second_point, no_go_bounds):
    """Return True when two points sit inside the same no-go rectangle."""
    if first_point is None or second_point is None:
        return False
    return any(
        point_inside_bounds(first_point, bounds)
        and point_inside_bounds(second_point, bounds)
        for bounds in no_go_bounds
    )


def room_entry_blocked_by_no_go_zones(room, no_go_zones=None):
    """Return True when an operator no-go zone blocks the room doorway."""
    if not no_go_zones or room not in ROOM_TASKS:
        return False

    room_inside, room_hub = room_doorway_waypoints(room)
    for zone in no_go_zones:
        if not isinstance(zone, dict) or zone.get("bounds") is None:
            continue
        bounds = zone["bounds"]
        if (
            point_inside_bounds(room_inside, bounds)
            or point_inside_bounds(room_hub, bounds)
            or segment_intersects_bounds(room_hub, room_inside, bounds)
        ):
            return True
    return False


def route_segment_around_no_go_zones(start_point, end_point, no_go_bounds):
    """Return waypoints from start to end after rechecking all no-go boxes."""
    if point_inside_any_bounds(end_point, no_go_bounds):
        return []

    _, first_blocked_bounds = first_no_go_segment_hit(
        [start_point, end_point],
        no_go_bounds,
    )
    if first_blocked_bounds is None:
        return [end_point]

    candidates = []

    def add_candidate(point, allow_blocked=False):
        rounded_point = [round(point[0], 3), round(point[1], 3)]
        if not allow_blocked and point_inside_any_bounds(rounded_point, no_go_bounds):
            return
        if rounded_point not in candidates:
            candidates.append(rounded_point)

    add_candidate(start_point, allow_blocked=True)
    add_candidate(end_point, allow_blocked=True)
    for bounds in no_go_bounds:
        min_x, max_x, min_y, max_y = expanded_bounds(
            bounds,
            OPERATOR_NO_GO_ROUTE_MARGIN_M,
        )
        for candidate in (
            [min_x, min_y],
            [min_x, max_y],
            [max_x, min_y],
            [max_x, max_y],
        ):
            add_candidate(candidate)

    start_candidate = candidates[0]

    def segment_is_clear(first_point, second_point):
        for bounds in no_go_bounds:
            if not segment_intersects_bounds(first_point, second_point, bounds):
                continue
            if (
                first_point == start_candidate
                and point_inside_bounds(first_point, bounds)
                and not point_inside_bounds(second_point, bounds)
            ):
                continue
            return False
        return True

    start_index = 0
    end_index = 1
    distances = [math.inf for _ in candidates]
    previous_indexes = [None for _ in candidates]
    visited = set()
    distances[start_index] = 0.0

    while len(visited) < len(candidates):
        current_index = None
        current_distance = math.inf
        for index, distance in enumerate(distances):
            if index not in visited and distance < current_distance:
                current_index = index
                current_distance = distance
        if current_index is None or math.isinf(current_distance):
            break
        if current_index == end_index:
            break

        visited.add(current_index)
        current_point = candidates[current_index]
        for neighbor_index, neighbor_point in enumerate(candidates):
            if neighbor_index == current_index or neighbor_index in visited:
                continue
            if not segment_is_clear(current_point, neighbor_point):
                continue
            candidate_distance = current_distance + point_distance_m(
                current_point,
                neighbor_point,
            )
            if candidate_distance < distances[neighbor_index]:
                distances[neighbor_index] = candidate_distance
                previous_indexes[neighbor_index] = current_index

    if math.isinf(distances[end_index]):
        return []

    path_indexes = []
    current_index = end_index
    while current_index is not None:
        path_indexes.append(current_index)
        current_index = previous_indexes[current_index]
    path_indexes.reverse()
    return [candidates[index] for index in path_indexes[1:]]


def route_around_no_go_zones(
    route,
    no_go_zones=None,
    start_point=None,
    allow_initial_blocked_waypoints=False,
):
    """Return a route with extra waypoints inserted around operator no-go zones."""
    if not no_go_zones:
        return list(route)

    no_go_bounds = [
        zone.get("bounds")
        for zone in no_go_zones
        if isinstance(zone, dict) and zone.get("bounds") is not None
    ]
    if not no_go_bounds:
        return list(route)

    adjusted_route = []
    previous_point = start_point
    for waypoint in route:
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) != 2:
            continue
        target_point = [round(float(waypoint[0]), 3), round(float(waypoint[1]), 3)]
        if point_inside_any_bounds(target_point, no_go_bounds):
            if (
                allow_initial_blocked_waypoints
                and points_share_no_go_bounds(previous_point, target_point, no_go_bounds)
            ):
                append_route_waypoint(adjusted_route, target_point)
                previous_point = target_point
                continue
            break

        if previous_point is None:
            append_route_waypoint(adjusted_route, target_point)
            previous_point = target_point
            continue

        segment_points = route_segment_around_no_go_zones(
            previous_point,
            target_point,
            no_go_bounds,
        )
        if not segment_points:
            break
        for segment_point in segment_points:
            append_route_waypoint(adjusted_route, segment_point)
            previous_point = segment_point

    return adjusted_route


def route_reaches_final_waypoint(original_route, adjusted_route):
    """Return True when a no-go adjusted route still reaches its intended end."""
    if not original_route:
        return True
    if not adjusted_route:
        return False
    return adjusted_route[-1] == original_route[-1]


def point_blocked_by_no_go_zones(point, no_go_zones=None):
    """Return True when a waypoint sits inside an operator no-go zone."""
    if not no_go_zones:
        return False

    return any(
        isinstance(zone, dict)
        and zone.get("bounds") is not None
        and point_inside_bounds(point, zone["bounds"])
        for zone in no_go_zones
    )


def coverage_plan_around_no_go_zones(coverage_plan, no_go_zones=None, start_point=None):
    """Return coverage waypoints with blocked targets skipped and segments detoured."""
    if not no_go_zones:
        return list(coverage_plan)

    adjusted_plan = []
    previous_point = start_point
    for waypoint in coverage_plan:
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) != 2:
            continue
        target_point = [round(float(waypoint[0]), 3), round(float(waypoint[1]), 3)]
        if point_blocked_by_no_go_zones(target_point, no_go_zones):
            continue

        segment_points = route_around_no_go_zones(
            [target_point],
            no_go_zones,
            start_point=previous_point,
        )
        if segment_points and segment_points[-1] != target_point:
            continue
        for segment_point in segment_points:
            append_route_waypoint(adjusted_plan, segment_point)
        if segment_points:
            previous_point = adjusted_plan[-1]

    return adjusted_plan


def route_target_for_empty_coverage(room, no_go_zones=None, pose=None):
    """Return a fallback target that does not sit inside an operator no-go zone."""
    center_x_m, center_y_m = ROOM_TASKS[room]["center"]
    room_center = [round(center_x_m, 3), round(center_y_m, 3)]
    room_inside, room_hub = room_doorway_waypoints(room)
    if room_entry_blocked_by_no_go_zones(room, no_go_zones):
        candidates = [room_hub, pose_point(pose), HUB_ROUTE_WAYPOINT]
    else:
        candidates = [
            room_center,
            room_inside,
            room_hub,
            pose_point(pose),
            HUB_ROUTE_WAYPOINT,
        ]
    for candidate in candidates:
        if candidate is not None and not point_blocked_by_no_go_zones(
            candidate,
            no_go_zones,
        ):
            return candidate
    return pose_point(pose) or room_hub


def normalize_operator_zone(raw_zone, zone_kind):
    """Normalize one operator priority or no-go zone definition."""
    if not isinstance(raw_zone, dict):
        return None

    room = raw_zone.get("room")
    bounds = normalized_bounds(raw_zone.get("bounds"))
    if room in ROOM_TASKS and bounds is None:
        bounds = list(ROOM_TASKS[room]["bounds"])
    if bounds is None:
        return None

    name = raw_zone.get("name")
    if not isinstance(name, str) or not name.strip():
        name = room if room in ROOM_TASKS else f"{zone_kind}_zone"

    try:
        weight = float(raw_zone.get("weight", 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    weight = max(0.0, weight)

    return {
        "kind": zone_kind,
        "name": name.strip(),
        "room": room if room in ROOM_TASKS else None,
        "bounds": bounds,
        "weight": round(weight, 3),
    }


def priority_weight_for_room(priority_zones, room):
    """Return the strongest operator priority weight affecting one room."""
    if not priority_zones or room not in ROOM_TASKS:
        return 0.0

    room_bounds = ROOM_TASKS[room]["bounds"]
    priority_weight = 0.0
    for zone in priority_zones:
        if not isinstance(zone, dict):
            continue
        zone_bounds = zone.get("bounds")
        if zone_bounds is None:
            continue
        if zone.get("room") == room or bounds_overlap(zone_bounds, room_bounds):
            priority_weight = max(priority_weight, zone.get("weight", 1.0))
    return priority_weight


def operator_zone_signature(zones):
    """Return a stable signature for detecting operator zone changes."""
    return tuple(
        (
            zone.get("name"),
            tuple(zone.get("bounds", [])),
            zone.get("room"),
            zone.get("weight", 1.0),
        )
        for zone in zones
        if isinstance(zone, dict)
    )


class TrafficReservationBook:
    """
    Coarsely stagger robots that would use the same doorway or hub corridor.

    This is intentionally simple: it is like telling kids to go through a
    narrow doorway one at a time. It does not reserve exact geometry forever;
    it just spaces route starts enough to prevent the first few seconds of a
    shared route from becoming a pileup.
    """

    def __init__(self):
        self.next_available_steps = {}
        self.total_planned_distance_m = 0.0
        self.total_wait_steps = 0
        self.route_conflicts = 0
        self.doorway_conflicts = 0

    def reserve(self, robot_name, resources, step_count):
        """Reserve resources and return the start delay and conflict metrics."""
        unique_resources = list(dict.fromkeys(resources))
        start_step = step_count
        conflict_resources = []
        for resource in unique_resources:
            available_step = self.next_available_steps.get(resource, step_count)
            if available_step > step_count:
                conflict_resources.append(resource)
            if available_step > start_step:
                start_step = available_step

        start_delay_steps = max(0, start_step - step_count)
        start_delay_steps = min(start_delay_steps, TRAFFIC_MAX_START_DELAY_STEPS)
        reserved_until = step_count + start_delay_steps + TRAFFIC_RESOURCE_STAGGER_STEPS
        for resource in unique_resources:
            self.next_available_steps[resource] = max(
                self.next_available_steps.get(resource, step_count),
                reserved_until,
            )

        doorway_conflicts = sum(
            1 for resource in conflict_resources
            if resource.startswith("doorway:")
        )
        self.total_wait_steps += start_delay_steps
        self.route_conflicts += len(conflict_resources)
        self.doorway_conflicts += doorway_conflicts
        return {
            "robot": robot_name,
            "start_delay_steps": start_delay_steps,
            "conflict_count": len(conflict_resources),
            "doorway_conflicts": doorway_conflicts,
            "conflict_resources": conflict_resources,
            "reserved_resources": unique_resources,
        }

    def record_distance(self, distance_m):
        """Add one planned assignment distance to the run metrics."""
        self.total_planned_distance_m += distance_m

    def summary(self):
        """Return rounded path-efficiency metrics for logging."""
        return {
            "planned_distance_m": round(self.total_planned_distance_m, 2),
            "waiting_steps": self.total_wait_steps,
            "route_conflicts": self.route_conflicts,
            "doorway_conflicts": self.doorway_conflicts,
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

    def iter_update_observations(self, map_update):
        """Yield map observations as kind, x, y, count tuples."""
        if not isinstance(map_update, dict):
            return

        if "observations" in map_update:
            for observation in map_update.get("observations", []):
                if self.is_valid_ordered_observation(observation):
                    yield tuple(observation)
            return

        for observation in map_update.get("free_observations", []):
            if self.is_valid_observation(observation):
                grid_x, grid_y, count = observation
                yield "free", grid_x, grid_y, count

        for observation in map_update.get("wall_observations", []):
            if self.is_valid_observation(observation):
                grid_x, grid_y, count = observation
                yield "wall", grid_x, grid_y, count

        if (
            "free_observations" in map_update
            or "wall_observations" in map_update
            or "observations" in map_update
        ):
            return

        for cell in map_update.get("free_cells", []):
            if self.is_valid_cell(cell):
                grid_x, grid_y = cell
                yield "free", grid_x, grid_y, 1

        for cell in map_update.get("wall_cells", []):
            if self.is_valid_cell(cell):
                grid_x, grid_y = cell
                yield "wall", grid_x, grid_y, 1

    def scan_match_score(self, map_update, offset_x_cells, offset_y_cells):
        """
        Score how well one robot update lines up with the existing global map.

        This is a small translation-only scan match. It tries sliding the new
        mini-map by a few grid cells and scores only wall features, because
        open floor readings are common enough that they can make a bad offset
        look good. Think of it like nudging tracing paper until the wall marks
        line up with the shared map underneath.
        """
        wall_observations = list(self.iter_wall_update_observations(map_update))
        return self.scan_match_observation_score(
            wall_observations,
            offset_x_cells,
            offset_y_cells,
        )

    def iter_wall_update_observations(self, map_update):
        """Yield only wall observations, since free cells are not scored."""
        for kind, grid_x, grid_y, count in self.iter_update_observations(map_update):
            if kind == "wall":
                yield grid_x, grid_y, count

    def scan_match_observation_score(
        self,
        wall_observations,
        offset_x_cells,
        offset_y_cells,
    ):
        """Score pre-filtered wall observations for one candidate offset."""
        score = 0
        informative_cells = 0
        for grid_x, grid_y, count in wall_observations:
            shifted_x = grid_x + offset_x_cells
            shifted_y = grid_y + offset_y_cells
            weight = min(count, 3)
            if not self.is_valid_cell([shifted_x, shifted_y]):
                score -= weight
                continue

            existing_state = self.data[shifted_y][shifted_x]
            if existing_state == GRID_UNKNOWN:
                continue

            informative_cells += 1
            if existing_state == GRID_WALL:
                score += 3 * weight
            elif existing_state == GRID_FREE:
                score -= 4 * weight

        score -= abs(offset_x_cells) + abs(offset_y_cells)
        return score, informative_cells

    def shifted_map_update(self, map_update, offset_x_cells, offset_y_cells):
        """Return a copy of a map update with every cell shifted by an offset."""
        if not isinstance(map_update, dict):
            return {}

        shifted_update = {}
        if "observations" in map_update:
            observations = []
            for observation in map_update.get("observations", []):
                if not self.is_valid_ordered_observation(observation):
                    continue
                kind, grid_x, grid_y, count = observation
                shifted_x = grid_x + offset_x_cells
                shifted_y = grid_y + offset_y_cells
                if self.is_valid_cell([shifted_x, shifted_y]):
                    observations.append([kind, shifted_x, shifted_y, count])
            shifted_update["observations"] = observations

        if "free_observations" in map_update:
            shifted_update["free_observations"] = self.shifted_observations(
                map_update.get("free_observations", []),
                offset_x_cells,
                offset_y_cells,
            )
        if "wall_observations" in map_update:
            shifted_update["wall_observations"] = self.shifted_observations(
                map_update.get("wall_observations", []),
                offset_x_cells,
                offset_y_cells,
            )
        if "free_cells" in map_update:
            shifted_update["free_cells"] = self.shifted_cells(
                map_update.get("free_cells", []),
                offset_x_cells,
                offset_y_cells,
            )
        if "wall_cells" in map_update:
            shifted_update["wall_cells"] = self.shifted_cells(
                map_update.get("wall_cells", []),
                offset_x_cells,
                offset_y_cells,
            )
        return shifted_update

    def shifted_observations(self, observations, offset_x_cells, offset_y_cells):
        """Shift [x, y, count] observations while dropping cells outside the map."""
        shifted = []
        for observation in observations:
            if not self.is_valid_observation(observation):
                continue
            grid_x, grid_y, count = observation
            shifted_x = grid_x + offset_x_cells
            shifted_y = grid_y + offset_y_cells
            if self.is_valid_cell([shifted_x, shifted_y]):
                shifted.append([shifted_x, shifted_y, count])
        return shifted

    def shifted_cells(self, cells, offset_x_cells, offset_y_cells):
        """Shift [x, y] cells while dropping cells outside the map."""
        shifted = []
        for cell in cells:
            if not self.is_valid_cell(cell):
                continue
            grid_x, grid_y = cell
            shifted_x = grid_x + offset_x_cells
            shifted_y = grid_y + offset_y_cells
            if self.is_valid_cell([shifted_x, shifted_y]):
                shifted.append([shifted_x, shifted_y])
        return shifted

    def scan_match_update(self, map_update):
        """Return a scan-matched map update and the accepted cell offset."""
        empty_match = {
            "accepted": False,
            "map_update": map_update if isinstance(map_update, dict) else {},
            "offset_cells": [0, 0],
            "offset_m": [0.0, 0.0],
            "score": 0,
            "score_gain": 0,
            "informative_cells": 0,
        }
        if not isinstance(map_update, dict):
            return empty_match

        wall_observations = list(self.iter_wall_update_observations(map_update))
        if not wall_observations:
            return empty_match

        baseline_score, baseline_informative = self.scan_match_observation_score(
            wall_observations,
            0,
            0,
        )
        best = {
            "offset_cells": [0, 0],
            "score": baseline_score,
            "informative_cells": baseline_informative,
        }
        for offset_x in range(-SCAN_MATCH_MAX_OFFSET_CELLS, SCAN_MATCH_MAX_OFFSET_CELLS + 1):
            for offset_y in range(
                -SCAN_MATCH_MAX_OFFSET_CELLS,
                SCAN_MATCH_MAX_OFFSET_CELLS + 1,
            ):
                if offset_x == 0 and offset_y == 0:
                    continue
                score, informative_cells = self.scan_match_observation_score(
                    wall_observations,
                    offset_x,
                    offset_y,
                )
                if (
                    score > best["score"]
                    or (
                        score == best["score"]
                        and abs(offset_x) + abs(offset_y)
                        < abs(best["offset_cells"][0]) + abs(best["offset_cells"][1])
                    )
                ):
                    best = {
                        "offset_cells": [offset_x, offset_y],
                        "score": score,
                        "informative_cells": informative_cells,
                    }

        score_gain = best["score"] - baseline_score
        accepted = (
            best["offset_cells"] != [0, 0]
            and score_gain >= SCAN_MATCH_MIN_SCORE_GAIN
            and best["informative_cells"] >= SCAN_MATCH_MIN_INFORMATIVE_CELLS
        )
        if not accepted:
            return {
                "accepted": False,
                "map_update": map_update,
                "offset_cells": [0, 0],
                "offset_m": [0.0, 0.0],
                "score": baseline_score,
                "score_gain": 0,
                "informative_cells": baseline_informative,
            }

        offset_x, offset_y = best["offset_cells"]
        return {
            "accepted": True,
            "map_update": self.shifted_map_update(map_update, offset_x, offset_y),
            "offset_cells": best["offset_cells"],
            "offset_m": [
                round(offset_x * self.cell_size_m, 4),
                round(offset_y * self.cell_size_m, 4),
            ],
            "score": best["score"],
            "score_gain": score_gain,
            "informative_cells": best["informative_cells"],
        }

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

    def assignment_cost(self, robot_status, room, no_go_zones=None, priority_zones=None):
        """Return the cost of sending one robot to one room."""
        pose = robot_status["pose"]
        base_route = generate_assignment_route(pose, room)
        route = route_around_no_go_zones(
            base_route,
            no_go_zones,
            start_point=pose_point(pose),
        )
        if not route_reaches_final_waypoint(base_route, route):
            return float("inf")

        distance_m = route_distance_m(pose, route)
        area_m2 = self.rooms[room]["area_m2"]
        priority_weight = priority_weight_for_room(priority_zones, room)
        return (
            self.distance_weight * distance_m
            + self.area_weight * area_m2
            - OPERATOR_PRIORITY_ROUTE_WEIGHT * priority_weight
        )

    def assign(self, robot_statuses, no_go_zones=None, priority_zones=None):
        """Return room assignments that minimize total cost for this small team."""
        robots = [robot for robot in EXPECTED_ROBOTS if robot in robot_statuses]
        rooms = list(self.rooms)
        if not robots or len(robots) > len(rooms):
            return {}

        best_assignments = None
        best_total_cost = None
        best_assigned_count = -1
        remaining_rooms = set(rooms)

        def search(robot_index, current_assignments, current_cost):
            nonlocal best_assignments, best_total_cost, best_assigned_count

            if robot_index >= len(robots):
                assigned_count = len(current_assignments)
                if (
                    assigned_count > best_assigned_count
                    or (
                        assigned_count == best_assigned_count
                        and (
                            best_total_cost is None
                            or current_cost < best_total_cost
                        )
                    )
                ):
                    best_assigned_count = assigned_count
                    best_total_cost = current_cost
                    best_assignments = dict(current_assignments)
                return

            robot = robots[robot_index]
            for room in sorted(remaining_rooms):
                room_cost = self.assignment_cost(
                    robot_statuses[robot],
                    room,
                    no_go_zones,
                    priority_zones,
                )
                if math.isinf(room_cost):
                    continue
                total_cost = current_cost + room_cost

                remaining_rooms.remove(room)
                base_route = generate_assignment_route(
                    robot_statuses[robot]["pose"],
                    room,
                )
                route = route_around_no_go_zones(
                    base_route,
                    no_go_zones,
                    start_point=pose_point(robot_statuses[robot]["pose"]),
                )
                current_assignments[robot] = {
                    "room": room,
                    "cost": round(room_cost, 3),
                    "target": self.rooms[room]["center"],
                    "route": route,
                }
                search(robot_index + 1, current_assignments, total_cost)
                current_assignments.pop(robot)
                remaining_rooms.add(room)

            search(robot_index + 1, current_assignments, current_cost)

        search(0, {}, 0.0)
        return best_assignments or {}


def normalized_lane(lane_index=0, lane_count=1):
    """Return a safe lane index/count pair."""
    try:
        lane_index = int(lane_index)
        lane_count = int(lane_count)
    except (TypeError, ValueError):
        return 0, 1

    lane_count = max(1, lane_count)
    lane_index = max(0, min(lane_count - 1, lane_index))
    return lane_index, lane_count


def generate_coverage_waypoints(room, lane_index=0, lane_count=1):
    """Create simple back-and-forth sweep waypoints for one room or lane."""
    min_x, max_x, min_y, max_y = ROOM_TASKS[room]["bounds"]
    sweep_min_x = min_x + COVERAGE_MARGIN_M
    sweep_max_x = max_x - COVERAGE_MARGIN_M
    sweep_min_y = min_y + COVERAGE_MARGIN_M
    sweep_max_y = max_y - COVERAGE_MARGIN_M
    lane_index, lane_count = normalized_lane(lane_index, lane_count)
    sweep_width_m = sweep_max_x - sweep_min_x
    if lane_count > 1 and sweep_width_m > 0.0:
        lane_width_m = sweep_width_m / lane_count
        sweep_min_x += lane_width_m * lane_index
        sweep_max_x = sweep_min_x + lane_width_m

    waypoints = []
    row_positions = []
    y_m = sweep_min_y
    while y_m <= sweep_max_y + 1e-6:
        row_positions.append(y_m)
        y_m += COVERAGE_ROW_SPACING_M
    if not row_positions or row_positions[-1] < sweep_max_y - 1e-6:
        row_positions.append(sweep_max_y)
    if ROOM_TASKS[room]["center"][1] < 0.0:
        row_positions.reverse()

    start_from_left = lane_index % 2 == 0
    for row_index, y_m in enumerate(row_positions):
        if (row_index % 2 == 0) == start_from_left:
            waypoints.append([round(sweep_min_x, 3), round(y_m, 3)])
            waypoints.append([round(sweep_max_x, 3), round(y_m, 3)])
        else:
            waypoints.append([round(sweep_max_x, 3), round(y_m, 3)])
            waypoints.append([round(sweep_min_x, 3), round(y_m, 3)])

    return waypoints


def cleanup_waypoint_for_tile_center(room, tile_center):
    """Return a reachable robot waypoint that can clean one dirty tile."""
    min_x, max_x, min_y, max_y = ROOM_TASKS[room]["bounds"]
    target_x = min(
        max(float(tile_center[0]), min_x + COVERAGE_MARGIN_M),
        max_x - COVERAGE_MARGIN_M,
    )
    target_y = min(
        max(float(tile_center[1]), min_y + COVERAGE_MARGIN_M),
        max_y - COVERAGE_MARGIN_M,
    )
    return [round(target_x, 3), round(target_y, 3)]


def room_robot_names(room_assignments, room):
    """Return robots currently assigned to one room in a stable order."""
    return sorted(
        robot_name
        for robot_name, assignment in room_assignments.items()
        if assignment is not None and assignment["room"] == room
    )


def coverage_lane_for_robot(robot_name, room_assignments):
    """Return the sweep lane assigned to one robot inside its current room."""
    assignment = room_assignments.get(robot_name)
    if assignment is None:
        return 0, 1

    robots_in_room = room_robot_names(room_assignments, assignment["room"])
    if robot_name not in robots_in_room:
        return 0, 1

    return robots_in_room.index(robot_name), len(robots_in_room)


def pause_robot_assignment(robot_name, room_assignments, paused_assignments):
    """Move a paused robot's task out of active room accounting."""
    assignment = room_assignments.pop(robot_name, None)
    if assignment is not None:
        paused_assignments[robot_name] = assignment
    else:
        paused_assignments.pop(robot_name, None)
    return assignment


def resume_robot_assignment(robot_name, room_assignments, paused_assignments):
    """Restore a paused robot's task when the operator resumes it."""
    assignment = room_assignments.get(robot_name)
    paused_assignment = paused_assignments.pop(robot_name, None)
    if assignment is None and paused_assignment is not None:
        room_assignments[robot_name] = paused_assignment
        return paused_assignment
    return assignment


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


def reset_webots_robot_poses(supervisor):
    """Move each robot back to its initial world pose without reloading Webots."""
    reset_count = 0
    for robot_name, def_name in ROBOT_DEF_NAMES.items():
        robot_node = supervisor.getFromDef(def_name)
        start_pose = ROBOT_START_POSES.get(robot_name)
        if robot_node is None or start_pose is None:
            continue

        x_m, y_m, theta_rad = start_pose
        translation_field = robot_node.getField("translation")
        if translation_field is not None:
            translation_field.setSFVec3f([x_m, y_m, ROBOT_START_Z_M])

        rotation_field = robot_node.getField("rotation")
        if rotation_field is not None:
            rotation_field.setSFRotation([0.0, 0.0, 1.0, theta_rad])

        reset_physics = getattr(robot_node, "resetPhysics", None)
        if callable(reset_physics):
            reset_physics()
        reset_count += 1
    return reset_count


def send_sim_reset_command(emitter):
    """Tell every robot controller to clear local state for a dashboard reset."""
    emitter.send(
        json.dumps(
            {
                "type": "sim_reset",
                "robot": "all",
            }
        )
    )


def pose_error(status_pose, reference_pose):
    """Return position and heading error between two poses."""
    if status_pose is None or reference_pose is None:
        return None

    try:
        position_error_m = math.hypot(
            reference_pose["x_m"] - status_pose["x_m"],
            reference_pose["y_m"] - status_pose["y_m"],
        )
        heading_error_rad = abs(
            normalize_angle(
                reference_pose.get("theta_rad", 0.0)
                - status_pose.get("theta_rad", 0.0)
            )
        )
    except (KeyError, TypeError):
        return None

    return position_error_m, heading_error_rad


def localization_confidence(status):
    """Read the robot-reported pose confidence, defaulting to zero."""
    localization = status.get("localization", {})
    try:
        return float(localization.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def should_send_ground_truth_pose_correction(status, actual_pose):
    """
    Return True when Webots truth should correct the robot pose.

    Ground truth is now a fallback guardrail. The supervisor lets odometry and
    scan matching carry normal localization, then uses Webots truth only when
    the robot says its estimate is stale or measured drift is too large.
    """
    if status is None or actual_pose is None:
        return False

    error = pose_error(status.get("pose"), actual_pose)
    if error is None:
        return True

    position_error_m, heading_error_rad = error
    return (
        position_error_m >= GROUND_TRUTH_DRIFT_CORRECTION_M
        or heading_error_rad >= GROUND_TRUTH_HEADING_CORRECTION_RAD
        or localization_confidence(status) < LOCALIZATION_MIN_CONFIDENCE
    )


def send_pose_correction(emitter, robot_name, pose, source, blend=1.0):
    """Send a pose correction command with its source and blend strength."""
    emitter.send(
        json.dumps(
            {
                "type": "pose_correction",
                "robot": robot_name,
                "source": source,
                "blend": round(blend, 3),
                "pose": {
                    "x_m": round(pose["x_m"], 4),
                    "y_m": round(pose["y_m"], 4),
                    "theta_rad": round(pose.get("theta_rad", 0.0), 4),
                },
            }
        )
    )


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

    def reset_progress(self):
        """Return the cleaning overlay to its initial all-dirty hidden state."""
        self.dirty_tiles = set(self.tile_centers)
        self.tile_claims = {}
        self.active_rooms = set()
        if not self.enabled:
            return

        for appearance in self.tile_appearances.values():
            appearance.getField("baseColor").setSFColor(DIRTY_TILE_COLOR)
            appearance.getField("transparency").setSFFloat(HIDDEN_TILE_TRANSPARENCY)

    def dirty_tile_count(
        self,
        room,
        no_go_zones=None,
        exclude_entry_blocked=True,
    ):
        """Return how many reachable cleaning tiles in one room are still dirty."""
        can_filter_no_go = bool(no_go_zones) and (
            hasattr(self, "rooms") or hasattr(self, "tile_centers")
        )
        if (
            exclude_entry_blocked
            and can_filter_no_go
            and room_entry_blocked_by_no_go_zones(room, no_go_zones)
        ):
            return 0

        dirty_count = 0
        for tile_key in self.dirty_tiles:
            tile_room, _, _ = tile_key
            if tile_room != room:
                continue
            if can_filter_no_go and point_blocked_by_no_go_zones(
                self.tile_center(tile_key),
                no_go_zones,
            ):
                continue
            dirty_count += 1
        return dirty_count

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

    def cleaned_tile_count(
        self,
        room,
        no_go_zones=None,
        exclude_entry_blocked=True,
    ):
        """Return how many cleaning tiles in one room have been cleaned."""
        total_tiles = self.room_tile_counts.get(room, 0)
        return max(
            0,
            total_tiles
            - self.dirty_tile_count(room, no_go_zones, exclude_entry_blocked),
        )

    def room_progress_percent(
        self,
        room,
        no_go_zones=None,
        exclude_entry_blocked=True,
    ):
        """Return how much of one room is clean, from 0 to 100."""
        total_tiles = self.room_tile_counts.get(room, 0)
        if total_tiles <= 0:
            return 0.0
        return (
            100.0
            * self.cleaned_tile_count(room, no_go_zones, exclude_entry_blocked)
            / total_tiles
        )

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

    def release_robot_room_claims(self, robot_name, room):
        """Forget dirty-tile claims owned by one robot in one room."""
        self.tile_claims = {
            tile_key: owner
            for tile_key, owner in self.tile_claims.items()
            if owner != robot_name or tile_key[0] != room
        }

    def release_robot_room_claims_except_centers(self, robot_name, room, centers):
        """Keep only this robot's room claims whose tile centers are still planned."""
        center_keys = {
            tuple(round(float(coordinate), 3) for coordinate in center)
            for center in centers
            if isinstance(center, (list, tuple)) and len(center) == 2
        }
        self.tile_claims = {
            tile_key: owner
            for tile_key, owner in self.tile_claims.items()
            if (
                owner != robot_name
                or tile_key[0] != room
                or tuple(self.tile_center(tile_key)) in center_keys
            )
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
        prefer_edges=False,
        no_go_zones=None,
        entry_blocked_is_reachable=False,
    ):
        """
        Reserve a small cleanup batch for one robot.

        A claim is like putting a robot's name on a dirty square, so another
        robot does not drive to the same square unless the claim is released.
        """
        if room not in self.rooms or max_tiles <= 0:
            return []
        if (
            not entry_blocked_is_reachable
            and room_entry_blocked_by_no_go_zones(room, no_go_zones)
        ):
            return []

        def tile_is_allowed(tile_key):
            return not point_blocked_by_no_go_zones(
                self.tile_center(tile_key),
                no_go_zones,
            )

        self.prune_cleaned_claims()
        claimed_by_robot = [
            tile_key
            for tile_key, owner in self.tile_claims.items()
            if owner == robot_name
            and tile_key in self.dirty_tiles
            and tile_key[0] == room
            and tile_is_allowed(tile_key)
        ]

        open_tiles = [
            tile_key
            for tile_key in self.dirty_tiles
            if tile_key[0] == room
            and self.tile_claims.get(tile_key, robot_name) == robot_name
            and tile_key not in claimed_by_robot
            and tile_is_allowed(tile_key)
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
            if not prefer_edges:
                return (distance_m, row_index, col_index)

            min_x, max_x, min_y, max_y = self.rooms[room]["bounds"]
            edge_distance_m = min(
                center_x_m - min_x,
                max_x - center_x_m,
                center_y_m - min_y,
                max_y - center_y_m,
            )
            corner_distance_m = min(
                math.hypot(center_x_m - corner_x_m, center_y_m - corner_y_m)
                for corner_x_m, corner_y_m in (
                    (min_x, min_y),
                    (min_x, max_y),
                    (max_x, min_y),
                    (max_x, max_y),
                )
            )
            is_edge_tile = edge_distance_m <= CLEANUP_EDGE_PRIORITY_MARGIN_M
            return (
                0 if is_edge_tile else 1,
                corner_distance_m,
                distance_m,
                row_index,
                col_index,
            )

        selected_tiles = sorted(claimed_by_robot, key=sort_key)
        selected_tiles.extend(
            sorted(open_tiles, key=sort_key)[:max(0, max_tiles - len(selected_tiles))]
        )
        selected_tiles = selected_tiles[:max_tiles]

        for tile_key in selected_tiles:
            self.tile_claims[tile_key] = robot_name

        return [self.tile_center(tile_key) for tile_key in selected_tiles]


def parse_operator_control_config(raw_config):
    """Return normalized operator controls from a JSON object."""
    if not isinstance(raw_config, dict):
        raw_config = {}

    def config_list(key):
        value = raw_config.get(key, [])
        return value if isinstance(value, list) else []

    paused_robots = set()
    for robot_name in config_list("paused_robots"):
        if robot_name == "all" or robot_name in EXPECTED_ROBOTS:
            paused_robots.add(robot_name)

    priority_zones = []
    for raw_zone in config_list("priority_zones"):
        zone = normalize_operator_zone(raw_zone, "priority")
        if zone is not None:
            priority_zones.append(zone)

    no_go_zones = []
    for raw_zone in config_list("no_go_zones"):
        zone = normalize_operator_zone(raw_zone, "no_go")
        if zone is not None:
            no_go_zones.append(zone)

    redirects = []
    raw_redirects = []
    raw_redirects.extend(config_list("redirects"))
    raw_redirects.extend(config_list("manual_redirects"))
    for index, raw_redirect in enumerate(raw_redirects):
        if not isinstance(raw_redirect, dict):
            continue
        robot_name = raw_redirect.get("robot")
        room = raw_redirect.get("room")
        if robot_name not in EXPECTED_ROBOTS or room not in ROOM_TASKS:
            continue

        redirect_id = raw_redirect.get("id")
        if not isinstance(redirect_id, str) or not redirect_id.strip():
            redirect_id = f"{robot_name}:{room}:{index}"
        redirects.append(
            {
                "id": redirect_id.strip(),
                "robot": robot_name,
                "room": room,
            }
        )

    sim_reset_request = None
    raw_reset_request = raw_config.get("sim_reset_request")
    if isinstance(raw_reset_request, dict):
        reset_id = raw_reset_request.get("id")
        if isinstance(reset_id, str) and reset_id.strip():
            sim_reset_request = {"id": reset_id.strip()}
            requested_at = raw_reset_request.get("requested_at")
            try:
                sim_reset_request["requested_at"] = round(float(requested_at), 3)
            except (TypeError, ValueError):
                pass

    return {
        "paused_robots": paused_robots,
        "priority_zones": priority_zones[:OPERATOR_MAX_PRIORITY_ZONES],
        "no_go_zones": no_go_zones[:OPERATOR_MAX_NO_GO_ZONES],
        "redirects": redirects,
        "sim_reset_request": sim_reset_request,
    }


def default_operator_control_config():
    """Return the empty operator command file used at sim startup."""
    return {
        "paused_robots": [],
        "priority_zones": [],
        "no_go_zones": [],
        "redirects": [],
        "sim_reset_request": None,
    }


def default_operator_state_snapshot():
    """Return the dashboard map state for a freshly reset simulation."""
    return {
        "step": None,
        "rooms": [
            {
                "name": room,
                "bounds": list(config["bounds"]),
                "progress_percent": 0.0,
                "dirty_tiles": 0,
                "assigned_robots": [],
                "completed": False,
                "priority_zones": [],
                "no_go_zones": [],
            }
            for room, config in ROOM_TASKS.items()
        ],
        "robots": [],
        "operator": {
            "paused_robots": [],
            "priority_zones": [],
            "no_go_zones": [],
            "pending_redirects": [],
            "sim_reset_request": None,
            "last_error": "",
        },
        "traffic": {
            "planned_distance_m": 0.0,
            "waiting_steps": 0,
            "doorway_conflicts": 0,
        },
        "metrics": default_evaluation_metrics_snapshot(),
    }


def reset_operator_control_file(control_path, preserve_zones=False):
    """Clear stale operator commands when the Webots simulation starts over."""
    path = Path(control_path)
    preserved_config = default_operator_control_config()
    if preserve_zones and path.exists():
        try:
            parsed = parse_operator_control_config(
                json.loads(path.read_text(encoding="utf-8"))
            )
            preserved_config["priority_zones"] = parsed["priority_zones"]
            preserved_config["no_go_zones"] = parsed["no_go_zones"]
        except (OSError, json.JSONDecodeError):
            pass

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(preserved_config, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError as exc:
        print(f"[supervisor] Could not reset operator controls: {exc}")
        return False
    return True


class OperatorZoneOverlay:
    """Draw operator priority and no-go rectangles on the Webots floor."""

    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.enabled = True
        self.root_children = supervisor.getRoot().getField("children")
        self.priority_slots = self.create_zone_slots(
            "OPERATOR_PRIORITY",
            OPERATOR_MAX_PRIORITY_ZONES,
            OPERATOR_PRIORITY_COLOR,
        )
        self.no_go_slots = self.create_zone_slots(
            "OPERATOR_NO_GO",
            OPERATOR_MAX_NO_GO_ZONES,
            OPERATOR_NO_GO_COLOR,
        )

    def create_zone_slots(self, prefix, max_slots, color):
        """Create reusable flat rectangles for one zone type."""
        slots = []
        for index in range(max_slots):
            def_name = f"{prefix}_{index}"
            zone_source = (
                f"DEF {def_name} Solid {{ "
                f"translation 0 0 {OPERATOR_ZONE_Z_M:.3f} "
                "children [ Shape { "
                "appearance PBRAppearance { "
                f"baseColor {color[0]:.3f} {color[1]:.3f} {color[2]:.3f} "
                f"transparency {OPERATOR_ZONE_HIDDEN_TRANSPARENCY:.1f} "
                "roughness 0.9 "
                "} "
                "geometry Box { size 0.050 0.050 0.006 } "
                "} ] "
                "}"
            )
            try:
                self.root_children.importMFNodeFromString(-1, zone_source)
                node = self.supervisor.getFromDef(def_name)
                shape = node.getField("children").getMFNode(0)
                appearance = shape.getField("appearance").getSFNode()
                geometry = shape.getField("geometry").getSFNode()
                slots.append(
                    {
                        "translation": node.getField("translation"),
                        "appearance": appearance,
                        "geometry": geometry,
                    }
                )
            except Exception as exc:
                self.enabled = False
                print(f"[supervisor] Operator zone overlay disabled: {exc}")
                return slots
        return slots

    def update_zones(self, priority_zones, no_go_zones):
        """Refresh all visible operator zone rectangles."""
        if not self.enabled:
            return

        self.update_zone_slots(self.priority_slots, priority_zones, OPERATOR_PRIORITY_COLOR)
        self.update_zone_slots(self.no_go_slots, no_go_zones, OPERATOR_NO_GO_COLOR)

    def update_zone_slots(self, slots, zones, color):
        """Show active zones and hide unused rectangle slots."""
        for index, slot in enumerate(slots):
            if index >= len(zones):
                self.hide_slot(slot)
                continue

            bounds = zones[index]["bounds"]
            min_x, max_x, min_y, max_y = bounds
            center_x_m = (min_x + max_x) / 2.0
            center_y_m = (min_y + max_y) / 2.0
            width_m = max(0.05, max_x - min_x)
            height_m = max(0.05, max_y - min_y)
            slot["translation"].setSFVec3f(
                [center_x_m, center_y_m, OPERATOR_ZONE_Z_M]
            )
            slot["geometry"].getField("size").setSFVec3f(
                [width_m, height_m, 0.006]
            )
            slot["appearance"].getField("baseColor").setSFColor(color)
            slot["appearance"].getField("transparency").setSFFloat(
                OPERATOR_ZONE_TRANSPARENCY
            )

    def hide_slot(self, slot):
        """Hide an unused zone rectangle."""
        slot["translation"].setSFVec3f([0.0, 0.0, OPERATOR_ZONE_Z_M])
        slot["geometry"].getField("size").setSFVec3f([0.05, 0.05, 0.006])
        slot["appearance"].getField("transparency").setSFFloat(
            OPERATOR_ZONE_HIDDEN_TRANSPARENCY
        )


class OperatorControls:
    """Read operator commands and keep their Webots visual state current."""

    def __init__(self, supervisor, control_path=None):
        self.supervisor = supervisor
        if control_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            control_path = repo_root / OPERATOR_CONTROL_FILE
        self.control_path = Path(control_path)
        self.paused_robots = set()
        self.priority_zones = []
        self.no_go_zones = []
        self.redirects = []
        self.sim_reset_request = None
        self.processed_redirect_ids = set()
        self.processed_sim_reset_ids = set()
        self.last_mtime_ns = None
        self.last_poll_step = -OPERATOR_CONTROL_POLL_STEPS
        self.last_loaded_step = None
        self.last_error = ""
        try:
            self.zone_overlay = OperatorZoneOverlay(supervisor)
        except Exception as exc:
            self.zone_overlay = None
            print(f"[supervisor] Operator visual zones disabled: {exc}")

    def load_if_due(self, step_count):
        """Reload operator controls when the JSON file changes."""
        if step_count - self.last_poll_step < OPERATOR_CONTROL_POLL_STEPS:
            return False

        self.last_poll_step = step_count
        return self.load(step_count)

    def load(self, step_count):
        """Load controls from disk, returning True when the visible state changed."""
        if not self.control_path.exists():
            changed = (
                self.last_mtime_ns is not None
                or bool(self.paused_robots)
                or bool(self.priority_zones)
                or bool(self.no_go_zones)
                or bool(self.redirects)
                or self.sim_reset_request is not None
            )
            self.last_mtime_ns = None
            self.last_error = ""
            self.paused_robots = set()
            self.priority_zones = []
            self.no_go_zones = []
            self.redirects = []
            self.sim_reset_request = None
            return changed

        try:
            stat = self.control_path.stat()
        except OSError as exc:
            self.last_error = f"read error: {exc}"
            return True

        if stat.st_mtime_ns == self.last_mtime_ns:
            return False

        try:
            raw_config = json.loads(self.control_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.last_error = f"json error: {exc}"
            return True

        parsed = parse_operator_control_config(raw_config)
        self.paused_robots = parsed["paused_robots"]
        self.priority_zones = parsed["priority_zones"]
        self.no_go_zones = parsed["no_go_zones"]
        self.redirects = parsed["redirects"]
        self.sim_reset_request = parsed["sim_reset_request"]
        self.last_mtime_ns = stat.st_mtime_ns
        self.last_loaded_step = step_count
        self.last_error = ""
        return True

    def paused_robot_names(self):
        """Return the concrete robot names currently paused by the operator."""
        if "all" in self.paused_robots:
            return set(EXPECTED_ROBOTS)
        return {robot for robot in self.paused_robots if robot in EXPECTED_ROBOTS}

    def pending_redirects(self):
        """Return manual redirects that have not been applied yet."""
        return [
            redirect
            for redirect in self.redirects
            if redirect["id"] not in self.processed_redirect_ids
        ]

    def mark_redirect_processed(self, redirect_id):
        """Remember that one manual redirect has already been applied."""
        self.processed_redirect_ids.add(redirect_id)

    def pending_sim_reset_request(self):
        """Return a dashboard sim reset request that has not been used yet."""
        if self.sim_reset_request is None:
            return None
        reset_id = self.sim_reset_request["id"]
        if reset_id in self.processed_sim_reset_ids:
            return None
        return self.sim_reset_request

    def mark_sim_reset_processed(self, reset_id):
        """Remember that one dashboard sim reset request was handled."""
        self.processed_sim_reset_ids.add(reset_id)

    def update_visuals(self, step_count):
        """Refresh floor-zone overlays for operator controls."""
        if self.zone_overlay is not None:
            self.zone_overlay.update_zones(self.priority_zones, self.no_go_zones)


def room_reached_coverage_goal(
    cleaning_overlay,
    room,
    no_go_zones=None,
    exclude_entry_blocked=True,
):
    """Return True when no visible cleaning tiles remain dirty."""
    return (
        cleaning_overlay.dirty_tile_count(
            room,
            no_go_zones,
            exclude_entry_blocked,
        )
        == 0
    )


def reopen_completed_rooms_with_reachable_dirt(
    completed_rooms,
    completed_robot_rooms,
    cleaning_overlay,
    no_go_zones=None,
):
    """Reopen completed rooms when current no-go zones expose dirty tiles."""
    reopened_rooms = []
    for room in sorted(completed_rooms):
        if room_reached_coverage_goal(cleaning_overlay, room, no_go_zones):
            continue
        completed_rooms.discard(room)
        reopened_rooms.append(room)

    if reopened_rooms:
        reopened_room_set = set(reopened_rooms)
        for robot_name, room in list(completed_robot_rooms.items()):
            if room in reopened_room_set:
                completed_robot_rooms.pop(robot_name, None)

    return reopened_rooms


def cleanup_trigger_reason(
    cleaning_overlay,
    room,
    coverage_done_for_room,
    room_stalled=False,
    no_go_zones=None,
    exclude_entry_blocked=True,
    cleanup_robot_count=1,
):
    """Return why a robot should switch from sweep to cleanup, or None."""
    dirty_count = cleaning_overlay.dirty_tile_count(
        room,
        no_go_zones,
        exclude_entry_blocked,
    )
    if dirty_count <= 0:
        return None

    progress_percent = cleaning_overlay.room_progress_percent(
        room,
        no_go_zones,
        exclude_entry_blocked,
    )
    if coverage_done_for_room:
        return f"sweep complete; {dirty_count} dirty tile(s) remain"
    if dirty_count <= FINAL_CLEANUP_DIRTY_TILE_COUNT:
        return f"final {dirty_count} dirty tile(s) remain"
    cleanup_capacity = max(
        FINAL_CLEANUP_DIRTY_TILE_COUNT,
        CLEANUP_TARGETS_PER_ROBOT * max(1, cleanup_robot_count),
    )
    if cleanup_robot_count > 1 and dirty_count <= cleanup_capacity:
        return (
            f"{dirty_count} dirty tile(s) fit "
            f"{max(1, cleanup_robot_count)} cleanup robot(s)"
        )
    if progress_percent >= FINAL_CLEANUP_PROGRESS_PERCENT:
        return f"room is {progress_percent:.1f}% clean"
    if room_stalled and progress_percent >= STALLED_CLEANUP_PROGRESS_PERCENT:
        return f"room stalled at {progress_percent:.1f}% clean"
    if dirty_count <= CLEANUP_TARGETS_PER_ROBOT:
        return f"{dirty_count} dirty tile(s) fit one cleanup robot"

    return None


def room_progress_snapshot(
    cleaning_overlay,
    no_go_zones=None,
    exclude_entry_blocked=True,
    entry_blocked_reachable_rooms=None,
):
    """Return clean percentages for every room in a stable order."""
    if entry_blocked_reachable_rooms is None:
        entry_blocked_reachable_rooms = set()
    return {
        room: round(
            cleaning_overlay.room_progress_percent(
                room,
                no_go_zones,
                exclude_entry_blocked and room not in entry_blocked_reachable_rooms,
            ),
            1,
        )
        for room in ROOM_TASKS
    }


def format_room_progress(snapshot):
    """Format room progress for one compact supervisor log line."""
    return " ".join(
        f"{room}={snapshot.get(room, 0.0):.1f}%"
        for room in ROOM_TASKS
    )


def room_dirty_tile_count(
    cleaning_overlay,
    room,
    no_go_zones=None,
    exclude_entry_blocked=True,
):
    """Return how many visible dirty tiles remain in one room."""
    if hasattr(cleaning_overlay, "dirty_tile_count"):
        return cleaning_overlay.dirty_tile_count(
            room,
            no_go_zones,
            exclude_entry_blocked,
        )

    dirty_count = 0
    for tile_key in getattr(cleaning_overlay, "dirty_tiles", set()):
        tile_room, _, _ = tile_key
        if tile_room != room:
            continue
        dirty_count += 1
    return dirty_count


def room_dirty_percent(
    cleaning_overlay,
    room,
    no_go_zones=None,
    exclude_entry_blocked=True,
):
    """Return roughly how much of one room still needs work."""
    total_tiles = cleaning_overlay.room_tile_counts.get(room, 0)
    if total_tiles <= 0:
        return 0.0
    return 100.0 * room_dirty_tile_count(
        cleaning_overlay,
        room,
        no_go_zones,
        exclude_entry_blocked,
    ) / total_tiles


def overall_coverage_percent(cleaning_overlay, no_go_zones=None):
    """Return weighted whole-arena cleaning coverage from 0 to 100."""
    total_tiles = 0
    dirty_tiles = 0
    for room in ROOM_TASKS:
        total_tiles += cleaning_overlay.room_tile_counts.get(room, 0)
        dirty_tiles += room_dirty_tile_count(
            cleaning_overlay,
            room,
            no_go_zones,
            exclude_entry_blocked=False,
        )

    if total_tiles <= 0:
        return 0.0
    clean_tiles = max(0, total_tiles - dirty_tiles)
    return 100.0 * clean_tiles / total_tiles


def default_evaluation_metrics_snapshot():
    """Return empty run metrics for the dashboard and JSON exports."""
    return {
        "elapsed_cleaning_time_s": None,
        "coverage_percent": 0.0,
        "coverage_target_percent": EVALUATION_COVERAGE_TARGET_PERCENT,
        "coverage_target_met": False,
        "coverage_target_time_s": None,
        "complete_cleaning_time_s": None,
        "inter_robot_collisions": 0,
        "collision_free": True,
        "minimum_robot_spacing_m": None,
        "closest_robot_pair": [],
        "single_robot_baseline_time_s": None,
        "single_robot_baseline_metric": "elapsed_cleaning_time_s",
        "time_reduction_percent": None,
        "meets_time_reduction_target": None,
    }


def load_single_robot_baseline(baseline_path):
    """
    Read the optional single-robot baseline time.

    The baseline file can be either a number or a JSON object with
    cleaning_time_s, coverage_95_time_s, or complete_cleaning_time_s.
    """
    path = Path(baseline_path)
    if not path.exists():
        return None

    try:
        raw_baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    baseline_metric = "elapsed_cleaning_time_s"
    if isinstance(raw_baseline, (int, float)):
        baseline_time_s = float(raw_baseline)
    elif isinstance(raw_baseline, dict):
        baseline_time_s = None
        baseline_keys = (
            ("cleaning_time_s", "elapsed_cleaning_time_s"),
            ("coverage_95_time_s", "coverage_target_time_s"),
            ("coverage_target_time_s", "coverage_target_time_s"),
            ("complete_cleaning_time_s", "complete_cleaning_time_s"),
        )
        for key, metric in baseline_keys:
            value = raw_baseline.get(key)
            if isinstance(value, (int, float)):
                baseline_time_s = float(value)
                baseline_metric = metric
                break
    else:
        baseline_time_s = None

    if baseline_time_s is None or baseline_time_s <= 0.0:
        return None
    return {
        "time_s": round(baseline_time_s, 3),
        "metric": baseline_metric,
    }


class EvaluationMetrics:
    """Track the quick demo metrics used for scaling and evaluation."""

    def __init__(
        self,
        single_robot_baseline_time_s=None,
        single_robot_baseline_metric="elapsed_cleaning_time_s",
        coverage_target_percent=EVALUATION_COVERAGE_TARGET_PERCENT,
        collision_distance_m=INTER_ROBOT_COLLISION_DISTANCE_M,
    ):
        self.single_robot_baseline_time_s = single_robot_baseline_time_s
        self.single_robot_baseline_metric = single_robot_baseline_metric
        self.coverage_target_percent = coverage_target_percent
        self.collision_distance_m = collision_distance_m
        self.reset()

    def reset(self):
        """Clear all per-run measurements."""
        self.start_step = None
        self.start_time_s = None
        self.coverage_percent = 0.0
        self.coverage_target_step = None
        self.coverage_target_time_s = None
        self.complete_step = None
        self.complete_time_s = None
        self.inter_robot_collisions = 0
        self.active_collision_pairs = set()
        self.minimum_robot_spacing_m = None
        self.closest_robot_pair = []

    def mark_started(self, step_count, webots_time_s):
        """Start the cleaning timer once the first room tasks are sent."""
        if self.start_time_s is not None:
            return
        self.start_step = step_count
        self.start_time_s = webots_time_s

    def update_progress(
        self,
        cleaning_overlay,
        step_count,
        webots_time_s,
        no_go_zones=None,
    ):
        """Refresh coverage and return notable log events."""
        events = []
        self.coverage_percent = overall_coverage_percent(
            cleaning_overlay,
            no_go_zones,
        )
        if self.start_time_s is None:
            return events

        if self.coverage_percent < 100.0 and self.complete_time_s is not None:
            self.complete_step = None
            self.complete_time_s = None
        if (
            self.coverage_percent < self.coverage_target_percent
            and self.coverage_target_time_s is not None
        ):
            self.coverage_target_step = None
            self.coverage_target_time_s = None

        if (
            self.coverage_target_time_s is None
            and self.coverage_percent >= self.coverage_target_percent
        ):
            self.coverage_target_step = step_count
            self.coverage_target_time_s = webots_time_s
            elapsed_s = self.coverage_target_time_s - self.start_time_s
            events.append(
                f"Coverage target reached: {self.coverage_percent:.1f}% "
                f"in {elapsed_s:.1f}s"
            )

        if self.complete_time_s is None and self.coverage_percent >= 100.0:
            self.complete_step = step_count
            self.complete_time_s = webots_time_s
            elapsed_s = self.complete_time_s - self.start_time_s
            events.append(f"Full coverage reached in {elapsed_s:.1f}s")

        return events

    def update_collisions(self, robot_poses, step_count, webots_time_s):
        """Count new robot-pair contacts based on center-to-center distance."""
        del step_count, webots_time_s
        events = []
        current_collision_pairs = set()
        pose_items = [
            (robot_name, pose)
            for robot_name, pose in robot_poses.items()
            if pose is not None
        ]
        for first_index, (first_robot, first_pose) in enumerate(pose_items):
            for second_robot, second_pose in pose_items[first_index + 1:]:
                distance_m = math.hypot(
                    second_pose["x_m"] - first_pose["x_m"],
                    second_pose["y_m"] - first_pose["y_m"],
                )
                if (
                    self.minimum_robot_spacing_m is None
                    or distance_m < self.minimum_robot_spacing_m
                ):
                    self.minimum_robot_spacing_m = distance_m
                    self.closest_robot_pair = [first_robot, second_robot]

                if distance_m > self.collision_distance_m:
                    continue

                pair = tuple(sorted((first_robot, second_robot)))
                current_collision_pairs.add(pair)
                if pair in self.active_collision_pairs:
                    continue

                self.inter_robot_collisions += 1
                events.append(
                    f"Inter-robot collision detected: {pair[0]} and {pair[1]} "
                    f"within {distance_m:.3f}m"
                )

        self.active_collision_pairs = current_collision_pairs
        return events

    def elapsed_cleaning_time_s(self, current_time_s):
        """Return seconds since cleaning tasks started, or None before start."""
        if self.start_time_s is None:
            return None
        end_time_s = self.complete_time_s
        if end_time_s is None:
            end_time_s = current_time_s
        if end_time_s is None:
            return None
        return max(0.0, end_time_s - self.start_time_s)

    def elapsed_target_time_s(self):
        """Return seconds from assignment to the 95% target, if reached."""
        if self.start_time_s is None or self.coverage_target_time_s is None:
            return None
        return max(0.0, self.coverage_target_time_s - self.start_time_s)

    def elapsed_complete_time_s(self):
        """Return seconds from assignment to full coverage, if reached."""
        if self.start_time_s is None or self.complete_time_s is None:
            return None
        return max(0.0, self.complete_time_s - self.start_time_s)

    def comparison_time_s(self, current_time_s):
        """Pick the current run time that matches the baseline's meaning."""
        if self.single_robot_baseline_metric == "coverage_target_time_s":
            return self.elapsed_target_time_s()
        if self.single_robot_baseline_metric == "complete_cleaning_time_s":
            return self.elapsed_complete_time_s()
        return self.elapsed_cleaning_time_s(current_time_s)

    def snapshot(self, step_count=None, webots_time_s=None):
        """Return serializable metrics for logs, dashboard, and report data."""
        del step_count
        snapshot = default_evaluation_metrics_snapshot()
        elapsed_time_s = self.elapsed_cleaning_time_s(webots_time_s)
        target_time_s = self.elapsed_target_time_s()
        complete_time_s = self.elapsed_complete_time_s()
        comparison_time_s = self.comparison_time_s(webots_time_s)

        snapshot.update(
            {
                "elapsed_cleaning_time_s": (
                    None if elapsed_time_s is None else round(elapsed_time_s, 1)
                ),
                "coverage_percent": round(self.coverage_percent, 1),
                "coverage_target_percent": round(self.coverage_target_percent, 1),
                "coverage_target_met": target_time_s is not None,
                "coverage_target_time_s": (
                    None if target_time_s is None else round(target_time_s, 1)
                ),
                "complete_cleaning_time_s": (
                    None if complete_time_s is None else round(complete_time_s, 1)
                ),
                "inter_robot_collisions": self.inter_robot_collisions,
                "collision_free": self.inter_robot_collisions == 0,
                "minimum_robot_spacing_m": (
                    None
                    if self.minimum_robot_spacing_m is None
                    else round(self.minimum_robot_spacing_m, 3)
                ),
                "closest_robot_pair": list(self.closest_robot_pair),
                "single_robot_baseline_time_s": self.single_robot_baseline_time_s,
                "single_robot_baseline_metric": self.single_robot_baseline_metric,
            }
        )

        if (
            self.single_robot_baseline_time_s is not None
            and comparison_time_s is not None
        ):
            reduction_percent = (
                (self.single_robot_baseline_time_s - comparison_time_s)
                / self.single_robot_baseline_time_s
                * 100.0
            )
            snapshot["time_reduction_percent"] = round(reduction_percent, 1)
            snapshot["meets_time_reduction_target"] = reduction_percent >= 50.0

        return snapshot


def current_robot_poses(supervisor, latest_robot_status):
    """Return actual Webots poses when available, otherwise last reported poses."""
    robot_poses = {}
    for robot_name in EXPECTED_ROBOTS:
        pose = get_actual_robot_pose(supervisor, robot_name)
        if pose is None:
            status = latest_robot_status.get(robot_name)
            pose = status.get("pose") if isinstance(status, dict) else None
        if pose is not None:
            robot_poses[robot_name] = pose
    return robot_poses


def pose_for_operator_state(pose):
    """Return a small pose object for the operator dashboard."""
    if pose is None:
        return None
    return {
        "x_m": round(float(pose.get("x_m", 0.0)), 3),
        "y_m": round(float(pose.get("y_m", 0.0)), 3),
        "theta_rad": round(float(pose.get("theta_rad", 0.0)), 3),
    }


def zone_names_for_room(zones, room):
    """Return operator zone names that cover a room."""
    names = []
    room_bounds = ROOM_TASKS[room]["bounds"]
    for zone in zones:
        if zone.get("room") == room or bounds_overlap(zone["bounds"], room_bounds):
            names.append(zone["name"])
    return names


def build_operator_state(
    supervisor,
    step_count,
    latest_robot_status,
    room_assignments,
    cleaning_overlay,
    operator_controls,
    operator_paused_robots,
    completed_rooms,
    traffic_reservations,
    evaluation_metrics=None,
    webots_time_s=None,
):
    """Build the live snapshot consumed by the operator dashboard."""
    path_summary = traffic_reservations.summary()
    if evaluation_metrics is None:
        metrics_snapshot = default_evaluation_metrics_snapshot()
    else:
        metrics_snapshot = evaluation_metrics.snapshot(step_count, webots_time_s)
    paused_robots = sorted(operator_paused_robots)
    robots = []

    for robot_name in EXPECTED_ROBOTS:
        status = latest_robot_status.get(robot_name)
        assignment = room_assignments.get(robot_name)
        actual_pose = get_actual_robot_pose(supervisor, robot_name)
        reported_pose = status.get("pose") if status is not None else None
        pose = actual_pose if actual_pose is not None else reported_pose
        coverage = status.get("coverage", {}) if status is not None else {}
        route = status.get("assignment_route", {}) if status is not None else {}
        assigned_room = assignment["room"] if assignment is not None else None
        display_phase = status.get("phase", "offline") if status is not None else "offline"
        if robot_name in operator_paused_robots:
            display_phase = "paused"

        robots.append(
            {
                "name": robot_name,
                "label": robot_name.replace("epuck_", "E"),
                "connected": status is not None,
                "paused": robot_name in operator_paused_robots,
                "phase": display_phase,
                "reported_phase": status.get("phase", "offline") if status else "offline",
                "room": assigned_room,
                "helper": bool(assignment and assignment.get("helper", False)),
                "pose": pose_for_operator_state(pose),
                "coverage": {
                    "room": coverage.get("room"),
                    "index": int(coverage.get("waypoint_index", 0)),
                    "count": int(coverage.get("waypoint_count", 0)),
                    "complete": bool(coverage.get("complete", False)),
                    "plan_kind": coverage.get("plan_kind", "sweep"),
                },
                "route_wait": int(route.get("waiting_steps_remaining", 0)),
                "target_reached": bool(
                    status.get("assignment_target_reached", False)
                    if status is not None
                    else False
                ),
            }
        )

    rooms = []
    for room_name, room_config in ROOM_TASKS.items():
        room_has_inside_robot = assigned_robot_inside_room(
            supervisor,
            latest_robot_status,
            room_assignments,
            room_name,
        )
        exclude_entry_blocked = not room_has_inside_robot
        assigned_robots = [
            robot_name.replace("epuck_", "E")
            for robot_name, assignment in room_assignments.items()
            if assignment is not None and assignment["room"] == room_name
        ]
        rooms.append(
            {
                "name": room_name,
                "bounds": list(room_config["bounds"]),
                "progress_percent": round(
                    cleaning_overlay.room_progress_percent(
                        room_name,
                        operator_controls.no_go_zones,
                        exclude_entry_blocked,
                    ),
                    1,
                ),
                "dirty_tiles": room_dirty_tile_count(
                    cleaning_overlay,
                    room_name,
                    operator_controls.no_go_zones,
                    exclude_entry_blocked,
                ),
                "assigned_robots": assigned_robots,
                "completed": room_name in completed_rooms,
                "priority_zones": zone_names_for_room(
                    operator_controls.priority_zones,
                    room_name,
                ),
                "no_go_zones": zone_names_for_room(
                    operator_controls.no_go_zones,
                    room_name,
                ),
            }
        )

    return {
        "step": step_count,
        "rooms": rooms,
        "robots": robots,
        "operator": {
            "paused_robots": paused_robots,
            "priority_zones": operator_controls.priority_zones,
            "no_go_zones": operator_controls.no_go_zones,
            "pending_redirects": operator_controls.pending_redirects(),
            "sim_reset_request": operator_controls.pending_sim_reset_request(),
            "last_error": operator_controls.last_error,
        },
        "traffic": {
            "planned_distance_m": round(path_summary["planned_distance_m"], 2),
            "waiting_steps": path_summary["waiting_steps"],
            "doorway_conflicts": path_summary["doorway_conflicts"],
        },
        "metrics": metrics_snapshot,
    }


def write_operator_state_file(state_path, operator_state):
    """Write the live dashboard state with a small atomic replace."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(operator_state, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def write_evaluation_metrics_file(metrics_path, metrics_snapshot):
    """Write a compact metrics file that can be copied into the report."""
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(metrics_snapshot, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def reset_operator_state_file(state_path):
    """Clear the dashboard map state after a Webots reset."""
    try:
        write_operator_state_file(state_path, default_operator_state_snapshot())
    except OSError as exc:
        print(f"[supervisor] Could not reset operator state: {exc}")
        return False
    return True


def room_progress_rate(progress_monitors, room):
    """
    Return recent cleaning progress in percent per 1000 supervisor steps.

    Bigger means a room is still improving on its own. Near zero means help is
    more useful there, like noticing one chore has stopped moving while others
    are still getting done.
    """
    monitor = progress_monitors.get(room, {})
    rate = monitor.get("rate_percent_per_1000_steps")
    if isinstance(rate, (int, float)):
        return rate
    return 0.0


def route_cost_to_room(robot_status, room, no_go_zones=None):
    """Return planned route distance from one robot status to a room."""
    if not isinstance(robot_status, dict):
        return float("inf")
    pose = robot_status.get("pose")
    if pose is None:
        return float("inf")
    base_route = generate_assignment_route(pose, room)
    route = route_around_no_go_zones(
        base_route,
        no_go_zones,
        start_point=pose_point(pose),
    )
    if not route_reaches_final_waypoint(base_route, route):
        return float("inf")
    return route_distance_m(pose, route)


def pose_distance_m(first_pose, second_pose):
    """Return ground distance between two robot poses."""
    if first_pose is None or second_pose is None:
        return 0.0
    return math.hypot(
        second_pose["x_m"] - first_pose["x_m"],
        second_pose["y_m"] - first_pose["y_m"],
    )


def robot_should_be_moving(status, assignment):
    """Return True when the assigned robot is supposed to be driving."""
    if status is None or assignment is None:
        return False

    launch = status.get("launch", {})
    launch_done = launch.get("complete", False) or launch.get("timed_out", False)
    if not launch_done:
        return True

    if not status.get("assignment_target_reached", False):
        assignment_route = status.get("assignment_route", {})
        if assignment_route.get("waiting_steps_remaining", 0) > 0:
            return False
        return True

    coverage = status.get("coverage", {})
    if coverage.get("room") != assignment["room"]:
        return False

    return not coverage.get("complete", False)


def update_robot_motion_monitor(
    motion_monitors,
    robot_name,
    status,
    pose,
    assignment,
    step_count,
):
    """
    Track whether one robot is making enough movement while it has work.

    The monitor is like checking a toy car every few seconds: if it should be
    driving but it is still in almost the same spot after a while, call it
    stuck so the supervisor can try a new assignment route.
    """
    monitor = motion_monitors.setdefault(robot_name, {})
    if not robot_should_be_moving(status, assignment) or pose is None:
        monitor.clear()
        if pose is not None:
            monitor["anchor_pose"] = pose
            monitor["anchor_step"] = step_count
        return False, ""

    anchor_pose = monitor.get("anchor_pose")
    anchor_step = monitor.get("anchor_step")
    if anchor_pose is None or anchor_step is None:
        monitor["anchor_pose"] = pose
        monitor["anchor_step"] = step_count
        return False, ""

    moved_m = pose_distance_m(anchor_pose, pose)
    if moved_m >= ROBOT_STUCK_MIN_MOVE_M:
        monitor["anchor_pose"] = pose
        monitor["anchor_step"] = step_count
        return False, ""

    elapsed_steps = step_count - anchor_step
    if elapsed_steps < ROBOT_STUCK_WINDOW_STEPS:
        return False, ""

    last_alert_step = monitor.get("last_alert_step")
    if (
        last_alert_step is not None
        and step_count - last_alert_step < ROBOT_STUCK_ALERT_COOLDOWN_STEPS
    ):
        return False, ""

    monitor["last_alert_step"] = step_count
    motion = status.get("motion", {})
    if motion.get("corner_pressure") or motion.get("front_blocked"):
        reason = "blocked by nearby obstacle"
    else:
        reason = f"moved only {moved_m:.2f}m in {elapsed_steps} steps"
    return True, reason


def update_room_progress_monitors(
    progress_monitors,
    progress_snapshot,
    room_assignments,
    completed_rooms,
    step_count,
    progress_ready_rooms=None,
):
    """Return rooms whose visible cleaning progress has stopped improving."""
    active_rooms = {
        assignment["room"]
        for assignment in room_assignments.values()
        if assignment is not None
    }
    if progress_ready_rooms is None:
        progress_ready_rooms = active_rooms

    stalled_rooms = []
    for room in ROOM_TASKS:
        progress_percent = progress_snapshot.get(room, 0.0)
        monitor = progress_monitors.setdefault(room, {})
        if (
            room in completed_rooms
            or room not in active_rooms
            or room not in progress_ready_rooms
            or progress_percent >= COVERAGE_COMPLETE_PERCENT
        ):
            monitor.clear()
            monitor["progress"] = progress_percent
            monitor["step"] = step_count
            monitor["ready"] = False
            monitor["rate_percent_per_1000_steps"] = 0.0
            continue

        previous_progress = monitor.get("progress")
        previous_step = monitor.get("step")
        if (
            previous_progress is None
            or previous_step is None
            or not monitor.get("ready", False)
        ):
            monitor["progress"] = progress_percent
            monitor["step"] = step_count
            monitor["ready"] = True
            monitor["rate_percent_per_1000_steps"] = 0.0
            continue

        if progress_percent >= previous_progress + ROOM_PROGRESS_MIN_DELTA_PERCENT:
            elapsed_steps = max(1, step_count - previous_step)
            progress_gain = progress_percent - previous_progress
            monitor["rate_percent_per_1000_steps"] = (
                progress_gain / elapsed_steps * 1000.0
            )
            monitor["progress"] = progress_percent
            monitor["step"] = step_count
            monitor["ready"] = True
            continue

        elapsed_steps = step_count - previous_step
        if elapsed_steps >= ROOM_PROGRESS_RATE_WINDOW_STEPS:
            monitor["rate_percent_per_1000_steps"] = 0.0
        if elapsed_steps < ROOM_PROGRESS_STALL_STEPS:
            continue

        last_alert_step = monitor.get("last_alert_step")
        if (
            last_alert_step is not None
            and step_count - last_alert_step < ROOM_PROGRESS_ALERT_COOLDOWN_STEPS
        ):
            continue

        monitor["last_alert_step"] = step_count
        stalled_rooms.append(room)

    return stalled_rooms


def rooms_ready_for_progress_monitoring(supervisor, latest_robot_status, room_assignments):
    """Return rooms where at least one assigned robot has reached cleaning range."""
    ready_rooms = set()
    for robot_name, assignment in room_assignments.items():
        if assignment is None:
            continue

        room = assignment["room"]
        status = latest_robot_status.get(robot_name)
        if status is None:
            continue

        pose = get_actual_robot_pose(supervisor, robot_name)
        if pose is None:
            pose = status.get("pose")

        if (
            robot_can_mark_cleaning(pose, room)
            or status.get("assignment_target_reached", False)
        ):
            ready_rooms.add(room)

    return ready_rooms


def assigned_robot_inside_room(supervisor, latest_robot_status, room_assignments, room):
    """Return True when an assigned robot is already inside one room."""
    for robot_name, assignment in room_assignments.items():
        if assignment is None or assignment["room"] != room:
            continue

        status = latest_robot_status.get(robot_name)
        pose = get_actual_robot_pose(supervisor, robot_name)
        if pose is None and isinstance(status, dict):
            pose = status.get("pose")
        if room_for_pose(pose) == room:
            return True
    return False


def select_robot_for_stalled_room(
    stalled_room,
    room_assignments,
    cleaning_overlay,
    excluded_robots=None,
    latest_robot_status=None,
    no_go_zones=None,
    reserved_assignments=None,
):
    """Pick a robot that can be redirected to a stalled room without abandoning work."""
    if excluded_robots is None:
        excluded_robots = set()
    if latest_robot_status is None:
        latest_robot_status = {}

    if not room_has_active_capacity(
        room_assignments,
        stalled_room,
        reserved_assignments=reserved_assignments,
    ):
        return None

    idle_candidates = []
    for robot_name in EXPECTED_ROBOTS:
        if robot_name in excluded_robots:
            continue
        if room_assignments.get(robot_name) is None:
            route_distance = route_cost_to_room(
                latest_robot_status.get(robot_name),
                stalled_room,
                no_go_zones,
            )
            if no_go_zones and math.isinf(route_distance):
                continue
            idle_candidates.append((route_distance, robot_name))
    if idle_candidates:
        idle_candidates.sort()
        return idle_candidates[0][1]

    active_room_counts = {}
    for assignment in room_assignments.values():
        if assignment is None:
            continue
        room = assignment["room"]
        active_room_counts[room] = active_room_counts.get(room, 0) + 1

    helper_candidates = []
    backup_candidates = []
    for robot_name, assignment in room_assignments.items():
        if robot_name in excluded_robots:
            continue
        if assignment is None or assignment["room"] == stalled_room:
            continue

        source_room = assignment["room"]
        source_progress = cleaning_overlay.room_progress_percent(
            source_room,
            no_go_zones,
        )
        source_count = active_room_counts.get(source_room, 0)
        source_target_count = target_robot_count_for_room(source_room)
        if source_count <= source_target_count:
            continue

        route_distance_m = route_cost_to_room(
            latest_robot_status.get(robot_name),
            stalled_room,
            no_go_zones,
        )
        if no_go_zones and math.isinf(route_distance_m):
            continue
        candidate = (route_distance_m, -source_progress, robot_name)
        if assignment.get("helper", False):
            helper_candidates.append(candidate)
        else:
            backup_candidates.append(candidate)

    candidates = helper_candidates or backup_candidates
    if not candidates:
        return None

    candidates.sort()
    return candidates[0][-1]


def target_robot_count_for_room(room):
    """
    Return how many robots should normally work in one room.

    Small rooms are like a small bedroom: one robot is usually enough before
    larger rooms get help. Medium and large rooms can use two robots sooner.
    """
    if room_is_small(room):
        return SMALL_ROOM_MAX_PRIMARY_ROBOTS
    return LARGE_ROOM_MAX_PRIMARY_ROBOTS


def late_helper_robot_count_for_room(room):
    """Return the most robots allowed when only already-staffed work remains."""
    return target_robot_count_for_room(room)


def room_is_small(room):
    """Return True when a room is too small to benefit from extra robots."""
    area_m2 = ROOM_TASKS[room]["area_m2"]
    return area_m2 <= ROOM_TASKS["nw_small"]["area_m2"] + 1e-9


def room_assignments_with_reserved_tasks(room_assignments, reserved_assignments=None):
    """Return active assignments plus paused tasks that still reserve a room."""
    combined_assignments = dict(room_assignments)
    if reserved_assignments is None:
        return combined_assignments

    for robot_name, assignment in reserved_assignments.items():
        if assignment is not None and combined_assignments.get(robot_name) is None:
            combined_assignments[robot_name] = assignment
    return combined_assignments


def active_robot_count(
    room_assignments,
    room,
    ignored_robot=None,
    reserved_assignments=None,
):
    """Return how many robots are currently assigned to one room."""
    assignment_pool = room_assignments_with_reserved_tasks(
        room_assignments,
        reserved_assignments,
    )
    return sum(
        1 for robot_name, assignment in assignment_pool.items()
        if (
            robot_name != ignored_robot
            and assignment is not None
            and assignment["room"] == room
        )
    )


def room_has_active_capacity(
    room_assignments,
    room,
    ignored_robot=None,
    reserved_assignments=None,
):
    """Return True when one more active robot can work without crowding."""
    return (
        active_robot_count(
            room_assignments,
            room,
            ignored_robot,
            reserved_assignments,
        )
        < target_robot_count_for_room(room)
    )


def operator_redirect_has_capacity(
    room_assignments,
    robot_name,
    next_room,
    previous_room=None,
    reserved_assignments=None,
):
    """Return True when an operator redirect will not overcrowd a room."""
    return room_has_active_capacity(
        room_assignments,
        next_room,
        ignored_robot=robot_name,
        reserved_assignments=reserved_assignments,
    )


def unfinished_rooms_below_target(
    room_assignments,
    completed_rooms,
    cleaning_overlay,
    ignored_robot=None,
    no_go_zones=None,
    reserved_assignments=None,
):
    """Return unfinished rooms that have fewer robots than their normal target."""
    assignment_pool = room_assignments_with_reserved_tasks(
        room_assignments,
        reserved_assignments,
    )
    active_room_counts = {}
    for robot, assignment in assignment_pool.items():
        if robot == ignored_robot or assignment is None:
            continue
        room = assignment["room"]
        active_room_counts[room] = active_room_counts.get(room, 0) + 1

    rooms_below_target = []
    for room in ROOM_TASKS:
        if room in completed_rooms:
            continue
        progress_percent = cleaning_overlay.room_progress_percent(room, no_go_zones)
        if progress_percent >= COVERAGE_COMPLETE_PERCENT:
            continue
        active_count = active_room_counts.get(room, 0)
        target_count = target_robot_count_for_room(room)
        if active_count < target_count:
            rooms_below_target.append(room)

    return rooms_below_target


def select_redirect_room_for_stalled_progress(
    stalled_room,
    room_assignments,
    completed_rooms,
    cleaning_overlay,
):
    """Choose where an extra robot should go after a room stops improving."""
    return stalled_room


def select_reassignment_room(
    robot_name,
    room_assignments,
    completed_rooms,
    cleaning_overlay,
    latest_robot_status=None,
    progress_monitors=None,
    priority_zones=None,
    no_go_zones=None,
    reserved_assignments=None,
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
    if latest_robot_status is None:
        latest_robot_status = {}
    if progress_monitors is None:
        progress_monitors = {}

    active_room_counts = {}
    reserved_room_counts = {}
    assignment_pool = room_assignments_with_reserved_tasks(
        room_assignments,
        reserved_assignments,
    )
    for robot, assignment in assignment_pool.items():
        if robot == robot_name or assignment is None:
            continue
        room = assignment["room"]
        active_room_counts[room] = active_room_counts.get(room, 0) + 1
        if room_assignments.get(robot) is None:
            reserved_room_counts[room] = reserved_room_counts.get(room, 0) + 1

    preferred_rooms = unfinished_rooms_below_target(
        room_assignments,
        completed_rooms,
        cleaning_overlay,
        ignored_robot=robot_name,
        no_go_zones=no_go_zones,
        reserved_assignments=reserved_assignments,
    )
    preferred_rooms = [
        room
        for room in preferred_rooms
        if room != current_room
        and (
            not no_go_zones
            or not math.isinf(
                route_cost_to_room(
                    latest_robot_status.get(robot_name),
                    room,
                    no_go_zones,
                )
            )
        )
    ]
    candidate_rooms = []
    late_helper_rooms = []
    for room in ROOM_TASKS:
        if room == current_room or room in completed_rooms:
            continue
        dirty_count = cleaning_overlay.dirty_tile_count(room, no_go_zones)
        active_count = active_room_counts.get(room, 0)
        if active_count > 0 and dirty_count < FINAL_CLEANUP_DIRTY_TILE_COUNT:
            continue
        progress_percent = cleaning_overlay.room_progress_percent(room, no_go_zones)
        if progress_percent >= COVERAGE_COMPLETE_PERCENT:
            continue
        if preferred_rooms and room not in preferred_rooms:
            continue
        target_count = target_robot_count_for_room(room)
        late_helper = False
        if active_count >= target_count:
            if (
                preferred_rooms
                or active_count >= late_helper_robot_count_for_room(room)
                or (room_is_small(room) and reserved_room_counts.get(room, 0) > 0)
            ):
                continue
            late_helper = True
        support_ratio = active_count / target_count
        progress_rate = room_progress_rate(progress_monitors, room)
        route_distance = route_cost_to_room(
            latest_robot_status.get(robot_name),
            room,
            no_go_zones,
        )
        if no_go_zones and math.isinf(route_distance):
            continue
        dirty_percent = room_dirty_percent(cleaning_overlay, room, no_go_zones)
        priority_weight = priority_weight_for_room(priority_zones, room)
        score = (
            support_ratio
            + MRTA_PROGRESS_RATE_WEIGHT * progress_rate
            + MRTA_ROUTE_DISTANCE_WEIGHT * route_distance
            - MRTA_DIRTY_PERCENT_WEIGHT * dirty_percent
            - OPERATOR_PRIORITY_ROUTE_WEIGHT * priority_weight
        )
        candidate = (
            support_ratio,
            score,
            -priority_weight,
            progress_rate,
            progress_percent,
            route_distance,
            room,
        )
        if late_helper:
            late_helper_rooms.append(candidate)
        else:
            candidate_rooms.append(candidate)

    if not candidate_rooms:
        candidate_rooms = late_helper_rooms
    if not candidate_rooms:
        return None

    candidate_rooms.sort()
    return candidate_rooms[0][6]


def select_stuck_recovery_room(
    robot_name,
    room_assignments,
    completed_rooms,
    cleaning_overlay,
    latest_robot_status=None,
    progress_monitors=None,
    recovery_count=0,
    priority_zones=None,
    no_go_zones=None,
    robot_inside_room=False,
    reserved_assignments=None,
):
    """Choose whether a stuck robot should retry its room or help elsewhere."""
    assignment = room_assignments.get(robot_name)
    if assignment is None:
        return None

    room = assignment["room"]
    room_unfinished = (
        room not in completed_rooms
        and not room_reached_coverage_goal(
            cleaning_overlay,
            room,
            no_go_zones,
            exclude_entry_blocked=not robot_inside_room,
        )
    )
    if (
        room_unfinished
        and (
            recovery_count <= STUCK_ROUTE_RETRY_LIMIT
            or active_robot_count(room_assignments, room) <= 1
        )
    ):
        return room

    fallback_room = select_reassignment_room(
        robot_name,
        room_assignments,
        completed_rooms,
        cleaning_overlay,
        latest_robot_status,
        progress_monitors,
        priority_zones,
        no_go_zones,
        reserved_assignments,
    )
    if fallback_room is not None:
        return fallback_room
    if room_unfinished:
        return room
    return None


def build_assignment(robot_status, room, helper=False):
    """Create one assignment dictionary from the robot pose to a room."""
    pose = robot_status["pose"]
    route = generate_assignment_route(pose, room)
    distance_m = route_distance_m(pose, route)
    area_m2 = ROOM_TASKS[room]["area_m2"]
    cost = MRTA_DISTANCE_WEIGHT * distance_m + MRTA_AREA_WEIGHT * area_m2
    return {
        "room": room,
        "cost": round(cost, 3),
        "target": ROOM_TASKS[room]["center"],
        "route": route,
        "helper": helper,
    }


def prepare_assignment_for_dispatch(
    robot_name,
    assignment,
    robot_status=None,
    room_assignments=None,
    traffic_reservations=None,
    step_count=0,
    no_go_zones=None,
):
    """Refresh route, lane, delay, and metric fields before sending a task."""
    lane_index = 0
    lane_count = 1
    if room_assignments is not None:
        lane_index, lane_count = coverage_lane_for_robot(
            robot_name,
            room_assignments,
        )

    pose = None
    if isinstance(robot_status, dict):
        pose = robot_status.get("pose")

    coverage_plan = generate_coverage_waypoints(
        assignment["room"],
        lane_index,
        lane_count,
    )
    entry_blocked = room_entry_blocked_by_no_go_zones(
        assignment["room"],
        no_go_zones,
    )
    if entry_blocked:
        coverage_plan = []
    else:
        coverage_plan = coverage_plan_around_no_go_zones(
            coverage_plan,
            no_go_zones,
        )
    coverage_available = bool(coverage_plan)
    route_target = (
        coverage_plan[0]
        if coverage_available
        else route_target_for_empty_coverage(
            assignment["room"],
            no_go_zones,
            pose,
        )
    )

    route = assignment.get("route", [assignment["target"]])
    if pose is not None:
        if coverage_available:
            route = generate_assignment_route(
                pose,
                assignment["room"],
                final_waypoint=route_target,
            )
        else:
            route = generate_route_to_room_hub(
                pose,
                assignment["room"],
                final_waypoint=route_target,
            )
    route = route_around_no_go_zones(
        route,
        no_go_zones,
        start_point=pose_point(pose),
        allow_initial_blocked_waypoints=True,
    )
    if (
        no_go_zones
        and pose is not None
        and route_target is not None
        and (not route or route[-1] != route_target)
    ):
        coverage_plan = []
        route_target = route[-1] if route else route_target_for_empty_coverage(
            assignment["room"],
            no_go_zones,
            pose,
        )
        route = [route_target] if route_target is not None and not route else route

    planned_distance_m = route_distance_m(pose, route)
    resources = traffic_resource_keys(pose, assignment["room"])
    reservation = {
        "start_delay_steps": 0,
        "conflict_count": 0,
        "doorway_conflicts": 0,
        "conflict_resources": [],
        "reserved_resources": resources,
    }
    if traffic_reservations is not None:
        reservation = traffic_reservations.reserve(
            robot_name,
            resources,
            step_count,
        )
        traffic_reservations.record_distance(planned_distance_m)

    assignment["target"] = route_target
    assignment["route"] = route
    assignment["lane_index"] = lane_index
    assignment["lane_count"] = lane_count
    assignment["start_delay_steps"] = reservation["start_delay_steps"]
    assignment["path_metrics"] = {
        "planned_distance_m": round(planned_distance_m, 3),
        "start_delay_steps": reservation["start_delay_steps"],
        "route_conflicts": reservation["conflict_count"],
        "doorway_conflicts": reservation["doorway_conflicts"],
        "resources": reservation["reserved_resources"],
        "conflict_resources": reservation["conflict_resources"],
        "operator_no_go_zones": [
            zone["name"]
            for zone in (no_go_zones or [])
        ],
    }
    return coverage_plan


def send_assignment_commands(
    emitter,
    robot_name,
    assignment,
    room_assignments=None,
    send_coverage=True,
    robot_status=None,
    traffic_reservations=None,
    step_count=0,
    no_go_zones=None,
):
    """Send the room target and coverage path to one robot."""
    coverage_plan = prepare_assignment_for_dispatch(
        robot_name,
        assignment,
        robot_status,
        room_assignments,
        traffic_reservations,
        step_count,
        no_go_zones,
    )
    emitter.send(
        json.dumps(
            {
                "type": "task_assignment",
                "robot": robot_name,
                "room": assignment["room"],
                "target": assignment["target"],
                "route": assignment.get("route", [assignment["target"]]),
                "cost": assignment["cost"],
                "start_delay_steps": assignment.get("start_delay_steps", 0),
                "path_metrics": assignment.get("path_metrics", {}),
            }
        )
    )
    if not send_coverage:
        return []

    send_coverage_plan(
        emitter,
        robot_name,
        assignment["room"],
        coverage_plan,
        plan_kind="sweep",
        lane_index=assignment["lane_index"],
        lane_count=assignment["lane_count"],
    )
    return coverage_plan


def send_coverage_plan(
    emitter,
    robot_name,
    room,
    coverage_plan,
    plan_kind="sweep",
    lane_index=0,
    lane_count=1,
):
    """Send coverage waypoints to one robot."""
    lane_index, lane_count = normalized_lane(lane_index, lane_count)
    emitter.send(
        json.dumps(
            {
                "type": "coverage_plan",
                "robot": robot_name,
                "room": room,
                "waypoints": coverage_plan,
                "plan_kind": plan_kind,
                "lane_index": lane_index,
                "lane_count": lane_count,
            }
        )
    )
    return coverage_plan


def send_room_coverage_plans(
    emitter,
    room_assignments,
    room,
    cleaning_overlay=None,
    cleanup_plan_signatures=None,
    cleanup_plan_steps=None,
    no_go_zones=None,
    latest_robot_status=None,
):
    """Resend split sweep lanes to every robot currently assigned to one room."""
    coverage_plans = {}
    for robot_name in room_robot_names(room_assignments, room):
        lane_index, lane_count = coverage_lane_for_robot(
            robot_name,
            room_assignments,
        )
        coverage_plan = generate_coverage_waypoints(room, lane_index, lane_count)
        status = (
            latest_robot_status.get(robot_name)
            if isinstance(latest_robot_status, dict)
            else None
        )
        pose = status.get("pose") if isinstance(status, dict) else None
        robot_inside_room = pose is not None and room_for_pose(pose) == room
        start_point = pose_point(pose) if robot_inside_room else None
        if room_entry_blocked_by_no_go_zones(room, no_go_zones) and not robot_inside_room:
            coverage_plan = []
        else:
            coverage_plan = coverage_plan_around_no_go_zones(
                coverage_plan,
                no_go_zones,
                start_point=start_point,
            )
        if cleanup_plan_signatures is not None:
            cleanup_plan_signatures.pop(robot_name, None)
        if cleanup_plan_steps is not None:
            cleanup_plan_steps.pop(robot_name, None)
        if cleaning_overlay is not None:
            cleaning_overlay.release_robot_claims(robot_name)
        coverage_plans[robot_name] = send_coverage_plan(
            emitter,
            robot_name,
            room,
            coverage_plan,
            plan_kind="sweep",
            lane_index=lane_index,
            lane_count=lane_count,
        )
    return coverage_plans


def status_pose_inside_room(status, room):
    """Return True when a robot status reports the robot inside one room."""
    if not isinstance(status, dict):
        return False
    pose = status.get("pose")
    return pose is not None and room_for_pose(pose) == room


def assignment_route_needs_refresh(status, assignment, pose):
    """Return True when route constraints changed before the robot is in-room."""
    if not isinstance(status, dict) or assignment is None:
        return False
    if not status.get("assignment_target_reached", False):
        return True
    return room_for_pose(pose) != assignment["room"]


def room_has_status_robot_inside(latest_robot_status, room_assignments, room):
    """Return True when any assigned robot status is already inside the room."""
    if not isinstance(latest_robot_status, dict):
        return False
    return any(
        status_pose_inside_room(latest_robot_status.get(robot_name), room)
        for robot_name in room_robot_names(room_assignments, room)
    )


def room_has_completed_sweep_status(latest_robot_status, room_assignments, room):
    """Return True when any assigned robot reports a finished sweep for the room."""
    if not isinstance(latest_robot_status, dict):
        return False
    for robot_name in room_robot_names(room_assignments, room):
        status = latest_robot_status.get(robot_name)
        if not isinstance(status, dict):
            continue
        coverage = status.get("coverage", {})
        if (
            coverage.get("room") == room
            and coverage.get("complete", False)
            and coverage.get("plan_kind", "sweep") == "sweep"
        ):
            return True
    return False


def room_cleanup_completed_by_all(latest_robot_status, room_assignments, room):
    """Return True when every robot in one room finished a cleanup plan."""
    robot_names = room_robot_names(room_assignments, room)
    if not robot_names or not isinstance(latest_robot_status, dict):
        return False

    for robot_name in robot_names:
        status = latest_robot_status.get(robot_name)
        if not isinstance(status, dict):
            return False
        coverage = status.get("coverage", {})
        if (
            coverage.get("room") != room
            or not coverage.get("complete", False)
            or coverage.get("plan_kind") != "cleanup"
        ):
            return False
    return True


def send_room_cleanup_retry_if_all_done(
    emitter,
    room_assignments,
    room,
    cleaning_overlay,
    cleanup_plan_signatures,
    cleanup_plan_steps,
    step_count,
    reason,
    no_go_zones=None,
    latest_robot_status=None,
):
    """Retry room cleanup when finished cleanup plans left dirt behind."""
    if not room_cleanup_completed_by_all(latest_robot_status, room_assignments, room):
        return {}

    for robot_name in room_robot_names(room_assignments, room):
        cleanup_plan_signatures.pop(robot_name, None)
        cleanup_plan_steps.pop(robot_name, None)
    cleaning_overlay.release_room_claims(room)
    return send_room_cleanup_plans(
        emitter,
        room_assignments,
        room,
        cleaning_overlay,
        cleanup_plan_signatures,
        cleanup_plan_steps,
        step_count,
        reason,
        no_go_zones,
        latest_robot_status,
    )


def cleanup_robot_names_for_room(
    robot_names,
    latest_robot_status,
    cleaning_overlay,
    room,
    no_go_zones=None,
    room_assignments=None,
):
    """Choose which robots should move during this room cleanup pass."""
    if len(robot_names) <= 1:
        return list(robot_names)

    dirty_count = cleaning_overlay.dirty_tile_count(room, no_go_zones)
    if dirty_count >= FINAL_CLEANUP_DIRTY_TILE_COUNT:
        return list(robot_names)

    dirty_centers = [
        center for center in cleaning_overlay.dirty_tile_centers(room)
        if not point_blocked_by_no_go_zones(center, no_go_zones)
    ]
    if not dirty_centers:
        return list(robot_names)

    if room_assignments is None:
        room_assignments = {}
    primary_robot_names = [
        robot_name for robot_name in robot_names
        if not room_assignments.get(robot_name, {}).get("helper", False)
    ]
    candidate_robot_names = primary_robot_names or robot_names

    def distance_to_dirty(robot_name):
        status = (
            latest_robot_status.get(robot_name)
            if isinstance(latest_robot_status, dict)
            else None
        )
        pose = status.get("pose") if isinstance(status, dict) else None
        if pose is None:
            return (float("inf"), robot_name)
        nearest_distance = min(
            math.hypot(center[0] - pose["x_m"], center[1] - pose["y_m"])
            for center in dirty_centers
        )
        return (nearest_distance, robot_name)

    return [min(candidate_robot_names, key=distance_to_dirty)]


def retry_completed_cleanup_rooms(
    emitter,
    room_assignments,
    completed_rooms,
    cleaning_overlay,
    cleanup_plan_signatures,
    cleanup_plan_steps,
    step_count,
    no_go_zones=None,
    latest_robot_status=None,
):
    """Restart room cleanup when every assigned robot has stopped too early."""
    retried_rooms = set()
    coverage_plans = {}
    for room in ROOM_TASKS:
        if room in completed_rooms:
            continue
        dirty_count = cleaning_overlay.dirty_tile_count(room, no_go_zones)
        if dirty_count <= 0:
            continue
        retry_plans = send_room_cleanup_retry_if_all_done(
            emitter,
            room_assignments,
            room,
            cleaning_overlay,
            cleanup_plan_signatures,
            cleanup_plan_steps,
            step_count,
            f"completed cleanup left {dirty_count} dirty tile(s)",
            no_go_zones,
            latest_robot_status,
        )
        if not retry_plans:
            continue
        coverage_plans.update(retry_plans)
        retried_rooms.add(room)
        print(
            f"[supervisor] Retrying cleanup for {room}; "
            f"{dirty_count} dirty tile(s) remain after completed plans"
        )

    return retried_rooms, coverage_plans


def send_room_cleanup_plans(
    emitter,
    room_assignments,
    room,
    cleaning_overlay,
    cleanup_plan_signatures,
    cleanup_plan_steps,
    step_count,
    reason,
    no_go_zones=None,
    latest_robot_status=None,
):
    """Send targeted cleanup claims to every robot assigned to one room."""
    coverage_plans = {}
    robot_names = room_robot_names(room_assignments, room)
    for robot_name in robot_names:
        cleanup_plan_signatures.pop(robot_name, None)
        cleanup_plan_steps.pop(robot_name, None)
        if hasattr(cleaning_overlay, "release_robot_room_claims"):
            cleaning_overlay.release_robot_room_claims(robot_name, room)
        else:
            cleaning_overlay.release_robot_claims(robot_name)

    moving_robot_names = set(
        cleanup_robot_names_for_room(
            robot_names,
            latest_robot_status,
            cleaning_overlay,
            room,
            no_go_zones,
            room_assignments,
        )
    )
    for robot_name in robot_names:
        if robot_name not in moving_robot_names:
            empty_plan = send_empty_cleanup_plan_if_needed(
                emitter,
                robot_name,
                room,
                cleanup_plan_signatures,
                cleanup_plan_steps,
                step_count,
            )
            if empty_plan is not None:
                coverage_plans[robot_name] = empty_plan
            continue

        status = (
            latest_robot_status.get(robot_name)
            if isinstance(latest_robot_status, dict)
            else None
        )
        pose = status.get("pose") if isinstance(status, dict) else None
        robot_inside_room = pose is not None and room_for_pose(pose) == room
        cleanup_plan = send_cleanup_plan_if_needed(
            emitter,
            cleaning_overlay,
            robot_name,
            room,
            pose,
            cleanup_plan_signatures,
            cleanup_plan_steps,
            step_count,
            reason,
            no_go_zones,
            entry_blocked_is_reachable=robot_inside_room,
        )
        if cleanup_plan is not None:
            coverage_plans[robot_name] = cleanup_plan
            continue

        empty_plan = send_empty_cleanup_plan_if_needed(
            emitter,
            robot_name,
            room,
            cleanup_plan_signatures,
            cleanup_plan_steps,
            step_count,
        )
        if empty_plan is not None:
            coverage_plans[robot_name] = empty_plan

    return coverage_plans


def send_room_work_plans(
    emitter,
    room_assignments,
    room,
    cleaning_overlay=None,
    cleanup_plan_signatures=None,
    cleanup_plan_steps=None,
    no_go_zones=None,
    latest_robot_status=None,
    step_count=0,
):
    """Send either normal sweep lanes or targeted cleanup claims for a room."""
    if (
        cleaning_overlay is None
        or cleanup_plan_signatures is None
        or cleanup_plan_steps is None
        or not hasattr(cleaning_overlay, "claim_dirty_tile_centers")
    ):
        return send_room_coverage_plans(
            emitter,
            room_assignments,
            room,
            cleaning_overlay,
            cleanup_plan_signatures,
            cleanup_plan_steps,
            no_go_zones,
            latest_robot_status,
        )

    robot_names = room_robot_names(room_assignments, room)
    room_has_inside_robot = room_has_status_robot_inside(
        latest_robot_status,
        room_assignments,
        room,
    )
    cleanup_reason = cleanup_trigger_reason(
        cleaning_overlay,
        room,
        room_has_completed_sweep_status(latest_robot_status, room_assignments, room),
        no_go_zones=no_go_zones,
        exclude_entry_blocked=not room_has_inside_robot,
        cleanup_robot_count=len(robot_names),
    )
    if cleanup_reason is not None:
        return send_room_cleanup_plans(
            emitter,
            room_assignments,
            room,
            cleaning_overlay,
            cleanup_plan_signatures,
            cleanup_plan_steps,
            step_count,
            cleanup_reason,
            no_go_zones,
            latest_robot_status,
        )

    return send_room_coverage_plans(
        emitter,
        room_assignments,
        room,
        cleaning_overlay,
        cleanup_plan_signatures,
        cleanup_plan_steps,
        no_go_zones,
        latest_robot_status,
    )


def send_recovery_command(emitter, robot_name, reason):
    """Ask one robot to back up and turn before trying its route again."""
    emitter.send(
        json.dumps(
            {
                "type": "recovery",
                "robot": robot_name,
                "reason": reason,
                "reverse_steps": ROBOT_RECOVERY_REVERSE_STEPS,
                "turn_steps": ROBOT_RECOVERY_TURN_STEPS,
            }
        )
    )


def assign_idle_robots_to_unfinished_rooms(
    emitter,
    room_assignments,
    completed_rooms,
    cleaning_overlay,
    latest_robot_status,
    operator_paused_robots,
    coverage_plans,
    cleanup_plan_signatures,
    cleanup_plan_steps,
    last_cleaning_poses,
    robot_recovery_counts,
    traffic_reservations=None,
    step_count=0,
    priority_zones=None,
    no_go_zones=None,
    operator_paused_assignments=None,
):
    """Send idle robots back to rooms that became unfinished again."""
    assigned_rooms = {}
    for robot_name in EXPECTED_ROBOTS:
        if robot_name in operator_paused_robots:
            continue
        if room_assignments.get(robot_name) is not None:
            continue

        status = latest_robot_status.get(robot_name)
        if status is None:
            continue

        next_room = select_reassignment_room(
            robot_name,
            room_assignments,
            completed_rooms,
            cleaning_overlay,
            latest_robot_status,
            priority_zones=priority_zones,
            no_go_zones=no_go_zones,
            reserved_assignments=operator_paused_assignments,
        )
        if next_room is None:
            continue

        next_assignment = build_assignment(status, next_room, helper=True)
        room_assignments[robot_name] = next_assignment
        cleanup_plan_signatures.pop(robot_name, None)
        cleanup_plan_steps.pop(robot_name, None)
        cleaning_overlay.release_robot_claims(robot_name)
        last_cleaning_poses.pop(robot_name, None)
        robot_recovery_counts.pop(robot_name, None)
        cleaning_overlay.show_dirty_room(next_room)
        send_assignment_commands(
            emitter,
            robot_name,
            next_assignment,
            room_assignments,
            send_coverage=False,
            robot_status=status,
            traffic_reservations=traffic_reservations,
            step_count=step_count,
            no_go_zones=no_go_zones,
        )
        coverage_plans.update(
            send_room_work_plans(
                emitter,
                room_assignments,
                next_room,
                cleaning_overlay,
                cleanup_plan_signatures,
                cleanup_plan_steps,
                no_go_zones,
                latest_robot_status,
                step_count,
            )
        )
        assigned_rooms[robot_name] = next_room

    return assigned_rooms


def handle_operator_no_go_change(
    emitter,
    room_assignments,
    completed_rooms,
    completed_robot_rooms,
    cleaning_overlay,
    latest_robot_status,
    operator_paused_robots,
    coverage_plans,
    cleanup_plan_signatures,
    cleanup_plan_steps,
    last_cleaning_poses,
    robot_recovery_counts,
    traffic_reservations=None,
    step_count=0,
    priority_zones=None,
    no_go_zones=None,
    operator_paused_assignments=None,
):
    """Refresh reachable work after the operator changes no-go zones."""
    reopened_rooms = reopen_completed_rooms_with_reachable_dirt(
        completed_rooms,
        completed_robot_rooms,
        cleaning_overlay,
        no_go_zones,
    )
    for room in reopened_rooms:
        cleaning_overlay.show_dirty_room(room)

    assigned_idle_rooms = assign_idle_robots_to_unfinished_rooms(
        emitter,
        room_assignments,
        completed_rooms,
        cleaning_overlay,
        latest_robot_status,
        operator_paused_robots,
        coverage_plans,
        cleanup_plan_signatures,
        cleanup_plan_steps,
        last_cleaning_poses,
        robot_recovery_counts,
        traffic_reservations,
        step_count,
        priority_zones,
        no_go_zones,
        operator_paused_assignments,
    )
    return reopened_rooms, assigned_idle_rooms


def send_cleanup_plan_if_needed(
    emitter,
    cleaning_overlay,
    robot_name,
    room,
    pose,
    cleanup_plan_signatures,
    cleanup_plan_steps,
    step_count,
    reason,
    no_go_zones=None,
    entry_blocked_is_reachable=False,
):
    """Send a cleanup pass when the robot has new or stale dirty-tile targets."""
    claimed_cleanup_plan = cleaning_overlay.claim_dirty_tile_centers(
        room,
        robot_name,
        pose,
        prefer_edges=True,
        no_go_zones=no_go_zones,
        entry_blocked_is_reachable=entry_blocked_is_reachable,
    )
    cleanup_targets = [
        (
            center,
            cleanup_waypoint_for_tile_center(room, center),
        )
        for center in claimed_cleanup_plan
    ]
    cleanup_plan = coverage_plan_around_no_go_zones(
        [waypoint for _, waypoint in cleanup_targets],
        no_go_zones,
        start_point=pose_point(pose),
    )
    if claimed_cleanup_plan:
        cleanup_waypoint_keys = {
            tuple(round(float(coordinate), 3) for coordinate in waypoint)
            for waypoint in cleanup_plan
            if isinstance(waypoint, (list, tuple)) and len(waypoint) == 2
        }
        reachable_claim_centers = [
            center for center, waypoint in cleanup_targets
            if tuple(round(float(coordinate), 3) for coordinate in waypoint)
            in cleanup_waypoint_keys
        ]
        cleaning_overlay.release_robot_room_claims_except_centers(
            robot_name,
            room,
            reachable_claim_centers,
        )
    if not cleanup_plan:
        return None

    cleanup_signature = (
        room,
        tuple(tuple(waypoint) for waypoint in cleanup_plan),
    )
    previous_step = cleanup_plan_steps.get(robot_name)
    resend_due = (
        previous_step is not None
        and step_count - previous_step >= CLEANUP_RESEND_STEPS
    )
    if cleanup_plan_signatures.get(robot_name) == cleanup_signature and not resend_due:
        return None

    cleanup_plan_steps[robot_name] = step_count
    cleanup_plan_signatures[robot_name] = cleanup_signature
    send_coverage_plan(
        emitter,
        robot_name,
        room,
        cleanup_plan,
        plan_kind="cleanup",
    )
    print(
        f"[supervisor] Sent {robot_name} cleanup pass for {room}: "
        f"{len(cleanup_plan)} claimed dirty tile(s); reason={reason}"
    )
    return cleanup_plan


def send_empty_cleanup_plan_if_needed(
    emitter,
    robot_name,
    room,
    cleanup_plan_signatures,
    cleanup_plan_steps,
    step_count,
):
    """Send an empty cleanup plan so a robot with no claim stops roaming."""
    cleanup_signature = (room, ())
    previous_step = cleanup_plan_steps.get(robot_name)
    resend_due = (
        previous_step is not None
        and step_count - previous_step >= CLEANUP_RESEND_STEPS
    )
    if cleanup_plan_signatures.get(robot_name) == cleanup_signature and not resend_due:
        return None

    cleanup_plan_steps[robot_name] = step_count
    cleanup_plan_signatures[robot_name] = cleanup_signature
    send_coverage_plan(
        emitter,
        robot_name,
        room,
        [],
        plan_kind="cleanup",
    )
    return []


def has_active_cleanup_plan(cleanup_plan_signatures, robot_name, room):
    """Return True when this robot already has non-empty cleanup targets."""
    cleanup_signature = cleanup_plan_signatures.get(robot_name)
    return (
        isinstance(cleanup_signature, tuple)
        and len(cleanup_signature) == 2
        and cleanup_signature[0] == room
        and bool(cleanup_signature[1])
    )


def reset_completed_cleanup_plan_if_dirty(
    robot_name,
    room,
    coverage,
    cleaning_overlay,
    cleanup_plan_signatures,
    cleanup_plan_steps,
):
    """Clear a finished cleanup target so remaining dirt gets a fresh pass."""
    if (
        coverage.get("room") != room
        or not coverage.get("complete", False)
        or coverage.get("plan_kind") != "cleanup"
        or not has_active_cleanup_plan(cleanup_plan_signatures, robot_name, room)
    ):
        return False

    cleanup_plan_signatures.pop(robot_name, None)
    cleanup_plan_steps.pop(robot_name, None)
    cleaning_overlay.release_robot_room_claims(robot_name, room)
    return True


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

    repo_root = Path(__file__).resolve().parents[2]
    operator_control_path = repo_root / OPERATOR_CONTROL_FILE
    reset_operator_control_file(operator_control_path, preserve_zones=True)

    global_grid = GlobalOccupancyGrid(world_size_m=6.0, cell_size_m=0.05)
    cleaning_overlay = CleaningOverlay(supervisor, ROOM_TASKS)
    operator_controls = OperatorControls(supervisor, operator_control_path)
    operator_state_path = repo_root / OPERATOR_STATE_FILE
    evaluation_metrics_path = repo_root / EVALUATION_METRICS_FILE
    single_robot_baseline = load_single_robot_baseline(
        repo_root / SINGLE_ROBOT_BASELINE_FILE
    )
    if single_robot_baseline is None:
        single_robot_baseline_time_s = None
        single_robot_baseline_metric = "elapsed_cleaning_time_s"
    else:
        single_robot_baseline_time_s = single_robot_baseline["time_s"]
        single_robot_baseline_metric = single_robot_baseline["metric"]
    evaluation_metrics = EvaluationMetrics(
        single_robot_baseline_time_s,
        single_robot_baseline_metric,
    )
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
    cleanup_plan_steps = {}
    robot_motion_monitors = {}
    robot_recovery_counts = {}
    room_progress_monitors = {}
    last_scan_match_log_steps = {}
    last_map_update_log_steps = {}
    traffic_reservations = TrafficReservationBook()
    operator_paused_robots = set()
    operator_paused_assignments = {}
    last_operator_no_go_signature = operator_zone_signature(operator_controls.no_go_zones)
    last_operator_state_error_step = -COMMUNICATION_SUMMARY_INTERVAL_STEPS
    last_webots_time_s = supervisor.getTime()
    completed_rooms = set()
    completed_robot_rooms = {}
    assignments_sent = False
    step_count = 0
    reset_operator_state_file(operator_state_path)

    def clear_runtime_after_sim_reset():
        """Clear supervisor memory that should not survive a sim reset."""
        nonlocal global_grid, traffic_reservations, assignments_sent
        nonlocal last_operator_no_go_signature

        global_grid = GlobalOccupancyGrid(world_size_m=6.0, cell_size_m=0.05)
        latest_robot_status.clear()
        last_cleaning_poses.clear()
        room_assignments.clear()
        coverage_plans.clear()
        cleanup_plan_signatures.clear()
        cleanup_plan_steps.clear()
        robot_motion_monitors.clear()
        robot_recovery_counts.clear()
        room_progress_monitors.clear()
        last_scan_match_log_steps.clear()
        last_map_update_log_steps.clear()
        traffic_reservations = TrafficReservationBook()
        operator_paused_robots.clear()
        operator_paused_assignments.clear()
        completed_rooms.clear()
        completed_robot_rooms.clear()
        evaluation_metrics.reset()
        assignments_sent = False
        last_operator_no_go_signature = operator_zone_signature(
            operator_controls.no_go_zones
        )
        try:
            cleaning_overlay.reset_progress()
        except Exception as exc:
            print(f"[supervisor] Could not reset cleaning overlay: {exc}")
        reset_operator_state_file(operator_state_path)

    print("[supervisor] Starting central communication hub")
    print(
        f"[supervisor] Global map ready: "
        f"{global_grid.width}x{global_grid.height} cells"
    )
    if single_robot_baseline_time_s is None:
        print(
            "[supervisor] Single-robot baseline not loaded; "
            f"add {SINGLE_ROBOT_BASELINE_FILE} with cleaning_time_s to compare"
        )
    else:
        print(
            "[supervisor] Single-robot baseline loaded: "
            f"{single_robot_baseline_time_s:.1f}s"
        )

    while supervisor.step(timestep) != -1:
        step_count += 1
        webots_time_s = supervisor.getTime()
        if webots_time_s + 1e-9 < last_webots_time_s:
            print(
                "[supervisor] Webots simulation reset detected; "
                "clearing operator controls"
            )
            reset_operator_control_file(operator_controls.control_path)
            operator_controls.last_mtime_ns = None
            operator_controls.load(step_count)
            operator_controls.update_visuals(step_count)
            clear_runtime_after_sim_reset()
        last_webots_time_s = webots_time_s

        operator_controls.load_if_due(step_count)
        sim_reset_request = operator_controls.pending_sim_reset_request()
        if sim_reset_request is not None:
            reset_id = sim_reset_request["id"]
            print(f"[supervisor] Dashboard requested simulation reset: {reset_id}")
            operator_controls.mark_sim_reset_processed(reset_id)
            reset_operator_control_file(operator_controls.control_path)
            operator_controls.last_mtime_ns = None
            operator_controls.load(step_count)
            operator_controls.update_visuals(step_count)
            clear_runtime_after_sim_reset()
            send_sim_reset_command(emitter)
            reset_count = reset_webots_robot_poses(supervisor)
            print(
                "[supervisor] Dashboard soft reset complete: "
                f"{reset_count}/{len(EXPECTED_ROBOTS)} robot pose(s) restored"
            )
            continue

        operator_no_go_signature = operator_zone_signature(operator_controls.no_go_zones)
        operator_no_go_changed = (
            operator_no_go_signature != last_operator_no_go_signature
        )
        last_operator_no_go_signature = operator_no_go_signature
        operator_controls.update_visuals(step_count)

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
            raw_map_update = message.get("map_update", {})
            scan_match = global_grid.scan_match_update(raw_map_update)
            map_update = scan_match["map_update"]
            if scan_match["accepted"]:
                message["map_update"] = map_update
                message["localization"] = dict(message.get("localization", {}))
                message["localization"]["scan_match"] = {
                    "offset_cells": scan_match["offset_cells"],
                    "offset_m": scan_match["offset_m"],
                    "score_gain": scan_match["score_gain"],
                    "informative_cells": scan_match["informative_cells"],
                }
                last_scan_step = last_scan_match_log_steps.get(robot_name, -999999)
                if step_count - last_scan_step >= SCAN_MATCH_LOG_INTERVAL_STEPS:
                    last_scan_match_log_steps[robot_name] = step_count
                    print(
                        f"[supervisor] Scan matched map update from {robot_name}: "
                        f"offset=({scan_match['offset_m'][0]:.2f}, "
                        f"{scan_match['offset_m'][1]:.2f})m "
                        f"score_gain={scan_match['score_gain']}"
                    )

            latest_robot_status[robot_name] = message
            new_free_cells, new_wall_cells = global_grid.merge_update(map_update)

            if first_message:
                print(f"[supervisor] Connected to {robot_name}")
            if new_free_cells or new_wall_cells:
                last_map_log_step = last_map_update_log_steps.get(
                    robot_name,
                    -MAP_UPDATE_LOG_INTERVAL_STEPS,
                )
                if step_count - last_map_log_step >= MAP_UPDATE_LOG_INTERVAL_STEPS:
                    last_map_update_log_steps[robot_name] = step_count
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

            if robot_name in operator_paused_robots:
                last_cleaning_poses.pop(robot_name, None)
            elif assignment is not None and robot_can_mark_cleaning(pose, assignment["room"]):
                cleaned_count = cleaning_overlay.mark_clean_trail(
                    assignment["room"],
                    last_cleaning_poses.get(robot_name),
                    pose,
                )
                if cleaned_count:
                    robot_recovery_counts.pop(robot_name, None)
                    room_progress = cleaning_overlay.room_progress_percent(
                        assignment["room"],
                        operator_controls.no_go_zones,
                    )
                    print(
                        f"[supervisor] Cleaned {cleaned_count} tile(s) in "
                        f"{assignment['room']} with {robot_name}; "
                        f"room_clean={room_progress:.1f}%"
                    )
                last_cleaning_poses[robot_name] = pose
            else:
                last_cleaning_poses.pop(robot_name, None)

        for metrics_event in evaluation_metrics.update_collisions(
            current_robot_poses(supervisor, latest_robot_status),
            step_count,
            webots_time_s,
        ):
            print(f"[supervisor] {metrics_event}")
        for metrics_event in evaluation_metrics.update_progress(
            cleaning_overlay,
            step_count,
            webots_time_s,
            operator_controls.no_go_zones,
        ):
            print(f"[supervisor] {metrics_event}")

        if step_count % GROUND_TRUTH_POSE_CORRECTION_INTERVAL_STEPS == 0:
            for robot_name in EXPECTED_ROBOTS:
                status = latest_robot_status.get(robot_name)
                pose = get_actual_robot_pose(supervisor, robot_name)
                if not should_send_ground_truth_pose_correction(status, pose):
                    continue

                send_pose_correction(
                    emitter,
                    robot_name,
                    pose,
                    source="ground_truth",
                    blend=1.0,
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
            room_assignments = task_allocator.assign(
                latest_robot_status,
                no_go_zones=operator_controls.no_go_zones,
                priority_zones=operator_controls.priority_zones,
            )
            if room_assignments:
                for robot_name, assignment in room_assignments.items():
                    cleaning_overlay.show_dirty_room(assignment["room"])
                    coverage_plans[robot_name] = send_assignment_commands(
                        emitter,
                        robot_name,
                        assignment,
                        room_assignments,
                        robot_status=latest_robot_status.get(robot_name),
                        traffic_reservations=traffic_reservations,
                        step_count=step_count,
                        no_go_zones=operator_controls.no_go_zones,
                    )
                    print(
                        f"[supervisor] Assigned {robot_name} -> "
                        f"{assignment['room']} cost={assignment['cost']:.3f} "
                        f"coverage_waypoints={len(coverage_plans[robot_name])} "
                        f"path={assignment['path_metrics']['planned_distance_m']:.2f}m "
                        f"wait={assignment['start_delay_steps']} step(s)"
                    )
                assignments_sent = True
                evaluation_metrics.mark_started(step_count, webots_time_s)

        if assignments_sent:
            if operator_no_go_changed:
                reopened_rooms, assigned_idle_rooms = handle_operator_no_go_change(
                    emitter,
                    room_assignments,
                    completed_rooms,
                    completed_robot_rooms,
                    cleaning_overlay,
                    latest_robot_status,
                    operator_paused_robots,
                    coverage_plans,
                    cleanup_plan_signatures,
                    cleanup_plan_steps,
                    last_cleaning_poses,
                    robot_recovery_counts,
                    traffic_reservations,
                    step_count,
                    operator_controls.priority_zones,
                    operator_controls.no_go_zones,
                    operator_paused_assignments,
                )
                if reopened_rooms:
                    print(
                        "[supervisor] Reopened room(s) after operator no-go change: "
                        + ", ".join(reopened_rooms)
                    )
                for robot_name, room in assigned_idle_rooms.items():
                    room_source = "reopened" if room in reopened_rooms else "available"
                    print(
                        f"[supervisor] Reassigned idle {robot_name} "
                        f"to {room_source} {room}"
                    )
                for robot_name, assignment in room_assignments.items():
                    if robot_name in operator_paused_robots or assignment is None:
                        continue
                    status = latest_robot_status.get(robot_name)
                    if status is None:
                        continue
                    pose = get_actual_robot_pose(supervisor, robot_name)
                    if pose is None:
                        pose = status.get("pose")
                    if not assignment_route_needs_refresh(status, assignment, pose):
                        continue
                    send_assignment_commands(
                        emitter,
                        robot_name,
                        assignment,
                        room_assignments,
                        send_coverage=False,
                        robot_status=status,
                        traffic_reservations=traffic_reservations,
                        step_count=step_count,
                        no_go_zones=operator_controls.no_go_zones,
                    )
                refreshed_rooms = sorted(
                    {
                        assignment["room"]
                        for robot_name, assignment in room_assignments.items()
                        if (
                            robot_name not in operator_paused_robots
                            and assignment is not None
                        )
                    }
                )
                for room in refreshed_rooms:
                    coverage_plans.update(
                        send_room_work_plans(
                            emitter,
                            room_assignments,
                            room,
                            cleaning_overlay,
                            cleanup_plan_signatures,
                            cleanup_plan_steps,
                            operator_controls.no_go_zones,
                            latest_robot_status,
                            step_count,
                        )
                    )
                print("[supervisor] Refreshed active routes and sweeps for operator no-go zone changes")

            for redirect in operator_controls.pending_redirects():
                robot_name = redirect["robot"]
                next_room = redirect["room"]
                status = latest_robot_status.get(robot_name)
                if status is None:
                    continue

                robot_is_paused = robot_name in operator_paused_robots
                if robot_is_paused:
                    previous_assignment = operator_paused_assignments.get(robot_name)
                else:
                    previous_assignment = room_assignments.get(robot_name)
                previous_room = None
                if previous_assignment is not None:
                    previous_room = previous_assignment["room"]

                if not operator_redirect_has_capacity(
                    room_assignments,
                    robot_name,
                    next_room,
                    previous_room,
                    operator_paused_assignments,
                ):
                    operator_controls.mark_redirect_processed(redirect["id"])
                    print(
                        f"[supervisor] Ignored operator redirect for {robot_name} "
                        f"to {next_room}; small room already has an active robot"
                    )
                    continue

                cleaning_overlay.release_robot_claims(robot_name)
                cleanup_plan_signatures.pop(robot_name, None)
                cleanup_plan_steps.pop(robot_name, None)
                last_cleaning_poses.pop(robot_name, None)
                completed_robot_rooms.pop(robot_name, None)
                robot_recovery_counts.pop(robot_name, None)
                robot_motion_monitors.pop(robot_name, None)
                if cleaning_overlay.dirty_tile_count(
                    next_room,
                    operator_controls.no_go_zones,
                ) > 0:
                    completed_rooms.discard(next_room)

                next_assignment = build_assignment(status, next_room, helper=True)
                if robot_is_paused:
                    operator_paused_assignments[robot_name] = next_assignment
                    room_assignments.pop(robot_name, None)
                    coverage_plans.pop(robot_name, None)
                    cleaning_overlay.show_dirty_room(next_room)
                    operator_controls.mark_redirect_processed(redirect["id"])
                    print(
                        f"[supervisor] Operator queued paused {robot_name} "
                        f"from {previous_room or 'idle'} to {next_room}"
                    )
                    continue

                room_assignments[robot_name] = next_assignment
                cleaning_overlay.show_dirty_room(next_room)
                send_assignment_commands(
                    emitter,
                    robot_name,
                    next_assignment,
                    room_assignments,
                    send_coverage=False,
                    robot_status=status,
                    traffic_reservations=traffic_reservations,
                    step_count=step_count,
                    no_go_zones=operator_controls.no_go_zones,
                )
                coverage_plans.update(
                    send_room_work_plans(
                        emitter,
                        room_assignments,
                        next_room,
                        cleaning_overlay,
                        cleanup_plan_signatures,
                        cleanup_plan_steps,
                        operator_controls.no_go_zones,
                        latest_robot_status,
                        step_count,
                    )
                )
                if previous_room is not None and previous_room != next_room:
                    coverage_plans.update(
                        send_room_work_plans(
                            emitter,
                            room_assignments,
                            previous_room,
                            cleaning_overlay,
                            cleanup_plan_signatures,
                            cleanup_plan_steps,
                            operator_controls.no_go_zones,
                            latest_robot_status,
                            step_count,
                        )
                    )

                operator_controls.mark_redirect_processed(redirect["id"])
                print(
                    f"[supervisor] Operator redirected {robot_name} "
                    f"from {previous_room or 'idle'} to {next_room}"
                )

            requested_paused_robots = operator_controls.paused_robot_names()
            for robot_name in EXPECTED_ROBOTS:
                if (
                    robot_name in requested_paused_robots
                    and robot_name not in operator_paused_robots
                ):
                    send_idle_command(emitter, robot_name)
                    operator_paused_robots.add(robot_name)
                    pause_robot_assignment(
                        robot_name,
                        room_assignments,
                        operator_paused_assignments,
                    )
                    coverage_plans.pop(robot_name, None)
                    robot_motion_monitors.pop(robot_name, None)
                    cleanup_plan_signatures.pop(robot_name, None)
                    cleanup_plan_steps.pop(robot_name, None)
                    cleaning_overlay.release_robot_claims(robot_name)
                    last_cleaning_poses.pop(robot_name, None)
                    print(f"[supervisor] Operator paused {robot_name}")
                elif (
                    robot_name not in requested_paused_robots
                    and robot_name in operator_paused_robots
                ):
                    operator_paused_robots.remove(robot_name)
                    assignment = resume_robot_assignment(
                        robot_name,
                        room_assignments,
                        operator_paused_assignments,
                    )
                    status = latest_robot_status.get(robot_name)
                    if assignment is None or status is None:
                        print(f"[supervisor] Operator resumed {robot_name}; no task queued")
                        continue

                    cleaning_overlay.show_dirty_room(assignment["room"])
                    send_assignment_commands(
                        emitter,
                        robot_name,
                        assignment,
                        room_assignments,
                        send_coverage=False,
                        robot_status=status,
                        traffic_reservations=traffic_reservations,
                        step_count=step_count,
                        no_go_zones=operator_controls.no_go_zones,
                    )
                    coverage_plans.update(
                        send_room_work_plans(
                            emitter,
                            room_assignments,
                            assignment["room"],
                            cleaning_overlay,
                            cleanup_plan_signatures,
                            cleanup_plan_steps,
                            operator_controls.no_go_zones,
                            latest_robot_status,
                            step_count,
                        )
                    )
                    robot_motion_monitors.pop(robot_name, None)
                    print(
                        f"[supervisor] Operator resumed {robot_name} "
                        f"toward {assignment['room']}"
                    )

            assigned_idle_rooms = assign_idle_robots_to_unfinished_rooms(
                emitter,
                room_assignments,
                completed_rooms,
                cleaning_overlay,
                latest_robot_status,
                operator_paused_robots,
                coverage_plans,
                cleanup_plan_signatures,
                cleanup_plan_steps,
                last_cleaning_poses,
                robot_recovery_counts,
                traffic_reservations,
                step_count,
                operator_controls.priority_zones,
                operator_controls.no_go_zones,
                operator_paused_assignments,
            )
            for robot_name, room in assigned_idle_rooms.items():
                print(
                    f"[supervisor] Reassigned idle {robot_name} "
                    f"to available {room}"
                )

            cleanup_retry_rooms, cleanup_retry_plans = retry_completed_cleanup_rooms(
                emitter,
                room_assignments,
                completed_rooms,
                cleaning_overlay,
                cleanup_plan_signatures,
                cleanup_plan_steps,
                step_count,
                operator_controls.no_go_zones,
                latest_robot_status,
            )
            coverage_plans.update(cleanup_retry_plans)

            for robot_name in EXPECTED_ROBOTS:
                status = latest_robot_status.get(robot_name)
                assignment = room_assignments.get(robot_name)
                if (
                    robot_name in operator_paused_robots
                    or status is None
                    or assignment is None
                ):
                    continue

                pose = get_actual_robot_pose(supervisor, robot_name)
                if pose is None:
                    pose = status["pose"]
                robot_stuck, stuck_reason = update_robot_motion_monitor(
                    robot_motion_monitors,
                    robot_name,
                    status,
                    pose,
                    assignment,
                    step_count,
                )
                if not robot_stuck:
                    continue

                room = assignment["room"]
                recovery_count = robot_recovery_counts.get(robot_name, 0) + 1
                robot_recovery_counts[robot_name] = recovery_count
                next_room = select_stuck_recovery_room(
                    robot_name,
                    room_assignments,
                    completed_rooms,
                    cleaning_overlay,
                    latest_robot_status,
                    room_progress_monitors,
                    recovery_count,
                    operator_controls.priority_zones,
                    operator_controls.no_go_zones,
                    robot_inside_room=room_for_pose(pose) == room,
                    reserved_assignments=operator_paused_assignments,
                )
                if next_room is None:
                    continue

                cleaning_overlay.release_robot_claims(robot_name)
                cleanup_plan_signatures.pop(robot_name, None)
                cleanup_plan_steps.pop(robot_name, None)
                last_cleaning_poses.pop(robot_name, None)
                completed_robot_rooms.pop(robot_name, None)
                send_recovery_command(emitter, robot_name, stuck_reason)
                helper = assignment.get("helper", False) or next_room != room
                next_assignment = build_assignment(status, next_room, helper=helper)
                room_assignments[robot_name] = next_assignment
                cleaning_overlay.show_dirty_room(next_room)
                send_assignment_commands(
                    emitter,
                    robot_name,
                    next_assignment,
                    room_assignments,
                    send_coverage=False,
                    robot_status=status,
                    traffic_reservations=traffic_reservations,
                    step_count=step_count,
                    no_go_zones=operator_controls.no_go_zones,
                )
                coverage_plans.update(
                    send_room_work_plans(
                        emitter,
                        room_assignments,
                        next_room,
                        cleaning_overlay,
                        cleanup_plan_signatures,
                        cleanup_plan_steps,
                        operator_controls.no_go_zones,
                        latest_robot_status,
                        step_count,
                    )
                )
                if room != next_room:
                    robot_recovery_counts.pop(robot_name, None)
                    coverage_plans.update(
                        send_room_work_plans(
                            emitter,
                            room_assignments,
                            room,
                            cleaning_overlay,
                            cleanup_plan_signatures,
                            cleanup_plan_steps,
                            operator_controls.no_go_zones,
                            latest_robot_status,
                            step_count,
                        )
                    )
                robot_motion_monitors.pop(robot_name, None)
                print(
                    f"[supervisor] {robot_name} appears stuck "
                    f"({stuck_reason}); sent recovery nudge and "
                    f"routing to {next_room} "
                    f"(recovery_attempt={recovery_count})"
                )

            monitored_room_assignments = {
                robot_name: assignment
                for robot_name, assignment in room_assignments.items()
                if robot_name not in operator_paused_robots
            }
            progress_ready_rooms = rooms_ready_for_progress_monitoring(
                supervisor,
                latest_robot_status,
                monitored_room_assignments,
            )
            entry_blocked_reachable_rooms = {
                room
                for room in ROOM_TASKS
                if assigned_robot_inside_room(
                    supervisor,
                    latest_robot_status,
                    monitored_room_assignments,
                    room,
                )
            }
            progress_snapshot = room_progress_snapshot(
                cleaning_overlay,
                operator_controls.no_go_zones,
                entry_blocked_reachable_rooms=entry_blocked_reachable_rooms,
            )
            stalled_rooms = update_room_progress_monitors(
                room_progress_monitors,
                progress_snapshot,
                monitored_room_assignments,
                completed_rooms,
                step_count,
                progress_ready_rooms,
            )
            stalled_room_set = set(stalled_rooms)
            redirected_robots = set()
            for stalled_room in stalled_rooms:
                redirect_room = select_redirect_room_for_stalled_progress(
                    stalled_room,
                    room_assignments,
                    completed_rooms,
                    cleaning_overlay,
                )

                robot_to_redirect = select_robot_for_stalled_room(
                    redirect_room,
                    room_assignments,
                    cleaning_overlay,
                    excluded_robots=redirected_robots | operator_paused_robots,
                    latest_robot_status=latest_robot_status,
                    no_go_zones=operator_controls.no_go_zones,
                    reserved_assignments=operator_paused_assignments,
                )
                if robot_to_redirect is None:
                    print(
                        f"[supervisor] Room {stalled_room} progress stalled at "
                        f"{progress_snapshot.get(stalled_room, 0.0):.1f}%; "
                        "no spare robot is available"
                    )
                    continue

                status = latest_robot_status.get(robot_to_redirect)
                if status is None:
                    continue

                previous_assignment = room_assignments.get(robot_to_redirect)
                previous_room = "idle"
                if previous_assignment is not None:
                    previous_room = previous_assignment["room"]
                cleaning_overlay.release_robot_claims(robot_to_redirect)
                cleanup_plan_signatures.pop(robot_to_redirect, None)
                cleanup_plan_steps.pop(robot_to_redirect, None)
                last_cleaning_poses.pop(robot_to_redirect, None)
                completed_robot_rooms.pop(robot_to_redirect, None)
                robot_recovery_counts.pop(robot_to_redirect, None)
                next_assignment = build_assignment(
                    status,
                    redirect_room,
                    helper=True,
                )
                room_assignments[robot_to_redirect] = next_assignment
                redirected_robots.add(robot_to_redirect)
                cleaning_overlay.show_dirty_room(redirect_room)
                send_assignment_commands(
                    emitter,
                    robot_to_redirect,
                    next_assignment,
                    room_assignments,
                    send_coverage=False,
                    robot_status=status,
                    traffic_reservations=traffic_reservations,
                    step_count=step_count,
                    no_go_zones=operator_controls.no_go_zones,
                )
                coverage_plans.update(
                    send_room_work_plans(
                        emitter,
                        room_assignments,
                        redirect_room,
                        cleaning_overlay,
                        cleanup_plan_signatures,
                        cleanup_plan_steps,
                        operator_controls.no_go_zones,
                        latest_robot_status,
                        step_count,
                    )
                )
                if previous_room != "idle" and previous_room != redirect_room:
                    coverage_plans.update(
                        send_room_work_plans(
                            emitter,
                            room_assignments,
                            previous_room,
                            cleaning_overlay,
                            cleanup_plan_signatures,
                            cleanup_plan_steps,
                            operator_controls.no_go_zones,
                            latest_robot_status,
                            step_count,
                        )
                    )
                robot_motion_monitors.pop(robot_to_redirect, None)
                print(
                    f"[supervisor] Room {stalled_room} progress stalled at "
                    f"{progress_snapshot.get(stalled_room, 0.0):.1f}%; "
                    f"redirected {robot_to_redirect} from {previous_room} "
                    f"to {redirect_room}"
                )

            for robot_name in EXPECTED_ROBOTS:
                status = latest_robot_status.get(robot_name)
                assignment = room_assignments.get(robot_name)
                if (
                    robot_name in operator_paused_robots
                    or status is None
                    or assignment is None
                ):
                    continue

                room = assignment["room"]
                pose = get_actual_robot_pose(supervisor, robot_name)
                if pose is None:
                    pose = status["pose"]
                robot_inside_room = room_for_pose(pose) == room
                coverage = status.get("coverage", {})
                coverage_done_for_room = (
                    coverage.get("room") == room
                    and coverage.get("complete", False)
                )
                room_has_inside_robot = assigned_robot_inside_room(
                    supervisor,
                    latest_robot_status,
                    room_assignments,
                    room,
                )
                cleanup_reason = cleanup_trigger_reason(
                    cleaning_overlay,
                    room,
                    coverage_done_for_room,
                    room in stalled_room_set,
                    operator_controls.no_go_zones,
                    exclude_entry_blocked=not room_has_inside_robot,
                    cleanup_robot_count=len(room_robot_names(room_assignments, room)),
                )
                if cleanup_reason is not None:
                    if room in cleanup_retry_rooms:
                        continue
                    retry_plans = send_room_cleanup_retry_if_all_done(
                        emitter,
                        room_assignments,
                        room,
                        cleaning_overlay,
                        cleanup_plan_signatures,
                        cleanup_plan_steps,
                        step_count,
                        cleanup_reason,
                        operator_controls.no_go_zones,
                        latest_robot_status,
                    )
                    if retry_plans:
                        coverage_plans.update(retry_plans)
                        cleanup_retry_rooms.add(room)
                        print(
                            f"[supervisor] Retrying cleanup for {room}; "
                            "dirty tiles remain after completed plans"
                        )
                        continue
                    if reset_completed_cleanup_plan_if_dirty(
                        robot_name,
                        room,
                        coverage,
                        cleaning_overlay,
                        cleanup_plan_signatures,
                        cleanup_plan_steps,
                    ):
                        print(
                            f"[supervisor] Refreshing cleanup target for "
                            f"{robot_name} in {room}; dirty tiles remain"
                        )
                    cleanup_plan = send_cleanup_plan_if_needed(
                        emitter,
                        cleaning_overlay,
                        robot_name,
                        room,
                        pose,
                        cleanup_plan_signatures,
                        cleanup_plan_steps,
                        step_count,
                        cleanup_reason,
                        operator_controls.no_go_zones,
                        entry_blocked_is_reachable=robot_inside_room,
                    )
                    if cleanup_plan is not None:
                        coverage_plans[robot_name] = cleanup_plan
                    elif not has_active_cleanup_plan(
                        cleanup_plan_signatures,
                        robot_name,
                        room,
                    ):
                        empty_plan = send_empty_cleanup_plan_if_needed(
                            emitter,
                            robot_name,
                            room,
                            cleanup_plan_signatures,
                            cleanup_plan_steps,
                            step_count,
                        )
                        if empty_plan is not None:
                            coverage_plans[robot_name] = empty_plan
                    continue

                if not room_reached_coverage_goal(
                    cleaning_overlay,
                    room,
                    operator_controls.no_go_zones,
                    exclude_entry_blocked=not room_has_inside_robot,
                ):
                    continue

                cleanup_plan_signatures.pop(robot_name, None)
                cleanup_plan_steps.pop(robot_name, None)
                cleaning_overlay.release_robot_claims(robot_name)
                if room not in completed_rooms:
                    completed_rooms.add(room)
                    cleaning_overlay.release_room_claims(room)
                    room_progress = cleaning_overlay.room_progress_percent(
                        room,
                        operator_controls.no_go_zones,
                        not room_has_inside_robot,
                    )
                    print(
                        f"[supervisor] Room {room} reached "
                        f"{room_progress:.1f}% clean"
                    )

                if completed_robot_rooms.get(robot_name) == room:
                    continue

                completed_robot_rooms[robot_name] = room
                next_room = select_reassignment_room(
                    robot_name,
                    room_assignments,
                    completed_rooms,
                    cleaning_overlay,
                    latest_robot_status,
                    room_progress_monitors,
                    operator_controls.priority_zones,
                    operator_controls.no_go_zones,
                    operator_paused_assignments,
                )
                if next_room is None:
                    send_idle_command(emitter, robot_name)
                    room_assignments[robot_name] = None
                    coverage_plans[robot_name] = []
                    cleanup_plan_signatures.pop(robot_name, None)
                    cleanup_plan_steps.pop(robot_name, None)
                    cleaning_overlay.release_robot_claims(robot_name)
                    last_cleaning_poses.pop(robot_name, None)
                    robot_recovery_counts.pop(robot_name, None)
                    print(
                        f"[supervisor] {robot_name} has no unfinished room to help; "
                        "all visible dirty tiles are clean, holding idle"
                    )
                    continue

                next_assignment = build_assignment(status, next_room, helper=True)
                robot_recovery_counts.pop(robot_name, None)
                room_assignments[robot_name] = next_assignment
                cleaning_overlay.show_dirty_room(next_room)
                send_assignment_commands(
                    emitter,
                    robot_name,
                    next_assignment,
                    room_assignments,
                    send_coverage=False,
                    robot_status=status,
                    traffic_reservations=traffic_reservations,
                    step_count=step_count,
                    no_go_zones=operator_controls.no_go_zones,
                )
                coverage_plans.update(
                    send_room_work_plans(
                        emitter,
                        room_assignments,
                        next_room,
                        cleaning_overlay,
                        cleanup_plan_signatures,
                        cleanup_plan_steps,
                        operator_controls.no_go_zones,
                        latest_robot_status,
                        step_count,
                    )
                )
                last_cleaning_poses.pop(robot_name, None)
                next_room_progress = cleaning_overlay.room_progress_percent(
                    next_room,
                    operator_controls.no_go_zones,
                )
                print(
                    f"[supervisor] Reassigned {robot_name} to help {next_room}; "
                    f"room_clean={next_room_progress:.1f}% "
                    f"coverage_waypoints={len(coverage_plans[robot_name])}"
                )

        if step_count % OPERATOR_STATE_WRITE_STEPS == 0:
            try:
                operator_state = build_operator_state(
                    supervisor,
                    step_count,
                    latest_robot_status,
                    room_assignments,
                    cleaning_overlay,
                    operator_controls,
                    operator_paused_robots,
                    completed_rooms,
                    traffic_reservations,
                    evaluation_metrics,
                    webots_time_s,
                )
                write_operator_state_file(
                    operator_state_path,
                    operator_state,
                )
                write_evaluation_metrics_file(
                    evaluation_metrics_path,
                    operator_state["metrics"],
                )
            except OSError as exc:
                if (
                    step_count - last_operator_state_error_step
                    >= COMMUNICATION_SUMMARY_INTERVAL_STEPS
                ):
                    last_operator_state_error_step = step_count
                    print(f"[supervisor] Could not write dashboard metrics: {exc}")

        if step_count % COMMUNICATION_SUMMARY_INTERVAL_STEPS == 0:
            connected_count = len(latest_robot_status)
            progress_snapshot = room_progress_snapshot(
                cleaning_overlay,
                operator_controls.no_go_zones,
            )
            path_summary = traffic_reservations.summary()
            metrics_snapshot = evaluation_metrics.snapshot(
                step_count,
                webots_time_s,
            )
            clean_time_s = metrics_snapshot["elapsed_cleaning_time_s"]
            clean_time_text = "--" if clean_time_s is None else f"{clean_time_s:.1f}s"
            reduction_percent = metrics_snapshot["time_reduction_percent"]
            reduction_text = (
                "--"
                if reduction_percent is None
                else f"{reduction_percent:.1f}%"
            )
            print(
                f"[supervisor] Robot status: "
                f"{connected_count}/{len(EXPECTED_ROBOTS)} reporting, "
                f"global_map free={global_grid.free_cell_count} "
                f"walls={global_grid.wall_cell_count} "
                f"path_distance={path_summary['planned_distance_m']:.2f}m "
                f"path_wait={path_summary['waiting_steps']} step(s) "
                f"doorway_conflicts={path_summary['doorway_conflicts']} "
                f"coverage={metrics_snapshot['coverage_percent']:.1f}% "
                f"target95={metrics_snapshot['coverage_target_met']} "
                f"clean_time={clean_time_text} "
                f"collisions={metrics_snapshot['inter_robot_collisions']} "
                f"baseline_reduction={reduction_text} "
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
                plan_kind = coverage.get("plan_kind", "sweep")
                lane_index, lane_count = normalized_lane(
                    coverage.get("lane_index", 0),
                    coverage.get("lane_count", 1),
                )
                room_progress = 0.0
                if assignment is not None:
                    room_progress = cleaning_overlay.room_progress_percent(
                        assignment["room"],
                        operator_controls.no_go_zones,
                    )
                route = status.get("assignment_route", {})
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
                    f"phase={status.get('phase', 'unknown')} "
                    f"room={room} "
                    f"route_wait={route.get('waiting_steps_remaining', 0)} "
                    f"coverage={coverage.get('waypoint_index', 0)}/"
                    f"{coverage.get('waypoint_count', 0)} "
                    f"{plan_kind} lane={lane_index + 1}/{lane_count} "
                    f"clean={room_progress:.1f}% "
                    f"free={status.get('free_cell_count', 0)} "
                    f"walls={status['wall_cell_count']}"
                )


if __name__ == "__main__":
    run()
