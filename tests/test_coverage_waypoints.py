import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = (
    REPO_ROOT / "controllers" / "roomba_supervisor" / "roomba_supervisor.py"
)
CONTROLLER_PATH = (
    REPO_ROOT / "controllers" / "roomba_controller" / "roomba_controller.py"
)


def load_supervisor_module():
    sys.modules["controller"] = types.SimpleNamespace(Supervisor=object)
    spec = importlib.util.spec_from_file_location("roomba_supervisor", SUPERVISOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_controller_module():
    sys.modules["controller"] = types.SimpleNamespace(Robot=object)
    spec = importlib.util.spec_from_file_location("roomba_controller", CONTROLLER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoverageWaypointTests(unittest.TestCase):
    def test_coverage_waypoints_include_final_sweep_row(self):
        supervisor = load_supervisor_module()

        for room, config in supervisor.ROOM_TASKS.items():
            with self.subTest(room=room):
                _, _, _, max_y = config["bounds"]
                sweep_max_y = round(max_y - supervisor.COVERAGE_MARGIN_M, 3)
                waypoints = supervisor.generate_coverage_waypoints(room)
                row_positions = sorted({waypoint[1] for waypoint in waypoints})

                self.assertIn(sweep_max_y, row_positions)
                self.assertEqual(sweep_max_y, row_positions[-1])

    def test_coverage_waypoints_keep_robot_center_away_from_walls(self):
        supervisor = load_supervisor_module()

        for room, config in supervisor.ROOM_TASKS.items():
            with self.subTest(room=room):
                min_x, max_x, min_y, max_y = config["bounds"]
                for x_m, y_m in supervisor.generate_coverage_waypoints(room):
                    self.assertGreaterEqual(
                        x_m - min_x + 1e-9,
                        supervisor.COVERAGE_MARGIN_M,
                    )
                    self.assertGreaterEqual(
                        max_x - x_m + 1e-9,
                        supervisor.COVERAGE_MARGIN_M,
                    )
                    self.assertGreaterEqual(
                        y_m - min_y + 1e-9,
                        supervisor.COVERAGE_MARGIN_M,
                    )
                    self.assertGreaterEqual(
                        max_y - y_m + 1e-9,
                        supervisor.COVERAGE_MARGIN_M,
                    )

    def test_coverage_waypoints_can_split_room_into_lanes(self):
        supervisor = load_supervisor_module()

        full_plan = supervisor.generate_coverage_waypoints("ne_large")
        left_lane = supervisor.generate_coverage_waypoints(
            "ne_large",
            lane_index=0,
            lane_count=2,
        )
        right_lane = supervisor.generate_coverage_waypoints(
            "ne_large",
            lane_index=1,
            lane_count=2,
        )

        full_min_x = min(waypoint[0] for waypoint in full_plan)
        full_max_x = max(waypoint[0] for waypoint in full_plan)
        left_max_x = max(waypoint[0] for waypoint in left_lane)
        right_min_x = min(waypoint[0] for waypoint in right_lane)

        self.assertEqual(full_min_x, min(waypoint[0] for waypoint in left_lane))
        self.assertEqual(full_max_x, max(waypoint[0] for waypoint in right_lane))
        self.assertLessEqual(left_max_x, right_min_x)

    def test_south_room_sweep_starts_near_hub_doorway(self):
        supervisor = load_supervisor_module()

        first_waypoint = supervisor.generate_coverage_waypoints("s_medium")[0]
        _, _, _, max_y = supervisor.ROOM_TASKS["s_medium"]["bounds"]

        self.assertEqual(
            round(max_y - supervisor.COVERAGE_MARGIN_M, 3),
            first_waypoint[1],
        )

    def test_split_lanes_start_on_different_sides(self):
        supervisor = load_supervisor_module()

        left_lane = supervisor.generate_coverage_waypoints(
            "ne_large",
            lane_index=0,
            lane_count=2,
        )
        right_lane = supervisor.generate_coverage_waypoints(
            "ne_large",
            lane_index=1,
            lane_count=2,
        )

        self.assertLess(left_lane[0][0], right_lane[0][0])
        self.assertNotEqual(left_lane[0], right_lane[0])

    def test_coverage_lane_for_robot_uses_stable_room_order(self):
        supervisor = load_supervisor_module()
        room_assignments = {
            "epuck_4": {"room": "ne_large"},
            "epuck_2": {"room": "n_medium"},
            "epuck_1": {"room": "ne_large"},
            "epuck_3": {"room": "ne_large"},
        }

        self.assertEqual(
            (0, 3),
            supervisor.coverage_lane_for_robot("epuck_1", room_assignments),
        )
        self.assertEqual(
            (1, 3),
            supervisor.coverage_lane_for_robot("epuck_3", room_assignments),
        )
        self.assertEqual(
            (2, 3),
            supervisor.coverage_lane_for_robot("epuck_4", room_assignments),
        )

    def test_coverage_reach_distance_cleans_endpoint_tiles(self):
        controller = load_controller_module()
        supervisor = load_supervisor_module()

        endpoint_to_edge_tile_center_m = (
            supervisor.COVERAGE_MARGIN_M - 0.5 * supervisor.CLEAN_TILE_SIZE_M
        )
        max_reach_distance_m = (
            supervisor.CLEAN_RADIUS_M - endpoint_to_edge_tile_center_m
        )

        self.assertLessEqual(
            controller.COVERAGE_WAYPOINT_REACHED_M,
            max_reach_distance_m,
        )

    def test_coverage_reach_distance_cleans_corner_tiles(self):
        controller = load_controller_module()
        supervisor = load_supervisor_module()

        corner_offset_m = (
            supervisor.COVERAGE_MARGIN_M - 0.5 * supervisor.CLEAN_TILE_SIZE_M
        )
        distance_when_reached_m = (
            (corner_offset_m + controller.COVERAGE_WAYPOINT_REACHED_M) ** 2
            + corner_offset_m ** 2
        ) ** 0.5

        self.assertLessEqual(distance_when_reached_m, supervisor.CLEAN_RADIUS_M)

    def test_odometry_uses_midpoint_heading_for_turning_motion(self):
        controller = load_controller_module()

        left_speed = 1.0
        right_speed = 2.0
        dt_s = 1.0
        x_m, y_m, theta_rad, distance_delta_m, turn_delta_rad = (
            controller.integrate_odometry_pose(
                0.0,
                0.0,
                0.0,
                left_speed,
                right_speed,
                dt_s,
            )
        )

        left_linear_mps = left_speed * controller.WHEEL_RADIUS_M
        right_linear_mps = right_speed * controller.WHEEL_RADIUS_M
        forward_mps = 0.5 * (left_linear_mps + right_linear_mps)
        heading_delta_rad = (
            (right_linear_mps - left_linear_mps)
            / controller.AXLE_LENGTH_M
            * dt_s
        )

        self.assertAlmostEqual(
            forward_mps * controller.math.cos(0.5 * heading_delta_rad),
            x_m,
        )
        self.assertAlmostEqual(
            forward_mps * controller.math.sin(0.5 * heading_delta_rad),
            y_m,
        )
        self.assertAlmostEqual(heading_delta_rad, theta_rad)
        self.assertAlmostEqual(forward_mps, distance_delta_m)
        self.assertAlmostEqual(abs(heading_delta_rad), turn_delta_rad)

    def test_lidar_geometry_uses_forward_offset_and_beam_fan(self):
        controller = load_controller_module()

        origin_x_m, origin_y_m = controller.lidar_sensor_origin(1.0, 2.0, 0.0)
        rays = controller.lidar_free_ray_endpoints(1.0, 2.0, 0.0, 0.5)
        hit_x_m, hit_y_m = controller.lidar_wall_hit_point(1.0, 2.0, 0.0, 0.5)

        self.assertAlmostEqual(1.0 + controller.LIDAR_FORWARD_OFFSET_M, origin_x_m)
        self.assertAlmostEqual(2.0, origin_y_m)
        self.assertEqual(3, len(rays))
        self.assertTrue(all(ray[0] == origin_x_m for ray in rays))
        self.assertTrue(all(ray[1] == origin_y_m for ray in rays))
        self.assertAlmostEqual(origin_x_m + 0.5, hit_x_m)
        self.assertAlmostEqual(origin_y_m, hit_y_m)
        self.assertLess(rays[0][3], origin_y_m)
        self.assertGreater(rays[2][3], origin_y_m)

    def test_pose_confidence_decays_with_odometry_distance_and_turning(self):
        controller = load_controller_module()

        confident = controller.pose_confidence_from_odometry(0.0, 0.0)
        stale = controller.pose_confidence_from_odometry(
            controller.POSE_CONFIDENCE_DISTANCE_DECAY_M,
            controller.POSE_CONFIDENCE_TURN_DECAY_RAD,
        )

        self.assertEqual(1.0, confident)
        self.assertEqual(controller.MIN_POSE_CONFIDENCE, stale)

    def test_robot_holds_after_launch_until_assignment_arrives(self):
        controller = load_controller_module()

        self.assertTrue(
            controller.should_hold_for_assignment(
                launch_finished=True,
                launch_timed_out=False,
                assigned_target=None,
            )
        )
        self.assertTrue(
            controller.should_hold_for_assignment(
                launch_finished=False,
                launch_timed_out=True,
                assigned_target=None,
            )
        )
        self.assertFalse(
            controller.should_hold_for_assignment(
                launch_finished=True,
                launch_timed_out=False,
                assigned_target=(1.0, 1.0),
            )
        )

    def test_robot_holds_after_task_completion(self):
        controller = load_controller_module()

        self.assertTrue(
            controller.should_hold_after_task(
                assigned_target=(1.0, 1.0),
                assignment_target_reached=True,
                coverage_complete=True,
            )
        )

    def test_robot_phase_names_explain_current_behavior(self):
        controller = load_controller_module()

        self.assertEqual(
            "launching",
            controller.robot_phase(False, False, None, False, False, "sweep"),
        )
        self.assertEqual(
            "waiting_for_assignment",
            controller.robot_phase(True, False, None, False, False, "sweep"),
        )
        self.assertEqual(
            "driving_to_room",
            controller.robot_phase(True, False, (1.0, 1.0), False, False, "sweep"),
        )
        self.assertEqual(
            "waiting_for_route",
            controller.robot_phase(
                True,
                False,
                (1.0, 1.0),
                False,
                False,
                "sweep",
                assignment_wait_steps_remaining=4,
            ),
        )
        self.assertEqual(
            "sweeping",
            controller.robot_phase(True, False, (1.0, 1.0), True, False, "sweep"),
        )
        self.assertEqual(
            "cleanup",
            controller.robot_phase(True, False, (1.0, 1.0), True, False, "cleanup"),
        )
        self.assertEqual(
            "holding_after_task",
            controller.robot_phase(True, False, (1.0, 1.0), True, True, "cleanup"),
        )
        self.assertFalse(
            controller.should_hold_after_task(
                assigned_target=(1.0, 1.0),
                assignment_target_reached=False,
                coverage_complete=True,
            )
        )

    def test_launch_waypoints_stage_robots_in_central_hub(self):
        controller = load_controller_module()

        for robot_name, waypoints in controller.ROBOT_LAUNCH_WAYPOINTS.items():
            with self.subTest(robot=robot_name):
                self.assertEqual(1, len(waypoints))
                x_m, y_m = waypoints[0]
                self.assertLessEqual(abs(x_m), 0.5)
                self.assertLessEqual(abs(y_m), 0.5)

    def test_actual_robot_pose_uses_webots_translation(self):
        supervisor_module = load_supervisor_module()

        class FakeTranslationField:
            def getSFVec3f(self):
                return [1.25, -2.5, 0.0]

        class FakeRotationField:
            def getSFRotation(self):
                return [0.0, 0.0, 1.0, 1.5]

        class FakeRobotNode:
            def getField(self, field_name):
                if field_name == "translation":
                    return FakeTranslationField()
                if field_name == "rotation":
                    return FakeRotationField()
                return None

        class FakeSupervisor:
            def getFromDef(self, def_name):
                if def_name == "EPUCK_3":
                    return FakeRobotNode()
                return None

        pose = supervisor_module.get_actual_robot_pose(FakeSupervisor(), "epuck_3")

        self.assertEqual({"x_m": 1.25, "y_m": -2.5, "theta_rad": 1.5}, pose)
        self.assertIsNone(
            supervisor_module.get_actual_robot_pose(FakeSupervisor(), "missing")
        )

    def test_cleaning_path_interpolation_covers_status_update_gaps(self):
        supervisor = load_supervisor_module()
        start_pose = {"x_m": 0.0, "y_m": 0.0}
        end_pose = {"x_m": 0.24, "y_m": 0.0}

        points = supervisor.interpolate_cleaning_path(start_pose, end_pose)
        largest_gap_m = max(
            (
                ((points[index][0] - points[index - 1][0]) ** 2
                 + (points[index][1] - points[index - 1][1]) ** 2) ** 0.5
                for index in range(1, len(points))
            ),
            default=0.0,
        )

        self.assertEqual((0.0, 0.0), points[0])
        self.assertEqual((0.24, 0.0), points[-1])
        self.assertLessEqual(
            largest_gap_m,
            supervisor.CLEAN_TRAIL_SAMPLE_SPACING_M + 1e-9,
        )

    def test_cleaning_marks_only_after_robot_enters_assigned_room(self):
        supervisor = load_supervisor_module()
        room = "nw_small"
        min_x, max_x, min_y, max_y = supervisor.ROOM_TASKS[room]["bounds"]
        inside_pose = {
            "x_m": 0.5 * (min_x + max_x),
            "y_m": 0.5 * (min_y + max_y),
        }
        outside_pose = {"x_m": 0.0, "y_m": 0.0}

        self.assertFalse(supervisor.robot_can_mark_cleaning(outside_pose, room))
        self.assertTrue(supervisor.robot_can_mark_cleaning(inside_pose, room))
        self.assertFalse(supervisor.robot_can_mark_cleaning(None, room))
        self.assertFalse(
            supervisor.robot_can_mark_cleaning(inside_pose, "not_a_room")
        )

    def test_room_progress_percent_uses_dirty_tiles(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {"nw_small": 4}
        overlay.dirty_tiles = {
            ("nw_small", 0, 0),
            ("nw_small", 1, 0),
        }

        self.assertEqual(2, overlay.cleaned_tile_count("nw_small"))
        self.assertEqual(50.0, overlay.room_progress_percent("nw_small"))

    def test_room_reached_coverage_goal_requires_no_dirty_tiles(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {"nw_small": 20}
        overlay.dirty_tiles = {("nw_small", 0, 0)}

        self.assertFalse(
            supervisor.room_reached_coverage_goal(overlay, "nw_small")
        )

        overlay.dirty_tiles = set()
        self.assertTrue(
            supervisor.room_reached_coverage_goal(overlay, "nw_small")
        )

    def test_dirty_tile_centers_targets_remaining_tiles(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.rooms = supervisor.ROOM_TASKS
        overlay.dirty_tiles = {
            ("nw_small", 0, 0),
            ("nw_small", 1, 0),
            ("n_medium", 0, 0),
        }

        centers = overlay.dirty_tile_centers("nw_small")

        self.assertEqual([[-2.875, 0.875], [-2.625, 0.875]], centers)

    def test_dirty_tile_centers_align_to_floor_checkerboard(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.rooms = supervisor.ROOM_TASKS
        overlay.dirty_tiles = {
            ("ne_large", 0, 0),
            ("s_medium", 0, 0),
        }

        self.assertEqual([[0.875, 0.875]], overlay.dirty_tile_centers("ne_large"))
        self.assertEqual([[-0.625, -2.875]], overlay.dirty_tile_centers("s_medium"))

    def test_cleanup_claims_give_robots_different_dirty_tiles(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.rooms = supervisor.ROOM_TASKS
        overlay.tile_claims = {}
        overlay.dirty_tiles = {
            ("nw_small", 0, 0),
            ("nw_small", 1, 0),
            ("nw_small", 2, 0),
            ("nw_small", 3, 0),
        }

        first_plan = overlay.claim_dirty_tile_centers(
            "nw_small",
            "epuck_1",
            {"x_m": -2.9, "y_m": 0.9},
            max_tiles=2,
        )
        second_plan = overlay.claim_dirty_tile_centers(
            "nw_small",
            "epuck_2",
            {"x_m": -2.1, "y_m": 0.9},
            max_tiles=2,
        )

        self.assertEqual(2, len(first_plan))
        self.assertEqual(2, len(second_plan))
        self.assertEqual(
            set(),
            set(map(tuple, first_plan)) & set(map(tuple, second_plan)),
        )
        self.assertEqual(
            {
                ("nw_small", 0, 0): "epuck_1",
                ("nw_small", 1, 0): "epuck_1",
                ("nw_small", 2, 0): "epuck_2",
                ("nw_small", 3, 0): "epuck_2",
            },
            overlay.tile_claims,
        )

    def test_cleanup_claims_prefer_nearby_dirty_tiles(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.rooms = supervisor.ROOM_TASKS
        overlay.tile_claims = {}
        overlay.dirty_tiles = {
            ("nw_small", 0, 0),
            ("nw_small", 1, 0),
            ("nw_small", 2, 0),
        }

        plan = overlay.claim_dirty_tile_centers(
            "nw_small",
            "epuck_1",
            {"x_m": -2.62, "y_m": 0.88},
            max_tiles=1,
        )

        self.assertEqual([[-2.625, 0.875]], plan)
        self.assertEqual({("nw_small", 1, 0): "epuck_1"}, overlay.tile_claims)

    def test_cleanup_claims_can_prioritize_wall_and_corner_tiles(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.rooms = supervisor.ROOM_TASKS
        overlay.tile_claims = {}
        overlay.dirty_tiles = {
            ("nw_small", 0, 0),
            ("nw_small", 3, 3),
        }

        plan = overlay.claim_dirty_tile_centers(
            "nw_small",
            "epuck_1",
            {"x_m": -2.12, "y_m": 1.62},
            max_tiles=1,
            prefer_edges=True,
        )

        self.assertEqual([[-2.875, 0.875]], plan)
        self.assertEqual({("nw_small", 0, 0): "epuck_1"}, overlay.tile_claims)

    def test_cleanup_trigger_reason_starts_final_cleanup_before_sweep_end(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {"nw_small": 100}
        overlay.dirty_tiles = {
            ("nw_small", 0, 0),
            ("nw_small", 1, 0),
            ("nw_small", 2, 0),
            ("nw_small", 3, 0),
        }

        reason = supervisor.cleanup_trigger_reason(
            overlay,
            "nw_small",
            coverage_done_for_room=False,
            room_stalled=False,
        )

        self.assertEqual("final 4 dirty tile(s) remain", reason)

    def test_cleanup_trigger_reason_starts_stalled_late_cleanup(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {"nw_small": 100}
        overlay.dirty_tiles = {
            ("nw_small", col_index, 0)
            for col_index in range(7)
        }

        reason = supervisor.cleanup_trigger_reason(
            overlay,
            "nw_small",
            coverage_done_for_room=False,
            room_stalled=True,
        )

        self.assertEqual("room stalled at 93.0% clean", reason)

    def test_cleanup_pass_resends_stale_identical_plan(self):
        supervisor = load_supervisor_module()

        class FakeEmitter:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        emitter = FakeEmitter()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.rooms = supervisor.ROOM_TASKS
        overlay.tile_claims = {("nw_small", 0, 0): "epuck_1"}
        overlay.dirty_tiles = {("nw_small", 0, 0)}
        cleanup_plan_signatures = {
            "epuck_1": ("nw_small", ((-2.875, 0.875),))
        }
        cleanup_plan_steps = {"epuck_1": 10}

        plan = supervisor.send_cleanup_plan_if_needed(
            emitter,
            overlay,
            "epuck_1",
            "nw_small",
            {"x_m": -2.8, "y_m": 0.9},
            cleanup_plan_signatures,
            cleanup_plan_steps,
            10 + supervisor.CLEANUP_RESEND_STEPS,
            "room stalled at 99.0% clean",
        )

        self.assertEqual([[-2.875, 0.875]], plan)
        self.assertEqual(1, len(emitter.messages))
        command = json.loads(emitter.messages[0])
        self.assertEqual("cleanup", command["plan_kind"])

    def test_cleaning_a_tile_releases_its_claim(self):
        supervisor = load_supervisor_module()

        class FakeField:
            def setSFColor(self, value):
                self.value = value

            def setSFFloat(self, value):
                self.value = value

        class FakeAppearance:
            def getField(self, field_name):
                return FakeField()

        tile_key = ("nw_small", 0, 0)
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.enabled = True
        overlay.active_rooms = {"nw_small"}
        overlay.rooms = supervisor.ROOM_TASKS
        overlay.dirty_tiles = {tile_key}
        overlay.tile_claims = {tile_key: "epuck_1"}
        overlay.tile_appearances = {tile_key: FakeAppearance()}

        cleaned_count = overlay.mark_clean_near("nw_small", -2.875, 0.875)

        self.assertEqual(1, cleaned_count)
        self.assertEqual(set(), overlay.dirty_tiles)
        self.assertEqual({}, overlay.tile_claims)

    def test_releasing_robot_claims_keeps_other_robot_claims(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.tile_claims = {
            ("nw_small", 0, 0): "epuck_1",
            ("nw_small", 1, 0): "epuck_2",
            ("n_medium", 0, 0): "epuck_1",
        }

        overlay.release_robot_claims("epuck_1")

        self.assertEqual({("nw_small", 1, 0): "epuck_2"}, overlay.tile_claims)

    def test_room_progress_snapshot_reports_every_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 10 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("n_medium", 0, 0),
            ("n_medium", 1, 0),
            ("ne_large", 0, 0),
            ("ne_large", 1, 0),
            ("ne_large", 2, 0),
        }

        snapshot = supervisor.room_progress_snapshot(overlay)

        expected = {room: 100.0 for room in supervisor.ROOM_TASKS}
        expected["n_medium"] = 80.0
        expected["ne_large"] = 70.0
        self.assertEqual(expected, snapshot)

        formatted = supervisor.format_room_progress(snapshot)
        self.assertIn("n_medium=80.0%", formatted)
        self.assertIn("ne_large=70.0%", formatted)
        self.assertIn("nw_small=100.0%", formatted)

    def test_robot_motion_monitor_flags_robot_that_should_move_but_does_not(self):
        supervisor = load_supervisor_module()
        monitors = {}
        assignment = {"room": "n_medium"}
        status = {
            "launch": {"complete": True, "timed_out": False},
            "assignment_target_reached": False,
            "coverage": {"room": "n_medium", "complete": False},
            "motion": {"front_blocked": True},
        }
        pose = {"x_m": 0.0, "y_m": 0.0}

        stuck, _ = supervisor.update_robot_motion_monitor(
            monitors,
            "epuck_1",
            status,
            pose,
            assignment,
            0,
        )
        self.assertFalse(stuck)

        stuck, reason = supervisor.update_robot_motion_monitor(
            monitors,
            "epuck_1",
            status,
            pose,
            assignment,
            supervisor.ROBOT_STUCK_WINDOW_STEPS,
        )

        self.assertTrue(stuck)
        self.assertEqual("blocked by nearby obstacle", reason)

    def test_route_start_delay_does_not_count_as_stuck_motion(self):
        supervisor = load_supervisor_module()
        status = {
            "launch": {"complete": True, "timed_out": False},
            "assignment_target_reached": False,
            "assignment_route": {"waiting_steps_remaining": 12},
            "coverage": {"room": "n_medium", "complete": False},
        }

        self.assertFalse(
            supervisor.robot_should_be_moving(status, {"room": "n_medium"})
        )

    def test_robot_motion_monitor_resets_when_robot_moves_enough(self):
        supervisor = load_supervisor_module()
        monitors = {}
        assignment = {"room": "n_medium"}
        status = {
            "launch": {"complete": True, "timed_out": False},
            "assignment_target_reached": True,
            "coverage": {"room": "n_medium", "complete": False},
            "motion": {},
        }

        supervisor.update_robot_motion_monitor(
            monitors,
            "epuck_1",
            status,
            {"x_m": 0.0, "y_m": 0.0},
            assignment,
            0,
        )
        stuck, _ = supervisor.update_robot_motion_monitor(
            monitors,
            "epuck_1",
            status,
            {"x_m": supervisor.ROBOT_STUCK_MIN_MOVE_M, "y_m": 0.0},
            assignment,
            supervisor.ROBOT_STUCK_WINDOW_STEPS,
        )

        self.assertFalse(stuck)
        self.assertEqual(
            supervisor.ROBOT_STUCK_WINDOW_STEPS,
            monitors["epuck_1"]["anchor_step"],
        )

    def test_room_progress_monitor_flags_active_room_without_cleaning_gain(self):
        supervisor = load_supervisor_module()
        monitors = {}
        room_assignments = {"epuck_1": {"room": "n_medium"}}
        progress = {room: 100.0 for room in supervisor.ROOM_TASKS}
        progress["n_medium"] = 40.0

        stalled_rooms = supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            0,
        )
        self.assertEqual([], stalled_rooms)

        stalled_rooms = supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            supervisor.ROOM_PROGRESS_STALL_STEPS,
        )

        self.assertEqual(["n_medium"], stalled_rooms)

    def test_room_progress_monitor_waits_until_robot_can_clean_room(self):
        supervisor = load_supervisor_module()
        monitors = {}
        room_assignments = {"epuck_1": {"room": "n_medium"}}
        progress = {room: 100.0 for room in supervisor.ROOM_TASKS}
        progress["n_medium"] = 40.0

        supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            0,
            progress_ready_rooms=set(),
        )
        stalled_rooms = supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            supervisor.ROOM_PROGRESS_STALL_STEPS,
            progress_ready_rooms=set(),
        )

        self.assertEqual([], stalled_rooms)

        stalled_rooms = supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            2 * supervisor.ROOM_PROGRESS_STALL_STEPS,
            progress_ready_rooms={"n_medium"},
        )
        self.assertEqual([], stalled_rooms)

        stalled_rooms = supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            3 * supervisor.ROOM_PROGRESS_STALL_STEPS,
            progress_ready_rooms={"n_medium"},
        )
        self.assertEqual(["n_medium"], stalled_rooms)

    def test_room_progress_monitor_resets_after_cleaning_gain(self):
        supervisor = load_supervisor_module()
        monitors = {}
        room_assignments = {"epuck_1": {"room": "n_medium"}}
        progress = {room: 100.0 for room in supervisor.ROOM_TASKS}
        progress["n_medium"] = 40.0
        supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            0,
        )

        progress["n_medium"] = 40.2
        stalled_rooms = supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            supervisor.ROOM_PROGRESS_STALL_STEPS,
        )

        self.assertEqual([], stalled_rooms)
        self.assertEqual(
            supervisor.ROOM_PROGRESS_STALL_STEPS,
            monitors["n_medium"]["step"],
        )

    def test_room_progress_monitor_tracks_recent_cleaning_rate(self):
        supervisor = load_supervisor_module()
        monitors = {}
        room_assignments = {"epuck_1": {"room": "n_medium"}}
        progress = {room: 100.0 for room in supervisor.ROOM_TASKS}
        progress["n_medium"] = 40.0

        supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            0,
        )

        progress["n_medium"] = 41.0
        supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            100,
        )
        self.assertAlmostEqual(
            10.0,
            supervisor.room_progress_rate(monitors, "n_medium"),
        )

        supervisor.update_room_progress_monitors(
            monitors,
            progress,
            room_assignments,
            set(),
            100 + supervisor.ROOM_PROGRESS_RATE_WINDOW_STEPS,
        )
        self.assertEqual(0.0, supervisor.room_progress_rate(monitors, "n_medium"))

    def test_stalled_room_redirect_prefers_idle_robot(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 10 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {("n_medium", 0, 0)}
        room_assignments = {
            "epuck_1": {"room": "n_medium"},
            "epuck_2": None,
            "epuck_3": {"room": "ne_large"},
            "epuck_4": {"room": "se_small"},
        }

        robot = supervisor.select_robot_for_stalled_room(
            "n_medium",
            room_assignments,
            overlay,
        )

        self.assertEqual("epuck_2", robot)

    def test_stalled_room_redirect_prefers_closest_idle_robot(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 10 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {("n_medium", 0, 0)}
        room_assignments = {
            "epuck_1": {"room": "n_medium"},
            "epuck_2": None,
            "epuck_3": None,
            "epuck_4": {"room": "se_small"},
        }
        latest_robot_status = {
            "epuck_2": {"pose": {"x_m": 2.6, "y_m": -2.6}},
            "epuck_3": {"pose": {"x_m": -0.2, "y_m": 0.2}},
        }

        robot = supervisor.select_robot_for_stalled_room(
            "n_medium",
            room_assignments,
            overlay,
            latest_robot_status=latest_robot_status,
        )

        self.assertEqual("epuck_3", robot)

    def test_stalled_room_redirect_can_reuse_helper_from_supported_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("n_medium", 0, 0),
            ("ne_large", 0, 0),
        }
        room_assignments = {
            "epuck_1": {"room": "n_medium"},
            "epuck_2": {"room": "ne_large", "helper": True},
            "epuck_3": {"room": "ne_large"},
            "epuck_4": {"room": "ne_large", "helper": True},
        }

        robot = supervisor.select_robot_for_stalled_room(
            "n_medium",
            room_assignments,
            overlay,
        )

        self.assertEqual("epuck_2", robot)

    def test_stalled_room_redirect_keeps_helper_in_supported_dirty_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("n_medium", 0, 0),
        } | {
            ("ne_large", col_index, 0)
            for col_index in range(8)
        } | {
            ("se_small", col_index, 0)
            for col_index in range(8)
        }
        room_assignments = {
            "epuck_1": {"room": "n_medium"},
            "epuck_2": {"room": "ne_large", "helper": True},
            "epuck_3": {"room": "ne_large"},
            "epuck_4": {"room": "se_small"},
        }

        robot = supervisor.select_robot_for_stalled_room(
            "n_medium",
            room_assignments,
            overlay,
        )

        self.assertIsNone(robot)

    def test_stalled_room_redirect_skips_robot_already_redirected_this_pass(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 10 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("n_medium", 0, 0),
            ("ne_large", 0, 0),
            ("sw_large", 0, 0),
        }
        room_assignments = {
            "epuck_1": {"room": "n_medium"},
            "epuck_2": {"room": "ne_large", "helper": True},
            "epuck_3": {"room": "ne_large"},
            "epuck_4": {"room": "ne_large", "helper": True},
        }

        first_robot = supervisor.select_robot_for_stalled_room(
            "n_medium",
            room_assignments,
            overlay,
        )
        self.assertEqual("epuck_2", first_robot)

        room_assignments[first_robot] = {"room": "n_medium", "helper": True}
        second_robot = supervisor.select_robot_for_stalled_room(
            "sw_large",
            room_assignments,
            overlay,
            excluded_robots={first_robot},
        )

        self.assertIsNone(second_robot)

    def test_stalled_small_room_redirect_stays_on_stalled_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("se_small", col_index, 0)
            for col_index in range(12)
        } | {
            ("ne_large", col_index, 0)
            for col_index in range(8)
        }
        room_assignments = {
            "epuck_1": {"room": "se_small"},
            "epuck_2": {"room": "se_small", "helper": True},
            "epuck_3": {"room": "ne_large"},
            "epuck_4": {"room": "s_medium"},
        }

        redirect_room = supervisor.select_redirect_room_for_stalled_progress(
            "se_small",
            room_assignments,
            {"nw_small", "n_medium", "sw_large", "s_medium"},
            overlay,
        )

        self.assertEqual("se_small", redirect_room)

    def test_reassignment_selects_least_supported_unfinished_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("n_medium", col_index, 0)
            for col_index in range(12)
        } | {
            ("ne_large", col_index, 0)
            for col_index in range(5)
        } | {
            ("se_small", col_index, 0)
            for col_index in range(8)
        }
        room_assignments = {
            "epuck_1": {"room": "nw_small"},
            "epuck_2": {"room": "n_medium"},
            "epuck_3": {"room": "ne_large"},
            "epuck_4": {"room": "se_small"},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            {"nw_small"},
            overlay,
        )

        self.assertEqual("n_medium", next_room)

    def test_reassignment_keeps_unstaffed_rooms_ahead_of_nearby_supported_rooms(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 100 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("s_medium", col_index, 0)
            for col_index in range(70)
        } | {
            ("ne_large", col_index, 0)
            for col_index in range(20)
        }
        room_assignments = {
            "epuck_1": {"room": "nw_small"},
            "epuck_2": {"room": "s_medium"},
            "epuck_3": {"room": "se_small"},
        }
        latest_robot_status = {
            "epuck_1": {"pose": {"x_m": 0.3, "y_m": -0.7}},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            {"nw_small", "n_medium", "sw_large", "se_small"},
            overlay,
            latest_robot_status=latest_robot_status,
        )

        self.assertEqual("ne_large", next_room)

    def test_reassignment_balances_helpers_before_progress(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("n_medium", col_index, 0)
            for col_index in range(12)
        } | {
            ("ne_large", col_index, 0)
            for col_index in range(6)
        }
        room_assignments = {
            "epuck_1": {"room": "nw_small"},
            "epuck_2": {"room": "n_medium"},
            "epuck_3": {"room": "n_medium", "helper": True},
            "epuck_4": {"room": "se_small"},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            {"nw_small"},
            overlay,
        )

        self.assertEqual("ne_large", next_room)

    def test_reassignment_prefers_large_room_below_target_over_small_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("se_small", col_index, 0)
            for col_index in range(14)
        } | {
            ("ne_large", col_index, 0)
            for col_index in range(4)
        }
        room_assignments = {
            "epuck_1": {"room": "nw_small"},
            "epuck_2": {"room": "se_small"},
            "epuck_3": {"room": "se_small", "helper": True},
            "epuck_4": {"room": "ne_large"},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            {"nw_small", "n_medium", "sw_large", "s_medium"},
            overlay,
        )

        self.assertEqual("ne_large", next_room)

    def test_reassignment_prefers_slow_progress_room_when_support_matches(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 100 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("ne_large", col_index, 0)
            for col_index in range(40)
        } | {
            ("sw_large", col_index, 0)
            for col_index in range(40)
        }
        room_assignments = {
            "epuck_1": {"room": "nw_small"},
            "epuck_2": {"room": "ne_large"},
            "epuck_3": {"room": "sw_large"},
        }
        progress_monitors = {
            "ne_large": {"rate_percent_per_1000_steps": 8.0},
            "sw_large": {"rate_percent_per_1000_steps": 0.0},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            {"nw_small", "n_medium", "s_medium", "se_small"},
            overlay,
            progress_monitors=progress_monitors,
        )

        self.assertEqual("sw_large", next_room)

    def test_reassignment_uses_route_cost_when_need_is_similar(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 100 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("n_medium", col_index, 0)
            for col_index in range(40)
        } | {
            ("s_medium", col_index, 0)
            for col_index in range(40)
        }
        room_assignments = {
            "epuck_1": {"room": "nw_small"},
            "epuck_2": {"room": "n_medium"},
            "epuck_3": {"room": "s_medium"},
        }
        latest_robot_status = {
            "epuck_1": {"pose": {"x_m": 0.2, "y_m": -0.2}},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            {"nw_small", "ne_large", "sw_large", "se_small"},
            overlay,
            latest_robot_status=latest_robot_status,
        )

        self.assertEqual("s_medium", next_room)

    def test_reassignment_can_join_room_that_already_has_helper(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("ne_large", col_index, 0)
            for col_index in range(12)
        } | {
            ("sw_large", col_index, 0)
            for col_index in range(10)
        }
        room_assignments = {
            "epuck_1": {"room": "ne_large", "helper": True},
            "epuck_2": {"room": "sw_large", "helper": True},
            "epuck_3": {"room": "s_medium"},
            "epuck_4": {"room": "se_small"},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_3",
            room_assignments,
            {"nw_small", "n_medium", "s_medium", "se_small"},
            overlay,
        )

        self.assertEqual("ne_large", next_room)

    def test_reassignment_returns_none_when_all_rooms_are_complete(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = set()
        room_assignments = {
            "epuck_1": {"room": "nw_small"},
            "epuck_2": {"room": "n_medium"},
            "epuck_3": {"room": "s_medium"},
            "epuck_4": {"room": "se_small"},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            set(supervisor.ROOM_TASKS),
            overlay,
        )

        self.assertIsNone(next_room)

    def test_stuck_recovery_retries_current_unfinished_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("nw_small", 0, 0),
            ("ne_large", 0, 0),
        }
        room_assignments = {
            "epuck_1": {"room": "nw_small"},
            "epuck_2": {"room": "n_medium"},
            "epuck_3": {"room": "s_medium"},
            "epuck_4": {"room": "se_small"},
        }

        recovery_room = supervisor.select_stuck_recovery_room(
            "epuck_1",
            room_assignments,
            {"n_medium", "s_medium", "se_small"},
            overlay,
        )

        self.assertEqual("nw_small", recovery_room)

    def test_repeated_stuck_robot_can_switch_from_supported_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {
            ("ne_large", 0, 0),
            ("n_medium", 0, 0),
        }
        room_assignments = {
            "epuck_1": {"room": "ne_large"},
            "epuck_2": {"room": "ne_large"},
            "epuck_3": {"room": "s_medium"},
            "epuck_4": {"room": "se_small"},
        }

        recovery_room = supervisor.select_stuck_recovery_room(
            "epuck_1",
            room_assignments,
            {"nw_small", "s_medium", "se_small"},
            overlay,
            recovery_count=supervisor.STUCK_ROUTE_RETRY_LIMIT + 1,
        )

        self.assertEqual("n_medium", recovery_room)

    def test_stuck_recovery_can_reassign_after_current_room_is_complete(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {room: 20 for room in supervisor.ROOM_TASKS}
        overlay.dirty_tiles = {("ne_large", 0, 0)}
        room_assignments = {
            "epuck_1": {"room": "nw_small"},
            "epuck_2": {"room": "n_medium"},
            "epuck_3": {"room": "s_medium"},
            "epuck_4": {"room": "se_small"},
        }

        recovery_room = supervisor.select_stuck_recovery_room(
            "epuck_1",
            room_assignments,
            {"nw_small", "n_medium", "s_medium", "se_small"},
            overlay,
        )

        self.assertEqual("ne_large", recovery_room)

    def test_build_assignment_cost_uses_current_robot_pose(self):
        supervisor = load_supervisor_module()
        center_x_m, center_y_m = supervisor.ROOM_TASKS["ne_large"]["center"]
        status = {"pose": {"x_m": center_x_m, "y_m": center_y_m}}

        assignment = supervisor.build_assignment(status, "ne_large", helper=True)

        self.assertEqual("ne_large", assignment["room"])
        self.assertEqual((center_x_m, center_y_m), assignment["target"])
        self.assertTrue(assignment["helper"])
        expected_cost = round(
            supervisor.MRTA_AREA_WEIGHT * supervisor.ROOM_TASKS["ne_large"]["area_m2"],
            3,
        )
        self.assertAlmostEqual(expected_cost, assignment["cost"])
        self.assertEqual(
            supervisor.generate_assignment_route(status["pose"], "ne_large"),
            assignment["route"],
        )

    def test_assignment_route_graph_shortens_room_to_room_travel(self):
        supervisor = load_supervisor_module()
        pose = {"x_m": -2.85, "y_m": 2.2}

        route = supervisor.generate_assignment_route(pose, "ne_large")
        old_center_route = [
            [-2.25, 0.93],
            [-2.25, 0.57],
            supervisor.HUB_ROUTE_WAYPOINT,
            [1.85, 0.57],
            [1.85, 0.93],
            [1.85, 1.875],
        ]

        self.assertEqual([-2.25, 0.93], route[0])
        self.assertEqual([-2.25, 0.57], route[1])
        self.assertNotIn(supervisor.HUB_ROUTE_WAYPOINT, route)
        self.assertEqual([1.85, 0.57], route[-3])
        self.assertEqual([1.85, 0.93], route[-2])
        self.assertEqual([1.85, 1.875], route[-1])
        self.assertLess(
            supervisor.route_distance_m(pose, route),
            supervisor.route_distance_m(pose, old_center_route),
        )

    def test_assignment_route_from_hub_skips_shared_center_waypoint(self):
        supervisor = load_supervisor_module()
        pose = {"x_m": -0.3, "y_m": 0.1}

        route = supervisor.generate_assignment_route(pose, "n_medium")

        self.assertNotIn(supervisor.HUB_ROUTE_WAYPOINT, route)
        self.assertEqual([-0.4, 0.57], route[0])
        self.assertEqual([-0.4, 0.93], route[1])
        self.assertEqual([-0.4, 1.875], route[2])

    def test_assignment_route_can_end_at_lane_start(self):
        supervisor = load_supervisor_module()
        pose = {"x_m": -0.3, "y_m": 0.1}
        lane_start = supervisor.generate_coverage_waypoints("n_medium")[0]

        route = supervisor.generate_assignment_route(
            pose,
            "n_medium",
            final_waypoint=lane_start,
        )

        self.assertEqual(lane_start, route[-1])

    def test_traffic_reservations_stagger_shared_hub_routes(self):
        supervisor = load_supervisor_module()
        reservations = supervisor.TrafficReservationBook()
        resources = ["hub:all", "hub:north", "doorway:n_medium"]

        first = reservations.reserve("epuck_1", resources, step_count=10)
        second = reservations.reserve("epuck_2", resources, step_count=10)

        self.assertEqual(0, first["start_delay_steps"])
        self.assertEqual(
            supervisor.TRAFFIC_RESOURCE_STAGGER_STEPS,
            second["start_delay_steps"],
        )
        self.assertGreater(second["conflict_count"], 0)
        self.assertEqual(1, second["doorway_conflicts"])

    def test_send_assignment_commands_targets_one_robot(self):
        supervisor = load_supervisor_module()

        class FakeEmitter:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        emitter = FakeEmitter()
        assignment = {
            "room": "n_medium",
            "target": supervisor.ROOM_TASKS["n_medium"]["center"],
            "cost": 0.5,
            "route": [[0.0, 0.0], [-0.4, 0.57], [-0.4, 0.93], [-0.4, 1.875]],
        }

        coverage_plan = supervisor.send_assignment_commands(
            emitter,
            "epuck_2",
            assignment,
        )

        self.assertEqual(
            supervisor.generate_coverage_waypoints("n_medium"),
            coverage_plan,
        )
        self.assertEqual(2, len(emitter.messages))

        task_command = json.loads(emitter.messages[0])
        plan_command = json.loads(emitter.messages[1])
        self.assertEqual("task_assignment", task_command["type"])
        self.assertEqual("coverage_plan", plan_command["type"])
        self.assertEqual("epuck_2", task_command["robot"])
        self.assertEqual("epuck_2", plan_command["robot"])
        self.assertEqual("n_medium", task_command["room"])
        self.assertEqual("n_medium", plan_command["room"])
        self.assertEqual(assignment["route"], task_command["route"])
        self.assertEqual("sweep", plan_command["plan_kind"])
        self.assertEqual(0, plan_command["lane_index"])
        self.assertEqual(1, plan_command["lane_count"])

    def test_send_assignment_commands_adds_route_delay_and_metrics(self):
        supervisor = load_supervisor_module()

        class FakeEmitter:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        reservations = supervisor.TrafficReservationBook()
        status = {"pose": {"x_m": -0.3, "y_m": 0.1}}
        first_assignment = supervisor.build_assignment(status, "n_medium")
        second_assignment = supervisor.build_assignment(status, "n_medium")

        supervisor.send_assignment_commands(
            FakeEmitter(),
            "epuck_1",
            first_assignment,
            {"epuck_1": first_assignment},
            robot_status=status,
            traffic_reservations=reservations,
            step_count=20,
        )

        emitter = FakeEmitter()
        supervisor.send_assignment_commands(
            emitter,
            "epuck_2",
            second_assignment,
            {"epuck_2": second_assignment},
            robot_status=status,
            traffic_reservations=reservations,
            step_count=20,
        )

        command = json.loads(emitter.messages[0])
        self.assertEqual(
            supervisor.TRAFFIC_RESOURCE_STAGGER_STEPS,
            command["start_delay_steps"],
        )
        self.assertGreater(command["path_metrics"]["planned_distance_m"], 0.0)
        self.assertIn("hub:all", command["path_metrics"]["resources"])
        self.assertGreater(command["path_metrics"]["doorway_conflicts"], 0)

    def test_send_assignment_commands_uses_split_lane_when_room_has_helpers(self):
        supervisor = load_supervisor_module()

        class FakeEmitter:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        emitter = FakeEmitter()
        room_assignments = {
            "epuck_1": {"room": "ne_large"},
            "epuck_2": {"room": "ne_large"},
        }
        assignment = {
            "room": "ne_large",
            "target": supervisor.ROOM_TASKS["ne_large"]["center"],
            "cost": 0.5,
            "route": [[0.0, 0.0], [1.85, 0.57], [1.85, 0.93], [1.85, 1.875]],
        }

        coverage_plan = supervisor.send_assignment_commands(
            emitter,
            "epuck_2",
            assignment,
            room_assignments,
        )

        plan_command = json.loads(emitter.messages[1])
        self.assertEqual(
            supervisor.generate_coverage_waypoints("ne_large", 1, 2),
            coverage_plan,
        )
        self.assertEqual(1, plan_command["lane_index"])
        self.assertEqual(2, plan_command["lane_count"])
        self.assertEqual(coverage_plan, plan_command["waypoints"])

    def test_send_room_coverage_plans_resplits_all_robots_in_room(self):
        supervisor = load_supervisor_module()

        class FakeEmitter:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        emitter = FakeEmitter()
        room_assignments = {
            "epuck_1": {"room": "ne_large"},
            "epuck_2": {"room": "ne_large", "helper": True},
            "epuck_3": {"room": "s_medium"},
        }

        plans = supervisor.send_room_coverage_plans(
            emitter,
            room_assignments,
            "ne_large",
        )

        self.assertEqual({"epuck_1", "epuck_2"}, set(plans))
        self.assertEqual(2, len(emitter.messages))
        first_plan = json.loads(emitter.messages[0])
        second_plan = json.loads(emitter.messages[1])
        self.assertEqual("epuck_1", first_plan["robot"])
        self.assertEqual(0, first_plan["lane_index"])
        self.assertEqual(2, first_plan["lane_count"])
        self.assertEqual("epuck_2", second_plan["robot"])
        self.assertEqual(1, second_plan["lane_index"])
        self.assertEqual(2, second_plan["lane_count"])

    def test_resplitting_room_clears_overwritten_cleanup_state(self):
        supervisor = load_supervisor_module()

        class FakeEmitter:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        emitter = FakeEmitter()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.tile_claims = {
            ("ne_large", 0, 0): "epuck_1",
            ("ne_large", 1, 0): "epuck_2",
            ("s_medium", 0, 0): "epuck_3",
        }
        cleanup_plan_signatures = {
            "epuck_1": ("ne_large", ((0.875, 0.875),)),
            "epuck_2": ("ne_large", ((1.125, 0.875),)),
            "epuck_3": ("s_medium", ((-0.625, -2.875),)),
        }
        cleanup_plan_steps = {
            "epuck_1": 10,
            "epuck_2": 20,
            "epuck_3": 30,
        }
        room_assignments = {
            "epuck_1": {"room": "ne_large"},
            "epuck_2": {"room": "ne_large", "helper": True},
            "epuck_3": {"room": "s_medium"},
        }

        supervisor.send_room_coverage_plans(
            emitter,
            room_assignments,
            "ne_large",
            overlay,
            cleanup_plan_signatures,
            cleanup_plan_steps,
        )

        self.assertEqual({"epuck_3"}, set(cleanup_plan_signatures))
        self.assertEqual({"epuck_3"}, set(cleanup_plan_steps))
        self.assertEqual(
            {("s_medium", 0, 0): "epuck_3"},
            overlay.tile_claims,
        )

    def test_send_recovery_command_targets_one_robot(self):
        supervisor = load_supervisor_module()

        class FakeEmitter:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        emitter = FakeEmitter()
        supervisor.send_recovery_command(emitter, "epuck_3", "blocked")

        self.assertEqual(1, len(emitter.messages))
        command = json.loads(emitter.messages[0])
        self.assertEqual("recovery", command["type"])
        self.assertEqual("epuck_3", command["robot"])
        self.assertEqual("blocked", command["reason"])
        self.assertEqual(
            supervisor.ROBOT_RECOVERY_REVERSE_STEPS,
            command["reverse_steps"],
        )
        self.assertEqual(supervisor.ROBOT_RECOVERY_TURN_STEPS, command["turn_steps"])

    def test_send_idle_command_targets_one_robot(self):
        supervisor = load_supervisor_module()

        class FakeEmitter:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        emitter = FakeEmitter()
        supervisor.send_idle_command(emitter, "epuck_4")

        self.assertEqual(1, len(emitter.messages))
        command = json.loads(emitter.messages[0])
        self.assertEqual({"type": "idle", "robot": "epuck_4"}, command)

    def test_robot_grid_uses_repeated_free_evidence(self):
        controller = load_controller_module()
        grid = controller.OccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        self.assertFalse(grid.mark_free_cell(5, 5))
        self.assertEqual(controller.GRID_UNKNOWN, grid.data[5][5])

        self.assertTrue(grid.mark_free_cell(5, 5))
        self.assertEqual(controller.GRID_FREE, grid.data[5][5])
        self.assertEqual(1, grid.free_cell_count)

    def test_robot_grid_wall_evidence_can_override_light_free_evidence(self):
        controller = load_controller_module()
        grid = controller.OccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        grid.mark_free_cell(5, 5)
        grid.mark_free_cell(5, 5)
        self.assertEqual(controller.GRID_FREE, grid.data[5][5])

        grid.mark_wall_cell(5, 5)
        self.assertEqual(controller.GRID_UNKNOWN, grid.data[5][5])
        self.assertEqual(0, grid.free_cell_count)
        self.assertEqual(0, grid.wall_cell_count)

        grid.mark_wall_cell(5, 5)
        self.assertEqual(controller.GRID_WALL, grid.data[5][5])
        self.assertEqual(1, grid.wall_cell_count)

    def test_robot_grid_bad_wall_reading_can_be_recovered_by_free_evidence(self):
        controller = load_controller_module()
        grid = controller.OccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        grid.mark_wall_cell(5, 5)
        self.assertEqual(controller.GRID_WALL, grid.data[5][5])

        for _ in range(5):
            grid.mark_free_cell(5, 5)

        self.assertEqual(controller.GRID_FREE, grid.data[5][5])
        self.assertEqual(1, grid.free_cell_count)
        self.assertEqual(0, grid.wall_cell_count)

    def test_robot_grid_sends_observation_counts_with_legacy_cells(self):
        controller = load_controller_module()
        grid = controller.OccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        grid.mark_free_cell(5, 5)
        grid.mark_free_cell(5, 5)
        grid.mark_wall_cell(6, 5)
        updates = grid.drain_pending_updates()

        self.assertEqual([[5, 5, 2]], updates["free_observations"])
        self.assertEqual([[6, 5, 1]], updates["wall_observations"])
        self.assertEqual([[5, 5]], updates["free_cells"])
        self.assertEqual([[6, 5]], updates["wall_cells"])
        self.assertEqual(
            {
                "free_cells": [],
                "wall_cells": [],
                "free_observations": [],
                "wall_observations": [],
                "observations": [],
            },
            grid.drain_pending_updates(),
        )

    def test_robot_grid_sends_observations_in_order(self):
        controller = load_controller_module()
        grid = controller.OccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        for _ in range(3):
            grid.mark_wall_cell(5, 5)
        for _ in range(10):
            grid.mark_free_cell(5, 5)
        updates = grid.drain_pending_updates()

        self.assertEqual(
            [
                ["wall", 5, 5, 3],
                ["free", 5, 5, 10],
            ],
            updates["observations"],
        )

    def test_supervisor_grid_merges_confidence_observations(self):
        supervisor = load_supervisor_module()
        grid = supervisor.GlobalOccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        free_count, wall_count = grid.merge_update(
            {"free_observations": [[5, 5, 1]]}
        )
        self.assertEqual((0, 0), (free_count, wall_count))
        self.assertEqual(supervisor.GRID_UNKNOWN, grid.data[5][5])

        free_count, wall_count = grid.merge_update(
            {"free_observations": [[5, 5, 1]]}
        )
        self.assertEqual((1, 0), (free_count, wall_count))
        self.assertEqual(supervisor.GRID_FREE, grid.data[5][5])

        free_count, wall_count = grid.merge_update(
            {"wall_observations": [[5, 5, 2]]}
        )
        self.assertEqual((0, 1), (free_count, wall_count))
        self.assertEqual(supervisor.GRID_WALL, grid.data[5][5])

    def test_supervisor_ordered_observations_match_robot_clamped_score(self):
        controller = load_controller_module()
        supervisor = load_supervisor_module()
        robot_grid = controller.OccupancyGrid(world_size_m=1.0, cell_size_m=0.1)
        global_grid = supervisor.GlobalOccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        for _ in range(3):
            robot_grid.mark_wall_cell(5, 5)
        for _ in range(10):
            robot_grid.mark_free_cell(5, 5)

        global_grid.merge_update(robot_grid.drain_pending_updates())

        self.assertEqual(robot_grid.scores[5][5], global_grid.scores[5][5])
        self.assertEqual(robot_grid.data[5][5], global_grid.data[5][5])
        self.assertEqual(supervisor.GRID_FREE, global_grid.data[5][5])

    def test_scan_match_shifts_update_toward_existing_wall_cells(self):
        supervisor = load_supervisor_module()
        grid = supervisor.GlobalOccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        for _ in range(2):
            grid.add_cell_evidence(5, 5, supervisor.WALL_EVIDENCE)
            grid.add_cell_evidence(5, 6, supervisor.WALL_EVIDENCE)
        map_update = {
            "observations": [
                ["wall", 6, 5, 2],
                ["wall", 6, 6, 2],
            ]
        }

        scan_match = grid.scan_match_update(map_update)

        self.assertTrue(scan_match["accepted"])
        self.assertEqual([-1, 0], scan_match["offset_cells"])
        self.assertEqual(
            [["wall", 5, 5, 2], ["wall", 5, 6, 2]],
            scan_match["map_update"]["observations"],
        )

    def test_scan_match_rejects_offsets_without_enough_existing_map_support(self):
        supervisor = load_supervisor_module()
        grid = supervisor.GlobalOccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        scan_match = grid.scan_match_update(
            {"observations": [["wall", 6, 5, 2]]}
        )

        self.assertFalse(scan_match["accepted"])
        self.assertEqual([0, 0], scan_match["offset_cells"])

    def test_scan_match_does_not_accept_free_space_only_offsets(self):
        supervisor = load_supervisor_module()
        grid = supervisor.GlobalOccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        for _ in range(2):
            grid.add_cell_evidence(5, 5, supervisor.FREE_EVIDENCE)
            grid.add_cell_evidence(5, 6, supervisor.FREE_EVIDENCE)

        scan_match = grid.scan_match_update(
            {
                "observations": [
                    ["free", 6, 5, 2],
                    ["free", 6, 6, 2],
                ]
            }
        )

        self.assertFalse(scan_match["accepted"])
        self.assertEqual([0, 0], scan_match["offset_cells"])

    def test_ground_truth_correction_is_only_fallback_for_stale_or_drifted_pose(self):
        supervisor = load_supervisor_module()
        actual_pose = {"x_m": 0.0, "y_m": 0.0, "theta_rad": 0.0}
        fresh_status = {
            "pose": {"x_m": 0.1, "y_m": 0.0, "theta_rad": 0.1},
            "localization": {"confidence": 0.9},
        }
        stale_status = {
            "pose": {"x_m": 0.1, "y_m": 0.0, "theta_rad": 0.1},
            "localization": {"confidence": 0.1},
        }
        drifted_status = {
            "pose": {"x_m": 1.0, "y_m": 0.0, "theta_rad": 0.1},
            "localization": {"confidence": 0.9},
        }

        self.assertFalse(
            supervisor.should_send_ground_truth_pose_correction(
                fresh_status,
                actual_pose,
            )
        )
        self.assertTrue(
            supervisor.should_send_ground_truth_pose_correction(
                stale_status,
                actual_pose,
            )
        )
        self.assertTrue(
            supervisor.should_send_ground_truth_pose_correction(
                drifted_status,
                actual_pose,
            )
        )

    def test_supervisor_grid_keeps_legacy_map_update_support(self):
        supervisor = load_supervisor_module()
        grid = supervisor.GlobalOccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        free_count, wall_count = grid.merge_update(
            {
                "free_cells": [[5, 5]],
                "wall_cells": [[6, 5]],
            }
        )

        self.assertEqual((1, 1), (free_count, wall_count))
        self.assertEqual(supervisor.GRID_FREE, grid.data[5][5])
        self.assertEqual(supervisor.GRID_WALL, grid.data[5][6])

    def test_supervisor_legacy_wall_update_still_overrides_free_cell(self):
        supervisor = load_supervisor_module()
        grid = supervisor.GlobalOccupancyGrid(world_size_m=1.0, cell_size_m=0.1)

        grid.merge_update({"free_cells": [[5, 5]]})
        free_count, wall_count = grid.merge_update({"wall_cells": [[5, 5]]})

        self.assertEqual((0, 1), (free_count, wall_count))
        self.assertEqual(supervisor.GRID_WALL, grid.data[5][5])
        self.assertEqual(0, grid.free_cell_count)
        self.assertEqual(1, grid.wall_cell_count)


if __name__ == "__main__":
    unittest.main()
