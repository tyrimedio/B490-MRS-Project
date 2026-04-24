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

    def test_cleaning_marks_only_after_assigned_room_is_reached(self):
        supervisor = load_supervisor_module()

        self.assertFalse(
            supervisor.robot_can_mark_cleaning(
                {"assignment_target_reached": False}
            )
        )
        self.assertTrue(
            supervisor.robot_can_mark_cleaning(
                {"assignment_target_reached": True}
            )
        )

    def test_room_progress_percent_uses_dirty_tiles(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {"northwest": 4}
        overlay.dirty_tiles = {
            ("northwest", 0, 0),
            ("northwest", 1, 0),
        }

        self.assertEqual(2, overlay.cleaned_tile_count("northwest"))
        self.assertEqual(50.0, overlay.room_progress_percent("northwest"))

    def test_room_reached_coverage_goal_uses_percent_threshold(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {"northwest": 20}
        overlay.dirty_tiles = {("northwest", 0, 0)}

        self.assertTrue(
            supervisor.room_reached_coverage_goal(overlay, "northwest")
        )

    def test_room_progress_snapshot_reports_every_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {
            "northwest": 10,
            "northeast": 10,
            "southeast": 10,
            "southwest": 10,
        }
        overlay.dirty_tiles = {
            ("northwest", 0, 0),
            ("northeast", 0, 0),
            ("northeast", 1, 0),
            ("southeast", 0, 0),
            ("southeast", 1, 0),
            ("southeast", 2, 0),
        }

        snapshot = supervisor.room_progress_snapshot(overlay)

        self.assertEqual(
            {
                "northwest": 90.0,
                "northeast": 80.0,
                "southeast": 70.0,
                "southwest": 100.0,
            },
            snapshot,
        )
        self.assertEqual(
            "northwest=90.0% northeast=80.0% southeast=70.0% southwest=100.0%",
            supervisor.format_room_progress(snapshot),
        )

    def test_reassignment_selects_least_clean_unfinished_room(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {
            "northwest": 20,
            "northeast": 20,
            "southeast": 20,
            "southwest": 20,
        }
        overlay.dirty_tiles = {
            ("northeast", col_index, 0)
            for col_index in range(12)
        } | {
            ("southeast", col_index, 0)
            for col_index in range(5)
        } | {
            ("southwest", col_index, 0)
            for col_index in range(8)
        }
        room_assignments = {
            "epuck_1": {"room": "northwest"},
            "epuck_2": {"room": "northeast"},
            "epuck_3": {"room": "southeast"},
            "epuck_4": {"room": "southwest"},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            {"northwest"},
            overlay,
        )

        self.assertEqual("northeast", next_room)

    def test_reassignment_skips_room_that_already_has_helper(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {
            "northwest": 20,
            "northeast": 20,
            "southeast": 20,
            "southwest": 20,
        }
        overlay.dirty_tiles = {
            ("northeast", col_index, 0)
            for col_index in range(12)
        } | {
            ("southeast", col_index, 0)
            for col_index in range(6)
        }
        room_assignments = {
            "epuck_1": {"room": "northwest"},
            "epuck_2": {"room": "northeast"},
            "epuck_3": {"room": "northeast", "helper": True},
            "epuck_4": {"room": "southwest"},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            {"northwest"},
            overlay,
        )

        self.assertEqual("southeast", next_room)

    def test_reassignment_returns_none_when_all_rooms_are_complete(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {
            "northwest": 20,
            "northeast": 20,
            "southeast": 20,
            "southwest": 20,
        }
        overlay.dirty_tiles = set()
        room_assignments = {
            "epuck_1": {"room": "northwest"},
            "epuck_2": {"room": "northeast"},
            "epuck_3": {"room": "southeast"},
            "epuck_4": {"room": "southwest"},
        }

        next_room = supervisor.select_reassignment_room(
            "epuck_1",
            room_assignments,
            {"northwest", "northeast", "southeast", "southwest"},
            overlay,
        )

        self.assertIsNone(next_room)

    def test_build_assignment_cost_uses_current_robot_pose(self):
        supervisor = load_supervisor_module()
        status = {"pose": {"x_m": 1.75, "y_m": 1.75}}

        assignment = supervisor.build_assignment(status, "northeast", helper=True)

        self.assertEqual("northeast", assignment["room"])
        self.assertEqual((1.75, 1.75), assignment["target"])
        self.assertTrue(assignment["helper"])
        self.assertEqual(0.312, assignment["cost"])

    def test_send_assignment_commands_targets_one_robot(self):
        supervisor = load_supervisor_module()

        class FakeEmitter:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        emitter = FakeEmitter()
        assignment = {
            "room": "northeast",
            "target": (1.75, 1.75),
            "cost": 0.312,
        }

        coverage_plan = supervisor.send_assignment_commands(
            emitter,
            "epuck_2",
            assignment,
        )

        self.assertEqual(
            supervisor.generate_coverage_waypoints("northeast"),
            coverage_plan,
        )
        self.assertEqual(2, len(emitter.messages))

        task_command = json.loads(emitter.messages[0])
        plan_command = json.loads(emitter.messages[1])
        self.assertEqual("task_assignment", task_command["type"])
        self.assertEqual("coverage_plan", plan_command["type"])
        self.assertEqual("epuck_2", task_command["robot"])
        self.assertEqual("epuck_2", plan_command["robot"])
        self.assertEqual("northeast", task_command["room"])
        self.assertEqual("northeast", plan_command["room"])


if __name__ == "__main__":
    unittest.main()
