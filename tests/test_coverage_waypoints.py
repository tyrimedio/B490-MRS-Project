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

    def test_room_reached_coverage_goal_uses_percent_threshold(self):
        supervisor = load_supervisor_module()
        overlay = supervisor.CleaningOverlay.__new__(supervisor.CleaningOverlay)
        overlay.room_tile_counts = {"nw_small": 20}
        overlay.dirty_tiles = {("nw_small", 0, 0)}

        self.assertTrue(
            supervisor.room_reached_coverage_goal(overlay, "nw_small")
        )

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

    def test_reassignment_selects_least_clean_unfinished_room(self):
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

    def test_reassignment_skips_room_that_already_has_helper(self):
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


if __name__ == "__main__":
    unittest.main()
