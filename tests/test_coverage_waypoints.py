import importlib.util
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


if __name__ == "__main__":
    unittest.main()
