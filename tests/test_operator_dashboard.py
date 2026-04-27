import importlib.util
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = REPO_ROOT / "operator_dashboard.py"


def load_dashboard_module():
    spec = importlib.util.spec_from_file_location("operator_dashboard", DASHBOARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OperatorDashboardTests(unittest.TestCase):
    def test_pause_and_resume_robot(self):
        dashboard = load_dashboard_module()
        config = dashboard.default_config()

        config = dashboard.apply_action(
            config,
            {"action": "set_paused", "robot": "epuck_2", "paused": True},
        )
        self.assertEqual(["epuck_2"], config["paused_robots"])

        config = dashboard.apply_action(
            config,
            {"action": "set_paused", "robot": "epuck_2", "paused": False},
        )
        self.assertEqual([], config["paused_robots"])

    def test_add_priority_room_zone(self):
        dashboard = load_dashboard_module()

        config = dashboard.apply_action(
            dashboard.default_config(),
            {
                "action": "add_priority",
                "name": "north focus",
                "room": "n_medium",
                "weight": 2.5,
            },
        )

        self.assertEqual(1, len(config["priority_zones"]))
        self.assertEqual("north focus", config["priority_zones"][0]["name"])
        self.assertEqual("n_medium", config["priority_zones"][0]["room"])
        self.assertEqual(2.5, config["priority_zones"][0]["weight"])

    def test_add_no_go_preset_zone(self):
        dashboard = load_dashboard_module()

        config = dashboard.apply_action(
            dashboard.default_config(),
            {
                "action": "add_no_go",
                "name": "block hub",
                "preset": "hub_center",
            },
        )

        self.assertEqual(1, len(config["no_go_zones"]))
        self.assertEqual("block hub", config["no_go_zones"][0]["name"])
        self.assertEqual(
            dashboard.ZONE_PRESETS["hub_center"]["bounds"],
            config["no_go_zones"][0]["bounds"],
        )

    def test_manual_redirect_gets_unique_dashboard_id(self):
        dashboard = load_dashboard_module()

        config = dashboard.apply_action(
            dashboard.default_config(),
            {
                "action": "redirect",
                "robot": "epuck_3",
                "room": "ne_large",
            },
        )

        self.assertEqual("epuck_3", config["redirects"][0]["robot"])
        self.assertEqual("ne_large", config["redirects"][0]["room"])
        self.assertTrue(config["redirects"][0]["id"].startswith("dashboard-"))

    def test_reset_sim_clears_controls_and_records_request(self):
        dashboard = load_dashboard_module()
        config = {
            "paused_robots": ["epuck_1"],
            "priority_zones": [{"room": "n_medium"}],
            "no_go_zones": [{"room": "s_medium"}],
            "redirects": [{"robot": "epuck_2", "room": "ne_large"}],
        }

        updated = dashboard.apply_action(config, {"action": "reset_sim"})

        self.assertEqual([], updated["paused_robots"])
        self.assertEqual([], updated["priority_zones"])
        self.assertEqual([], updated["no_go_zones"])
        self.assertEqual([], updated["redirects"])
        self.assertTrue(
            updated["sim_reset_request"]["id"].startswith("dashboard-reset-")
        )

    def test_reset_all_clears_sim_reset_request(self):
        dashboard = load_dashboard_module()
        config = dashboard.apply_action(dashboard.default_config(), {"action": "reset_sim"})

        updated = dashboard.apply_action(config, {"action": "reset_all"})

        self.assertIsNone(updated["sim_reset_request"])

    def test_action_endpoint_handles_non_object_json_payload(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "operator_controls.json"
            state_path = Path(temp_dir) / "operator_state.json"
            dashboard.write_config(config_path, dashboard.default_config())
            dashboard.reset_operator_state_file(state_path)
            original_config_path = dashboard.OperatorDashboardHandler.config_path
            original_state_path = dashboard.OperatorDashboardHandler.state_path
            dashboard.OperatorDashboardHandler.config_path = config_path
            dashboard.OperatorDashboardHandler.state_path = state_path
            server = dashboard.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                dashboard.OperatorDashboardHandler,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection(
                    server.server_address[0],
                    server.server_address[1],
                    timeout=5,
                )
                connection.request(
                    "POST",
                    "/api/action",
                    body="[]",
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                body = json.loads(response.read().decode("utf-8"))
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                dashboard.OperatorDashboardHandler.config_path = original_config_path
                dashboard.OperatorDashboardHandler.state_path = original_state_path

        self.assertEqual(200, response.status)
        self.assertEqual([], body["config"]["paused_robots"])
        self.assertIsNone(body["config"]["sim_reset_request"])

    def test_dashboard_state_uses_reset_map_while_sim_reset_pending(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "operator_controls.json"
            state_path = Path(temp_dir) / "operator_state.json"
            dashboard.write_config(
                config_path,
                dashboard.apply_action(
                    dashboard.default_config(),
                    {"action": "reset_sim"},
                ),
            )
            state_path.write_text(
                json.dumps(
                    {
                        "step": 999,
                        "rooms": [
                            {"name": "n_medium", "progress_percent": 70.0}
                        ],
                        "robots": [
                            {
                                "name": "epuck_1",
                                "pose": {"x_m": 1.0, "y_m": 1.0},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            state = dashboard.dashboard_state(config_path, state_path)

        self.assertIsNone(state["simState"]["step"])
        self.assertEqual([], state["simState"]["robots"])
        self.assertEqual(0.0, state["simState"]["rooms"][0]["progress_percent"])

    def test_reset_operator_state_file_clears_stale_map_state(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "operator_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "step": 999,
                        "rooms": [{"name": "n_medium", "progress_percent": 70.0}],
                        "robots": [{"name": "epuck_1"}],
                    }
                ),
                encoding="utf-8",
            )

            changed = dashboard.reset_operator_state_file(state_path)
            reloaded = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertIsNone(reloaded["step"])
        self.assertEqual([], reloaded["robots"])
        self.assertEqual(0.0, reloaded["rooms"][0]["progress_percent"])

    def test_write_config_sanitizes_invalid_entries(self):
        dashboard = load_dashboard_module()
        raw_config = {
            "paused_robots": ["epuck_1", "ghost"],
            "priority_zones": [{"room": "bad_room"}],
            "no_go_zones": [{"name": "custom", "bounds": [0.5, -0.5, 0.2, -0.2]}],
            "redirects": [{"robot": "epuck_2", "room": "s_medium"}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "operator_controls.json"
            written = dashboard.write_config(config_path, raw_config)
            reloaded = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(written, reloaded)
        self.assertEqual(["epuck_1"], reloaded["paused_robots"])
        self.assertEqual([], reloaded["priority_zones"])
        self.assertEqual([-0.5, 0.5, -0.2, 0.2], reloaded["no_go_zones"][0]["bounds"])
        self.assertEqual("epuck_2", reloaded["redirects"][0]["robot"])

    def test_dashboard_caps_zones_to_supervisor_limits(self):
        dashboard = load_dashboard_module()
        raw_config = {
            "priority_zones": [
                {"name": f"priority {index}", "bounds": [index, index + 0.1, 0, 0.1]}
                for index in range(8)
            ],
            "no_go_zones": [
                {"name": f"no-go {index}", "bounds": [index, index + 0.1, 0, 0.1]}
                for index in range(8)
            ],
        }

        clean_config = dashboard.sanitize_config(raw_config)

        self.assertEqual(
            dashboard.OPERATOR_MAX_PRIORITY_ZONES,
            len(clean_config["priority_zones"]),
        )
        self.assertEqual(
            dashboard.OPERATOR_MAX_NO_GO_ZONES,
            len(clean_config["no_go_zones"]),
        )
        self.assertEqual("priority 5", clean_config["priority_zones"][-1]["name"])
        self.assertEqual("no-go 5", clean_config["no_go_zones"][-1]["name"])

    def test_compact_dashboard_view_is_available(self):
        dashboard = load_dashboard_module()

        self.assertIn("/compact", dashboard.HTML)
        self.assertIn("Operator Popout", dashboard.COMPACT_HTML)
        self.assertIn("floorMap", dashboard.COMPACT_HTML)
        self.assertIn("applyRoomAction", dashboard.COMPACT_HTML)
        self.assertIn("resetSimulation", dashboard.COMPACT_HTML)
        self.assertIn("/api/action", dashboard.COMPACT_HTML)

    def test_main_dashboard_zone_rendering_escapes_user_text(self):
        dashboard = load_dashboard_module()

        self.assertIn("function esc(value)", dashboard.HTML)
        self.assertIn("${esc(zone.name)}", dashboard.HTML)
        self.assertIn("zone.bounds.map(esc).join(', ')", dashboard.HTML)
        self.assertIn("room=${esc(zone.room)}", dashboard.HTML)
        self.assertIn("weight=${esc(zone.weight)}", dashboard.HTML)

    def test_main_dashboard_redirect_rendering_escapes_user_text(self):
        dashboard = load_dashboard_module()

        self.assertIn("${esc(redirect.robot)} to ${esc(redirect.room)}", dashboard.HTML)
        self.assertIn("${esc(redirect.id)}", dashboard.HTML)
        self.assertNotIn("${redirect.id}", dashboard.HTML)

    def test_dashboard_state_includes_live_operator_state(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "operator_controls.json"
            state_path = Path(temp_dir) / "operator_state.json"
            dashboard.write_config(config_path, dashboard.default_config())
            state_path.write_text(
                json.dumps(
                    {
                        "step": 123,
                        "rooms": [{"name": "n_medium", "progress_percent": 40.0}],
                        "robots": [{"name": "epuck_1", "phase": "sweeping"}],
                    }
                ),
                encoding="utf-8",
            )

            state = dashboard.dashboard_state(config_path, state_path)

        self.assertEqual(123, state["simState"]["step"])
        self.assertEqual("n_medium", state["simState"]["rooms"][0]["name"])
        self.assertTrue(state["statePath"].endswith("operator_state.json"))

    def test_dashboard_state_does_not_mark_edge_touching_room_no_go(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "operator_controls.json"
            state_path = Path(temp_dir) / "operator_state.json"
            dashboard.write_config(
                config_path,
                dashboard.apply_action(
                    dashboard.default_config(),
                    {
                        "action": "add_no_go",
                        "name": "north block",
                        "room": "n_medium",
                    },
                ),
            )
            state_path.write_text(
                json.dumps(
                    {
                        "step": 10,
                        "rooms": [
                            {
                                "name": "nw_small",
                                "bounds": dashboard.ROOM_BOUNDS["nw_small"],
                                "no_go_zones": ["stale bad label"],
                            },
                            {
                                "name": "n_medium",
                                "bounds": dashboard.ROOM_BOUNDS["n_medium"],
                                "no_go_zones": [],
                            },
                        ],
                        "robots": [],
                    }
                ),
                encoding="utf-8",
            )

            state = dashboard.dashboard_state(config_path, state_path)

        rooms = {room["name"]: room for room in state["simState"]["rooms"]}
        self.assertEqual([], rooms["nw_small"]["no_go_zones"])
        self.assertEqual(["north block"], rooms["n_medium"]["no_go_zones"])

    def test_open_compact_window_builds_app_window_command(self):
        dashboard = load_dashboard_module()
        calls = []

        def fake_run(command, check=False):
            calls.append((command, check))

        original_run = dashboard.subprocess.run
        try:
            dashboard.subprocess.run = fake_run
            dashboard.open_compact_window("127.0.0.1", 8787, "Brave Browser")
        finally:
            dashboard.subprocess.run = original_run

        command, check = calls[0]
        self.assertFalse(check)
        self.assertIn("Brave Browser", command)
        self.assertIn("--app=http://127.0.0.1:8787/compact", command)
        self.assertIn("--window-size=390,760", command)

    def test_remove_webots_text_file_tabs_removes_operator_json_files_only(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / ".roomba_cluster.wbproj"
            project_path.write_text(
                "\n".join(
                    [
                        "Webots Project File version R2025a",
                        'textFiles: 0 "operator_controls.json" 1 "operator_state.json" 2 "notes.py"',
                        "consoles: Console:All:All",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            changed = dashboard.remove_webots_text_file_tabs(
                project_path,
                {"operator_controls.json", "operator_state.json"},
            )

            cleaned = project_path.read_text(encoding="utf-8")

        self.assertTrue(changed)
        self.assertIn('textFiles: 0 "notes.py"', cleaned)
        self.assertNotIn("operator_controls.json", cleaned)
        self.assertNotIn("operator_state.json", cleaned)

    def test_remove_webots_text_file_tabs_uses_webots_empty_marker(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / ".roomba_cluster.wbproj"
            project_path.write_text(
                "\n".join(
                    [
                        "Webots Project File version R2025a",
                        'textFiles: 0 "operator_controls.json"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            changed = dashboard.remove_webots_text_file_tabs(
                project_path,
                {"operator_controls.json"},
            )

            cleaned = project_path.read_text(encoding="utf-8")

        self.assertTrue(changed)
        self.assertIn("textFiles: -1", cleaned)

    def test_ensure_webots_sim_view_visible_restores_hidden_view(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / ".roomba_cluster.wbproj"
            project_path.write_text(
                "\n".join(
                    [
                        "Webots Project File version R2025a",
                        "centralWidgetVisible: 0",
                        "textFiles: -1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            changed = dashboard.ensure_webots_sim_view_visible(project_path)

            cleaned = project_path.read_text(encoding="utf-8")

        self.assertTrue(changed)
        self.assertIn("centralWidgetVisible: 1", cleaned)
        self.assertNotIn("centralWidgetVisible: 0", cleaned)

    def test_ensure_webots_sim_view_visible_keeps_visible_view_unchanged(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / ".roomba_cluster.wbproj"
            project_text = "\n".join(
                [
                    "Webots Project File version R2025a",
                    "centralWidgetVisible: 1",
                    "textFiles: -1",
                ]
            ) + "\n"
            project_path.write_text(project_text, encoding="utf-8")

            changed = dashboard.ensure_webots_sim_view_visible(project_path)

            cleaned = project_path.read_text(encoding="utf-8")

        self.assertFalse(changed)
        self.assertEqual(project_text, cleaned)

    def test_remove_webots_rendering_device_overlays_removes_camera_previews(self):
        dashboard = load_dashboard_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / ".roomba_cluster.wbproj"
            project_path.write_text(
                "\n".join(
                    [
                        "Webots Project File version R2025a",
                        "renderingDevicePerspectives: epuck_1:camera;1;1;0;0",
                        "renderingDevicePerspectives: epuck_1:lidar;1;1;0;0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            changed = dashboard.remove_webots_rendering_device_overlays(
                project_path,
                {"camera"},
            )

            cleaned = project_path.read_text(encoding="utf-8")

        self.assertTrue(changed)
        self.assertNotIn("epuck_1:camera", cleaned)
        self.assertIn("epuck_1:lidar", cleaned)


if __name__ == "__main__":
    unittest.main()
