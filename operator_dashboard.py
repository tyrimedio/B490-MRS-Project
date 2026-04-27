#!/usr/bin/env python3
"""Local browser dashboard for Webots operator controls."""

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


VALID_ROBOTS = ("epuck_1", "epuck_2", "epuck_3", "epuck_4")
VALID_ROOMS = (
    "nw_small",
    "n_medium",
    "ne_large",
    "sw_large",
    "s_medium",
    "se_small",
)
OPERATOR_MAX_PRIORITY_ZONES = 6
OPERATOR_MAX_NO_GO_ZONES = 6
OPERATOR_STATE_FILE = "operator_state.json"
ROOM_BOUNDS = {
    "nw_small": [-3.0, -1.5, 0.75, 3.0],
    "n_medium": [-1.5, 0.7, 0.75, 3.0],
    "ne_large": [0.7, 3.0, 0.75, 3.0],
    "sw_large": [-3.0, -0.7, -3.0, -0.75],
    "s_medium": [-0.7, 1.5, -3.0, -0.75],
    "se_small": [1.5, 3.0, -3.0, -0.75],
}
ZONE_PRESETS = {
    "hub_center": {
        "label": "Hub center",
        "bounds": [-0.35, 0.35, -0.35, 0.35],
    },
    "north_doorways": {
        "label": "North doorways",
        "bounds": [-2.55, 2.15, 0.45, 1.05],
    },
    "south_doorways": {
        "label": "South doorways",
        "bounds": [-2.15, 2.55, -1.05, -0.45],
    },
    "west_hub_lane": {
        "label": "West hub lane",
        "bounds": [-1.6, -0.35, -0.45, 0.45],
    },
    "east_hub_lane": {
        "label": "East hub lane",
        "bounds": [0.35, 1.6, -0.45, 0.45],
    },
}
WEBOTS_PROJECT_PATH = Path(__file__).resolve().parent / "worlds" / ".roomba_cluster.wbproj"
WEBOTS_TEXT_FILE_BLOCKLIST = {"operator_controls.json", OPERATOR_STATE_FILE}
WEBOTS_RENDERING_DEVICE_BLOCKLIST = {"camera"}


def default_config():
    """Return an empty operator-control config."""
    return {
        "paused_robots": [],
        "priority_zones": [],
        "no_go_zones": [],
        "redirects": [],
        "sim_reset_request": None,
    }


def normalize_bounds(value):
    """Return [min_x, max_x, min_y, max_y] when bounds are usable."""
    if not isinstance(value, list) or len(value) != 4:
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


def bounds_overlap(first_bounds, second_bounds):
    """Return True when two rectangles share floor area, not just a wall edge."""
    first_min_x, first_max_x, first_min_y, first_max_y = first_bounds
    second_min_x, second_max_x, second_min_y, second_max_y = second_bounds
    return not (
        first_max_x <= second_min_x
        or second_max_x <= first_min_x
        or first_max_y <= second_min_y
        or second_max_y <= first_min_y
    )


def sanitize_zone(raw_zone, zone_kind):
    """Return a clean zone dictionary or None."""
    if not isinstance(raw_zone, dict):
        return None

    room = raw_zone.get("room")
    bounds = normalize_bounds(raw_zone.get("bounds"))
    if room in ROOM_BOUNDS and bounds is None:
        bounds = list(ROOM_BOUNDS[room])
    if bounds is None:
        return None

    name = raw_zone.get("name")
    if not isinstance(name, str) or not name.strip():
        name = room if room in ROOM_BOUNDS else zone_kind.replace("_", " ")

    zone = {
        "name": name.strip(),
        "bounds": bounds,
    }
    if room in ROOM_BOUNDS:
        zone["room"] = room
    if zone_kind == "priority_zones":
        try:
            weight = float(raw_zone.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        zone["weight"] = round(max(0.0, weight), 3)
    return zone


def sanitize_config(raw_config):
    """Return a supervisor-compatible config with invalid entries removed."""
    if not isinstance(raw_config, dict):
        return default_config()

    config = default_config()
    paused = raw_config.get("paused_robots", [])
    if isinstance(paused, list):
        for robot in paused:
            if robot == "all":
                config["paused_robots"] = ["all"]
                break
            if robot in VALID_ROBOTS and robot not in config["paused_robots"]:
                config["paused_robots"].append(robot)

    for key in ("priority_zones", "no_go_zones"):
        zones = raw_config.get(key, [])
        if not isinstance(zones, list):
            continue
        zone_limit = (
            OPERATOR_MAX_PRIORITY_ZONES
            if key == "priority_zones"
            else OPERATOR_MAX_NO_GO_ZONES
        )
        for raw_zone in zones:
            zone = sanitize_zone(raw_zone, key)
            if zone is not None:
                config[key].append(zone)
            if len(config[key]) >= zone_limit:
                break

    redirects = raw_config.get("redirects", [])
    if isinstance(redirects, list):
        for index, raw_redirect in enumerate(redirects):
            if not isinstance(raw_redirect, dict):
                continue
            robot = raw_redirect.get("robot")
            room = raw_redirect.get("room")
            if robot not in VALID_ROBOTS or room not in VALID_ROOMS:
                continue
            redirect_id = raw_redirect.get("id")
            if not isinstance(redirect_id, str) or not redirect_id.strip():
                redirect_id = f"dashboard-{robot}-{room}-{index}"
            config["redirects"].append(
                {
                    "id": redirect_id.strip(),
                    "robot": robot,
                    "room": room,
                }
            )

    raw_reset_request = raw_config.get("sim_reset_request")
    if isinstance(raw_reset_request, dict):
        reset_id = raw_reset_request.get("id")
        if isinstance(reset_id, str) and reset_id.strip():
            reset_request = {"id": reset_id.strip()}
            requested_at = raw_reset_request.get("requested_at")
            try:
                reset_request["requested_at"] = round(float(requested_at), 3)
            except (TypeError, ValueError):
                pass
            config["sim_reset_request"] = reset_request

    return config


def read_config(config_path):
    """Read the live operator config from disk."""
    path = Path(config_path)
    if not path.exists():
        return default_config()
    try:
        return sanitize_config(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return default_config()


def write_config(config_path, config):
    """Write the live operator config using a small atomic replace."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_config = sanitize_config(config)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(clean_config, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)
    return clean_config


def default_operator_state():
    """Return an empty live-state payload for when the sim is not running."""
    return {
        "step": None,
        "rooms": [
            {
                "name": room,
                "bounds": list(ROOM_BOUNDS[room]),
                "progress_percent": 0.0,
                "dirty_tiles": 0,
                "assigned_robots": [],
                "completed": False,
                "priority_zones": [],
                "no_go_zones": [],
            }
            for room in VALID_ROOMS
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
        "metrics": {
            "elapsed_cleaning_time_s": None,
            "coverage_percent": 0.0,
            "coverage_target_percent": 95.0,
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
        },
    }


def read_operator_state(state_path):
    """Read the live supervisor state used by the visual dashboard."""
    path = Path(state_path)
    if not path.exists():
        return default_operator_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_operator_state()
    if not isinstance(state, dict):
        return default_operator_state()
    return state


def zone_names_for_room(zones, room_name, room_bounds):
    """Return zone names that should visually apply to one dashboard room."""
    names = []
    for zone in zones:
        if zone.get("room") == room_name:
            names.append(zone["name"])
            continue
        zone_bounds = zone.get("bounds")
        if zone_bounds is not None and bounds_overlap(zone_bounds, room_bounds):
            names.append(zone["name"])
    return names


def apply_config_zones_to_room_state(sim_state, config):
    """Keep room zone colors tied to the current config, not stale sim labels."""
    if not isinstance(sim_state, dict):
        return default_operator_state()

    rooms = sim_state.get("rooms")
    if not isinstance(rooms, list):
        return sim_state

    for room in rooms:
        if not isinstance(room, dict):
            continue
        room_name = room.get("name")
        room_bounds = normalize_bounds(room.get("bounds"))
        if room_name not in ROOM_BOUNDS and room_bounds is None:
            continue
        if room_bounds is None:
            room_bounds = list(ROOM_BOUNDS[room_name])
        room["priority_zones"] = zone_names_for_room(
            config["priority_zones"],
            room_name,
            room_bounds,
        )
        room["no_go_zones"] = zone_names_for_room(
            config["no_go_zones"],
            room_name,
            room_bounds,
        )
    return sim_state


def reset_operator_state_file(state_path):
    """Clear the dashboard map state immediately after a sim reset request."""
    path = Path(state_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(default_operator_state(), indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError:
        return False
    return True


def remove_webots_text_file_tabs(project_path, blocked_file_names):
    """Remove dashboard config files from Webots' saved text editor tabs."""
    path = Path(project_path)
    try:
        project_text = path.read_text(encoding="utf-8")
    except OSError:
        return False

    blocked_names = {Path(name).name for name in blocked_file_names}
    changed = False
    output_lines = []
    for line in project_text.splitlines():
        if not line.startswith("textFiles:"):
            output_lines.append(line)
            continue

        entries = []
        tokens = line[len("textFiles:") :].strip().split('"')
        for token_index in range(1, len(tokens), 2):
            file_name = tokens[token_index]
            if Path(file_name).name in blocked_names:
                changed = True
                continue
            entries.append(file_name)

        rebuilt = "textFiles: -1"
        if entries:
            rebuilt = "textFiles:"
            for entry_index, file_name in enumerate(entries):
                rebuilt += f' {entry_index} "{file_name}"'
        output_lines.append(rebuilt)

    if not changed:
        return False

    try:
        path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def ensure_webots_sim_view_visible(project_path):
    """Restore the saved Webots layout when the main 3D sim view was hidden."""
    path = Path(project_path)
    try:
        project_text = path.read_text(encoding="utf-8")
    except OSError:
        return False

    changed = False
    found_setting = False
    output_lines = []
    for line in project_text.splitlines():
        if line.startswith("centralWidgetVisible:"):
            found_setting = True
            if line != "centralWidgetVisible: 1":
                line = "centralWidgetVisible: 1"
                changed = True
        output_lines.append(line)

    if not found_setting:
        output_lines.append("centralWidgetVisible: 1")
        changed = True
    if not changed:
        return False

    try:
        path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def remove_webots_rendering_device_overlays(project_path, blocked_device_names):
    """Remove saved Webots device preview overlays from the sim view."""
    path = Path(project_path)
    try:
        project_text = path.read_text(encoding="utf-8")
    except OSError:
        return False

    blocked_names = {str(name) for name in blocked_device_names}
    changed = False
    output_lines = []
    for line in project_text.splitlines():
        if not line.startswith("renderingDevicePerspectives:"):
            output_lines.append(line)
            continue

        device_spec = line[len("renderingDevicePerspectives:") :].strip()
        device_name = device_spec.split(";", 1)[0].split(":")[-1]
        if device_name in blocked_names:
            changed = True
            continue
        output_lines.append(line)

    if not changed:
        return False

    try:
        path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def zone_from_payload(payload, zone_kind):
    """Build a zone from a dashboard action payload."""
    room = payload.get("room")
    preset = payload.get("preset")
    bounds = payload.get("bounds")
    label = None

    if room in ROOM_BOUNDS:
        bounds = list(ROOM_BOUNDS[room])
        label = room
    elif preset in ZONE_PRESETS:
        preset_config = ZONE_PRESETS[preset]
        bounds = list(preset_config["bounds"])
        label = preset_config["label"]

    zone = {
        "name": payload.get("name") or label,
        "bounds": bounds,
    }
    if room in ROOM_BOUNDS:
        zone["room"] = room
    if zone_kind == "priority_zones":
        zone["weight"] = payload.get("weight", 1.0)
    return sanitize_zone(zone, zone_kind)


def remove_index(items, index):
    """Return a copy of items with one index removed when valid."""
    try:
        index = int(index)
    except (TypeError, ValueError):
        return list(items)
    if index < 0 or index >= len(items):
        return list(items)
    return [item for item_index, item in enumerate(items) if item_index != index]


def apply_action(config, action_payload):
    """Apply one dashboard action to an operator config."""
    config = sanitize_config(config)
    if not isinstance(action_payload, dict):
        return config

    action = action_payload.get("action")
    robot = action_payload.get("robot")
    room = action_payload.get("room")

    if action == "set_paused" and (robot in VALID_ROBOTS or robot == "all"):
        paused = bool(action_payload.get("paused"))
        if robot == "all":
            config["paused_robots"] = ["all"] if paused else []
        elif paused:
            if "all" not in config["paused_robots"] and robot not in config["paused_robots"]:
                config["paused_robots"].append(robot)
        else:
            config["paused_robots"] = [
                paused_robot
                for paused_robot in config["paused_robots"]
                if paused_robot != robot and paused_robot != "all"
            ]
    elif action == "pause_all":
        config["paused_robots"] = ["all"]
    elif action == "resume_all":
        config["paused_robots"] = []
    elif action == "add_priority":
        zone = zone_from_payload(action_payload, "priority_zones")
        if zone is not None:
            config["priority_zones"].append(zone)
    elif action == "add_no_go":
        zone = zone_from_payload(action_payload, "no_go_zones")
        if zone is not None:
            config["no_go_zones"].append(zone)
    elif action == "remove_priority":
        config["priority_zones"] = remove_index(
            config["priority_zones"],
            action_payload.get("index"),
        )
    elif action == "remove_no_go":
        config["no_go_zones"] = remove_index(
            config["no_go_zones"],
            action_payload.get("index"),
        )
    elif action == "clear_priority":
        config["priority_zones"] = []
    elif action == "clear_no_go":
        config["no_go_zones"] = []
    elif action == "redirect" and robot in VALID_ROBOTS and room in VALID_ROOMS:
        redirect_id = action_payload.get("id")
        if not isinstance(redirect_id, str) or not redirect_id.strip():
            redirect_id = f"dashboard-{robot}-{room}-{int(time.time() * 1000)}"
        config["redirects"].append(
            {
                "id": redirect_id.strip(),
                "robot": robot,
                "room": room,
            }
        )
    elif action == "clear_redirects":
        config["redirects"] = []
    elif action == "reset_all":
        config = default_config()
    elif action == "reset_sim":
        now = time.time()
        config = default_config()
        config["sim_reset_request"] = {
            "id": f"dashboard-reset-{int(now * 1000)}",
            "requested_at": round(now, 3),
        }
    elif action == "replace_config":
        config = sanitize_config(action_payload.get("config"))

    return sanitize_config(config)


def dashboard_state(config_path, state_path=None):
    """Return all state the browser needs."""
    if state_path is None:
        state_path = Path(config_path).with_name(OPERATOR_STATE_FILE)
    config = read_config(config_path)
    sim_state = (
        default_operator_state()
        if config.get("sim_reset_request") is not None
        else read_operator_state(state_path)
    )
    sim_state = apply_config_zones_to_room_state(sim_state, config)
    return {
        "config": config,
        "simState": sim_state,
        "robots": list(VALID_ROBOTS),
        "rooms": list(VALID_ROOMS),
        "roomBounds": ROOM_BOUNDS,
        "presets": ZONE_PRESETS,
        "configPath": str(Path(config_path).resolve()),
        "statePath": str(Path(state_path).resolve()),
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roomba Operator Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #121417;
      --panel: #1d2228;
      --panel-2: #232a31;
      --line: #38424c;
      --text: #eef3f7;
      --muted: #9aa8b5;
      --blue: #4c7dff;
      --red: #e2514b;
      --green: #43c77a;
      --yellow: #d9a441;
      --input: #0e1115;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #171b20;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(360px, 1fr);
      min-height: calc(100vh - 58px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      overflow: auto;
    }
    section {
      border-bottom: 1px solid var(--line);
      padding: 14px 0;
    }
    section:first-child { padding-top: 0; }
    h2 {
      margin: 0 0 10px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 700;
    }
    .grid {
      display: grid;
      gap: 8px;
    }
    .two {
      grid-template-columns: 1fr 1fr;
    }
    .three {
      grid-template-columns: repeat(3, 1fr);
    }
    .four {
      grid-template-columns: repeat(4, 1fr);
    }
    .robot-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    button, select, input {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--input);
      color: var(--text);
      font: inherit;
      font-size: 14px;
      padding: 7px 9px;
    }
    button {
      cursor: pointer;
      background: var(--panel-2);
      font-weight: 600;
    }
    button:hover { border-color: #667481; }
    button.active.pause {
      background: #3a1c1d;
      border-color: var(--red);
      color: #ffd8d5;
    }
    button.primary {
      background: #20335d;
      border-color: var(--blue);
    }
    button.danger {
      background: #3a1c1d;
      border-color: var(--red);
    }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .workspace {
      display: grid;
      grid-template-rows: auto 1fr;
      min-width: 0;
    }
    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #15191e;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .active-area {
      display: grid;
      grid-template-columns: repeat(2, minmax(260px, 1fr));
      gap: 14px;
      padding: 14px;
      overflow: auto;
      align-content: start;
    }
    .list {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      min-height: 130px;
      overflow: hidden;
    }
    .list h3 {
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    .item {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid #2b333b;
    }
    .item:last-child { border-bottom: 0; }
    .item-title {
      font-size: 14px;
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .item-meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
      overflow-wrap: anywhere;
    }
    .empty {
      color: var(--muted);
      padding: 12px;
      font-size: 14px;
    }
    .json {
      grid-column: 1 / -1;
    }
    pre {
      margin: 0;
      padding: 12px;
      background: #0d1014;
      color: #d9e1e8;
      overflow: auto;
      min-height: 220px;
      font-size: 12px;
      line-height: 1.45;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .active-area { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Roomba Operator Dashboard</h1>
    <div class="status" id="configPath"></div>
  </header>
  <main>
    <aside>
      <section>
        <h2>Pause</h2>
        <div class="grid robot-grid" id="robotButtons"></div>
        <div class="grid two" style="margin-top: 8px;">
          <button class="danger" onclick="sendAction({action: 'pause_all'})">Pause all</button>
          <button onclick="sendAction({action: 'resume_all'})">Resume all</button>
        </div>
      </section>

      <section>
        <h2>Priority Zone</h2>
        <div class="grid">
          <label>Name<input id="priorityName" value="operator priority"></label>
          <label>Room<select id="priorityRoom"></select></label>
          <label>Weight<input id="priorityWeight" type="number" min="0" step="0.5" value="2"></label>
          <button class="primary" onclick="addPriority()">Add priority</button>
        </div>
      </section>

      <section>
        <h2>No-Go Zone</h2>
        <div class="grid">
          <label>Preset<select id="noGoPreset"></select></label>
          <label>Name<input id="noGoName" value="operator no-go"></label>
          <button class="primary" onclick="addNoGoPreset()">Add no-go</button>
        </div>
      </section>

      <section>
        <h2>Custom No-Go</h2>
        <div class="grid two">
          <label>Min X<input id="minX" type="number" step="0.05" value="-0.35"></label>
          <label>Max X<input id="maxX" type="number" step="0.05" value="0.35"></label>
          <label>Min Y<input id="minY" type="number" step="0.05" value="-0.35"></label>
          <label>Max Y<input id="maxY" type="number" step="0.05" value="0.35"></label>
        </div>
        <div class="grid" style="margin-top: 8px;">
          <button class="primary" onclick="addCustomNoGo()">Add custom no-go</button>
        </div>
      </section>

      <section>
        <h2>Manual Redirect</h2>
        <div class="grid">
          <label>Robot<select id="redirectRobot"></select></label>
          <label>Room<select id="redirectRoom"></select></label>
          <button class="primary" onclick="redirectRobot()">Send redirect</button>
        </div>
      </section>
    </aside>

    <div class="workspace">
      <div class="toolbar">
        <div class="status" id="status">Loading</div>
        <div class="grid four" style="width: 480px;">
          <button onclick="openPopout()">Popout</button>
          <button onclick="loadState()">Refresh</button>
          <button onclick="resetSimulation()">Reset sim</button>
          <button class="danger" onclick="sendAction({action: 'reset_all'})">Clear controls</button>
        </div>
      </div>
      <div class="active-area">
        <div class="list">
          <h3>Paused Robots</h3>
          <div id="pausedList"></div>
        </div>
        <div class="list">
          <h3>Manual Redirects</h3>
          <div id="redirectList"></div>
        </div>
        <div class="list">
          <h3>Priority Zones</h3>
          <div id="priorityList"></div>
        </div>
        <div class="list">
          <h3>No-Go Zones</h3>
          <div id="noGoList"></div>
        </div>
        <div class="list json">
          <h3>Live JSON</h3>
          <pre id="jsonPreview">{}</pre>
        </div>
      </div>
    </div>
  </main>

  <script>
    let state = null;

    async function request(path, options) {
      const response = await fetch(path, options);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    async function loadState() {
      state = await request('/api/state');
      render();
    }

    async function sendAction(action) {
      state = await request('/api/action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(action)
      });
      render();
    }

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      })[char]);
    }

    function resetSimulation() {
      if (!window.confirm('Reset the Webots simulation and clear operator controls?')) return;
      sendAction({action: 'reset_sim'});
    }

    function optionList(values, labels = {}) {
      return values.map(value => `<option value="${value}">${labels[value] || value}</option>`).join('');
    }

    function openPopout() {
      window.open(
        '/compact',
        'roombaOperatorPopout',
        'popup=yes,width=390,height=760,menubar=no,toolbar=no,location=no,status=no,resizable=yes,scrollbars=yes'
      );
    }

    function render() {
      const config = state.config;
      document.getElementById('configPath').textContent = state.configPath;
      document.getElementById('status').textContent = 'Saved to operator_controls.json';
      document.getElementById('jsonPreview').textContent = JSON.stringify(config, null, 2);

      document.getElementById('priorityRoom').innerHTML = optionList(state.rooms);
      document.getElementById('redirectRobot').innerHTML = optionList(state.robots);
      document.getElementById('redirectRoom').innerHTML = optionList(state.rooms);
      const presetLabels = {};
      Object.entries(state.presets).forEach(([key, preset]) => presetLabels[key] = preset.label);
      document.getElementById('noGoPreset').innerHTML =
        optionList([...Object.keys(state.presets), ...state.rooms], presetLabels);

      const paused = new Set(config.paused_robots.includes('all') ? state.robots : config.paused_robots);
      document.getElementById('robotButtons').innerHTML = state.robots.map(robot => {
        const isPaused = paused.has(robot);
        return `<button class="${isPaused ? 'active pause' : ''}" onclick="sendAction({action: 'set_paused', robot: '${robot}', paused: ${!isPaused}})">${robot}</button>`;
      }).join('');

      renderPaused(config);
      renderRedirects(config);
      renderZones('priorityList', config.priority_zones, 'remove_priority');
      renderZones('noGoList', config.no_go_zones, 'remove_no_go');
    }

    function renderPaused(config) {
      const container = document.getElementById('pausedList');
      if (!config.paused_robots.length) {
        container.innerHTML = '<div class="empty">none</div>';
        return;
      }
      container.innerHTML = config.paused_robots.map(robot => `
        <div class="item">
          <div>
            <div class="item-title">${robot}</div>
            <div class="item-meta">holding idle until resumed</div>
          </div>
          <button onclick="sendAction({action: '${robot === 'all' ? 'resume_all' : 'set_paused'}', robot: '${robot}', paused: false})">Resume</button>
        </div>
      `).join('');
    }

    function renderRedirects(config) {
      const container = document.getElementById('redirectList');
      if (!config.redirects.length) {
        container.innerHTML = '<div class="empty">none</div>';
        return;
      }
      container.innerHTML = config.redirects.map(redirect => `
        <div class="item">
          <div>
            <div class="item-title">${esc(redirect.robot)} to ${esc(redirect.room)}</div>
            <div class="item-meta">${esc(redirect.id)}</div>
          </div>
          <button onclick="sendAction({action: 'clear_redirects'})">Clear</button>
        </div>
      `).join('');
    }

    function renderZones(elementId, zones, action) {
      const container = document.getElementById(elementId);
      if (!zones.length) {
        container.innerHTML = '<div class="empty">none</div>';
        return;
      }
      container.innerHTML = zones.map((zone, index) => {
        const weight = zone.weight === undefined ? '' : ` weight=${esc(zone.weight)}`;
        const room = zone.room ? ` room=${esc(zone.room)}` : '';
        const bounds = Array.isArray(zone.bounds) ? zone.bounds.map(esc).join(', ') : '';
        return `
          <div class="item">
            <div>
              <div class="item-title">${esc(zone.name)}</div>
              <div class="item-meta">${bounds}${room}${weight}</div>
            </div>
            <button onclick="sendAction({action: '${action}', index: ${index}})">Remove</button>
          </div>
        `;
      }).join('');
    }

    function addPriority() {
      sendAction({
        action: 'add_priority',
        name: document.getElementById('priorityName').value,
        room: document.getElementById('priorityRoom').value,
        weight: Number(document.getElementById('priorityWeight').value)
      });
    }

    function addNoGoPreset() {
      const value = document.getElementById('noGoPreset').value;
      const action = {
        action: 'add_no_go',
        name: document.getElementById('noGoName').value
      };
      if (state.rooms.includes(value)) action.room = value;
      else action.preset = value;
      sendAction(action);
    }

    function addCustomNoGo() {
      sendAction({
        action: 'add_no_go',
        name: document.getElementById('noGoName').value,
        bounds: [
          Number(document.getElementById('minX').value),
          Number(document.getElementById('maxX').value),
          Number(document.getElementById('minY').value),
          Number(document.getElementById('maxY').value)
        ]
      });
    }

    function redirectRobot() {
      sendAction({
        action: 'redirect',
        robot: document.getElementById('redirectRobot').value,
        room: document.getElementById('redirectRoom').value
      });
    }

    loadState().catch(error => {
      document.getElementById('status').textContent = error.message;
    });
  </script>
</body>
</html>
"""


COMPACT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roomba Operator Popout</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101317;
      --panel: #1b2127;
      --panel-2: #232a31;
      --line: #3b4652;
      --text: #edf3f8;
      --muted: #9aa7b4;
      --blue: #4d7dff;
      --blue-soft: rgba(77, 125, 255, 0.28);
      --red: #e05650;
      --red-soft: rgba(224, 86, 80, 0.30);
      --green: #42c574;
      --yellow: #d7a642;
      --input: #0c1014;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      width: 100vw;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
      overflow: hidden;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      height: 38px;
      width: min(100vw, 420px);
      margin: 0 auto;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      background: #151a1f;
      -webkit-app-region: drag;
    }
    h1 {
      margin: 0;
      font-size: 13px;
      font-weight: 700;
    }
    main {
      width: min(100vw, 420px);
      height: calc(100vh - 38px);
      margin: 0 auto;
      overflow: auto;
      padding: 8px;
    }
    section {
      padding: 8px;
      margin-bottom: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    h2 {
      margin: 0 0 7px;
      font-size: 10px;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 750;
    }
    button, select {
      width: 100%;
      min-height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--input);
      color: var(--text);
      font: inherit;
      font-size: 12px;
      padding: 6px 8px;
    }
    button {
      cursor: pointer;
      background: #242b33;
      font-weight: 700;
    }
    button:hover { border-color: #657384; }
    button.primary, button.mode.active {
      background: #20335e;
      border-color: var(--blue);
    }
    button.danger, button.pause-active {
      background: #3a1d1f;
      border-color: var(--red);
    }
    label {
      display: grid;
      gap: 3px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
    }
    .status {
      min-width: 76px;
      text-align: right;
      font-size: 11px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7px;
    }
    .modes {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-bottom: 7px;
    }
    .map-panel {
      padding: 7px;
    }
    .map-wrap {
      border: 1px solid #495562;
      border-radius: 8px;
      background: #0e1216;
      overflow: hidden;
    }
    svg {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
    }
    .room {
      fill: #202831;
      stroke: #697584;
      stroke-width: 1.2;
      cursor: pointer;
    }
    .room:hover {
      stroke: #d7e2ed;
      stroke-width: 2;
    }
    .room.priority { fill: var(--blue-soft); stroke: var(--blue); }
    .room.no-go { fill: var(--red-soft); stroke: var(--red); }
    .room.complete { fill: rgba(66, 197, 116, 0.18); }
    .hub {
      fill: #151b21;
      stroke: #586472;
      stroke-dasharray: 4 3;
    }
    .room-text {
      pointer-events: none;
      fill: var(--text);
      font-size: 10px;
      font-weight: 750;
      text-anchor: middle;
    }
    .progress-text {
      pointer-events: none;
      fill: #c5d0da;
      font-size: 9px;
      text-anchor: middle;
    }
    .robot-dot {
      stroke: #f4f8fb;
      stroke-width: 1.5;
    }
    .robot-cleaning { fill: var(--green); }
    .robot-traveling { fill: var(--yellow); }
    .robot-paused { fill: #8995a1; }
    .robot-offline { fill: #4f5963; }
    .robot-label {
      pointer-events: none;
      fill: #061014;
      font-size: 8px;
      font-weight: 850;
      text-anchor: middle;
      dominant-baseline: central;
    }
    .map-key {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 5px;
      margin-top: 6px;
      font-size: 10px;
      color: var(--muted);
    }
    .key-item {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      min-width: 0;
    }
    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 2px;
      border: 1px solid var(--line);
      flex: 0 0 auto;
    }
    .fleet {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 6px;
    }
    .robot-card {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 7px;
      background: #151b21;
    }
    .robot-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      margin-bottom: 5px;
      font-size: 12px;
      font-weight: 800;
    }
    .phase {
      color: var(--muted);
      font-size: 10px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .robot-card button {
      min-height: 26px;
      padding: 4px 6px;
      font-size: 11px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 6px;
      background: #151a20;
      color: var(--text);
      font-size: 10px;
    }
    .chip button {
      width: auto;
      min-height: 20px;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 10px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 5px;
      margin-top: 7px;
      color: var(--muted);
      font-size: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px;
      background: #13191f;
      min-width: 0;
    }
    .metric strong {
      display: block;
      color: var(--text);
      font-size: 12px;
      margin-top: 2px;
    }
    .empty {
      color: var(--muted);
      font-size: 11px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Operator Popout</h1>
    <div class="status" id="status">loading</div>
  </header>
  <main>
    <section class="map-panel">
      <div class="modes">
        <button class="mode active" data-mode="priority" onclick="setMode('priority')">Priority</button>
        <button class="mode" data-mode="no_go" onclick="setMode('no_go')">No-go</button>
        <button class="mode" data-mode="redirect" onclick="setMode('redirect')">Redirect</button>
      </div>
      <div class="row" style="margin-bottom: 7px;">
        <label>Robot<select id="targetRobot" onchange="selectedRobot = this.value"></select></label>
        <button onclick="resetSimulation()">Reset sim</button>
      </div>
      <div style="margin-bottom: 7px;">
        <button class="danger" onclick="sendAction({action: 'reset_all'})">Clear controls</button>
      </div>
      <div class="map-wrap">
        <svg id="floorMap" viewBox="0 0 300 300" role="img" aria-label="Operator floor map"></svg>
      </div>
      <div class="map-key">
        <span class="key-item"><span class="swatch" style="background: var(--blue-soft);"></span>priority</span>
        <span class="key-item"><span class="swatch" style="background: var(--red-soft);"></span>no-go</span>
        <span class="key-item"><span class="swatch" style="background: var(--green);"></span>robots</span>
      </div>
      <div class="metrics" id="metrics"></div>
    </section>

    <section>
      <h2>Fleet</h2>
      <div class="fleet" id="fleet"></div>
      <div class="row" style="margin-top: 7px;">
        <button class="danger" onclick="sendAction({action: 'pause_all'})">Pause all</button>
        <button onclick="sendAction({action: 'resume_all'})">Resume all</button>
      </div>
    </section>

    <section>
      <h2>Active</h2>
      <div class="chips" id="activeChips"></div>
    </section>
  </main>

  <script>
    const WORLD_MIN = -3;
    const WORLD_MAX = 3;
    const MAP_SIZE = 300;
    const ROOM_LABELS = {
      nw_small: 'NW',
      n_medium: 'N',
      ne_large: 'NE',
      sw_large: 'SW',
      s_medium: 'S',
      se_small: 'SE'
    };
    let state = null;
    let selectedMode = 'priority';
    let selectedRobot = 'epuck_1';

    async function request(path, options) {
      const response = await fetch(path, options);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    async function loadState() {
      state = await request('/api/state');
      render();
    }

    async function sendAction(action) {
      state = await request('/api/action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(action)
      });
      render();
    }

    function resetSimulation() {
      if (!window.confirm('Reset the Webots simulation and clear operator controls?')) return;
      sendAction({action: 'reset_sim'});
    }

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      })[char]);
    }

    function setMode(mode) {
      selectedMode = mode;
      document.querySelectorAll('button.mode').forEach(button => {
        button.classList.toggle('active', button.dataset.mode === selectedMode);
      });
    }

    function sx(x) {
      return ((x - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)) * MAP_SIZE;
    }

    function sy(y) {
      return MAP_SIZE - ((y - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)) * MAP_SIZE;
    }

    function roomRows() {
      const liveRooms = state.simState && Array.isArray(state.simState.rooms)
        ? state.simState.rooms
        : [];
      if (liveRooms.length) return liveRooms;
      return state.rooms.map(room => ({
        name: room,
        bounds: state.roomBounds[room],
        progress_percent: 0,
        dirty_tiles: 0,
        assigned_robots: [],
        completed: false,
        priority_zones: [],
        no_go_zones: []
      }));
    }

    function roomClass(room) {
      const classes = ['room'];
      if ((room.no_go_zones || []).length) classes.push('no-go');
      else if ((room.priority_zones || []).length) classes.push('priority');
      else if (room.completed) classes.push('complete');
      return classes.join(' ');
    }

    function robotClass(robot) {
      if (!robot.connected) return 'robot-dot robot-offline';
      if (robot.paused || robot.phase === 'paused') return 'robot-dot robot-paused';
      if (robot.phase === 'sweeping' || robot.phase === 'cleanup') return 'robot-dot robot-cleaning';
      return 'robot-dot robot-traveling';
    }

    function renderMap() {
      const svg = document.getElementById('floorMap');
      const rooms = roomRows();
      const robots = (state.simState && state.simState.robots) || [];
      const roomMarkup = rooms.map(room => {
        const [minX, maxX, minY, maxY] = room.bounds;
        const x = sx(minX);
        const y = sy(maxY);
        const width = sx(maxX) - sx(minX);
        const height = sy(minY) - sy(maxY);
        const cx = x + width / 2;
        const cy = y + height / 2;
        const progress = Number(room.progress_percent || 0);
        const assigned = (room.assigned_robots || []).join(',');
        return `
          <g class="room-hit" data-room="${esc(room.name)}">
            <rect class="${roomClass(room)}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${width.toFixed(1)}" height="${height.toFixed(1)}" rx="3"></rect>
            <text class="room-text" x="${cx.toFixed(1)}" y="${(cy - 5).toFixed(1)}">${ROOM_LABELS[room.name] || esc(room.name)}</text>
            <text class="progress-text" x="${cx.toFixed(1)}" y="${(cy + 9).toFixed(1)}">${Math.round(progress)}% ${esc(assigned)}</text>
          </g>
        `;
      }).join('');
      const robotMarkup = robots.filter(robot => robot.pose).map(robot => {
        const x = sx(robot.pose.x_m);
        const y = sy(robot.pose.y_m);
        return `
          <g>
            <circle class="${robotClass(robot)}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="9"></circle>
            <text class="robot-label" x="${x.toFixed(1)}" y="${y.toFixed(1)}">${esc(robot.label)}</text>
          </g>
        `;
      }).join('');
      svg.innerHTML = `
        <rect class="hub" x="${sx(-0.7)}" y="${sy(0.75)}" width="${(sx(0.7) - sx(-0.7)).toFixed(1)}" height="${(sy(-0.75) - sy(0.75)).toFixed(1)}" rx="4"></rect>
        ${roomMarkup}
        ${robotMarkup}
      `;
      svg.querySelectorAll('.room-hit').forEach(roomElement => {
        roomElement.addEventListener('click', () => applyRoomAction(roomElement.dataset.room));
      });
    }

    function applyRoomAction(room) {
      if (selectedMode === 'priority') {
        sendAction({action: 'add_priority', name: `priority ${room}`, room, weight: 2});
      } else if (selectedMode === 'no_go') {
        sendAction({action: 'add_no_go', name: `no-go ${room}`, room});
      } else {
        sendAction({action: 'redirect', robot: selectedRobot, room});
      }
    }

    function renderRobotSelect() {
      const select = document.getElementById('targetRobot');
      const previous = selectedRobot;
      select.innerHTML = state.robots.map(robot => {
        const label = robot.replace('epuck_', 'E');
        return `<option value="${robot}">${label}</option>`;
      }).join('');
      selectedRobot = state.robots.includes(previous) ? previous : state.robots[0];
      select.value = selectedRobot;
    }

    function renderFleet() {
      const config = state.config;
      const paused = new Set(config.paused_robots.includes('all') ? state.robots : config.paused_robots);
      const liveRobots = new Map(((state.simState && state.simState.robots) || []).map(robot => [robot.name, robot]));
      document.getElementById('fleet').innerHTML = state.robots.map(robotName => {
        const robot = liveRobots.get(robotName) || {
          name: robotName,
          label: robotName.replace('epuck_', 'E'),
          phase: 'offline',
          room: null,
          connected: false
        };
        const isPaused = paused.has(robotName);
        const room = robot.room ? robot.room : 'unassigned';
        const phase = isPaused ? 'paused' : robot.phase;
        return `
          <div class="robot-card">
            <div class="robot-top"><span>${esc(robot.label)}</span><span class="phase">${esc(phase)}</span></div>
            <div class="phase">${esc(room)}</div>
            <button class="${isPaused ? 'pause-active' : ''}" onclick="sendAction({action: 'set_paused', robot: '${robotName}', paused: ${!isPaused}})">${isPaused ? 'Resume' : 'Pause'}</button>
          </div>
        `;
      }).join('');
    }

    function renderActiveChips() {
      const config = state.config;
      const chips = [];
      config.paused_robots.forEach(robot => {
        chips.push(`<span class="chip">pause ${esc(robot)}<button onclick="sendAction({action: '${robot === 'all' ? 'resume_all' : 'set_paused'}', robot: '${robot}', paused: false})">x</button></span>`);
      });
      config.priority_zones.forEach((zone, index) => {
        chips.push(`<span class="chip">priority ${esc(zone.room || zone.name)}<button onclick="sendAction({action: 'remove_priority', index: ${index}})">x</button></span>`);
      });
      config.no_go_zones.forEach((zone, index) => {
        chips.push(`<span class="chip">no-go ${esc(zone.room || zone.name)}<button onclick="sendAction({action: 'remove_no_go', index: ${index}})">x</button></span>`);
      });
      config.redirects.forEach(redirect => {
        chips.push(`<span class="chip">${esc(redirect.robot.replace('epuck_', 'E'))} to ${esc(redirect.room)}<button onclick="sendAction({action: 'clear_redirects'})">x</button></span>`);
      });
      if (config.sim_reset_request) {
        chips.push('<span class="chip">reset pending</span>');
      }
      document.getElementById('activeChips').innerHTML =
        chips.length ? chips.join('') : '<div class="empty">none</div>';
    }

    function renderMetrics() {
      const sim = state.simState || {};
      const metrics = sim.metrics || {};
      const robots = sim.robots || [];
      const connected = robots.filter(robot => robot.connected).length;
      const step = sim.step === null || sim.step === undefined ? '--' : sim.step;
      const coverage = Number(metrics.coverage_percent || 0).toFixed(1);
      const cleanTime = metrics.elapsed_cleaning_time_s === null
        || metrics.elapsed_cleaning_time_s === undefined
        ? '--'
        : `${Number(metrics.elapsed_cleaning_time_s).toFixed(1)}s`;
      const collisions = metrics.inter_robot_collisions || 0;
      const baseline = metrics.time_reduction_percent === null
        || metrics.time_reduction_percent === undefined
        ? '--'
        : `${Number(metrics.time_reduction_percent).toFixed(1)}%`;
      document.getElementById('metrics').innerHTML = `
        <div class="metric">step<strong>${esc(step)}</strong></div>
        <div class="metric">robots<strong>${connected}/${state.robots.length}</strong></div>
        <div class="metric">coverage<strong>${esc(coverage)}%</strong></div>
        <div class="metric">clean time<strong>${esc(cleanTime)}</strong></div>
        <div class="metric">collisions<strong>${esc(collisions)}</strong></div>
        <div class="metric">baseline<strong>${esc(baseline)}</strong></div>
      `;
    }

    function render() {
      const sim = state.simState || {};
      document.getElementById('status').textContent =
        sim.step === null || sim.step === undefined ? 'waiting' : `step ${sim.step}`;
      renderRobotSelect();
      renderMap();
      renderFleet();
      renderActiveChips();
      renderMetrics();
      setMode(selectedMode);
    }

    loadState().catch(error => {
      document.getElementById('status').textContent = error.message;
    });
    setInterval(() => loadState().catch(() => {}), 1000);
  </script>
</body>
</html>
"""


class OperatorDashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the operator dashboard."""

    config_path = Path("operator_controls.json")
    state_path = Path(OPERATOR_STATE_FILE)

    def do_GET(self):
        """Serve the dashboard HTML or current JSON state."""
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/compact":
            self.send_bytes(COMPACT_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/state":
            self.send_json(dashboard_state(self.config_path, self.state_path))
        else:
            self.send_error(404, "not found")

    def do_POST(self):
        """Apply one dashboard action."""
        parsed = urlparse(self.path)
        if parsed.path != "/api/action":
            self.send_error(404, "not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        if not isinstance(payload, dict):
            payload = {}

        config = read_config(self.config_path)
        updated_config = apply_action(config, payload)
        write_config(self.config_path, updated_config)
        if payload.get("action") == "reset_sim":
            reset_operator_state_file(self.state_path)
        self.send_json(dashboard_state(self.config_path, self.state_path))

    def send_json(self, payload):
        """Send a JSON response."""
        self.send_bytes(
            json.dumps(payload, indent=2).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def send_bytes(self, body, content_type):
        """Send a plain byte response."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        """Keep server logs compact."""
        print(f"[operator_dashboard] {self.address_string()} {format_string % args}")


def make_handler(config_path, state_path=None):
    """Return a handler class bound to a config path."""
    if state_path is None:
        state_path = Path(config_path).with_name(OPERATOR_STATE_FILE)
    return type(
        "ConfiguredOperatorDashboardHandler",
        (OperatorDashboardHandler,),
        {
            "config_path": Path(config_path),
            "state_path": Path(state_path),
        },
    )


def open_compact_window(host, port, browser_name="Brave Browser"):
    """Open the compact dashboard as a small app-style browser window on macOS."""
    url = f"http://{host}:{port}/compact"
    try:
        subprocess.run(
            [
                "open",
                "-na",
                browser_name,
                "--args",
                f"--app={url}",
                "--window-size=390,760",
                "--window-position=930,90",
            ],
            check=False,
        )
    except OSError as exc:
        print(f"[operator_dashboard] Could not open compact window: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Run the Roomba operator dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--open-compact",
        action="store_true",
        help="Open a small app-style popout window for use over Webots.",
    )
    parser.add_argument(
        "--browser",
        default="Brave Browser",
        help="macOS browser app name used by --open-compact.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "operator_controls.json"),
    )
    parser.add_argument(
        "--state",
        default=str(Path(__file__).resolve().parent / OPERATOR_STATE_FILE),
        help="Live supervisor state JSON read by the dashboard.",
    )
    parser.add_argument(
        "--webots-project",
        default=str(WEBOTS_PROJECT_PATH),
        help="Webots project file whose saved text-editor tabs should be cleaned.",
    )
    parser.add_argument(
        "--keep-webots-text-tabs",
        action="store_true",
        help="Do not remove operator dashboard JSON files from Webots' saved text editor tabs.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    state_path = Path(args.state)
    if not config_path.exists():
        write_config(config_path, default_config())
    restored_sim_view = ensure_webots_sim_view_visible(args.webots_project)
    removed_device_overlay = remove_webots_rendering_device_overlays(
        args.webots_project,
        WEBOTS_RENDERING_DEVICE_BLOCKLIST,
    )
    if not args.keep_webots_text_tabs:
        removed_tab = remove_webots_text_file_tabs(
            args.webots_project,
            WEBOTS_TEXT_FILE_BLOCKLIST | {config_path.name, state_path.name},
        )
        if removed_tab:
            print(
                "[operator_dashboard] Removed operator dashboard JSON files from "
                f"{Path(args.webots_project).resolve()} text editor tabs"
            )
    if restored_sim_view:
        print(
            "[operator_dashboard] Restored the saved Webots 3D sim view in "
            f"{Path(args.webots_project).resolve()}"
        )
    if removed_device_overlay:
        print(
            "[operator_dashboard] Removed saved camera previews from "
            f"{Path(args.webots_project).resolve()}"
        )

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(config_path, state_path),
    )
    print(f"[operator_dashboard] Serving http://{args.host}:{args.port}")
    print(f"[operator_dashboard] Compact view http://{args.host}:{args.port}/compact")
    print(f"[operator_dashboard] Writing {config_path.resolve()}")
    print(f"[operator_dashboard] Reading {state_path.resolve()}")
    if args.open_compact:
        threading.Timer(
            0.3,
            open_compact_window,
            args=(args.host, args.port, args.browser),
        ).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
