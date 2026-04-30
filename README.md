# Roomba Cluster

Roomba Cluster is a Webots simulation of a small team of four e-puck robots
cleaning a six-room indoor floor plan. The project shows how a central
supervisor can divide work between robots, watch their progress, and send help
when a room still has dirty tiles left.

The simplest way to think about it is:

- The supervisor is the team manager.
- Each e-puck is a worker.
- Each room is a cleaning task.
- The floor overlay is the visible checklist of dirty and clean tiles.

The project does not use ROS. The controllers are Python scripts run by Webots.

## What The Simulation Does

The world has a 6 m by 6 m arena with six named rooms around a central hub:

- `nw_small`
- `n_medium`
- `ne_large`
- `sw_large`
- `s_medium`
- `se_small`

At startup, the robots move to staging points in the central hub. After all
robots are ready, the supervisor assigns rooms using a simple cost score. A
cost score means "how expensive this room is for this robot." It mostly depends
on how far the robot has to drive and how large the room is.

Each robot then drives a back-and-forth sweep path through its assigned room.
If a room still has dirty tiles after the sweep, the supervisor sends cleanup
targets. Cleanup targets use tile claims, which is like putting a robot's name
on a dirty square so another robot does not chase the same square.

The supervisor also:

- merges local map updates into a shared grid map
- sends pose corrections from Webots so the demo stays understandable
- reroutes robots through doorways instead of straight through walls
- detects stuck robots and sends recovery commands
- watches whether room progress has stalled
- reassigns finished robots to help unfinished rooms
- writes live and archived evaluation metrics

## Requirements

- Webots R2025a or a compatible recent Webots version
- Python 3
- A shell from the repository root

No Python package install is required for the main Webots simulation or the
operator dashboard. The tests use Python's built-in `unittest` runner.

## Run The Webots Simulation

Open the world in Webots:

```bash
open worlds/roomba_cluster.wbt
```

Then press the Webots run button. The supervisor and robot controllers start
automatically from the world file.

For faster evaluation runs, use Webots' fast simulation button.

## Run The Operator Dashboard

Start the dashboard from the repository root:

```bash
python3 operator_dashboard.py
```

Open:

```text
http://127.0.0.1:8787
```

For the small app-style popout window:

```bash
python3 operator_dashboard.py --open-compact
```

The dashboard reads `operator_state.json` and writes operator commands to
`operator_controls.json`. The supervisor updates those files while Webots is
running.

## Run A Single-Robot Baseline

Create `evaluation_run_config.json` in the repository root:

```json
{
  "run_label": "single_robot_baseline",
  "robot_count": 1
}
```

Reload the Webots world and run the simulation. The supervisor removes the
inactive robots at runtime, so you do not need to delete robots from the world
file by hand.

When the run finishes, copy the clean time into `single_robot_baseline.json`:

```json
{
  "cleaning_time_s": 1912.3
}
```

That baseline lets later four-robot runs show the dashboard's `baseline`
number. In the dashboard, `baseline` means the percent faster than the
single-robot run.

## Run The Normal Four-Robot Comparison

Use this `evaluation_run_config.json`:

```json
{
  "run_label": "four_robot_comparison",
  "active_robots": [
    "epuck_1",
    "epuck_2",
    "epuck_3",
    "epuck_4"
  ]
}
```

Reload the Webots world and run the simulation again.

## Evaluation Outputs

The supervisor writes the latest dashboard metrics to:

```text
evaluation_metrics.json
```

It also saves each separate run under:

```text
evaluation_runs/
```

That folder is useful because a new run does not overwrite the old one.

The saved comparison files from the current evaluation are:

- `evaluation_runs/20260429_195125_422304_single_robot_baseline.json`
- `evaluation_runs/20260429_195626_435933_four_robot_comparison.json`
- `evaluation_runs/single_vs_four_comparison.md`
- `evaluation_runs/single_vs_four_comparison.json`

## Key Result

The current single-robot and four-robot comparison used the same six-room world.
Both runs reached full coverage and reported zero inter-robot collisions.

| Metric | 1 robot | 4 robots | Difference |
| --- | ---: | ---: | ---: |
| Final coverage | 100.0% | 100.0% | Same |
| Full clean time | 1912.3 s | 535.7 s | 1376.6 s faster |
| 95% coverage time | 1806.2 s | 444.5 s | 1361.7 s faster |
| Collisions | 0 | 0 | Same |
| Speedup | 1.00x | 3.57x | 4 robots were 3.57x faster |
| Time reduction | Baseline | 72.0% | Meets the 50% target |

In plain terms: four robots cleaned the same arena in about 28% of the
single robot's time. That shows the multi-robot coordination is doing useful
work instead of just adding more robots to the screen.

## Run Tests

Run the full test suite with:

```bash
python3 -m unittest discover -s tests
```

The tests cover coverage waypoint bounds, room-entry cleaning gates, tile
claiming, overlay alignment, assignment routing, helper reassignment, stuck
detection, room progress monitoring, operator dashboard behavior, and metrics
logging.

## Known Limits

This is still a simulation project. The supervisor uses Webots ground-truth
robot poses to keep the demo stable. A real robot would need a real localization
system, such as wheel odometry, sensor-based localization, SLAM, or an external
tracking system.

The cleaning overlay is also a visual model. It marks dirty tiles clean when a
robot's path passes over them; it is not a physical vacuum model.
