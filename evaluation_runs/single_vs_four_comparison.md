# Single Robot vs Four Robot Comparison

Source files:
- Single robot: `evaluation_runs/20260429_195125_422304_single_robot_baseline.json`
- Four robots: `evaluation_runs/20260429_195626_435933_four_robot_comparison.json`

| Metric | Single robot | Four robots | Difference |
| --- | ---: | ---: | ---: |
| Final coverage | 100.0% | 100.0% | 0.0% |
| Full-clean time | 1912.3s | 535.7s | 1376.6s faster |
| 95% coverage time | 1806.2s | 444.5s | 1361.7s faster |
| Collisions | 0 | 0 | 0 |
| Minimum spacing | None | 0.198 | n/a |

Four robots finished 1376.6s faster, which is a 72.0% cleaning-time reduction and about 3.57x the single-robot speed.
