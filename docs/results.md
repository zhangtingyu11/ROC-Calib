# Paper results (2026-10-26)

The current C++ configuration is `all-bounded-contour-no-gradient-sampling`. All ten archived solves completed, with all 220 selected associations and 34,178 points retained. No new full benchmark is claimed by this source release.

| Group | RRE (deg) | RTE (m) |
|---|---:|---:|
| KITTI09 left | 0.424986 | 0.031226 |
| KITTI09 right | 0.235792 | 0.024936 |
| KITTI10 left | 0.390620 | 0.030563 |
| KITTI10 right | 0.373059 | 0.045566 |
| nuScenes1 | 1.196095 | 0.073641 |
| nuScenes2 | 0.520367 | 0.048637 |
| CARLA | 0.095255 | 0.010277 |
| KITTI Raw | 0.354249 | 0.042614 |
| PandaSet | 0.157593 | 0.107167 |
| Waymo | 0.314100 | 0.050856 |
| Mean | **0.406212** | **0.046548** |

RRE is the sum of absolute intrinsic ZYX Euler angles of the relative rotation in degrees. RTE is translation L2 distance in meters. Equal weighting across groups. The groups were used during development; this is not an independent annotation study.

[Machine-readable archived metrics](../configs/paper-20261026-results.json). The public CARLA example preserves the same20 pairs and full clouds. Other datasets require licensed raw data and their prepared annotations; this release does not redistribute those inputs or claim one-command reproduction of all ten groups.

[Legacy region-only results](legacy-results.md) remain separate and use a different rotation metric. The workbench now uses the paper C++ solver.
