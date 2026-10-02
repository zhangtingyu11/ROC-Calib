# Results

ROC-Calib region objective, ten frozen annotation groups. Errors are rotation angle and translation L2 distance.

| Group | Pairs | Rotation (°) | Translation (cm) |
| --- | ---: | ---: | ---: |
| KITTI Odometry 09 left | 20 | 0.2816 | 3.2029 |
| KITTI Odometry 09 right | 20 | 0.0339 | 2.1304 |
| KITTI Odometry 10 left | 20 | 0.3144 | 3.6882 |
| KITTI Odometry 10 right | 20 | 0.2259 | 5.0433 |
| nuScenes test group1 | 30 | 0.7197 | 6.0879 |
| nuScenes test group2 | 30 | 0.5464 | 7.2677 |
| CARLA static | 20 | 0.1113 | 1.6136 |
| KITTI Raw | 20 | 0.2559 | 4.6780 |
| PandaSet | 20 | 0.1256 | 9.3020 |
| Waymo validation | 20 | 0.2240 | 6.0349 |
| **Mean** | **220 total** | **0.2839** | **4.9049** |

Each group contributes equally to the mean. The total is 34,178 selected points. Configuration: [paper.json](../configs/paper.json). Machine-readable results: [results.json](../configs/results.json).

## Reproduce CARLA

Run the commands in the [demo](../demo/carla/README.md). The standalone package reproduces the saved CARLA matrix from the frozen object-derived PnP start. The expected matrix is read after fitting.

The remaining groups require licensed source data and the original prepared inputs. Their summary metrics are provided here; the CARLA command does not reproduce all ten groups.

## Scope

Frame and object selection are manual. Reference information was available during preparation of some original annotation groups. These results measure fitting on the fixed observations; they do not measure reference-blind human repeatability or annotation time.

The objective uses all selected points, equal object weights and exact distances to foreground pixel cells. It has no contour-attraction term. A low region loss alone does not establish physical accuracy or a unique pose.

The later Cauchy experiment and the optional structural fitting tools are not included in this result table.
